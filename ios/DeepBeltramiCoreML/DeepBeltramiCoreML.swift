import CoreML
import Foundation

public struct DeepBeltramiCoreMLResult: Sendable {
    public let pixels: [Float]
    public let milliseconds: Double
    /// MLComputePlan's anticipated preferred-operation counts, not a hardware trace.
    public let plan: [String: Int]
    public let tileCount: Int
    public let modelName: String
    public let precision: String
}

public enum DeepBeltramiCoreMLError: Error, LocalizedError {
    case invalidImage, missingModel(String), invalidModel(String), nonFiniteOutput
    public var errorDescription: String? {
        switch self {
        case .invalidImage: return "Expected finite interleaved pixels with positive width/height and one or three channels."
        case .missingModel(let name): return "Core ML model not found: \(name)"
        case .invalidModel(let reason): return "Unsupported Core ML model: \(reason)"
        case .nonFiniteOutput: return "The network produced a non-finite value. No determinant regularization was applied."
        }
    }
}

/// A serialized, reusable native engine for the original five-stage network.
/// Image arrays are natural H,W interleaved; model tensors are native N,C,W,H.
@available(iOS 18.0, macOS 15.0, *)
public final class DeepBeltramiCoreMLEngine: @unchecked Sendable {
    public static let halo = 45
    public let sigma: Int
    public let computeUnits: MLComputeUnits
    public let modelName: String
    public let precision: String
    public let modelURL: URL
    private let model: MLModel
    private let configuration: MLModelConfiguration
    private let tileWidth: Int
    private let tileHeight: Int
    private let boundaryMasked: Bool
    private struct Companion {
        let model: MLModel
        let url: URL
        let width: Int
        let height: Int
        let boundaryMasked: Bool
    }
    private var companions: [Companion] = []
    private var qualityFallback: DeepBeltramiCoreMLEngine?
    private let executionLock = NSLock()
    private var preferredPlan: [String: Int] = [:]
    private var companionPlans: [String: [String: Int]] = [:]

    private struct AxisRegion {
        let origin: Int
        let lower: Int
        let upper: Int
    }

    private static func regions(length: Int, tile: Int) -> [AxisRegion] {
        guard length > tile else { return [AxisRegion(origin: 0, lower: 0, upper: length)] }
        let advance = tile-2*halo, last = length-tile
        var origins = [0]
        while origins.last! < last { origins.append(min(origins.last!+advance,last)) }
        // True image edges do not need an artificial halo. Every other core
        // starts 45 pixels into its window and ends within the previous valid
        // margin. Adjacent cores cover the image exactly once.
        return origins.enumerated().map { index,origin in
            AxisRegion(origin: origin, lower: index == 0 ? 0 : origin+halo,
                       upper: index+1 == origins.count ? length : origins[index+1]+halo)
        }
    }

    /// `precision` can select fp32 or mixed16 when both variants share a directory.
    /// A directory containing only one variant selects it automatically.
    public init(modelDirectory: URL, sigma: Int = 25,
                computeUnits: MLComputeUnits = .cpuAndGPU,
                precision requestedPrecision: String? = nil) throws {
        let factors: [Int: Float] = [5:3,10:2,15:1,20:1,25:1,30:1,40:1,50:0.5]
        guard factors[sigma] != nil else { throw DeepBeltramiCoreMLError.invalidModel("No checkpoint for sigma \(sigma)") }
        self.sigma = sigma
        self.computeUnits = computeUnits
        let stem = "DeepBeltrami_sigma\(sigma)_"
        var candidates: [URL] = []
        if modelDirectory.pathExtension == "mlmodelc" || modelDirectory.pathExtension == "mlpackage" {
            candidates = [modelDirectory]
        } else if let enumerator = FileManager.default.enumerator(at: modelDirectory, includingPropertiesForKeys: nil) {
            for case let url as URL in enumerator {
                if (url.pathExtension == "mlmodelc" || url.pathExtension == "mlpackage") && url.lastPathComponent.hasPrefix(stem) {
                    candidates.append(url)
                    enumerator.skipDescendants()
                }
            }
        }
        let allCandidates = candidates
        if let requestedPrecision {
            let exact = candidates.filter { $0.deletingPathExtension().lastPathComponent == stem + requestedPrecision }
            candidates = exact.isEmpty ? candidates.filter {
                $0.deletingPathExtension().lastPathComponent.hasPrefix(stem + requestedPrecision + "_fixed") ||
                $0.deletingPathExtension().lastPathComponent.hasPrefix(stem + requestedPrecision + "_masked")
            } : exact
        }
        candidates.sort { a,b in
            let aFixed = a.lastPathComponent.contains("_fixed")
            let bFixed = b.lastPathComponent.contains("_fixed")
            if aFixed != bFixed { return !aFixed }
            let aPreferred = a.lastPathComponent.contains("_fp32.")
            let bPreferred = b.lastPathComponent.contains("_fp32.")
            return aPreferred != bPreferred ? aPreferred : a.lastPathComponent < b.lastPathComponent
        }
        guard var selected = candidates.first else { throw DeepBeltramiCoreMLError.missingModel(stem + (requestedPrecision ?? "fp32")) }
        let sourceURL = selected
        if selected.pathExtension == "mlpackage" { selected = try MLModel.compileModel(at: selected) }
        modelURL = selected
        modelName = selected.deletingPathExtension().lastPathComponent
        let config = MLModelConfiguration()
        config.computeUnits = computeUnits
        config.allowLowPrecisionAccumulationOnGPU = false
        var hints = MLOptimizationHints()
        hints.reshapeFrequency = .infrequent
        hints.specializationStrategy = .fastPrediction
        config.optimizationHints = hints
        configuration = config
        model = try MLModel(contentsOf: selected, configuration: config)
        let metadata = model.modelDescription.metadata[.creatorDefinedKey] as? [String: String] ?? [:]
        precision = metadata["precision"] ?? requestedPrecision ?? (modelName.contains("mixed16") ? "mixed16" : "fp32")
        boundaryMasked = metadata["boundary_mode"] == "masked_rectangle"
        guard metadata["offset_external"] != nil else {
            throw DeepBeltramiCoreMLError.invalidModel("Expected a raw network with external grayscale offset/clipping")
        }
        guard let input = model.modelDescription.inputDescriptionsByName["noisy"]?.multiArrayConstraint,
              input.shape.count == 4, input.dataType == .float32 else {
            throw DeepBeltramiCoreMLError.invalidModel("Expected float32 noisy input in N,C,W,H")
        }
        tileWidth = input.shape[2].intValue
        tileHeight = input.shape[3].intValue
        guard tileWidth > 2*Self.halo && tileHeight > 2*Self.halo else {
            throw DeepBeltramiCoreMLError.invalidModel("Default tile is too small for the 45-pixel halo")
        }
        let permitsNeuralEngine = computeUnits == .all || computeUnits == .cpuAndNeuralEngine
        // A14/iOS18.0.1 produced incorrect finite values for a partial rectangle
        // in the masked512 ANE path. Keep large masked tiles GPU/CPU-only until
        // that backend has passed device validation; masked256 hybrid is valid.
        if permitsNeuralEngine && boundaryMasked && max(tileWidth,tileHeight) > 256 {
            throw DeepBeltramiCoreMLError.invalidModel("Masked tiles above 256 require CPU/GPU compute units; use the 256 tile for Neural Engine execution")
        }
        // Fixed companions retain static shapes for the accelerator compiler.
        // A real image rectangle smaller than the tile uses the flexible model;
        // extending it once would change the network's per-stage boundaries.
        for url in allCandidates where url != sourceURL &&
            (url.deletingPathExtension().lastPathComponent.hasPrefix(stem + precision + "_fixed") ||
             url.deletingPathExtension().lastPathComponent.hasPrefix(stem + precision + "_masked")) {
            let compiled = url.pathExtension == "mlpackage" ? try MLModel.compileModel(at: url) : url
            let candidate = try MLModel(contentsOf: compiled, configuration: config)
            guard let constraint = candidate.modelDescription.inputDescriptionsByName["noisy"]?.multiArrayConstraint,
                  constraint.shape.count == 4, constraint.dataType == .float32 else { continue }
            let width = constraint.shape[2].intValue, height = constraint.shape[3].intValue
            guard width > 2*Self.halo && height > 2*Self.halo else { continue }
            let metadata = candidate.modelDescription.metadata[.creatorDefinedKey] as? [String: String] ?? [:]
            let masked = metadata["boundary_mode"] == "masked_rectangle"
            if permitsNeuralEngine && masked && max(width,height) > 256 { continue }
            companions.append(Companion(model: candidate, url: compiled, width: width, height: height,
                                        boundaryMasked: masked))
        }
        companions.sort { $0.width*$0.height > $1.width*$1.height }
        if precision == "mixed16",
           let qualityURL = allCandidates.first(where: { $0.deletingPathExtension().lastPathComponent == stem + "fp32" }) {
            // Full-size tiny rectangles are faster and more accurate in FP32
            // than evaluating the unused area of a 256-pixel half tile.
            let units: MLComputeUnits = computeUnits == .cpuOnly || computeUnits == .cpuAndNeuralEngine ? .cpuOnly : .cpuAndGPU
            qualityFallback = try DeepBeltramiCoreMLEngine(modelDirectory: qualityURL, sigma: sigma,
                                                         computeUnits: units, precision: "fp32")
        }
    }

    /// Return anticipated preferred device counts from Core ML's loaded plan.
    /// Actual execution dispatch must be established separately with Instruments.
    public func computePlan() async throws -> [String: Int] {
        func deviceName(_ device: MLComputeDevice) -> String {
            switch device {
            case .cpu: return "cpu"
            case .gpu: return "gpu"
            case .neuralEngine: return "neuralEngine"
            @unknown default: return "unknown"
            }
        }
        func countsFor(_ url: URL) async throws -> [String: Int] {
            let plan = try await MLComputePlan.load(contentsOf: url, configuration: configuration)
            var counts: [String: Int] = [:]
            func visit(_ block: MLModelStructure.Program.Block) {
                for operation in block.operations {
                    let key = plan.deviceUsage(for: operation).map { deviceName($0.preferred) } ?? "unknown"
                    counts[key, default: 0] += 1
                    for child in operation.blocks { visit(child) }
                }
            }
            if case .program(let program) = plan.modelStructure {
                for (_, function) in program.functions { visit(function.block) }
            }
            return counts
        }
        let counts = try await countsFor(modelURL)
        var otherPlans: [String: [String: Int]] = [:]
        for companion in companions {
            otherPlans[companion.url.path] = try await countsFor(companion.url)
        }
        if let qualityFallback { _ = try await qualityFallback.computePlan() }
        executionLock.withLock {
            preferredPlan = counts
            companionPlans = otherPlans
        }
        return counts
    }

    public func denoiseAsync(pixels: [Float], width: Int, height: Int, channels: Int = 1) async throws -> DeepBeltramiCoreMLResult {
        try await Task.detached(priority: .userInitiated) {
            try self.denoise(pixels: pixels, width: width, height: height, channels: channels)
        }.value
    }

    public func denoise(pixels: [Float], width: Int, height: Int, channels: Int = 1) throws -> DeepBeltramiCoreMLResult {
        try executionLock.withLock {
            guard width > 0, height > 0, channels == 1 || channels == 3,
                  width <= Int.max / height, width*height <= Int.max / channels,
                  pixels.count == width*height*channels, pixels.allSatisfy({ $0.isFinite }) else {
                throw DeepBeltramiCoreMLError.invalidImage
            }
            if let qualityFallback, width*height <= 160*160 || min(width,height) < 16 {
                return try qualityFallback.denoise(pixels: pixels, width: width, height: height, channels: channels)
            }
            let start = ContinuousClock.now
            let factors: [Int: Float] = [5:3,10:2,15:1,20:1,25:1,30:1,40:1,50:0.5]
            let offset: Float = channels == 1 ? Float(sigma)*factors[sigma]! : 0
            func tileWork(_ candidate: Companion) -> Int {
                let nx = Self.regions(length: width, tile: candidate.width).count
                let ny = Self.regions(length: height, tile: candidate.height).count
                return nx*ny*candidate.width*candidate.height
            }
            var options = companions.filter { $0.boundaryMasked || (width >= $0.width && height >= $0.height) }
            if boundaryMasked {
                options.append(Companion(model: model, url: modelURL, width: tileWidth, height: tileHeight, boundaryMasked: true))
            }
            let selected = options.min { tileWork($0) < tileWork($1) }
            let companion = selected?.url == modelURL ? nil : selected
            let activeModel = companion?.model ?? model
            let activeURL = companion?.url ?? modelURL
            let activeWidth = companion?.width ?? tileWidth, activeHeight = companion?.height ?? tileHeight
            let activeMasked = companion?.boundaryMasked ?? boundaryMasked
            let inputWidth = min(width,activeWidth), inputHeight = min(height,activeHeight)
            let columns = Self.regions(length: width, tile: activeWidth)
            let rows = Self.regions(length: height, tile: activeHeight)
            var result = [Float](repeating: 0, count: pixels.count)
            var tileCount = 0
            var boundaryInputs: [String: Any] = [:]
            if activeMasked {
                let mask = try MLMultiArray(shape: [1,1,NSNumber(value: activeWidth),NSNumber(value: activeHeight)], dataType: .float32)
                let pointer = mask.dataPointer.bindMemory(to: Float.self, capacity: mask.count)
                pointer.update(repeating: 0, count: mask.count)
                let strides = mask.strides.map { $0.intValue }
                for x in 0..<inputWidth {
                    for y in 0..<inputHeight { pointer[x*strides[2]+y*strides[3]] = 1 }
                }
                let ix = try MLMultiArray(shape: [NSNumber(value: activeWidth)], dataType: .int32)
                let iy = try MLMultiArray(shape: [NSNumber(value: activeHeight)], dataType: .int32)
                let px = ix.dataPointer.bindMemory(to: Int32.self, capacity: ix.count)
                let py = iy.dataPointer.bindMemory(to: Int32.self, capacity: iy.count)
                for x in 0..<activeWidth { px[x*ix.strides[0].intValue] = Int32(min(x,inputWidth-1)) }
                for y in 0..<activeHeight { py[y*iy.strides[0].intValue] = Int32(min(y,inputHeight-1)) }
                boundaryInputs = ["valid_mask":mask,"idx_x":ix,"idx_y":iy]
            }
            for channel in 0..<channels {
                for row in rows {
                    let coreY = row.lower, copyHeight = row.upper-row.lower, originY = row.origin
                    for column in columns {
                        let coreX = column.lower, copyWidth = column.upper-column.lower, originX = column.origin
                        let input = try MLMultiArray(shape: [1,1,NSNumber(value: activeMasked ? activeWidth : inputWidth),NSNumber(value: activeMasked ? activeHeight : inputHeight)], dataType: .float32)
                        let inputPointer = input.dataPointer.bindMemory(to: Float.self,capacity: input.count)
                        if activeMasked { inputPointer.update(repeating: 0, count: input.count) }
                        let inputStrides = input.strides.map { $0.intValue }
                        for x in 0..<inputWidth {
                            for y in 0..<inputHeight {
                                inputPointer[x*inputStrides[2]+y*inputStrides[3]] = pixels[((originY+y)*width+originX+x)*channels+channel]+offset
                            }
                        }
                        var features = boundaryInputs
                        features["noisy"] = input
                        let provider = try MLDictionaryFeatureProvider(dictionary: features)
                        let prediction = try activeModel.prediction(from: provider)
                        guard let output = prediction.featureValue(for: "denoised")?.multiArrayValue, output.dataType == .float32 else {
                            throw DeepBeltramiCoreMLError.invalidModel("Expected denoised float32 output")
                        }
                        let outputPointer = output.dataPointer.bindMemory(to: Float.self,capacity: output.count)
                        let outputStrides = output.strides.map { $0.intValue }
                        for y in 0..<copyHeight {
                            for x in 0..<copyWidth {
                                let localX = coreX-originX+x, localY = coreY-originY+y
                                let value = outputPointer[localX*outputStrides[2]+localY*outputStrides[3]]-offset
                                guard value.isFinite else { throw DeepBeltramiCoreMLError.nonFiniteOutput }
                                result[((coreY+y)*width+coreX+x)*channels+channel] = min(255,max(0,value))
                            }
                        }
                        tileCount += 1
                    }
                }
            }
            let duration = start.duration(to: .now).components
            let milliseconds = Double(duration.seconds)*1000 + Double(duration.attoseconds)/1e15
            return DeepBeltramiCoreMLResult(pixels: result, milliseconds: milliseconds,
                                            plan: companion == nil ? preferredPlan : (companionPlans[activeURL.path] ?? [:]), tileCount: tileCount,
                                            modelName: activeURL.deletingPathExtension().lastPathComponent, precision: precision)
        }
    }
}

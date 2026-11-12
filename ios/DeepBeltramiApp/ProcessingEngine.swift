import Foundation
import CoreML
import Metal
import os

struct FilterOptions: Sendable {
    var sigma = 25
    var beta: Float = 800
    var alpha: Float = 0.21
    var iterations = 40
    var radius = 2
    var precision = "fp32"
    var compute = "gpu"
    var memoryBudgetBytes = 192 * 1024 * 1024
}

struct FilterStatistics: Codable, Sendable {
    var wallMilliseconds: Double
    var gpuMilliseconds: Double?
    var tileCount: Int
    var workspaceBytes: Int?
    var backend: String
    var precision: String
    var model: String?
    var anticipatedOperationDevices: [String: Int]?
}

struct FilteredRaster: Sendable {
    let image: ImageRaster
    let statistics: FilterStatistics
}

actor ProcessingEngine {
    private let signposter = OSSignposter(subsystem: Bundle.main.bundleIdentifier ?? "Beltrami", category: "Denoising")
    #if DEEP_BELTRAMI
    private var engines: [String: DeepBeltramiCoreMLEngine] = [:]
    #else
    private var engines: [Int: BeltramiMetal] = [:]
    #endif

    func run(_ image: ImageRaster, options: FilterOptions) async throws -> FilteredRaster {
        let interval = signposter.beginInterval("Denoise")
        defer { signposter.endInterval("Denoise", interval) }
        #if DEEP_BELTRAMI
        let key = "\(options.sigma)-\(options.precision)-\(options.compute)"
        let model: DeepBeltramiCoreMLEngine
        if let cached = engines[key] { model = cached }
        else {
            let units: MLComputeUnits
            switch options.compute { case "all": units = .all; case "ane": units = .cpuAndNeuralEngine; case "cpu": units = .cpuOnly; default: units = .cpuAndGPU }
            let directory = Bundle.main.bundleURL.appendingPathComponent("ModelAssets", isDirectory: true)
            model = try DeepBeltramiCoreMLEngine(modelDirectory: directory, sigma: options.sigma,
                                               computeUnits: units, precision: options.precision == "mixed" ? "mixed16" : "fp32")
            let arguments = ProcessInfo.processInfo.arguments
            if arguments.contains("--benchmark") || arguments.contains("--validate") {
                _ = try? await model.computePlan()
            }
            if engines.count >= 2 { engines.removeAll() }
            engines[key] = model
        }
        let result = try model.denoise(pixels: image.pixels, width: image.width, height: image.height, channels: image.channels)
        return FilteredRaster(image: ImageRaster(width: image.width, height: image.height, channels: image.channels, pixels: result.pixels),
            statistics: FilterStatistics(wallMilliseconds: result.milliseconds, tileCount: result.tileCount,
                                         backend: options.compute, precision: result.precision, model: result.modelName,
                                         anticipatedOperationDevices: result.plan))
        #else
        if engines[options.memoryBudgetBytes] == nil { engines[options.memoryBudgetBytes] = try BeltramiMetal(memoryBudgetBytes: options.memoryBudgetBytes) }
        let result = try engines[options.memoryBudgetBytes]!.denoise(image.pixels, width: image.width, height: image.height, channels: image.channels,
                                        beta: options.beta, alpha: options.alpha, iterations: options.iterations, radius: options.radius)
        return FilteredRaster(image: ImageRaster(width: image.width, height: image.height, channels: image.channels, pixels: result.pixels),
            statistics: FilterStatistics(wallMilliseconds: result.wallMilliseconds, gpuMilliseconds: result.gpuMilliseconds,
                                         tileCount: result.tileCount, workspaceBytes: result.temporaryBytes,
                                         backend: "Metal", precision: "float32"))
        #endif
    }
}

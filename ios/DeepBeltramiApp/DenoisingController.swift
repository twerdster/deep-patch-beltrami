import Foundation
import UIKit
import SwiftUI

struct DisplayMetrics {
    let time: String
    let quality: String
    let method: String
}

@MainActor
final class DenoisingController: ObservableObject {
    @Published var sigma = 25
    @Published var iterations = 40
    @Published var radius = 2
    @Published var beta: Double = 800
    @Published var precision = "mixed"
    @Published var compute = "all"
    @Published var isProcessing = false
    @Published var inputImage: UIImage?
    @Published var outputImage: UIImage?
    @Published var metrics: DisplayMetrics?
    @Published var error: String?
    private let engine = ProcessingEngine()
    private var input: ImageRaster?
    private var clean: ImageRaster?
    private var started = false
    private var selectedPhoto: UIImage?

    #if DEEP_BELTRAMI
    let title = "Deep Patch\nBeltrami"
    let footer = "Original five-stage learned model · Thesis §4.5\nProcessed on your iPhone, at full image resolution."
    #else
    let title = "Patch Beltrami"
    let footer = "Patch-space geometric flow · Metal GPU\nProcessed on your iPhone, at full image resolution."
    #endif

    var backendLabel: String {
        #if DEEP_BELTRAMI
        return precision == "fp32" ? "Core ML · Float32" : "Core ML · Mixed precision"
        #else
        return "Metal · Float32"
        #endif
    }
    var dimensions: String { input.map { "\($0.width) × \($0.height) · \($0.channels == 1 ? "Gray" : "RGB")" } ?? "" }
    var aspectRatio: Double { input.map { Double($0.width) / Double($0.height) } ?? 1.2 }
    var options: FilterOptions { FilterOptions(sigma: sigma, beta: Float(beta), iterations: iterations, radius: radius, precision: precision, compute: compute) }

    init() { loadSample() }

    func loadSample() {
        guard !isProcessing else { return }
        selectedPhoto = nil
        let sample = ImageRaster.sample(sigma: Float(sigma))
        clean = sample.clean; input = sample.noisy
        inputImage = sample.noisy.image(); outputImage = nil; metrics = nil; error = nil
    }
    func noiseChanged() { if selectedPhoto == nil { loadSample() } }
    func loadPhoto(_ image: UIImage) {
        guard !isProcessing else { return }
        do {
            let raster = try ImageRaster.load(image)
            selectedPhoto = image; input = raster; clean = nil
            inputImage = raster.image(); outputImage = nil; metrics = nil; error = nil
        } catch { self.error = error.localizedDescription }
    }
    func process() {
        guard !isProcessing, let input else { return }
        let settings = options
        isProcessing = true; error = nil
        Task {
            do {
                let filtered = try await engine.run(input, options: settings)
                outputImage = filtered.image.image()
                let quality = clean.flatMap { $0.psnr(comparedTo: filtered.image) }.map { String(format: "%.2f dB", $0) } ?? "Full resolution"
                metrics = DisplayMetrics(time: String(format: "%.1f ms", filtered.statistics.wallMilliseconds), quality: quality,
                                         method: "\(filtered.statistics.tileCount) tile\(filtered.statistics.tileCount == 1 ? "" : "s")")
                if selectedPhoto == nil { try savePreview(input: input, output: filtered.image, statistics: filtered.statistics) }
                isProcessing = false
                if ProcessInfo.processInfo.arguments.contains("--screenshot") { captureAfterLayout() }
            } catch { self.error = error.localizedDescription; isProcessing = false; print("DENOISING_ERROR \(error)") }
        }
    }
    func startIfRequested() {
        guard !started else { return }; started = true
        let arguments = ProcessInfo.processInfo.arguments
        if arguments.contains("--check-image-io") {
            do {
                let checks = try ImageRaster.imageIOChecks()
                try JSONSerialization.data(withJSONObject: checks, options: [.prettyPrinted, .sortedKeys])
                    .write(to: documents.appendingPathComponent("image-io-checks.json"))
                print("IMAGE_IO_CHECKS_PASSED")
                exit(0)
            } catch { print("IMAGE_IO_CHECKS_FAILED \(error)"); exit(1) }
        }
        if let i = arguments.firstIndex(of: "--compute"), i+1 < arguments.count { compute = arguments[i+1] }
        if let i = arguments.firstIndex(of: "--precision"), i+1 < arguments.count { precision = arguments[i+1] }
        if arguments.contains("--benchmark") || arguments.contains("--validate") {
            isProcessing = true
            Task { await benchmark(validateOnly: arguments.contains("--validate")) }
        } else if arguments.contains("--auto-run") { process() }
    }

    private var documents: URL { FileManager.default.urls(for: .documentDirectory, in: .userDomainMask)[0] }
    private func savePreview(input: ImageRaster, output: ImageRaster, statistics: FilterStatistics) throws {
        try output.writeRaw(to: documents.appendingPathComponent("demo-output.f32"))
        try input.writeRaw(to: documents.appendingPathComponent("demo-input.f32"))
        try input.image().pngData()?.write(to: documents.appendingPathComponent("demo-input.png"))
        try output.image().pngData()?.write(to: documents.appendingPathComponent("demo-output.png"))
        if let clean { try clean.image().pngData()?.write(to: documents.appendingPathComponent("demo-clean.png")); try clean.writeRaw(to: documents.appendingPathComponent("demo-clean.f32")) }
        var record: [String: Any] = ["width": input.width, "height": input.height, "channels": input.channels,
                                     "sigma": sigma, "iterations": iterations, "radius": radius, "beta": beta,
                                     "statistics": try JSONSerialization.jsonObject(with: JSONEncoder().encode(statistics))]
        #if DEEP_BELTRAMI
        for key in ["iterations", "radius", "beta"] { record.removeValue(forKey: key) }
        record["requested_precision"] = precision; record["requested_compute"] = compute
        #endif
        if let clean, let noisyScore = clean.psnr(comparedTo: input), let filteredScore = clean.psnr(comparedTo: output) {
            record["noisy_psnr_db"] = noisyScore; record["denoised_psnr_db"] = filteredScore
        }
        try JSONSerialization.data(withJSONObject: record, options: [.prettyPrinted, .sortedKeys]).write(to: documents.appendingPathComponent("demo.json"))
    }
    private func captureAfterLayout() {
        Task {
            try? await Task.sleep(for: .seconds(1))
            guard let scene = UIApplication.shared.connectedScenes.first as? UIWindowScene,
                  let window = scene.windows.first(where: { $0.isKeyWindow }) else { return }
            let renderer = UIGraphicsImageRenderer(bounds: window.bounds)
            let image = renderer.image { _ in window.drawHierarchy(in: window.bounds, afterScreenUpdates: true) }
            try? image.pngData()?.write(to: documents.appendingPathComponent("device-screen.png"))
            print("SCREENSHOT_READY")
        }
    }

    private func benchmark(validateOnly: Bool) async {
        var records: [[String: Any]] = []
        let thermalStart = ProcessInfo.processInfo.thermalState.rawValue
        var fixtureList: [[String: Any]] = []
        let fixtureRoot = Bundle.main.bundleURL.appendingPathComponent("Fixtures", isDirectory: true)
        if let data = try? Data(contentsOf: fixtureRoot.appendingPathComponent("manifest.json")),
           let manifest = try? JSONSerialization.jsonObject(with: data) as? [String: Any],
           let cases = manifest["cases"] as? [[String: Any]] { fixtureList = cases }
        if !validateOnly {
            for (width, height, channels) in [(128,128,1),(256,256,1),(385,257,1),(512,512,1),(1024,768,1),(257,193,3)] {
                fixtureList.append(["id": "performance-\(width)x\(height)c\(channels)", "width": width, "height": height, "channels": channels])
            }
        }
        #if !DEEP_BELTRAMI
        if !validateOnly {
            fixtureList.append(["id":"performance-2048x1536c1", "width":2048, "height":1536, "channels":1, "iterations":40])
            fixtureList.append(["id":"performance-4032x3024c1", "width":4032, "height":3024, "channels":1, "iterations":5])
        }
        #endif
        let arguments = ProcessInfo.processInfo.arguments
        if let index = arguments.firstIndex(of: "--case"), index + 1 < arguments.count {
            let selected = Set(arguments[index + 1].split(separator: ",").map(String.init))
            fixtureList = fixtureList.filter { selected.contains($0["id"] as? String ?? "") }
        }
        if validateOnly && fixtureList.isEmpty {
            error = "No validation fixtures were selected. Generate fixtures, rebuild the app, and check the requested case IDs."
            isProcessing = false
            print("VALIDATION_ERROR no selected fixtures")
            if !arguments.contains("--stay-open") { exit(2) }
            return
        }
        var repeatCount = validateOnly ? 1 : 3
        if let index = arguments.firstIndex(of: "--repeats"), index + 1 < arguments.count,
           let count = Int(arguments[index + 1]), (1...1000).contains(count) { repeatCount = count }
        for fixture in fixtureList {
            guard let width = fixture["width"] as? Int, let height = fixture["height"] as? Int else { continue }
            let channels = fixture["channels"] as? Int ?? 1
            let name = fixture["id"] as? String ?? "\(width)x\(height)"
            do {
                var settings = options
                if let v = fixture["sigma"] as? Int { settings.sigma = v }
                if let v = fixture["beta"] as? Double { settings.beta = Float(v) }
                if let v = fixture["alpha"] as? Double { settings.alpha = Float(v) }
                if let v = fixture["iterations"] as? Int { settings.iterations = v }
                if let v = fixture["radius"] as? Int { settings.radius = v }
                if fixture["forceTiling"] as? Bool == true { settings.memoryBudgetBytes = 65536 }
                let raster: ImageRaster
                let expected: ImageRaster?
                if let source = fixture["input"] as? [String: Any], let filename = source["file"] as? String {
                    let folder = fixtureRoot.appendingPathComponent(name)
                    raster = try ImageRaster.readRaw(folder.appendingPathComponent(filename), width: width, height: height, channels: channels)
                    expected = try ImageRaster.readRaw(folder.appendingPathComponent("expected.f32"), width: width, height: height, channels: channels)
                } else {
                    raster = ImageRaster.sample(width: width, height: height, channels: channels, sigma: Float(settings.sigma)).noisy
                    expected = nil
                }
                let cold = try await engine.run(raster, options: settings)
                var result = cold
                var times: [Double] = []
                var gpuTimes: [Double] = []
                for _ in 0..<repeatCount {
                    result = try await engine.run(raster, options: settings)
                    times.append(result.statistics.wallMilliseconds)
                    if let gpu = result.statistics.gpuMilliseconds { gpuTimes.append(gpu) }
                }
                let outputName = "\(name)-\(settings.precision)-\(settings.compute).f32"
                try result.image.writeRaw(to: documents.appendingPathComponent(outputName))
                var row: [String: Any] = ["id": name, "width": width, "height": height, "channels": channels,
                                          "sigma": settings.sigma, "beta": settings.beta, "alpha": settings.alpha,
                                          "iterations": settings.iterations, "radius": settings.radius,
                                          "cold_ms": cold.statistics.wallMilliseconds, "warm_ms": times,
                                          "median_ms": times.sorted()[times.count/2], "gpu_ms": gpuTimes,
                                          "finite": result.image.pixels.allSatisfy({ $0.isFinite }), "output_file": outputName,
                                          "statistics": try JSONSerialization.jsonObject(with: JSONEncoder().encode(result.statistics))]
                #if DEEP_BELTRAMI
                for key in ["iterations", "radius", "beta", "alpha"] { row.removeValue(forKey: key) }
                #endif
                if let expected {
                    var maximum = 0.0; var squared = 0.0
                    for i in expected.pixels.indices { let d = Double(result.image.pixels[i] - expected.pixels[i]); maximum = max(maximum,abs(d)); squared += d*d }
                    let rms = sqrt(squared/Double(expected.pixels.count))
                    row["max_absolute_error"] = maximum; row["rms_error"] = rms
                    #if DEEP_BELTRAMI
                    let maxTolerance = settings.precision == "fp32" ? 0.05 : 1.0
                    let rmsTolerance = settings.precision == "fp32" ? 0.01 : 0.05
                    #else
                    let maxTolerance = 0.01, rmsTolerance = 0.001
                    #endif
                    let passed = maximum <= maxTolerance && rms <= rmsTolerance && result.image.pixels.allSatisfy({ $0.isFinite })
                    row["max_tolerance"] = maxTolerance; row["rms_tolerance"] = rmsTolerance
                    row["reference_passed"] = passed
                    if !passed { row["error"] = "Numerical reference tolerance exceeded" }
                }
                if row["finite"] as? Bool != true { row["error"] = "Non-finite output" }
                records.append(row)
                print("CASE \(name) \(row["median_ms"]!) ms")
            } catch { records.append(["id": name, "width": width, "height": height, "channels": channels, "error": error.localizedDescription]); print("CASE_ERROR \(name): \(error)") }
            let report: [String: Any] = ["schema": 1, "platform": "physical iOS device", "os_version": UIDevice.current.systemVersion,
                                         "compute": compute, "precision": precision, "thermal_start": thermalStart,
                                         "thermal_current": ProcessInfo.processInfo.thermalState.rawValue,
                                         "low_power_mode": ProcessInfo.processInfo.isLowPowerModeEnabled, "records": records]
            if let data = try? JSONSerialization.data(withJSONObject: report, options: [.prettyPrinted,.sortedKeys]) {
                try? data.write(to: documents.appendingPathComponent("benchmark-\(precision)-\(compute).json"), options: .atomic)
            }
        }
        isProcessing = false
        print("BENCHMARK_COMPLETE \(records.count)")
        if !ProcessInfo.processInfo.arguments.contains("--stay-open") { exit(records.contains(where: { $0["error"] != nil }) ? 1 : 0) }
    }
}

import CoreML
import CryptoKit
import Foundation

enum BSD68RunError: Error, LocalizedError {
    case invalid(String)
    var errorDescription: String? {
        switch self { case .invalid(let reason): return reason }
    }
}

@main
struct RunBSD68CoreML {
    static let sigmas = [5, 10, 15, 20, 25, 30, 40, 50]
    static let published: [Int: Double] = [5:37.70, 10:33.52, 15:31.37, 20:29.92,
                                         25:28.87, 30:28.04, 40:26.81, 50:25.95]
    static let offsets: [Int: Double] = [5:15, 10:20, 15:15, 20:20, 25:25, 30:30, 40:40, 50:25]
    static let usage = """
    Usage: RunBSD68CoreML --models DIR --fixtures DIR --output FILE
           [--limit N] [--sigmas 5,10,15,20,25,30,40,50]
    Runs native Core ML FP32 with CPU+GPU allowed. --limit is per sigma.
    Only all eight noise levels with 68 images each is the full published suite.
    """

    struct Options {
        let models: URL
        let fixtures: URL
        let output: URL
        let selectedSigmas: [Int]
        let limit: Int?
    }

    static func parse() throws -> Options? {
        let args = Array(CommandLine.arguments.dropFirst())
        if args.contains("--help") || args.contains("-h") { print(usage); return nil }
        var values: [String: String] = [:]
        let allowed: Set<String> = ["--models", "--fixtures", "--output", "--limit", "--sigmas"]
        var i = 0
        while i < args.count {
            guard allowed.contains(args[i]), i+1 < args.count,
                  !args[i+1].hasPrefix("--"), values[args[i]] == nil else {
                throw BSD68RunError.invalid(usage)
            }
            values[args[i]] = args[i+1]; i += 2
        }
        guard let models = values["--models"], let fixtures = values["--fixtures"],
              let output = values["--output"] else { throw BSD68RunError.invalid(usage) }
        var selected = sigmas
        if let text = values["--sigmas"] {
            let parts = text.split(separator: ",", omittingEmptySubsequences: false)
            let parsed = parts.compactMap { Int($0) }
            guard !parsed.isEmpty, parsed.count == parts.count,
                  Set(parsed).count == parsed.count, parsed.allSatisfy({ sigmas.contains($0) }) else {
                throw BSD68RunError.invalid("--sigmas must contain unique supported comma-separated noise levels")
            }
            selected = sigmas.filter { parsed.contains($0) }
        }
        var limit: Int?
        if let text = values["--limit"] {
            guard let count = Int(text), count > 0 else { throw BSD68RunError.invalid("--limit must be positive") }
            limit = count
        }
        return Options(models: URL(fileURLWithPath: models), fixtures: URL(fileURLWithPath: fixtures),
                       output: URL(fileURLWithPath: output), selectedSigmas: selected, limit: limit)
    }

    static func sha(_ data: Data) -> String {
        SHA256.hash(data: data).map { String(format: "%02x", $0) }.joined()
    }

    static func floats(_ data: Data) throws -> [Float] {
        guard data.count % MemoryLayout<Float>.size == 0 else {
            throw BSD68RunError.invalid("Invalid float32 file length")
        }
        var result = [Float](repeating: 0, count: data.count/MemoryLayout<Float>.size)
        _ = result.withUnsafeMutableBytes { data.copyBytes(to: $0) }
        return result
    }

    static func relativeFile(_ value: Any?, root: URL) throws -> URL {
        guard let name = value as? String, !name.isEmpty, !name.hasPrefix("/"),
              !name.split(separator: "/").contains("..") else {
            throw BSD68RunError.invalid("Fixture paths must be relative")
        }
        return root.appendingPathComponent(name)
    }

    static func write(_ report: [String: Any], to output: URL) throws {
        try FileManager.default.createDirectory(at: output.deletingLastPathComponent(), withIntermediateDirectories: true)
        try JSONSerialization.data(withJSONObject: report, options: [.prettyPrinted, .sortedKeys])
            .write(to: output, options: .atomic)
    }

    static func run(_ options: Options) async throws {
        let manifestData = try Data(contentsOf: options.fixtures.appendingPathComponent("manifest.json"))
        guard let manifest = try JSONSerialization.jsonObject(with: manifestData) as? [String: Any],
              manifest["complete"] as? Bool == true,
              let allCases = manifest["cases"] as? [[String: Any]] else {
            throw BSD68RunError.invalid("Expected a completed BSD68 fixture manifest")
        }
        let full = options.limit == nil && options.selectedSigmas == sigmas
        var records: [[String: Any]] = [], means: [[String: Any]] = []
        var plans: [String: Any] = [:]
        var report: [String: Any] = [
            "schema": 1, "dataset": "bsd68", "backend": "native Core ML", "precision": "fp32",
            "compute_units": "cpuAndGPU", "offset_protocol": "archived", "checkpoint_step": 40000,
            "fixture_manifest_sha256": sha(manifestData), "selected_sigmas": options.selectedSigmas,
            "full_published_suite": false, "complete": false,
            "timing_caution": "One sequential validation including startup and shape effects; not a warmed benchmark.",
            "plan_caution": "Anticipated preferred operation devices, not a hardware execution trace."
        ]
        if let limit = options.limit { report["limit_per_sigma"] = limit }
        let started = Date()
        try write(report, to: options.output)
        do {
            for sigma in options.selectedSigmas {
                let candidates = allCases.filter { ($0["sigma"] as? Int) == sigma }
                guard !candidates.isEmpty, let expectedOffset = offsets[sigma] else {
                    throw BSD68RunError.invalid("Missing selected noise-level fixtures")
                }
                if full && candidates.count != 68 { throw BSD68RunError.invalid("Full suite requires 68 cases per noise level") }
                let group = options.limit.map { Array(candidates.prefix($0)) } ?? candidates
                let names = group.compactMap { $0["name"] as? String }
                guard names.count == group.count, Set(names).count == group.count else {
                    throw BSD68RunError.invalid("Missing or duplicate image names")
                }
                let engine = try DeepBeltramiCoreMLEngine(modelDirectory: options.models, sigma: sigma,
                                                        computeUnits: .cpuAndGPU, precision: "fp32")
                plans[String(sigma)] = try await engine.computePlan()
                let groupStart = Date()
                for item in group {
                    let record: [String: Any] = try autoreleasepool {
                        guard let name = item["name"] as? String,
                              name.range(of: "^[A-Za-z0-9_-]+$", options: .regularExpression) != nil,
                              let width = item["width"] as? Int,
                              let height = item["height"] as? Int, width > 0, height > 0,
                              width <= Int.max/height, let offset = item["offset"] as? Double,
                              offset == expectedOffset, let reference = item["reference_psnr"] as? Double else {
                            throw BSD68RunError.invalid("Invalid image metadata or archived offset")
                        }
                        let input = try Data(contentsOf: relativeFile(item["input"], root: options.fixtures))
                        guard sha(input) == item["input_natural_sha256"] as? String else {
                            throw BSD68RunError.invalid("Natural input SHA256 mismatch")
                        }
                        let pixels = try floats(input)
                        guard pixels.count == width*height, pixels.allSatisfy({ $0.isFinite }) else {
                            throw BSD68RunError.invalid("Invalid grayscale input dimensions or values")
                        }
                        var native = [Float](repeating: 0, count: pixels.count)
                        for x in 0..<width { for y in 0..<height { native[x*height+y] = pixels[y*width+x] } }
                        let nativeHash = sha(native.withUnsafeBytes { Data($0) })
                        guard nativeHash == item["noisy_native_sha256"] as? String else {
                            throw BSD68RunError.invalid("Original native input SHA256 mismatch")
                        }
                        let labelData = try Data(contentsOf: relativeFile(item["label"], root: options.fixtures))
                        guard sha(labelData) == item["label_natural_sha256"] as? String else {
                            throw BSD68RunError.invalid("Label SHA256 mismatch")
                        }
                        let label = try floats(labelData)
                        guard label.count == pixels.count, label.allSatisfy({ $0.isFinite }) else {
                            throw BSD68RunError.invalid("Invalid label dimensions or values")
                        }
                        let actual = try engine.denoise(pixels: pixels, width: width, height: height, channels: 1)
                        guard actual.pixels.count == label.count, actual.pixels.allSatisfy({ $0.isFinite }),
                              actual.modelName == "DeepBeltrami_sigma\(sigma)_fp32", actual.precision == "fp32" else {
                            throw BSD68RunError.invalid("Invalid prediction or incorrect selected model")
                        }
                        var squared = 0.0
                        for (value, target) in zip(actual.pixels, label) { let d = Double(value)-Double(target); squared += d*d }
                        let mse = squared/Double(label.count), score = 10*log10(255*255/mse)
                        guard mse.isFinite, mse > 0, score.isFinite else { throw BSD68RunError.invalid("Invalid MSE or PSNR") }
                        var row: [String: Any] = [
                            "name": name, "sigma": sigma, "width": width, "height": height, "channels": 1,
                            "offset": offset, "mse": mse, "psnr": score, "reference_psnr": reference,
                            "reference_delta_db": score-reference, "finite": true,
                            "noisy_native_sha256": nativeHash, "input_natural_sha256": sha(input),
                            "original_input_hash_verified": true, "label_hash_verified": true,
                            "tileCount": actual.tileCount, "milliseconds": actual.milliseconds,
                            "modelName": actual.modelName, "precision": actual.precision
                        ]
                        if let archived = item["archived_psnr"] as? Double {
                            row["archived_psnr"] = archived; row["archived_delta_db"] = score-archived
                        }
                        return row
                    }
                    records.append(record)
                    report["cases"] = records; report["means"] = means; report["plans"] = plans
                    try write(report, to: options.output)
                    print(String(decoding: try JSONSerialization.data(withJSONObject: record, options: .sortedKeys), as: UTF8.self))
                    fflush(stdout)
                }
                let subset = records.filter { ($0["sigma"] as? Int) == sigma }
                let mean = subset.reduce(0.0) { $0+($1["psnr"] as! Double) }/Double(subset.count)
                var summary: [String: Any] = ["sigma": sigma, "images": subset.count, "mean_psnr": mean,
                                             "seconds": Date().timeIntervalSince(groupStart)]
                if subset.count == 68, let target = published[sigma] {
                    summary["published_psnr"] = target
                    summary["matches_published_rounding"] = String(format: "%.2f", mean) == String(format: "%.2f", target)
                }
                means.append(summary); report["means"] = means
                try write(report, to: options.output)
                print("MEAN "+String(decoding: try JSONSerialization.data(withJSONObject: summary, options: .sortedKeys), as: UTF8.self))
                fflush(stdout)
            }
            let fullSuite = options.selectedSigmas == sigmas && records.count == 544 && means.count == 8 &&
                means.allSatisfy { $0["images"] as? Int == 68 }
            let matches = fullSuite && means.allSatisfy { $0["matches_published_rounding"] as? Bool == true }
            report["full_published_suite"] = fullSuite
            report["complete"] = true; report["all_finite"] = true
            report["all_original_input_hashes_verified"] = true
            report["matches_all_published_means"] = matches
            report["wall_seconds"] = Date().timeIntervalSince(started)
            try write(report, to: options.output)
            if fullSuite && !matches { throw BSD68RunError.invalid("Full run completed but published means did not all match at two decimals") }
            print("COMPLETE cases=\(records.count) full_published_suite=\(fullSuite)")
        } catch {
            report["validation_failed"] = true
            report["cases"] = records; report["means"] = means; report["plans"] = plans
            report["wall_seconds"] = Date().timeIntervalSince(started)
            try? write(report, to: options.output)
            throw error
        }
    }

    static func main() async {
        do { if let options = try parse() { try await run(options) } }
        catch {
            FileHandle.standardError.write(Data(("BSD68 validation failed: \(error.localizedDescription)\n").utf8))
            exit(1)
        }
    }
}

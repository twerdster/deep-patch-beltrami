import UIKit
import Foundation

struct ImageRaster: Sendable {
    let width: Int
    let height: Int
    let channels: Int
    var pixels: [Float]

    static func sample(width: Int = 384, height: Int = 320, channels: Int = 1, sigma: Float = 25) -> (clean: ImageRaster, noisy: ImageRaster) {
        var clean = [Float](repeating: 0, count: width * height * channels)
        var noisy = clean
        var rng = GaussianGenerator(seed: 2017)
        for y in 0..<height {
            for x in 0..<width {
                let row = Float(y) * 192 / Float(height)
                let col = Float(x) * 224 / Float(width)
                var value: Float = 105 + 35 * sin(row / 19) + 25 * cos(col / 23)
                if pow(row - 96, 2) + pow(col - 112, 2) < 1600 { value += 45 }
                for c in 0..<channels {
                    let v = channels == 1 ? value : min(245, max(10, value + 20 * sin(col / 32 + Float(c) * 2)))
                    let i = (y * width + x) * channels + c
                    clean[i] = v
                    noisy[i] = v + sigma * rng.next()
                }
            }
        }
        return (ImageRaster(width: width, height: height, channels: channels, pixels: clean),
                ImageRaster(width: width, height: height, channels: channels, pixels: noisy))
    }

    func image() -> UIImage {
        var rgba = [UInt8](repeating: 255, count: width * height * 4)
        for i in 0..<(width * height) {
            for c in 0..<3 {
                rgba[i * 4 + c] = UInt8(min(255, max(0, pixels[i * channels + (channels == 1 ? 0 : c)])).rounded())
            }
        }
        let data = Data(rgba) as CFData
        let provider = CGDataProvider(data: data)!
        let cg = CGImage(width: width, height: height, bitsPerComponent: 8, bitsPerPixel: 32,
                         bytesPerRow: width * 4, space: CGColorSpaceCreateDeviceRGB(),
                         bitmapInfo: CGBitmapInfo(rawValue: CGImageAlphaInfo.last.rawValue),
                         provider: provider, decode: nil, shouldInterpolate: false, intent: .defaultIntent)!
        return UIImage(cgImage: cg)
    }

    static func load(_ image: UIImage, maxDimension: Int? = nil) throws -> ImageRaster {
        // CGImage dimensions describe encoded pixels. UIImage's orientation
        // changes the displayed axes without changing that backing pixel grid.
        let sourceWidth: CGFloat
        let sourceHeight: CGFloat
        if let cg = image.cgImage {
            switch image.imageOrientation {
            case .left, .leftMirrored, .right, .rightMirrored:
                sourceWidth = CGFloat(cg.height); sourceHeight = CGFloat(cg.width)
            default:
                sourceWidth = CGFloat(cg.width); sourceHeight = CGFloat(cg.height)
            }
        } else {
            sourceWidth = image.size.width * image.scale
            sourceHeight = image.size.height * image.scale
        }
        guard sourceWidth.isFinite, sourceHeight.isFinite, sourceWidth > 0, sourceHeight > 0,
              (maxDimension ?? 1) > 0 else { throw RasterError.invalidImage }
        let scale = maxDimension.map { min(1, CGFloat($0) / max(sourceWidth, sourceHeight)) } ?? 1
        let width = max(1, Int((sourceWidth * scale).rounded()))
        let height = max(1, Int((sourceHeight * scale).rounded()))
        let format = UIGraphicsImageRendererFormat()
        format.scale = 1
        format.opaque = true
        let upright = UIGraphicsImageRenderer(size: CGSize(width: width, height: height), format: format).image { _ in
            UIColor.white.setFill()
            UIRectFill(CGRect(x: 0, y: 0, width: width, height: height))
            image.draw(in: CGRect(x: 0, y: 0, width: width, height: height))
        }
        var bytes = [UInt8](repeating: 255, count: width * height * 4)
        guard let cg = upright.cgImage,
              let context = CGContext(data: &bytes, width: width, height: height, bitsPerComponent: 8,
                                      bytesPerRow: width * 4, space: CGColorSpaceCreateDeviceRGB(),
                                      bitmapInfo: CGImageAlphaInfo.premultipliedLast.rawValue) else { throw RasterError.invalidImage }
        context.draw(cg, in: CGRect(x: 0, y: 0, width: width, height: height))
        var values = [Float](repeating: 0, count: width * height * 3)
        for i in 0..<(width * height) { for c in 0..<3 { values[i * 3 + c] = Float(bytes[i * 4 + c]) } }
        return ImageRaster(width: width, height: height, channels: 3, pixels: values)
    }

    /// Device-side checks for all EXIF-style orientations, scale, axes and RGB.
    /// Every check compares all six colored pixels, then verifies a preview
    /// round trip. It throws on a dimension or channel/corner mismatch.
    static func imageIOChecks() throws -> [String: Any] {
        let colors: [[Float]] = [[255,0,0], [0,255,0], [0,0,255],
                                 [255,255,0], [255,0,255], [0,255,255]]
        let source = ImageRaster(width: 3, height: 2, channels: 3, pixels: colors.flatMap { $0 })
        guard let cg = source.image().cgImage else { throw RasterError.invalidImage }
        let cases: [(String, UIImage.Orientation, Int, Int, [Int])] = [
            ("up", .up, 3, 2, [0,1,2,3,4,5]),
            ("down", .down, 3, 2, [5,4,3,2,1,0]),
            ("left", .left, 2, 3, [2,5,1,4,0,3]),
            ("right", .right, 2, 3, [3,0,4,1,5,2]),
            ("upMirrored", .upMirrored, 3, 2, [2,1,0,5,4,3]),
            ("downMirrored", .downMirrored, 3, 2, [3,4,5,0,1,2]),
            ("leftMirrored", .leftMirrored, 2, 3, [0,3,1,4,2,5]),
            ("rightMirrored", .rightMirrored, 2, 3, [5,2,4,1,3,0])
        ]
        var records: [[String: Any]] = []
        for (name, orientation, width, height, order) in cases {
            for scale in [CGFloat(1), CGFloat(2)] {
                let oriented = UIImage(cgImage: cg, scale: scale, orientation: orientation)
                let raster = try load(oriented)
                guard raster.width == width, raster.height == height, raster.channels == 3 else {
                    throw RasterError.imageCheck("\(name), scale \(scale): incorrect oriented dimensions")
                }
                let expected = order.flatMap { colors[$0] }
                var maximum: Float = 0
                for i in expected.indices { maximum = max(maximum, abs(raster.pixels[i]-expected[i])) }
                guard maximum <= 2 else {
                    throw RasterError.imageCheck("\(name), scale \(scale): incorrect RGB pixel mapping (error \(maximum))")
                }
                let roundTrip = try load(raster.image())
                guard roundTrip.width == width, roundTrip.height == height else {
                    throw RasterError.imageCheck("\(name): preview round-trip dimensions changed")
                }
                var roundTripError: Float = 0
                for i in raster.pixels.indices { roundTripError = max(roundTripError, abs(roundTrip.pixels[i]-raster.pixels[i])) }
                guard roundTripError <= 2 else {
                    throw RasterError.imageCheck("\(name): preview round-trip pixels changed")
                }
                records.append(["orientation": name, "scale": Double(scale), "width": width,
                                "height": height, "max_absolute_error": maximum,
                                "roundtrip_max_absolute_error": roundTripError, "passed": true])
            }
        }
        return ["schema": 1, "passed": true, "orientations": cases.count,
                "checks": records.count, "channel_tolerance": 2, "cases": records]
    }

    func psnr(comparedTo other: ImageRaster) -> Double? {
        guard pixels.count == other.pixels.count else { return nil }
        var mse = 0.0
        for i in pixels.indices {
            let d = Double(pixels[i]) - Double(other.pixels[i])
            mse += d * d
        }
        mse /= Double(pixels.count)
        return mse > 0 ? 10 * log10(255 * 255 / mse) : nil
    }

    func writeRaw(to url: URL) throws {
        try pixels.withUnsafeBufferPointer { pointer in
            try Data(buffer: pointer).write(to: url, options: .atomic)
        }
    }

    static func readRaw(_ url: URL, width: Int, height: Int, channels: Int) throws -> ImageRaster {
        let bytes = try Data(contentsOf: url)
        guard bytes.count == width * height * channels * MemoryLayout<Float>.size else { throw RasterError.invalidImage }
        var values = [Float](repeating: 0, count: width * height * channels)
        _ = values.withUnsafeMutableBytes { bytes.copyBytes(to: $0) }
        return ImageRaster(width: width, height: height, channels: channels, pixels: values)
    }
}

enum RasterError: LocalizedError {
    case invalidImage
    case imageCheck(String)
    var errorDescription: String? {
        switch self {
        case .invalidImage: return "The image dimensions or pixel data are invalid."
        case .imageCheck(let reason): return "Image I/O check failed: \(reason)"
        }
    }
}

private struct GaussianGenerator {
    var state: UInt64
    var spare: Float?
    init(seed: UInt64) { state = seed }
    mutating func uniform() -> Double {
        state = state &* 6364136223846793005 &+ 1442695040888963407
        return (Double(state >> 11) + 0.5) / 9007199254740992.0
    }
    mutating func next() -> Float {
        if let v = spare { spare = nil; return v }
        let r = sqrt(-2 * log(uniform()))
        let angle = 2 * Double.pi * uniform()
        spare = Float(r * sin(angle))
        return Float(r * cos(angle))
    }
}

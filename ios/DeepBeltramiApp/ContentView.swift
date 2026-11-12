import SwiftUI
import PhotosUI

struct ContentView: View {
    @StateObject private var controller = DenoisingController()
    @State private var photo: PhotosPickerItem?
    @State private var split = 0.5
    @State private var showAdvanced = false
    private let accent = Color(red: 0.39, green: 0.91, blue: 0.74)

    var body: some View {
        ScrollView {
            VStack(alignment: .leading, spacing: 20) {
                HStack(alignment: .top) {
                    VStack(alignment: .leading, spacing: 6) {
                        Text(controller.title).font(.system(size: 30, weight: .bold, design: .rounded))
                        Text("Geometric image denoising").font(.subheadline).foregroundStyle(.secondary)
                    }
                    Spacer()
                    Image(systemName: "waveform.path.ecg").font(.title).foregroundStyle(accent).padding(.top, 5)
                }
                HStack(spacing: 8) {
                    Label(controller.backendLabel, systemImage: "cpu").font(.caption.weight(.semibold))
                    Spacer()
                    Text(controller.dimensions).font(.caption.monospacedDigit()).foregroundStyle(.secondary)
                }
                .padding(12).background(Color.white.opacity(0.06), in: RoundedRectangle(cornerRadius: 14))
                comparison
                if let metrics = controller.metrics {
                    HStack(spacing: 12) {
                        stat("FILTER TIME", metrics.time)
                        stat("QUALITY", metrics.quality)
                        stat("PROCESSING", metrics.method)
                    }
                }
                VStack(alignment: .leading, spacing: 14) {
                    HStack {
                        Text("Noise level").font(.headline)
                        Spacer()
                        Text("σ \(controller.sigma)").font(.headline.monospacedDigit()).foregroundStyle(accent)
                    }
                    Picker("Noise level", selection: $controller.sigma) {
                        ForEach([5, 10, 15, 20, 25, 30, 40, 50], id: \.self) { Text("\($0)").tag($0) }
                    }.pickerStyle(.segmented)
                    #if DEEP_BELTRAMI
                    Picker("Precision", selection: $controller.precision) {
                        Text("Accurate FP32").tag("fp32")
                        Text("Fast mixed").tag("mixed")
                    }.pickerStyle(.segmented)
                    Picker("Compute", selection: $controller.compute) {
                        Text("Automatic").tag("all")
                        Text("CPU + GPU").tag("gpu")
                        Text("CPU + ANE").tag("ane")
                        Text("CPU").tag("cpu")
                    }.font(.subheadline)
                    #else
                    HStack {
                        Text("Iterations").font(.subheadline)
                        Spacer()
                        Picker("Iterations", selection: $controller.iterations) {
                            ForEach([10, 25, 40, 50, 100, 150], id: \.self) { Text("\($0)").tag($0) }
                        }.pickerStyle(.menu)
                    }
                    DisclosureGroup("Flow settings", isExpanded: $showAdvanced) {
                        VStack(alignment: .leading) {
                            Picker("Patch radius", selection: $controller.radius) {
                                Text("Local").tag(0); Text("5 × 5").tag(2); Text("7 × 7").tag(3)
                            }.pickerStyle(.segmented)
                            HStack { Text("Beta"); Spacer(); TextField("Beta", value: $controller.beta, format: .number).multilineTextAlignment(.trailing).keyboardType(.decimalPad) }
                        }.padding(.top, 8)
                    }.font(.subheadline)
                    #endif
                }.padding(16).background(Color.white.opacity(0.06), in: RoundedRectangle(cornerRadius: 18)).disabled(controller.isProcessing)
                Button { controller.process() } label: {
                    HStack {
                        if controller.isProcessing { ProgressView().tint(.black) }
                        else { Image(systemName: "sparkle.magnifyingglass") }
                        Text(controller.isProcessing ? "Processing image…" : "Denoise image").fontWeight(.semibold)
                        Spacer()
                        Image(systemName: "arrow.right")
                    }.padding(17).foregroundStyle(.black).background(accent, in: RoundedRectangle(cornerRadius: 16))
                }.disabled(controller.isProcessing)
                HStack(spacing: 12) {
                    Button { controller.loadSample() } label: { Label("Sample", systemImage: "circle.lefthalf.filled").frame(maxWidth: .infinity) }
                    PhotosPicker(selection: $photo, matching: .images) { Label("Open photo", systemImage: "photo").frame(maxWidth: .infinity) }
                    if let output = controller.outputImage {
                        ShareLink(item: Image(uiImage: output), preview: SharePreview("Beltrami denoising", image: Image(uiImage: output))) { Image(systemName: "square.and.arrow.up") }
                    }
                }.buttonStyle(.bordered).tint(accent).disabled(controller.isProcessing)
                if let error = controller.error { Text(error).font(.footnote).foregroundStyle(.orange).textSelection(.enabled) }
                Text(controller.footer).font(.caption).foregroundStyle(.secondary).fixedSize(horizontal: false, vertical: true)
            }.padding(20)
        }
        .background(Color(red: 0.055, green: 0.075, blue: 0.09))
        .preferredColorScheme(.dark)
        .onAppear { controller.startIfRequested() }
        .onChange(of: controller.sigma) { _, _ in controller.noiseChanged() }
        .onChange(of: photo) { _, item in
            Task { if let data = try? await item?.loadTransferable(type: Data.self), photo == item, let image = UIImage(data: data) { controller.loadPhoto(image) } }
        }
    }

    private var comparison: some View {
        VStack(spacing: 10) {
            GeometryReader { geometry in
                ZStack(alignment: .leading) {
                    if let image = controller.outputImage ?? controller.inputImage { Image(uiImage: image).resizable().scaledToFit().frame(width: geometry.size.width, height: geometry.size.height) }
                    if let image = controller.inputImage {
                        Image(uiImage: image).resizable().scaledToFit().frame(width: geometry.size.width, height: geometry.size.height)
                            .mask(alignment: .leading) { Rectangle().frame(width: geometry.size.width * split) }
                    }
                    Rectangle().fill(.white.opacity(0.9)).frame(width: 2).offset(x: geometry.size.width * split)
                    HStack { Text("INPUT"); Spacer(); Text(controller.outputImage == nil ? "READY" : "DENOISED") }
                        .font(.caption2.weight(.bold)).padding(10).frame(maxHeight: .infinity, alignment: .top)
                    Image(systemName: "arrow.left.and.right").font(.caption.weight(.bold)).padding(8).background(.black.opacity(0.65), in: Circle()).offset(x: geometry.size.width * split - 16)
                }
                .contentShape(Rectangle())
                .gesture(DragGesture(minimumDistance: 0).onChanged { split = min(0.98, max(0.02, $0.location.x / geometry.size.width)) })
            }
            .aspectRatio(controller.aspectRatio, contentMode: .fit)
            .clipShape(RoundedRectangle(cornerRadius: 18))
            .overlay(RoundedRectangle(cornerRadius: 18).stroke(.white.opacity(0.1)))
            Text("Drag to compare input and output").font(.caption).foregroundStyle(.secondary)
        }
    }
    private func stat(_ title: String, _ value: String) -> some View {
        VStack(alignment: .leading, spacing: 5) {
            Text(title).font(.system(size: 8, weight: .bold)).foregroundStyle(.secondary)
            Text(value).font(.system(size: 14, weight: .semibold, design: .rounded)).lineLimit(1).minimumScaleFactor(0.6)
        }.frame(maxWidth: .infinity, alignment: .leading).padding(12).background(Color.white.opacity(0.06), in: RoundedRectangle(cornerRadius: 13))
    }
}

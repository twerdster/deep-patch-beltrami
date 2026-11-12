# Native inference validation

This is the validation protocol, not a claim that every device, size or backend below has passed. Record completed checks and measured errors separately. Reference behavior is the released Python/PyTorch implementation and the original learned tensors; no retraining, determinant regularization or test-image parameter selection is part of this port.

## Numerical contract

- Inputs are floating-point intensities on the 0–255 scale. Keep negative and above-255 noisy values. Decode/resize/display paths are separate from identical-tensor inference checks.
- Public images are H,W; the trained Caffe convention is N,C,W,H. Transpose exactly once in each direction. Use a non-square, asymmetric test image to detect accidental axis exchange.
- Grayscale inference applies the archived sigma-specific offset before the first stage and removes it after stage five. RGB evaluates channels separately without that grayscale offset. Clip only the final prediction to [0,255].
- Preserve each stage's repeated input boundary, zero-padded metric convolution and zero-padded divergence. These boundary rules are not interchangeable with reflection or one global padding operation.
- Record the original checkpoint digest, conversion-tool versions, conversion options, Core ML package digest, input shape, scalar type, compute-unit configuration and model load/prediction failures.

First establish converted FP32 CPU behavior against PyTorch FP32 using identical arrays. Then evaluate GPU/ANE-permitted configurations and any FP16 or mixed-precision export. Compare intermediate stages as well as final outputs. In particular, inspect finite counts and the determinant `g11*g22 - g12*g12` before inverse square root: cancellation, overflow and rounding can make a nominally similar FP16 graph fail. A successful conversion or finite final PNG does not validate this calculation.

Record max absolute pixel error, RMSE, mean absolute error, finite counts, clipped/unclipped PSNR and PSNR difference against the reference. Metrics use float64 accumulation, peak 255, no output rounding or border crop. Conversion tolerances must be declared with results; they are not tolerances specified by the thesis. An accelerated approximation must be identified if it fails the chosen fidelity gate. Keep the accurate backend available rather than silently altering the model.

The phone suite's declared acceptance gates require all output values to be finite. FP32 maximum absolute error must be ≤0.05 and RMS error ≤0.01; mixed-precision maximum absolute error must be ≤1.0 and RMS error ≤0.05. Errors use 0–255 pixel units. The runner writes gates according to the requested precision. Also check FP32 fallbacks in a requested mixed run against the stricter FP32 gate; the published final 22-reference-case run passes that independent check. Retain the actual selected precision and measured values alongside recorded pass/fail status.

The current mixed-precision engine uses the FP32 path when image area is at most 160×160 pixels or either dimension is below 16. Record the selected model and precision in each result so this shape-dependent fallback is visible. A masked512 model produced large, finite errors for a partial rectangle under ANE-permitted execution on iPhone 12/iOS 18.0.1. The engine blocks masked tiles larger than 256 whenever Neural Engine execution is permitted; GPU/CPU-only modes may select 512-pixel masked tiles. Preserve the failed configuration as validation evidence and rerun the full affected matrix before lifting this restriction.

## Geometry and content matrix

| Test group | Natural H×W | Purpose |
|---|---|---|
| Tiny/singleton | 1×1, 1×17, 19×1, 2×3, 7×9 | Boundary validity and shape checks |
| Odd and asymmetric | 31×47, 127×129, 129×127 | Axis order, stride and crop errors |
| Standard sizes | 128×128, 256×256, 512×512, 1024×1024 | Scaling and memory behavior |
| Natural non-square | 321×481, 481×321 | Physical orientation and rectangular execution |
| Tile boundaries | Dimensions just below, equal to and above a tile stride | Partial tiles and seam checks |

Use constant values, horizontal and vertical ramps, an asymmetric impulse near each edge, checkerboards, a procedural smooth surface with an intensity step, and deterministic Gaussian-noisy variants. Include pixels outside [0,255]. Synthetic constants/ramps are graph parity inputs, not an assumption that a learned denoiser must leave them unchanged. Run sigma 25 first, then all eight released sigmas before claiming support for every checkpoint. Explicitly reject unsupported shapes rather than resizing them invisibly in a validation run.

For a benchmark-quality check, run all 68 original BSD68 inputs at sigma 15/25/50, then all eight levels for thesis reproduction. Preserve the supplied noise and verify input hashes. Report per-image errors and per-sigma means; one synthetic picture or matching display previews is insufficient evidence for general conversion accuracy.

The completed [Mac FP32 record](benchmarks/bsd68-coreml-mac.json) covers all 544 BSD68 inputs and all eight checkpoints using one full-image native Core ML prediction per case. Its largest per-case PSNR difference from PyTorch is 0.0000361 dB, and all eight means reproduce thesis rounding. Input/label hashes and means were independently rechecked. This record establishes the stated Mac FP32 quality result; device, mixed-precision, boundary-mask and tiled execution require their own checks. Per-pixel maximum/RMS error is outside that full-dataset record's metric scope.

Use the [portable native Mac runner](Tests/run_bsd68_coreml.sh) to prepare the verified data and rerun the full check. The [setup commands](README.md#completed-mac-quality-reproduction) specify a separate Python environment and configurable data/build directories. A `--limit` or restricted `--sigmas` run is a partial check and must not be reported as all 544 cases.

Generate the portable 11-case procedural CPU reference suite from the repository root:

```bash
python -m pip install -e .
python ios/scripts/prepare_fixtures.py
```

The [generator](scripts/prepare_fixtures.py) writes `manifest.json` and per-case `input.f32`, `expected.f32`, `clean.f32` and `case.json` under `ios/DeepBeltramiApp/Fixtures`. Arrays are natural H,W, row-major, little-endian float32. Each file has a byte length and SHA256; the manifest also records the native transposed input digest, reference versions and checkpoint digest. Its sigma-25 cases include tiny/singleton, odd, non-square, 128×128, 256×256 and 321×481 shapes. Forced corner extremes exercise unclipped inputs, so these are parity fixtures rather than a pure Gaussian benchmark.

Use `--output PATH` for other staging locations; `--cache-dir`, `--asset-dir` and `--model-dir` use the package's verified model-loading paths. `--case h1-w1` generates a small smoke fixture and marks the manifest as a partial suite. The default generates all 11 cases. Generated arrays are build/test resources rather than committed source; keep their matching manifest with any temporarily bundled device fixtures. CPU/library versions can change floating accumulation, so preserve the recorded reference environment when comparing output hashes across generators.

Add `--include-smoke` for the combined 22-case suite: the 11 geometry cases, eight 193×257 grayscale cases (one per released sigma), and sigma-25 RGB cases at 2×3, 79×111 and 193×257. This exercises all checkpoints and zero-offset RGB processing alongside the archived-offset grayscale path. Require 22 completed records when running this combined suite. This procedural coverage is separate from the full BSD68 quality check.

## Exact tiling

The released graph's maximum spatial dependency radius is 9 per stage: radius 3 for input derivatives, another 3 for metric filtering and another 3 for divergence. Elementwise operations and the final 1×1 step add no radius. Five stages therefore have an architectural radius of 45 pixels, or a 91×91 receptive field.

An interior tile requires at least 45 pixels of actual neighboring input on every side. Run the whole graph and retain only the tile's valid core. Specify whether an implementation's “overlap” means 45 per side or the resulting 90-pixel shared area between adjacent tile inputs. Averaging insufficient-context tile predictions does not restore equivalence.

At true image boundaries, execute the original per-stage boundary rules. Padding an edge tile once to a fixed size and then cropping its prediction generally changes those rules: the real boundary becomes an interior location. Use models that support the actual edge-tile shape or implement equivalent per-stage masks. Test all four edges, all corners, odd final tiles, images smaller than one tile, and seams on ramps/impulses.

Compare tiled and untiled execution in the same precision/backend before comparing either with PyTorch. Record maximum/RMS error globally and separately in a narrow band around every seam and outer image boundary. Context establishes mathematical locality, while accumulation order and backend shape specialization can still produce numerical differences.

## Compute-device evidence on iOS 18

`MLModelConfiguration.computeUnits = .all` permits CPU, GPU and Neural Engine. It does not identify which device executed a prediction. `.cpuAndNeuralEngine` still permits CPU fallback; there is no ANE-only setting. Inventory APIs describe available hardware, and a compute plan describes anticipated placement. Keep these distinct from measured activity.

The following Swift API shape was typechecked against the iOS 18.1 SDK for an iOS 18 target. `MLComputePlan` is available from iOS 17.4:

```swift
let configuration = MLModelConfiguration()
configuration.computeUnits = .all
let plan = try await MLComputePlan.load(
    contentsOf: compiledModelURL, configuration: configuration)

if case .program(let program) = plan.modelStructure {
    for (_, function) in program.functions {
        for operation in function.block.operations {
            let usage = plan.deviceUsage(for: operation)
            let preferred = usage?.preferred
            let supported = usage?.supported
            let relativeCost = plan.estimatedCost(of: operation)?.weight
            // Save these as planned placement and relative cost.
        }
    }
}
```

`MLComputeDevice` cases are `.cpu`, `.gpu` and `.neuralEngine`; device inventory is available through `MLModel.availableComputeDevices` and `MLComputeDevice.allComputeDevices`. Recurse into operation blocks for a graph with control flow. Missing usage/cost information means unknown, not CPU execution. Relative cost is not a measured duration or utilization percentage. [Apple's compute-plan documentation](https://developer.apple.com/documentation/coreml/mlcomputeplan) describes this inspection interface.

For actual execution evidence, profile the physical phone using the Core ML instrument with GPU and Neural Engine instruments. Correlate the app's prediction interval with Core ML compute requests and hardware activity; retain the trace and identify the tested model/configuration. Apple demonstrates this distinction in [Optimize your Core ML usage](https://developer.apple.com/videos/play/wwdc2022/10027/). A screenshot of the `.all` selector or an available ANE core count does not establish ANE execution.

Filter runtime events and symbolicated call stacks to this app before drawing conclusions. Model-labeled Metal Compute intervals establish GPU execution. App call stacks in Core ML's ANE evaluation and program-request paths, correlated with active Neural Engine hardware intervals, can establish ANE dispatch even when the exported Core ML table omits a separate ANE request label. Model loading, compilation and compute-plan inspection alone do not establish dispatch. Sampled call-stack counts are not request counts, layer counts or accelerator execution durations.

Some Neural Engine hardware tables contain device-global activity without a process identifier. Background activity can overlap the app's prediction windows, so do not attribute all such intervals or their summed duration to the app. Preserve this limitation in any results. Record the tested graph or model digest and repeat profiling after changing an export. A runtime result for one shape, precision and compute configuration does not establish placement for the others. Publish sanitized app-level aggregates; keep raw traces private because they may contain device identifiers and other processes.

A conversion's typed precision constrains eligible hardware. Validate FP32 and lower-precision exports separately; do not infer backend placement from tensor input dtype alone. See Apple's [typed execution guide](https://apple.github.io/coremltools/docs-guides/source/typed-execution.html). Finite enumerated shapes or bounded ranges can support specialization; verify every selected shape and edge-tile policy using the [flexible-shape guide](https://apple.github.io/coremltools/docs-guides/source/flexible-inputs.html).

## Timing record

Separate download, compilation, model load, first prediction and repeated warm prediction. Keep the model loaded. Report input preparation, native inference, tiling/stitching, output conversion and display separately, plus an end-to-end duration with its exact boundaries. Record sample count, median and tail latency, actual processed dimensions, device/OS/build, precision, backend configuration, memory and thermal state. Measure Release builds on the phone; simulator or Mac results do not establish iPhone speed.

Use signposts around prediction and conversion. Await completion before ending a timer; submission time alone is not execution time. Compare configurations on identical inputs and randomize run order when measuring thermal drift. State when large images use tiles or a preview downscale. Maintain a finite-value check during repeated runs, and report any unsupported configuration or resource failure as such.

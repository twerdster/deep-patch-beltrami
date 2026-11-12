# Deep Patch-Beltrami for iPhone

The native SwiftUI app runs the original five-stage learned denoiser through Core ML. It provides a procedural sample, a draggable input/output comparison, noise-level selection, precision/compute choices and full-resolution photo import. Inference runs on the phone; no server or training is involved.

## Build and run

Use Xcode 16 or newer on macOS, its installed iPhoneOS SDK, and an iPhone running iOS 18 or newer. The phone must be trusted, unlocked for installation and have Developer Mode enabled. Select your own Apple development team for signing.

From the repository root, prepare the versioned, hash-verified Core ML models:

```bash
python3 ios/scripts/fetch_models.py
```

The fetcher downloads the Core ML model archive from release `v1.1.0`, verifies it, and compiles the models into `ios/DeepBeltramiApp/ModelAssets`. It uses the installed Xcode tools; no global Python package installation is required for this download/build path. Downloaded models and build products are not committed source.

Open `ios/DeepBeltramiApp.xcodeproj`, select the **DeepBeltramiApp** target, choose your team under **Signing & Capabilities**, select the attached phone, then run. The project uses automatic signing. If your account cannot register the bundled identifier, set a unique **Bundle Identifier** in that panel. The command-line helper accepts `BUNDLE_ID=your.unique.identifier`; use that identifier in subsequent `devicectl` launch/copy commands. If Xcode's destination discovery reports an unavailable platform despite an installed device SDK, use the target-based build helper:

```bash
TEAM_ID=YOUR_TEAM_ID ios/scripts/build_ios.sh
```

The helper requires `TEAM_ID` to sign for a physical phone. Omitting it produces an unsigned app for compilation checks, regardless of a team saved in Xcode. It builds Release/arm64 against `iphoneos`, without requiring a simulator. Its output is `.build-ios/Build/Products/Release-iphoneos/DeepBeltramiApp.app`.

For an unsigned source/CI compilation check before downloading models, run `ios/scripts/build_ios.sh --compile-only` with no `TEAM_ID`. This skips only the model-presence check; it does not validate inference. A fresh-checkout bundle has no denoising models. Fetch the models and rebuild normally before running on a phone. Combining `--compile-only` with `TEAM_ID` is rejected.

For command-line installation, set `DEVICE_ID` to your selected phone's identifier:

```bash
xcrun devicectl device install app --device "$DEVICE_ID" \
  .build-ios/Build/Products/Release-iphoneos/DeepBeltramiApp.app

xcrun devicectl device process launch --device "$DEVICE_ID" \
  org.twerdster.DeepPatchBeltrami
```

## Use the app

Tap **Denoise image** to process the sample, then drag the divider to compare the input and result. The sample has a known clean reference, so its quality field reports PSNR. The displayed time measures the native filter work defined by the engine; model loading and photo import are separate costs.

Tap **Open photo** to select an image with the system Photos picker. The importer keeps the source pixel dimensions and converts the image to upright RGB; the view scales only the displayed preview. The size shown above the image is the actual processed size. Imported photos have no clean reference, so their quality field says **Full resolution** instead of reporting a fabricated PSNR. Use the share button to export the displayed denoised image.

To check image import/export independently of model inference, launch with `--check-image-io`. It exercises all eight image orientations at two UIImage scales, preserving pixel dimensions and RGB order through a round trip. The report is `Documents/image-io-checks.json`; the [physical-phone record](benchmarks/image-io.json) contains 16 passing checks.

Choose the known noise level on the 0–255 scale. Each sigma uses its corresponding trained checkpoint. The original grayscale intensity offsets and H,W-to-W,H adapter are handled by the engine. RGB photos run the grayscale network separately on each channel, following the recovered color wrapper.

**Accurate FP32** selects the float32 model. **Fast mixed** selects the `mixed16` model variant, using lower-precision convolution where the conversion permits it while retaining sensitive arithmetic at higher precision. Compare the two modes' measured errors and timings on the target phone; the mode name alone does not establish a speed or accuracy result.

Fast mixed uses the FP32 path for small images (area at most 160×160 pixels, or either dimension below 16) to preserve tiny-image accuracy and avoid processing mostly unused tile area. The report records the model that actually ran. Larger images can use the boundary-masked mixed models.

The compute choices specify allowed processors:

| App choice | Launch flag | Core ML configuration |
|---|---|---|
| CPU + GPU | `--compute gpu` | `.cpuAndGPU` |
| Automatic | `--compute all` | `.all` |
| CPU + ANE | `--compute ane` | `.cpuAndNeuralEngine` |
| CPU | `--compute cpu` | `.cpuOnly` |

CPU fallback is permitted in the first three modes. Compute-plan operation counts describe anticipated placement. An ANE-capable model, the **CPU + ANE** selector, or an `.all` configuration does not prove actual ANE execution. Retain Core ML/GPU/Neural Engine Instruments traces before making that claim. See [the validation protocol](VALIDATION.md).

The engine uses a 45-pixel context halo and retains valid tile cores. Flexible models use the actual image/tile dimensions. Fixed-shape masked models preserve the real rectangle at every stage: they gather the current estimate to repeat the true image edge, mask gradients before the metric convolutions, and mask fluxes before the divergence convolutions. Padding once to an artificial canvas would change those rules. The [boundary implementation](../tools/coreml_boundary.py) preserves the learned tensors and introduces no determinant clamp. Large-photo latency depends on tile count, channels, precision and hardware.

The 512-pixel masked model failed numerical validation on a partial rectangle when ANE execution was permitted on iPhone 12/iOS 18.0.1, despite returning finite values. The engine therefore excludes masked tiles above 256 pixels in **Automatic** and **CPU + ANE** modes. Those modes use 256-pixel tiles for larger mixed-precision images. The 512-pixel masked model remains available with **CPU + GPU** or **CPU**, subject to the recorded quality checks. This restriction preserves numerical fidelity rather than relying on finite output alone.

## Numerical checks and benchmark records

Generate the procedural PyTorch CPU references before running a device validation build:

```bash
python -m pip install -e .
python ios/scripts/prepare_fixtures.py --include-smoke
```

With `--include-smoke`, the generator writes 22 cases under `ios/DeepBeltramiApp/Fixtures`: 11 sigma-25 geometry cases, grayscale checks at all eight released noise levels, and three RGB checks. Geometry cases include tiny, singleton, odd, non-square, 128×128, 256×256 and 321×481 inputs. The all-sigma cases use 193×257 grayscale inputs; the sigma-25 RGB inputs are 2×3, 79×111 and 193×257. Omitting `--include-smoke` generates the original 11 geometry cases.

The Xcode project copies this folder into the app. Rebuild after preparing or changing fixtures. Inputs and expected outputs are natural row-major float32 arrays with checksums; they are generated test resources, not photographs. The generator also accepts explicit cache, asset, model and output paths; see its `--help`.

Launch a numerical check using the bundled fixture suite:

```bash
xcrun devicectl device process launch --device "$DEVICE_ID" --terminate-existing \
  org.twerdster.DeepPatchBeltrami --validate --precision fp32 --compute gpu
```

`--validate` runs the reference cases. `--benchmark` adds procedural performance sizes up to 1024×768 and an RGB case. Precision flags are `fp32` and `mixed`; compute flags are `gpu`, `all`, `ane` and `cpu`. For example:

```bash
xcrun devicectl device process launch --device "$DEVICE_ID" --terminate-existing \
  org.twerdster.DeepPatchBeltrami --benchmark --precision mixed --compute gpu
```

Reports are saved in the app container as `Documents/benchmark-<precision>-<compute>.json`. Each record includes dimensions, finite checks, output filename, timings, precision, model identity and available reference errors. Inspect error rows and require the expected case count; process exit alone does not establish that reference validation ran. `--stay-open` leaves the app open after the run.

Export a report without changing its numerical values:

```bash
mkdir -p results/ios
xcrun devicectl device copy from --device "$DEVICE_ID" \
  --domain-type appDataContainer --domain-identifier org.twerdster.DeepPatchBeltrami \
  --source Documents/benchmark-fp32-gpu.json \
  --destination results/ios/benchmark-fp32-gpu.json
```

Raw per-case output files are stored beside the report in `Documents` and can be exported with the same command using the recorded `output_file`. Compare them with the fixture's exact input and expected hashes. Separate numerical checks from sustained performance measurements: report the sample count, cold versus warm time, processed size and thermal state rather than treating one warm sample as a stable speed estimate.

## Physical iPhone measurements

The [final iPhone 12 record](benchmarks/iphone12.json) uses a Release build on iOS 18.0.1. It contains exactly 22 procedural reference cases and six additional performance cases: all 28 are finite, and all 22 reference checks pass. The requested Fast mixed mode uses FP32 for the small-image fallback described above; each row records its actual model. The largest reference maximum error is 0.02744 and the largest reference RMS error is 0.008783, across different cases.

The declared gates, in 0–255 pixel units, require finite output and maximum absolute/RMS errors of ≤0.05/0.01 for requested FP32 or ≤1.0/0.05 for requested mixed precision. The runner records gates by requested mode; the FP32 fallbacks in this mixed run were also independently checked against the stricter FP32 gate and pass. These are port-validation gates, not thesis quality thresholds or a full phone BSD68 reproduction. With `--include-smoke`, require exactly 22 completed reference cases; a benchmark adds six performance cases without reference scores.

Final warm medians use three repetitions after one initial per-case prediction:

| Processed image W×H | Actual precision | Tile/channel predictions | Median ms |
|---|---|---:|---:|
| 128×128 grayscale | FP32 fallback | 1 | 264.0 |
| 256×256 grayscale | mixed16 | 1 | 536.3 |
| 385×257 grayscale | mixed16 | 4 | 2,140.4 |
| 512×512 grayscale | mixed16 | 9 | 4,780.7 |
| 1024×768 grayscale | mixed16 | 30 | 16,082.3 |
| 257×193 RGB | mixed16 | 6 | 3,123.6 |

The full run started at nominal thermal state and ended at fair, with Low Power Mode off. These timings are short device observations; tile counts include separate channel predictions.

A separate [compute-mode comparison](benchmarks/compute-modes.json) used the identical sigma-25 `h256-w256` reference input and masked256 model. Three warm repetitions after one initial per-case prediction gave:

| Allowed processors | Median ms | Max pixel error | RMS error |
|---|---:|---:|---:|
| CPU + GPU + ANE (`all`) | 507.8 | 0.01341 | 0.001254 |
| CPU + GPU (`gpu`) | 954.4 | 0.00999 | 0.001164 |
| CPU + ANE (`ane`) | 675.0 | 0.00894 | 0.001267 |
| CPU (`cpu`) | 897.1 | 0.18881 | 0.025832 |

These runs were sequential in the displayed order, with three warm samples per configuration. The Automatic run started at nominal thermal state and ended at fair; the later runs started and ended at fair. Their order was not randomized. Treat them as short device observations, with the recorded numerical differences, rather than a sustained-throughput or tail-latency comparison. Times include engine packing, completed prediction and stitching/readback; model initialization, image decode and display are excluded.

In the JSON, `cold_ms` is the first timed prediction for that case; model initialization is outside its timer. Earlier cases may already have loaded the engine. Preserve this distinction when comparing startup latency.

The [sanitized runtime trace summary](benchmarks/hybrid-runtime.json) separately establishes observed hybrid ANE/GPU dispatch for the released sigma-25 masked256 model under `all`. The time-limited recording contains 32 complete predictions. Every one has app-specific sampled ANE evaluation stacks; 208 samples in the ANE program-send path lie within 0.801 ms of active ANE hardware intervals. The process/model-filtered Metal table also contains 519 active Compute intervals labeled with this model. These are runtime observations rather than compute-plan permissions. Sample counts are not request counts, and device-global ANE interval totals are not attributed to the app. The trace applies to this shape, precision and configuration.

## Completed Mac quality reproduction

The native FP32 Core ML path completed all 544 original BSD68 cases on an arm64 Mac running macOS 15.1.1, using `.cpuAndGPU`: 68 images at each of the eight released noise levels. Every case used one full-image prediction, the original noisy input, the 40,000-step checkpoint and archived intensity offset. All eight mean PSNR values round to the thesis values. The largest per-image PSNR difference from the released PyTorch reference was 0.0000361 dB. The [full sanitized record](benchmarks/bsd68-coreml-mac.json) includes case scores, input/label hashes, model hashes and the independent verification counts.

This is a Mac FP32 quality result. Phone tests, mixed-precision checks, tiling checks and physical accelerator traces have their own coverage and results. The full-dataset record measures PSNR differences rather than per-pixel maximum/RMS conversion error; its single sequential timing run includes startup/shape effects and is not a warm performance benchmark.

Rerun it on macOS 15 or newer using the [portable native harness](Tests/run_bsd68_coreml.sh), Swift command-line tools and a Python environment with this project installed. Fetch the Core ML assets first. Choose a work directory on mounted storage with about 380 MB for prepared arrays, plus 324 MB if the noise asset is uncached. The Swift build directory defaults to the repository's native-filesystem `.build-ios/bsd68-native` directory.

```bash
python3 -m venv .venv
.venv/bin/python -m pip install -e .
python3 ios/scripts/fetch_models.py
bash ios/Tests/run_bsd68_coreml.sh \
  --work-dir /path/to/bsd68-coreml-work \
  --python .venv/bin/python
```

Replace the work-directory path with your chosen location. The default runs all 544 cases and checks all eight thesis means. Results, logs and fixture hashes are written to that work directory. For a partial smoke check, add `--limit 1 --sigmas 25`; partial coverage is recorded explicitly. Use `--help` for existing asset/cache/image directories and a custom native build directory. This runner uses the selected environment and installs no packages itself.

## Rebuild the Core ML exports

Fetching the release assets is sufficient to build the app. To reproduce conversion on macOS, use Python 3.11 and the export bootstrap below from the repository root. It creates a separate virtual environment and pins PyTorch 2.5.1, coremltools 8.3.0 and NumPy 1.26.4. Set `DPB_COREML_PYTHON` to choose a Python executable or `DPB_COREML_VENV` to choose that environment's location. The bootstrap installs no packages into the global or Conda environment.

```bash
bash ios/DeepBeltramiCoreML/export_models.sh \
  --output-dir BUILD/quality --tile-size 512 \
  --sigmas 5 10 15 20 25 30 40 50 --precision fp32
bash ios/DeepBeltramiCoreML/export_models.sh \
  --output-dir BUILD/fast256 --tile-size 256 \
  --sigmas 5 10 15 20 25 30 40 50 --precision mixed16 --masked-boundary
bash ios/DeepBeltramiCoreML/export_models.sh \
  --output-dir BUILD/fast512 --tile-size 512 \
  --sigmas 5 10 15 20 25 30 40 50 --precision mixed16 --masked-boundary
python3 tools/package_coreml_assets.py \
  --quality-dir BUILD/quality --fast256-dir BUILD/fast256 \
  --fast512-dir BUILD/fast512 \
  --original-model-dir "$HOME/.cache/deep-patch-beltrami/v1/models" \
  --output-dir BUILD/release
```

The original model directory is the package's default cache; supply the actual directory if using a custom cache. The commands produce eight sigma-specific archives, each containing a bounded flexible FP32 model, masked256 and masked512 mixed models, checksums, validation metadata and the MIT license. Fast models use FP16 for the ten expensive 80-to-80 spatial convolutions while keeping metric, determinant, inverse-square-root and update arithmetic in FP32. The 512-pixel ANE restriction above also applies to rebuilt models until that configuration passes device validation.

Keep compiled `.mlmodelc` caches and build products on a native filesystem. Exported `.mlpackage` files and archives can be stored separately. Conversion quality records, source/checkpoint hashes and package hashes belong with each exported model; the original checkpoints are not retrained by this process.

The [validation protocol](VALIDATION.md) covers precision, tiling, edge behavior and hardware evidence. The [research reproduction notes](../docs/REPRODUCIBILITY.md) preserve the 40,000-versus-100,000-step checkpoint discrepancy, unknown historical offset calibration and portrait-data limitations. The iOS conversion does not retrain the model or replace those conditions.

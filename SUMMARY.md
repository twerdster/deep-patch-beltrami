# Project state

`deep-patch-beltrami` is a standalone Python/PyTorch inference and reproduction
package for the original Deep Patch-Beltrami models from Aaron Wetzler's
2017 PhD thesis, *Topics in Computational Geometric Vision*, Chapter 4 §4.5.
The preferred citation is the thesis. Package version: 1.2.0.

## Learned-operator study

The optional `research/` package evaluates width, depth, metric arithmetic and
an explicit fidelity reaction while retaining the learned differential operator.
The original inference package, trained models and iOS app remain reproducible.
Research weights are separate Python/CUDA models; they are not deployed in iOS.

Five models and six comparisons were frozen before the new BSD68 evaluation.
At sigma 25 the historical model scores 28.865240 dB, continued frozen-metric
control 28.897944, reaction 28.908215, precision control 28.891054 and precision
depth 28.894934. Reaction improves its matched control by 0.010271 dB, 66/68 wins,
with conditional 95% image-bootstrap interval [0.008306, 0.012338]. Its total
historical-baseline gain is 0.042975 dB and includes continued-training effects.
Depth's mean gain of 0.003881 dB is outlier-sensitive: 32 wins/36 losses and a
negative median change. One training seed/fixed noise draw and checkpoint
selection do not establish seed-robust superiority or modern state of the art.

All four selected inference-only exports preserve every tensor byte and CPU
raw/safe/image-adapter predictions. A separate Linux released-weight replay
matches all 340 input/label hashes, MSE, PSNR and clipping records and all six
paired statistics exactly. Prediction pixel arrays were not compared in that
full replay. Runtime is measured separately on a shared RTX 4090, without any
phone speed claim. Total charged process wall time is 6.890961199 hours of 12,
including startup, failures, continuation and both final evaluations.

Reproduction entry points:

- `research/README.md` provides installation, weight downloads and image inference.
- `research/docs/RESULTS.md` explains measured effects, runtime and limitations.
- `research/docs/results/operator-sigma25/` contains all five means, six contrasts,
  complete accounting, per-image/timing CSVs and the public model manifest.
- `research/docs/released-weight-reproduction.json` records full replay checks.
- `research/docs/validation-screening.*` and validation-continuations.* preserve
  the pilot/continuation evidence, failures and actual exposure.
- `research/docs/MODERN_CONTEXT.md` reports earlier inference comparisons on
  exactly the same inputs, with unmatched training/data/compute caveats.

The model-manifest SHA256 is
`20a3ea6011dfde4facf8bbf94aad5ad71aff3bed006328a56c14e8ed2e83c40b`.
The stable numerical reference excludes accounting; refreshing the completed
ledger left the tested manifest/reference/scoring CSV bytes unchanged.

Install from root with `python -m pip install -e '.[test]' -e './research[test]'`.
The combined CPU suite passes 174 tests with 17 optional archival checks skipped;
source guards, artifact integrations and installed-wheel checks are recorded in
`research/docs/VERIFICATION.md`. Wheel checks explicitly reused existing Torch.
The optional research package is version 0.1.0. Numerical namespaces remain
byte-frozen: `beltrami_operator` uses legacy float32; `beltrami_precision` uses
float64 metric/flux with float32 parameters/dense operations, without epsilon
or determinant clamp. Precision depth requires its matching precision control.
Export/support code lives separately in `beltrami_research`; public tensors
load with `weights_only=True` and source/tensor/provenance guards.

Exact training continuation needs the author's recovered 400 grayscale crops;
only validation32 has a byte-verified upstream reconstruction command. No
photographs or raw optimizer/RNG state are distributed. Code/author weights
are MIT licensed; dataset image terms remain separate. Cite the 2017 thesis.

## Implementation and commands

- Package: `src/deep_patch_beltrami/`. CLI: `deep-patch-beltrami` or
  `python -m deep_patch_beltrami`.
- Runtime dependencies: NumPy, Pillow, PyTorch and h5py. No Caffe runtime or
  MATLAB is required; no other Beltrami package is imported.
- Eight original noise-specific models, five stages per model, 80 feature
  channels and 7-by-7 filters. The inference-only checkpoint export preserves
  all learned tensor bytes and the original deployment graphs.
- `deep-patch-beltrami reproduce bsd68` downloads two pinned release assets
  and the public BSD68 PNGs, reconstructs and hash-checks all 544 original
  noisy inputs, and checks all eight thesis Table 4.6 mean PSNR values.
- `deep-patch-beltrami denoise noisy.png filtered.png --sigma 25` provides
  grayscale/RGB inference. Numerical intensities use the 0–255 scale.
- `examples/demo.py` generates a synthetic image and reports its denoising
  PSNR. No benchmark photographs are included in source or release assets.
- The numerical network and metric modules retain the validated original
  Python implementation. The standalone packaging has its own namespace,
  command, cache and GitHub release URLs.

## Reproduction protocol

The eight saved models are 40,000-step checkpoints; the thesis describes
100,000 steps. Preserve the original grayscale intensity offsets, native
N,C,W,H interface, unclipped float32 inputs, final prediction clipping and
full-image float PSNR. Dataset scores average individual image dB values.
Public image commands transpose ordinary H,W arrays into the native interface.
No training or test-label parameter fitting is needed for a rerun.

BSD68 validation targets sigma 5/10/15/20/25/30/40/50 and means
37.70/33.52/31.37/29.92/28.87/28.04/26.81/25.95 dB at printed precision.
The per-image numerical record is `docs/results/bsd68.json`. Matching displayed
scores does not imply bit-identical prediction pixels across Caffe/PyTorch.

Portraits require separately supplied data. The preserved 100-image suite
reproduces the thesis table despite its 92-image caption. The documented
mixed-input discrepancies and unknown historical offset calibration remain
relevant; see `docs/REPRODUCIBILITY.md` and `docs/reproduction.md`.

## Artifacts and licensing

The public repository is `twerdster/deep-patch-beltrami`. Versioned release
assets are pinned in `src/deep_patch_beltrami/data/release-assets.json`:

- `deep-beltrami-models-v1.tar.gz`, 94,582,731 bytes, SHA256
  `88a65088b7852683657fca0c65dc4cb864ed5328231b36519ddbae4f6da8146b`.
- `bsd68-original-noise-v1.h5`, 323,965,111 bytes, SHA256
  `0c1c0227c71e90f5f08eead3ee18e5e5c9217187b4f5a7240f7376d45f460190`.

These portable artifacts preserve the previously verified model and noise
bytes. The model archive contains a full MIT license; the noise asset contains
residuals only. External BSD68 images retain their original dataset terms.
No private portraits, manuscript sources, benchmark photos, Caffe source or
third-party trained models are distributed. Code license: MIT.

Default cache: `~/.cache/deep-patch-beltrami/v1`, respecting XDG_CACHE_HOME.
`--cache-dir`, `--asset-dir` and `--clean-dir` support explicit/offline storage.
Tests use synthetic arrays; the full numerical benchmark is an explicit command.

## Standalone validation

- A Linux wheel was built and installed noneditably into a separate task
  environment, with `PYTHONPATH` unset and installed-module paths verified.
  The environment reused its existing PyTorch via system-site-packages.
- All eight synthetic tests pass, covering protobuf tensor parsing, pinned
  manifests, atomic download integrity, padding, correlation axes, nonlinear
  graph math, grouped filters, branch lifetimes, offsets, channel handling,
  final clipping and non-finite diagnostics.
- Both installed command and `python -m` entry points passed smoke checks.
- The procedural CPU demo at sigma 25 improved PSNR from 20.154489 to
  39.315465 dB, with finite outputs. Its README preview uses those arrays.
- Verified versions: Python 3.13.5, NumPy 2.1.3, Pillow 11.1.0,
  PyTorch 2.7.1 and h5py 3.12.1.
- `CITATION.cff`, including its preferred thesis citation, validates against
  the official Citation File Format 1.2.0 schema.

## Native iOS release 1.1.0

- `ios/DeepBeltramiApp.xcodeproj` builds the iOS 18+ SwiftUI app with Xcode
  16.1+. It supports full-resolution upright RGB photos, procedural samples,
  noise/precision/compute selection, comparison and sharing. Imported photos
  are not automatically retained as diagnostic files.
- `ios/DeepBeltramiCoreML/` provides a serialized reusable Core ML engine.
  Accurate FP32 preserves the original graph. Fast mixed precision casts
  only ten heavy 80-to-80 convolutions; sensitive metric/determinant/update
  arithmetic remains FP32. No model is retrained or determinant clamped.
- Eight SHA256-pinned release archives in `ios/models.json` contain three
  portable `.mlpackage` variants per sigma: flexible FP32, masked256 and
  masked512 mixed. `ios/scripts/fetch_models.py` verifies downloads and
  compiles all 24 locally using Xcode, without requiring PyTorch.
- Exact 45-pixel context halos and per-stage masks preserve arbitrary image
  rectangles and true boundaries. Small/narrow images use the FP32 fallback.
  A physical A14 masked512+ANE parity failure caused that route to be
  excluded: Automatic/CPU+ANE use masked256; CPU+GPU may select masked512.
- The accurate native Mac run verified all 544 original BSD68 inputs,
  produced finite predictions, and reproduced all eight printed thesis
  means. Maximum per-case difference from public PyTorch PSNR was
  3.60965e-5 dB. The sanitized audit is `ios/benchmarks/bsd68-coreml-mac.json`.
- Physical-phone checks cover 22 cases: 11 sigma-25 geometries, eight
  grayscale noise-specific models and three RGB sizes. Fast/accurate modes
  have declared pixel tolerances; benchmark records distinguish requested
  compute settings from actual returned model/precision and planned devices.
- Actual GPU/ANE execution is measured with Instruments for the final
  masked256 mixed/all graph. Only sanitized app aggregates are published;
  global ANE hardware totals are not reported as application utilization.
- The UI image adapter checks eight orientations at two scales with exact
  pixel/corner/round-trip assertions. UI input/settings are held during work.
- Native documentation and the top README demo describe installation,
  limitations and measured phone timings. Python inference/protocol/model
  and noise release pins remain unchanged. CI includes unsigned iOS compilation.

- Boundary-aware tile partitions retain the true-edge margins and use the
  minimum window count; 512x512 fast Automatic needs 9 predictions instead
  of 16. Exhaustive axis coverage/context checks and seam/RGB checks passed.
- The portable full Mac command is `ios/Tests/run_bsd68_coreml.sh`; it checks
  original input hashes at preparation and inference, label hashes, finite
  predictions and the eight means. Datasets/results may use external storage;
  Swift builds default to the native `.build-ios/bsd68-native` directory.
- Physical-device optimized benchmark: all 28 cases finite, all 22 references
  pass; three warm samples per case, thermal nominal at start and fair at end.
  256x256 fast Automatic median 536.27 ms, 512x512 4,780.68 ms, 1024x768 16,082.26 ms.
  These are filter times, not model-load/UI times or sustained/tail estimates.

- The final public Mac harness rerun passed all 544 cases;
  per-image PSNR values were identical to the independently audited native
  record. The final-source summary records source/model/report hashes.
- The actual phone procedural demo at 384x320, sigma 25, fast Automatic
  improved raw-float PSNR from 20.18 to 41.21 dB in 2,056 ms of filter work.
  Its app-view capture and metrics are published beside the top README figure.

## Maintainer documentation — 2026-11-12

The README describes image input/output, independent grayscale passes for
RGB, the saved five-stage graph and the supported executor layer types.
The saved PyTorch/CUDA BSD68 result, native Mac FP32 CPU+GPU acceptance,
phone procedural checks and app-correlated runtime traces are separate claims.
Phone latency identifies the actual masked256 mixed16 model and excludes
loading/import/display. Figures distinguish phone and Python output, and the
pipeline is labeled schematic. The masked512/ANE limit and unrecovered
historical offset calibration remain explicit.

Reviews of the draft and substantial revision completed. Feedback
was checked against code and saved JSON; no learned tensors, release assets,
source algorithms or metrics changed. This pass verifies Markdown/resources
and wheel README metadata, without rerunning benchmarks or launching training.

Protocol-generated result CSVs retain their original CRLF bytes and SHA256
receipts. `.gitattributes` disables line-ending conversion and recognizes
those CSV terminators; do not normalize a frozen result file for formatting.

The operator release incorporates the concurrent upstream documentation update,
preserving its image-I/O, measurement-scope and limitation clarifications. No
numerical source or result identity changed during that integration.

Both distributions were rebuilt after the documentation integration. Installed
root/research wheel README metadata matches the current source exactly; source
hashes and four installed CLI checks pass with explicitly reused PyTorch. The
path-free record is `research/docs/final-wheel-verification.json`.

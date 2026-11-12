# Development and reproduction

Read `SUMMARY.md`, `README.md`, `docs/REPRODUCIBILITY.md`,
`docs/reproduction.md` and `THIRD_PARTY_NOTICES.md` before changing numerical
behavior or release workflows.

## Setup and verification

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -e '.[test]'
python -m pytest
python examples/demo.py
deep-patch-beltrami reproduce bsd68 --output results/bsd68.json
```

Use `--device cpu` for CPU execution. A subset run with `--sigmas 25 --limit 2`
is a smoke test; only 68 images at all eight sigmas is a full BSD68 reproduction.
Use a fresh cache when verifying public release URLs. Check each asset's
SHA256, every reconstructed input hash, finite predictions and all eight
printed means. Keep generated images, downloads and results untracked.

Build and install a wheel into a separate virtual environment, unset
`PYTHONPATH`, and verify imports/resources load from the installed wheel.
Check CLI and `python -m` entry points. CI runs synthetic tests on CPU;
full-model/benchmark verification is an explicit release check.

## Numerical boundaries

Preserve the archived five-stage graphs, tensor bytes, sigma-specific offsets,
native W,H network interface, unclipped float32 noisy inputs and final clipping.
Metrics use full-image float64 MSE with peak 255, averaging per-image dB values.
Do not select parameters or orientations using clean test scores. Report
40,000 saved training steps and the thesis's 100,000-step prose separately.
Portraits require externally supplied data and their provenance caveats.

Changes to numerical code require affected experiment checks; an unchanged
saved report does not validate a changed implementation. Do not silently
regularize learned determinants or substitute a filter for non-finite output.

## Distribution

Code and author checkpoint licenses are separate from dataset image terms.
Keep photos, portraits, meshes, papers, manuscript sources, credentials and
machine-specific paths out of tracked files and assets. Only synthetic images
and numerical validation records belong in documentation.

`tools/build_deep_assets.py` rebuilds inference-only models and noise residuals
from explicit author-supplied paths. Verify every learned tensor and original
input reconstruction before publishing changed assets. Preserve existing
versioned bytes; use a new explicit release version when artifacts change.

Before every commit, review and update `SUMMARY.md` and this file for changed
commands, behavior, artifact versions, validation status or handoff assumptions.
Keep README examples, citations, manifest URLs and license notices consistent.
The project provides pretrained inference; reproducing the released results
does not require launching training.

When a validation environment reuses system-site-packages for an existing
PyTorch installation, state that explicitly. An installed wheel import check
and clean CI installation establish different parts of packaging verification.
Keep the demo's procedural image, fixed seed and sigma explicit; its score is
an example, separate from the 544-case BSD68 acceptance check.

## Native iOS workflow

Read `ios/README.md` and `ios/VALIDATION.md` before changing native behavior.
Build with `ios/scripts/build_ios.sh` (unsigned), or set `TEAM_ID` to sign
using an existing Apple development account. Do not commit local signing
settings, device identifiers, generated fixtures/models, caches or builds.
Use Release builds on a physical phone for latency evidence; keep sample
counts, thermal state, precision and processed dimensions with the results.

Run `--check-image-io` on the phone after image-adapter changes. It must
pass all 16 orientation/scale checks. `--validate` must contain the expected
reference cases and pass declared finite/error gates; an empty suite fails.
Use `--benchmark`, `--case` and `--repeats` for controlled timings. Preserve
reports before another invocation replaces the same Documents filename.
`--auto-run --screenshot` exports only the procedural demo and an app-view
capture for documentation. Imported photos are shared only on user action.

Fetch all 24 Core ML variants with `python3 ios/scripts/fetch_models.py`.
Use `--sigmas 25` for a smaller development build. Generate the full 22
phone references with `python ios/scripts/prepare_fixtures.py --include-smoke`
(default without the flag is 11 geometry cases). Rebuild to bundle them.
Check FP32/GPU and mixed/Automatic separately; a tiny mixed request returns
FP32 by design, and model metadata must report the actual variant.

Keep the 45-pixel halo, per-stage true-boundary masks, native N,C,W,H axes,
sigma-specific grayscale offsets, zero RGB offsets and final clipping.
Never enable masked512 on an ANE-permitted configuration without resolving
and revalidating the observed physical A14 failure. Full FP16 was rejected
as non-finite. `MLComputePlan` is anticipated placement; actual processor
claims require app-correlated runtime traces for the exact released graph.
Rebuild portable packages with `ios/DeepBeltramiCoreML/export_models.sh`
and `tools/package_coreml_assets.py`; keep release SHA256 pins in sync.

For full native Mac quality acceptance, install the Python package and fetch
all Core ML models, then run `bash ios/Tests/run_bsd68_coreml.sh --work-dir
/path/to/data --python .venv/bin/python`. The work directory holds inputs,
logs and metrics; builds default to the native `.build-ios/bsd68-native`.
Use `--sigmas 25 --limit 1` for a subset, which is never full acceptance.
Only all 544 cases/eight rounded means establish the full BSD68 result.
When changing the spatial partition, retain at least 45 pixels before every
artificial boundary and cover every output pixel exactly once. Rerun seam,
RGB and geometry checks, then update phone tile counts/timings in the docs.
Unsigned CI uses `ios/scripts/build_ios.sh --compile-only`; this skips model
preflight and cannot produce an installed usable app without prepared models.

## README presentation (2026-11-12)

The README adds factual language, platform and existing-license badges while
preserving the original demonstration figures and all runtime, reproduction,
validation and attribution details. Keep badge claims aligned with these docs;
a presentation-only update does not establish a new build or runtime result.

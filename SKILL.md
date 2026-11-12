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

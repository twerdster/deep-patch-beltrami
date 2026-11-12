# Project state

`deep-patch-beltrami` is a standalone Python/PyTorch inference and reproduction
package for the original Deep Patch-Beltrami models from Aaron Wetzler's
2017 PhD thesis, *Topics in Computational Geometric Vision*, Chapter 4 §4.5.
The preferred citation is the thesis. Package version: 1.0.0.

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

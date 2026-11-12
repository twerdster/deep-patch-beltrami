# Deep Patch-Beltrami

The released inference code reproduces all eight BSD68 means in thesis Table 4.6
using the original learned parameters and the original Gaussian noise. Caffe and
MATLAB are not required. No training or parameter fitting occurs during a rerun. This standalone package contains its own inference code and resource manifests.

From an installed checkout:

```bash
pip install -e .
deep-patch-beltrami reproduce bsd68 --output results/deep-bsd68.json
```

This command downloads the author's checkpoints, a noise-only fixture, and the
68 clean BSD68 PNGs from a pinned public source. Every downloaded file is checked
against its SHA256. The fixture stores Gaussian residuals in double precision so
that adding each clean image reconstructs the original float32 noisy array
bit-for-bit. All 544 reconstructed inputs are checked against their original
array hashes before inference. The original noisy inputs are unclipped.

Downloads and extracted checkpoints are cached under
`~/.cache/deep-patch-beltrami/v1`, or `$XDG_CACHE_HOME/deep-patch-beltrami/v1`.
`--cache-dir PATH` selects another task directory. The release downloads total
about 400 MiB, excluding clean PNGs. They contain no clean BSD68 images, no noisy
benchmark images and no portraits. BSD68 images retain their own dataset terms;
they are not covered by this repository's MIT license. The Berkeley dataset page
permits downloading for non-commercial research and educational purposes:
[Berkeley Segmentation Dataset](https://www2.eecs.berkeley.edu/Research/Projects/CS/vision/bsds/).
The PNG source is pinned to
[SCUNet commit 52e440a](https://github.com/cszn/SCUNet/tree/52e440a80a655b01e0b41e9dd9bfe599bc11625e/testsets/bsd68).

The command selects CUDA when available and otherwise uses the CPU. It uses four
CPU threads, disables TF32 and caps its CUDA allocation at 4 GiB. The verified
BSD68 run reserved 666 MiB. Options:

```bash
# Download and verify everything without running the network.
deep-patch-beltrami prepare-bsd68

# A short smoke run; its JSON is explicitly marked as a partial suite.
deep-patch-beltrami reproduce bsd68 --sigmas 25 --limit 2 --device cpu \
  --output results/deep-smoke.json

# Full CPU execution, with a separate cache.
deep-patch-beltrami reproduce bsd68 --device cpu --cache-dir /path/to/cache \
  --output results/deep-bsd68-cpu.json

# An existing clean-image folder avoids downloading the PNGs again.
deep-patch-beltrami reproduce bsd68 --clean-dir /path/to/bsd68
```

The JSON records per-image input hashes, finite-value checks, PSNR, archived
reference PSNR, differences, runtime and GPU memory. A complete historical run
checks the published rounded means and exits unsuccessfully if any fail.

| Noise sigma | Thesis Table 4.6 | Verified reproduced mean |
| --- | --- | --- |
| 5 | 37.70 | 37.695809 |
| 10 | 33.52 | 33.522676 |
| 15 | 31.37 | 31.365600 |
| 20 | 29.92 | 29.918060 |
| 25 | 28.87 | 28.865240 |
| 30 | 28.04 | 28.038062 |
| 40 | 26.81 | 26.811863 |
| 50 | 25.95 | 25.947914 |

All pixels are evaluated in floating point on the 0–255 scale, after clipping
the prediction to that range. There is no border crop, output quantization or
test-time augmentation. The output PNG command below rounds only when saving
the display image; reproduction metrics use the unrounded network result.

## Denoise an image

```bash
deep-patch-beltrami denoise noisy.png denoised.png --sigma 25
```

Released noise levels are 5, 10, 15, 20, 25, 30, 40 and 50. Grayscale inputs use
the historical intensity offset for that noise level. RGB inputs apply the
grayscale network separately to each channel without an intensity offset, as in
the recovered color wrapper. This is a historical model, not a modern denoising
performance claim. Non-finite outputs are rejected; the implementation does not
clamp learned determinants or substitute another filter.

## Model and protocol details

Each network has five distinct stages, 80 feature channels and 7×7 filters.
The port executes the five original deployment graphs using the original
learned tensors. Input padding repeats edge pixels; divergence padding uses
zeros. Tensor-convolution weights are shared inside each stage. Model files
were exported as inference-only Caffe protobufs: training-layer metadata,
absolute training paths and gradient buffers were removed. Every layer/blob
shape and learned float32 array was checked bit-for-bit against the original
checkpoint. Original file hashes, release file hashes and aggregate tensor
hashes are retained in the model manifest. The standalone model archive includes
the full MIT license.

The Caffe/MATLAB interface stores images as `N,C,W,H`. Public image commands
transpose normal `H,W` images into that interface and transpose predictions back.
The benchmark fixture follows the original native interface. Metrics always
compare the same physical image and noisy pixels.

Two historical discrepancies matter for an exact rerun:

- The saved checkpoints are named `iter_40000`, although the thesis prose says
  100,000 training steps. The saved 40,000-step tensors reproduce the tables.
- The grayscale wrapper adds an intensity offset before inference and subtracts
  it afterward. Factors for sigma 5/10/15/20/25/30/40/50 are
  3/2/1/1/1/1/1/0.5. The thesis prose omits these offsets, and their historical
  selection procedure was not recovered. `--offset-protocol none` retains an
  explicit ablation; it must not be described as the exact historical wrapper.

The float32 PyTorch and Caffe backends have small numerical differences. In the
verified BSD68 run, maximum per-image PSNR difference from archived Caffe output
was 0.0000373 dB. Matching the displayed thesis precision does not mean every
output pixel is bit-identical across numerical backends.

## Portrait table: user-supplied data

The portrait images are not redistributed. An existing compatible HDF5 suite
can be evaluated with:

```bash
deep-patch-beltrami reproduce portraits100 --suite /path/to/faces100.h5 \
  --output results/deep-portraits.json
```

The suite layout is `/label/NAME` and `/data/SIGMA/NAME`, with float32
`N,C,W,H` arrays. Input datasets can carry a `sha256` attribute and a `methods`
JSON attribute containing `{"Beltrami-Deep":{"psnr":VALUE}}` for archival
comparisons. Hash attributes are verified when present. The full portrait suite
contains 100 images at all eight noise levels. A dataset with a different image
count is reported as a partial suite.

Thesis Table 4.5 says “92 face portraits”, but all 100 preserved portraits
reproduce every displayed Deep and BM3D mean. No supported 92-image selection
rule was found. Report the actual 100-image count. Reproduced Deep means for
sigma 5/10/15/20/25/30/40/50 are
39.844635/36.298859/34.381960/33.025205/31.980666/31.149391/29.805467/28.829769.

For ten portraits at all eight noise levels, later result containers held
different noisy data from the corresponding original input files while clean
labels were unchanged. The original input files recover the saved predictions;
the mixed result-container inputs do not. A portrait suite must use the
corresponding original inputs. This affects IDs 00001, 00005, 00008, 00009,
00011, 00013, 00021, 00031, 00052 and 00092. The verified full portrait run had
800 finite cases and maximum per-image archived PSNR difference 0.0001206 dB.

## Offline files and rebuilding assets

Download the two files attached to [release `v1.0.0`](https://github.com/twerdster/deep-patch-beltrami/releases/tag/v1.0.0) and supply their directory
with `--asset-dir PATH`. File hashes are still checked. `--model-dir PATH`
accepts an already extracted, verified release model directory. The clean PNG
directory can be supplied with `--clean-dir`.

`tools/build_deep_assets.py` rebuilds the release from author-supplied original
checkpoints, a compatible BSD68 suite and matching public PNGs. It strips
non-inference metadata, checks every learned tensor, exports noise-only arrays
and verifies reconstruction of every original noisy input. Its arguments are
explicit source/output paths; it does not depend on the author's filesystem.

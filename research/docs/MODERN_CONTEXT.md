# Released denoiser comparison context

These are measurements from the completed **2026-11-11** comparison of
released checkpoints, made before the new operator study. The new study's
five-arm held-out evaluation completed on **2026-11-12**, after its joint
selection freeze. Its public released-weight replay passed all 340 metric and
input-identity checks; see [reproduction evidence](released-weight-reproduction.json).
No model was run or trained to prepare this context.

## Existing BSD68 results at sigma 25

All rows use the same 68 archived noisy images. PSNR is the arithmetic mean
of per-image, full-image PSNR, computed from float64 squared error with peak
255. Only final outputs are clipped to [0,255]; there is no quantization,
border crop or self-ensemble. These are measured values on the archived
noise, rather than transcribed paper scores.

| System | Mean PSNR (dB) | Gain over historical Deep (dB) | Paired 95% interval (dB) | Wins /68 |
|---|---:|---:|---:|---:|
| Historical Deep Patch-Beltrami | 28.865240 | — | — | — |
| SCUNet | 29.560451 | +0.695211 | [+0.614268, +0.781265] | 68 |
| Xformer | 29.554485 | +0.689245 | [+0.603470, +0.780667] | 68 |
| BERUNet v2, revised release | 29.554153 | +0.688913 | [+0.611324, +0.770749] | 68 |
| DRUNet | 29.490154 | +0.624914 | [+0.553704, +0.700928] | 68 |

Intervals use 10,000 paired-image bootstrap replicates, seed 20261111. They
describe variability across these 68 images, not uncertainty across new noise
draws. Small differences among modern rows do not establish a reliable ranking.
Exact means, intervals, model pins and original result-file receipts are in
[modern-context.json](modern-context.json).

## Input and scoring controls

The archive stores the original Caffe/MATLAB interface as `N,C,W,H`. Modern
author APIs expect `N,C,H,W`, so the comparison transposed noisy inputs for
those APIs and transposed predictions back before scoring against untouched
archive labels. This fixed interface conversion follows source conventions;
it was not selected by comparing scores. Earlier archive-native modern runs
were retained only as orientation sensitivity checks.

No noise was regenerated and negative or above-255 noisy samples were
preserved. Modern models receive intensities divided by 255 and the true
noise level through their published conditioning interface or matching-sigma
checkpoint. The historical Deep model retains its original fixed +25 input
offset and corresponding output subtraction. That offset was not fitted in
this comparison; its historical calibration procedure was not recovered.

All 340 sigma-25 case records were rechecked for finite outputs and identical
archived noisy/clean array hashes across these five systems. The original
audit also checked all 68 labels against official BSD68 PNGs. The JSON records
a reproducible digest of the 68 sorted input/label identity records, without
redistributing photographs.

The completed new operator evaluation was independently checked against those
records: all five arms match all 68 archived input/label hashes and native
shapes exactly, with common identity digest
`1b5e3a42b6b6093a5a1b49dc113d4865827a0f523d4523a9d65f47ac05266905`.
The private completed-result SHA256 is
`16fb05d2d083d1ced88e15a0082199aa517678f46ba453a86b4b9bf5d9c542d5`.
The JSON gives the exact field mapping, freeze/evaluation timestamps and all
five new means. This verifies physical input identity without reopening
photographs or labels.

The new reaction arm measured **28.908215 dB**, a modest improvement over its
declared control, while the modern references above remain at
29.490154–29.560451 dB. The old modern scores were not used to choose the new
checkpoints. These measurements support no state-of-the-art claim; training
data and compute remain unmatched. The separately run public released-weight
replay reproduced all per-image hashes, MSE, PSNR and paired statistics exactly.

## Pinned implementations and weights

| System | Official source at executed commit | Evaluated checkpoint |
|---|---|---|
| SCUNet | [52e440a](https://github.com/cszn/SCUNet/tree/52e440a80a655b01e0b41e9dd9bfe599bc11625e) | [scunet_gray_25.pth](https://github.com/cszn/KAIR/releases/download/v1.0/scunet_gray_25.pth) |
| Xformer | [ecc1e91](https://github.com/gladzhang/Xformer/tree/ecc1e910b923f35c8d46f5ffbc8ed2bc603f6567) | `xformer_gray_dn_sigma25.pth`, [author model folder](https://drive.google.com/drive/folders/1Ui_X1JoC5jxgE6k_hK7AZ2qrLo2I_OsD) |
| BERUNet v2 | [2589ed6](https://github.com/Xin-Ge/BERUNet-denoiser/tree/2589ed6644218cd802473e30e6a7d72ab82a779c) | `BERUNet_v2_gaussian_gray.pth`, [author archive](https://drive.google.com/file/d/1xqiHJjwvATTQc0WjkrfE7MM0oLqm7dPk/view) |
| DRUNet | [15bca3f](https://github.com/cszn/DPIR/tree/15bca3fcc1f3cc51a1f99ccf027691e278c19354) | [drunet_gray.pth](https://github.com/cszn/KAIR/releases/download/v1.0/drunet_gray.pth) |

| Checkpoint | SHA256 |
|---|---|
| Historical `PB_25_80_5__iter_40000.caffemodel` | `07a9438eb26fc0e70c425d31a460cf2d493f53ebf8044893e1e3a9d94fcede50` |
| `scunet_gray_25.pth` | `06e971c573c52a9a62c2796071fe7fae168542ead7225cfb91bc074d3350a657` |
| `xformer_gray_dn_sigma25.pth` | `8ea3680c2ffd7331e78bf9b66949d87b6879f02a20cfd05205fc4ec97a0e0890` |
| `BERUNet_v2_gaussian_gray.pth` | `3abbdab50adc7060f4c9c7a6d768e46d49fff71e6f504572c38850858b719e85` |
| `drunet_gray.pth` | `e27fb29456732c604c8ee3ac3e92ecadd2a7a4e36a8be675e92ebbf93a240de4` |

SCUNet and DRUNet used official overlapping one-split inference; Xformer used
full-frame reflection padding/cropping; BERUNet used its official full-frame
circular padding/cropping. The execution used an RTX 4090, FP32, TF32 disabled
and batch one. Different inference wrappers and a shared GPU prevent a strict
speed ranking from the older timing measurements.

## Interpretation limits

This is a released-system inference comparison. Training data, training
compute, parameter count and checkpoint selection are not matched to
Beltrami. Author declarations include substantially larger external datasets:
DRUNet and BERUNet list BSD, WED, DIV2K and Flickr2K; SCUNet lists WED, DIV2K
and Flickr2K; Xformer's official preparation source specifies BSD400, WED,
DIV2K and Flickr2K. The exact images seen by external checkpoints and possible
near duplicates have not been exhaustively audited. Xformer's umbrella BSD500
citation alone does not establish training on BSD68.

BERUNet's authors state that their public implementation was rechecked and
models retrained after publication. This row evaluates that **revised
released checkpoint**, rather than claiming to reproduce the original
paper's numbers. See the [official release notice](https://github.com/Xin-Ge/BERUNet-denoiser#notice).

GRL-B is excluded from the headline table because its released-model audit
did not reproduce its paper scores: the existing sigma-25 measurement was
29.173388 dB versus the paper's 29.54 dB. Official source/configuration and
attention-batching parity were checked, but the discrepancy remains
unresolved. It is a diagnostic result, not evidence that the paper claim was
replicated. See the [official GRL source](https://github.com/ofsoundof/GRL-Image-Restoration).

The methods' own papers use different noise realizations and sometimes output
rounding or cropping, so equality with their paper tables is not the target
of this archived-input experiment. The earlier bounded release survey did
not establish an exhaustive November 2026 frontier. These rows supply strong
available references; they do not support a state-of-the-art claim for the
new Beltrami operators or isolate an architectural advantage from training
scale.

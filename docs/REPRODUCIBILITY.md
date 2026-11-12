# Reproducibility scope and limits

This package executes the original trained Deep Patch-Beltrami models with Python and PyTorch. Its numerical target is the Deep model in Aaron Wetzler's 2017 PhD thesis, *Topics in Computational Geometric Vision*, Chapter 4, Section 4.5 and Tables 4.5–4.6. It contains the inference implementation and manifests needed for a standalone rerun. The reproduction commands require no new training. The separately installed [research package](../research/README.md) provides optional new optimization experiments with their own protocol and provenance; it does not alter the historical reproduction target.

The [reproduction guide](reproduction.md) gives executable commands, accepted input layouts, model provenance and verified numerical results. Reproducing a displayed thesis value, matching an archived per-image score and recovering bit-identical network predictions are distinct claims.

## Original models

Eight saved checkpoints correspond to known noise standard deviations 5, 10, 15, 20, 25, 30, 40 and 50 on the 0-to-255 intensity scale. Each model has five stages with 80 feature channels and 7-by-7 filters. The recovered files are 40,000-step checkpoints, while the thesis describes 100,000 training iterations. The released saved tensors reproduce the tables; this release does not recreate or verify the historical training run.

The inference-only export removes training metadata and absolute machine paths. Every learned float32 tensor shape and payload byte was independently checked against the original checkpoint; all five deployment graphs are unchanged. The model manifest records original file hashes, sanitized file hashes and aggregate tensor hashes. Metadata sanitization does not change the learned parameters.

The historical grayscale wrapper applies a fixed intensity offset before inference and subtracts it afterward. For sigma 5/10/15/20/25/30/40/50, the factors multiplying sigma are 3/2/1/1/1/1/1/0.5. Clean test labels are used only for scoring at runtime. The procedure used to select these factors was not recovered, so their historical calibration provenance remains unknown. Faithful results retain the offsets. The explicit no-offset option is an ablation rather than the archived wrapper. The historical RGB wrapper applies the grayscale model to each channel without the grayscale offset.

## Exact physical inputs and array axes

Noisy benchmark arrays are unclipped float32 values; some pixels lie below zero or above 255. Preserve those values during inference. Do not clip, quantize, resize or regenerate noise when claiming the historical comparison.

The original MATLAB/Caffe interface stores spatial axes in W,H order, the transpose of an ordinary Python H,W image. Public image commands transpose into that historical network convention and transpose predictions back. The benchmark suite uses native N,C,W,H arrays. This fixed interface adapter preserves the same physical image; it is not augmentation or selection of an orientation using clean-image scores.

The released noise fixture stores float64 residuals computed from the archived float32 noisy and clean arrays. Adding the residual to the matching clean array and converting to float32 reconstructs every historical noisy input byte-for-byte. Every reconstructed input is checked against its original SHA256. This precision claim concerns the recovered stored float32 inputs, not recovery of unavailable original random-generator double values. A fresh Gaussian draw is a different experiment, even if its seed has the same integer value.

The 68 clean BSD68 PNGs are obtained separately from a pinned public source. Both downloaded-file and decoded-label hashes are checked. Source photographs and private portraits are not present in the release assets. External images retain their own terms; see [attribution and external materials](../THIRD_PARTY_NOTICES.md).

## Metrics and verification

The full BSD68 reproduction evaluates 68 images at all eight noise levels, for 544 cases. It compares all eight Deep mean PSNR values in thesis Table 4.6 at the printed two-decimal precision. A smoke run or a subset of sigmas/images must be reported as partial.

Predictions are clipped to [0,255] and retain floating-point values for scoring. PSNR uses full-image float64 mean squared error and peak 255, with no border crop, resizing, integer rounding or test-time augmentation. Dataset PSNR is the arithmetic mean of individual image scores in dB. Saved PNG output is rounded for display and is not the numerical metric input.

The float32 PyTorch and archived Caffe computations have small backend-dependent numerical differences. The verified public BSD68 run's maximum per-image PSNR difference from the archived outputs was about 0.0000373 dB. Matching all displayed means does not establish bit-identical prediction pixels across backends. The implementation rejects non-finite outputs; it does not silently clamp learned determinants or substitute a different filter.

The release uses frozen historical weights, known sigma and fixed wrapper settings. It makes no claim of matched training exposure, contemporary denoising leadership or reproduction of another model's published result. Runtime depends on hardware, transfers, synchronization and software versions; an observed runtime is not a reconstruction of historical training or a controlled speed ranking.

## Portrait table

The portrait test data are not distributed. A compatible user-supplied HDF5 suite can evaluate the same released weights. Thesis Table 4.5 labels its test set as 92 portraits, but all 100 preserved portraits reproduce the displayed Deep means. No supported 92-image selection rule was recovered. Report the actual image count rather than selecting a subset to match the caption.

For ten portrait IDs at each of eight noise levels, later result containers contain different noisy inputs from the corresponding original input files, although the clean labels match. The original Deep input files recover the saved predictions; the mixed result-container inputs do not. The reproduction guide lists the affected IDs. A newly supplied portrait suite does not acquire the archived suite's provenance merely by matching its image count or names.

Historical calibration and test-development exposure cannot be established completely from the recovered files. This release performs no test-label parameter fitting, adaptation or retraining during inference. These limitations remain relevant when interpreting the original generalization claims.

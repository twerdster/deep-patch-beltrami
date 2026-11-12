# Recorded operator study

The σ = 25 study freezes five models and six comparisons before held-out BSD68
scoring. Independent validation32 selects every new checkpoint; the cohort and
stopping decision were declared before the final scores. The joint selection
was frozen on 2026-11-12 at 19:59:41 UTC; evaluation ran from 19:59:58 to
20:01:25 UTC. All 340 model/image predictions were finite before final clipping.

The fidelity reaction produces a **small, consistent improvement** over its
matched continued-training control. The deeper precision operator's small mean
advantage is sensitive to a few images and does not establish a robust depth
benefit. These results concern the operator study; the thesis reproduction and
existing iOS models retain their original checkpoints.

## Frozen BSD68 comparison (σ = 25)

| Model | Role | Parameters | New updates | Mean PSNR (dB) | Median forward (ms) |
| --- | --- | ---: | ---: | ---: | ---: |
| historical | baseline | 3,195,600 | historical 40,000 | 28.865240 | 63.904 |
| frozen-control | control | 3,195,600 | 52000 | 28.897944 | 58.891 |
| reaction | candidate | 3,195,605 | 52000 | 28.908215 | 58.964 |
| precision-control | control | 3,195,600 | 25000 | 28.891054 | 87.685 |
| precision-depth | candidate | 5,112,960 | 14500 | 28.894934 | 139.432 |

| Declared comparison | Mean ΔPSNR (dB) | Conditional 95% interval | Improved / tied / worse |
| --- | ---: | --- | --- |
| frozen-control (control) vs historical (baseline) | +0.032704 | [+0.027555, +0.038027] | 65 / 0 / 3 |
| reaction (candidate) vs historical (baseline) | +0.042975 | [+0.038241, +0.047804] | 66 / 0 / 2 |
| reaction (candidate) vs frozen-control (control) | +0.010271 | [+0.008306, +0.012338] | 66 / 0 / 2 |
| precision-control (control) vs historical (baseline) | +0.025814 | [+0.020777, +0.030680] | 64 / 0 / 4 |
| precision-depth (candidate) vs historical (baseline) | +0.029695 | [+0.024900, +0.034870] | 66 / 0 / 2 |
| precision-depth (candidate) vs precision-control (control) | +0.003881 | [+0.000096, +0.008672] | 32 / 0 / 36 |

## What the comparisons support

**Reaction versus its matched control:** +0.010271 dB, 66/68 image wins,
conditional 95% paired-image bootstrap interval [+0.008306, +0.012338] dB.
The median image difference is +0.008604 dB. Both validation-selected snapshots
have 52,000 new updates, 104,000 sampled patches and the same 160×160,
batch-two training protocol. Reaction adds five stored scalar parameters,
with four effective degrees of freedom. Its observed median forward time is
58.964 ms versus 58.891 ms for the control on the tested GPU; this measurement
does not establish a universal latency difference.

The reaction model's +0.042975 dB over the historical checkpoint includes the
effect of additional training. Continued training alone gives +0.032704 dB.
The matched +0.010271 dB comparison is the evidence for the reaction extension;
the larger historical-baseline gain must not all be attributed to the new term.

**Precision depth versus its matched precision control:** +0.003881 dB mean,
but 32 improved images and 36 worse, with median −0.000959 dB. The conditional
interval is [+0.000096, +0.008672] dB, barely above zero. The two largest image
gains account for 66.7% of the net summed improvement. Depth also has 60% more
parameters and its observed median forward time is 139.432 ms versus 87.685 ms.
Its selected snapshot has 14,500 updates/29,000 sampled patches, while the
precision control has 25,000 updates/50,000 patches. The continuation allowances
were equal within this cohort, so the more expensive operator gets fewer
updates. This is an exploratory result under a time allowance, not an isolated
proof that depth improves the operator at matched training exposure.

Every row and all six prespecified contrasts are reported, including weak
results. No architecture, stopping rule, intensity offset or checkpoint was
changed after these test scores were observed.

## Protocol and limits

Every model receives the identical archived, unclipped float32 noisy image in
the historical `N,C,W,H` interface and the fixed +25 offset. The evaluator
rejects nonfinite predictions, removes the offset, clips only the final float
output to [0,255], and computes full-image float64 MSE and per-image PSNR.
Reported quality is the mean of 68 image dB scores. There is no border crop,
resizing, output quantization, test-time augmentation or test-label fitting.
The [machine-readable record](results/operator-sigma25/results.json) includes
source/checkpoint/config/data/input hashes and all individual results.

The 95% intervals use 20,000 paired-image percentile bootstrap resamples with
seed 20261112. They condition on the fixed images, one archived noise draw and
the frozen models from one training seed. They exclude training-seed,
noise-draw and model-selection uncertainty; there is no multiple-comparison
correction. A small positive interval is not evidence of robustness across
new training runs, noise realizations or datasets.

The 400 recovered training crops map to BSDS500's official train and test
photographs. BSD68 maps to 68 official validation photographs; the other 32
provide source-disjoint validation32 for this round. This source-identity audit
does not exclude near-duplicate scenes or unknown historical development
exposure. Exact continuation requires the separately supplied recovered crops;
public inference and released-weight reproduction do not. See
[data provenance and permissions](DATA.md).

The new protocol uses fresh Gaussian noise, random 160×160 crops and D4
augmentation. The recovered historical generator instead used 1,600 fixed
100×100 corner crops with fixed noise. Original saved weights have 40,000
updates, although the thesis describes 100,000. The new update counts above
exclude that preserved pretraining; they are not replicas of the unknown
historical optimizer trajectory.

Earlier official released-model comparisons give substantially stronger modern
references on these same archived inputs, with unmatched training data and
compute. See [the recorded modern context](MODERN_CONTEXT.md) for source and
weight pins, numerical results and fairness limits. The operator study does
not establish state-of-the-art denoising.

## Runtime and numerical implementations

Timing uses a GeForce RTX 4090, PyTorch 2.7.1+cu126, deterministic cuDNN and TF32
disabled. Each whole-image forward has two warmups and three synchronized
timed repeats. The table reports the median of the 68 per-image medians;
images differ in dimensions. Transfers, clipping and scoring are excluded,
and timing depends on hardware, software and host load. This is neither
end-to-end application latency nor a new iPhone benchmark.

The float32 reaction and frozen-metric control use `beltrami_operator`.
The eight-stage positive-metric operator must be compared to its five-stage
`beltrami_precision` control: both use float64 metric convolution, determinant
and flux, retaining float32 dense convolutions and parameters. The namespaces
have separate frozen source hashes. No determinant epsilon or clamp was added.
The failed float32-depth pilot is preserved as a numerical failure, rather
than treated as a valid accuracy result. The
[full validation screen](validation-screening.md) includes the width and
patch-size trials and their confounds. The
[completed continuation review](validation-continuations.md) records validation
selection, parent-aware exposure and the predeclared resource stopping rule.

## Reproduce and inspect

The [README](../README.md#run-the-released-models) gives concrete installation,
weight download, single-image inference and full CPU reproduction commands.
The [released-weight guide](RELEASED_EVALUATION.md) also documents a bounded GPU
run. The four tensor-only files omit optimizer, RNG and private host metadata;
the public reader uses `torch.load(weights_only=True)` and verifies exact
file/tensor/source/provenance identities.

- [Release manifest](results/operator-sigma25/models.json):
  SHA256 `20a3ea6011dfde4facf8bbf94aad5ad71aff3bed006328a56c14e8ed2e83c40b`.
- [Stable numerical reference](results/operator-sigma25/reference-results.json):
  SHA256 `c7a3bcd3e3e47e6d361ee4503c7d6f24baafaeed2a7ed1022f0b12e7fb52404f`.
- [All model/image scores](results/operator-sigma25/per-image.csv),
  [paired image differences](results/operator-sigma25/paired-per-image.csv) and
  [every timing repeat](results/operator-sigma25/runtime.csv).
- [Generated report](results/operator-sigma25/REPORT.md) and
  [publication file hashes](results/operator-sigma25/files.json).

A separate public tensor-only replay completed all 340 model/image cases.
Input/label hashes and native shapes, MSE, PSNR, clipping counts and all six
paired statistical summaries matched the private frozen evaluation exactly.
Maximum MSE and PSNR differences were zero on the same Linux/PyTorch setup.
Prediction pixel arrays were not compared, and independent runtime samples are
not expected to be identical. The path-free
[reproduction record](released-weight-reproduction.json) pins both results.

Complete aggregate charged process wall time was **6.890961 hours of the
12-hour ceiling**, covering 22 leases, including three failed leases, startup,
training, recovery validation and the two final evaluations. The private final
evaluation charged 88.019 seconds and the released replay 87.522 seconds. This
conservative accounting measures process wall time, not GPU busy time; selected
checkpoint inner time does not establish complete experiment consumption.
The final ledger is idle. Its hash and aggregate counts are retained in the
[sanitized result record](results/operator-sigma25/results.json).

The draft and final `models.json`, `reference-results.json` and scoring/runtime
CSVs are byte-identical. Final accounting includes the public replay without
changing the manifest and scientific reference that were actually tested.

## Citation and license

Cite Aaron Wetzler, *Topics in Computational Geometric Vision*, Technion PhD
thesis, 2017, Chapter 4 §4.5. [CITATION.cff](../../CITATION.cff) and
[references.bib](../../references.bib) provide the thesis record. This is a new
experimental extension of that operator, not a separate peer-reviewed paper.
Code and author-trained parameters are [MIT licensed](../../LICENSE); dataset
photographs retain their own terms and are not included in this release.

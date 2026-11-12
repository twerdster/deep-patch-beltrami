# Reproduce research results from released inference weights

`tools/evaluate_released.py` evaluates the selected public tensor-only models,
matched continued controls and original historical checkpoint. It needs neither
raw training checkpoints nor the 400 recovered training crops. It does not
train, select checkpoints or fit parameters. The completed study's manifest is
[models.json](results/operator-sigma25/models.json), with SHA256
`20a3ea6011dfde4facf8bbf94aad5ad71aff3bed006328a56c14e8ed2e83c40b`.
The four tensor files are attached to
[release v1.2.0](https://github.com/twerdster/deep-patch-beltrami/releases/tag/v1.2.0).
The full Linux safe-weight replay matched every input/label hash, MSE, PSNR,
clipping count and paired statistic in all 340 image/model cases;
[verification evidence](released-weight-reproduction.json) separates that
claim from prediction-array byte identity and timing.

Install both packages from the repository root, then work in `research/`:

```bash
python -m pip install -e '.[test]' -e './research[test]'
cd research
```

Prepare the original pinned public inputs using the unchanged package:

```bash
deep-patch-beltrami prepare-bsd68 --cache-dir /path/to/benchmark-cache
```

Obtain the final research `models.json` and its listed `.pt` release assets.
Use the independently published manifest SHA256 as the command's explicit pin;
do not derive that pin from an unchecked download. Put the tensor files in a
local weights directory with the relative filenames recorded in the manifest.
The evaluator does not download research tensors, change release pins, or
require copies of private run folders. Original model, noise and clean-image
paths must already exist; all bytes are checked before scoring.

For a single image, select one of the downloaded tensor files explicitly:

```bash
beltrami-operator-denoise noisy.png denoised.png \
  --checkpoint /path/to/research-assets/dpb-reaction-sigma25-step52000.pt --device cpu
```

Use the actual asset filename listed in `models.json`. The saved image is an
8-bit preview; reproduce the numerical benchmark with the evaluator below,
which scores the original floating-point inputs and predictions directly.

## CPU reproduction

From `research/`, this complete command runs all five models on CPU. It does
not require a compute ledger and can take substantially longer than CUDA.
The shared lock also applies to CPU runs; use the same lock path for other
processes that access these evaluation resources.

```bash
python tools/evaluate_released.py \
  --manifest docs/results/operator-sigma25/models.json \
  --manifest-sha256 20a3ea6011dfde4facf8bbf94aad5ad71aff3bed006328a56c14e8ed2e83c40b \
  --weights-dir /path/to/research-assets \
  --model-dir /path/to/benchmark-cache/models \
  --asset-dir /path/to/benchmark-cache/downloads \
  --clean-dir /path/to/benchmark-cache/bsd68-clean \
  --cache-dir /path/to/benchmark-cache \
  --gpu-lock "$PWD/gpu.lock" --device cpu --threads 4 \
  --output evaluations/released-bsd68-cpu-001
```

## Bounded GPU reproduction

For a new, independent reproduction, choose an aggregate wall-time allowance
before launching. This example creates a one-hour ledger, then permits one
quarter-hour CUDA evaluation. Ledger creation is exclusive and refuses to
replace an existing file. The runner charges startup, evaluation and failures,
and retains a 60-second global margin; a later lease needs a unique run ID
and enough allowance remaining. These hours measure process wall time, not
GPU utilization.

```bash
# New independent experiment only; do not replace an existing experiment ledger.
python tools/create_budget.py --hours 1 --output BUDGET.json
```

For the original research or any continuing experiment, reuse its existing
ledger instead of creating another allowance. Choose the lease within the
remaining authorized total. CPU evaluations in that experiment also use its
lease runner, with enough time for CPU execution.

```bash
python tools/lease_runner.py --budget-file BUDGET.json \
  --run-id public-released-bsd68-001 --hours 0.25 \
  --log evaluations/public-released-bsd68-001.console.log -- \
  python tools/evaluate_released.py \
    --manifest docs/results/operator-sigma25/models.json \
    --manifest-sha256 20a3ea6011dfde4facf8bbf94aad5ad71aff3bed006328a56c14e8ed2e83c40b \
    --weights-dir /path/to/research-assets \
    --model-dir /path/to/benchmark-cache/models \
    --asset-dir /path/to/benchmark-cache/downloads \
    --clean-dir /path/to/benchmark-cache/bsd68-clean \
    --cache-dir /path/to/benchmark-cache \
    --gpu-lock "$PWD/gpu.lock" --device cuda:0 --max-cuda-gib 10 \
    --output evaluations/public-released-bsd68-001
```

Every output directory must be new. Preserve a failed or partial directory and
diagnose it before another attempt with a new run identity/output. The lease
command does not create or reset a budget.

## What is verified

The manifest uses schema 1 and `purpose: bsd68`. It contains one `baseline`
record, release `assets`, explicit paired `comparisons`, the prior frozen
selection/result digests and the unchanged `evaluate_finalists.py` digest.
Each asset pins `filename`, file `bytes`/`sha256`, tensor/source digests,
implementation, model configuration, sigma/offset, parameters, training step
and sanitized export `provenance`. Source, config, training/validation manifest
and original initialization identities remain explicit without exposing local
paths. The historical baseline pins the official sigma-specific Caffe weights
and all five deployment graphs by name and SHA256.

Before opening labels, the evaluator verifies the independent manifest pin,
original assets and every released artifact. It loads tensors with
`torch.load(weights_only=True)`, rejects extra private checkpoint fields,
strictly loads the model and compares all manifest/export metadata and tensor
content digests. Every candidate requires its correct matched continued
control: reaction versus frozen metric, and precision depth versus the
five-stage precision control. All released arms must share the recorded
training and validation manifests. Every arm is also compared with the
historical checkpoint. Artifact paths must remain inside the weights directory.

The wrapper imports the **unchanged** `evaluate_finalists.py` for original-model
execution, dataset loading, device/lock policy, scoring and paired bootstrap
statistics. `PublicBSD68` verifies all 68 clean labels, the archived residual
asset and each reconstructed float32 noisy-input hash. Each model receives the
same native `N,C,W,H` image plus the fixed archived offset (+25 at sigma 25),
without clipping or rounding the input. Predictions must be finite before
subtracting the offset and clipping the final float output to [0,255]. Full
image float64 MSE yields individual image PSNR in dB, then mean image dB;
there is no crop, resizing, augmentation or test-time fitting.

The manifest fixes warmups (at least 2), repetitions (at least 3), bootstrap
seed and bootstrap count (the research protocol uses 20,000). Synchronized
forward timings exclude transfers, clipping and scoring. Reported timing is
for whole-image CPU/CUDA inference; it does not describe the existing iOS app.
CUDA allocation is capped at no more than 10 GiB and execution holds the
shared nonblocking lock. It cannot reserve memory against other applications.

`results.json` retains `complete`, full provenance, environment, per-image
input/label hashes, MSE, PSNR, timing samples, paired differences and percentile
bootstrap intervals. `status.json` records process identity and current progress.
Only `complete: true`, all 68 images per model and a healthy terminal status
establish completion. Failure evidence remains available with `complete: false`.
The manifest's `expected_results` pins the stable `reference-results.json`,
which excludes compute accounting. It is a provenance reference, not an input
to scoring: compare the completed independent run against that record afterward.
Maintainers first build the manifest without a budget, run this leased replay,
then refresh `results.json` with the completed ledger. The tested `models.json`
and `reference-results.json` must remain byte-identical; accounting updates do
not require repeating inference. See the [publication pipeline](PUBLICATION.md).
Bootstrap intervals describe the fixed images/noise, not training-seed or
model-selection uncertainty. Exact timing is hardware/load dependent; small
floating-point PSNR differences across software/hardware must be reported.

## Synthetic verification

```bash
python -m pytest tests/test_released_evaluation.py
```

These CPU fixtures test explicit manifest/source/artifact guards, safe loading,
private-state rejection, matched-control/role checks, contained paths, numerical
protocol reuse, paired results, exclusive outputs and preserved failures. They
never open BSD68 labels or noise and are not evidence of held-out quality.

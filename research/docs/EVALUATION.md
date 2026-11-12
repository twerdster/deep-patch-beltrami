# Freeze selection, then evaluate

The evaluator supports both frozen namespaces and the unchanged original
inference graph. Use validation to decide architecture, checkpoint and stopping
rule, then freeze those choices before opening final BSD68 labels. Preserve
failed runs and their charges. Do not replace a selection after viewing test
scores to obtain another architecture ranking in the same research round.

Create `evaluations/spec.json` with the validation-only reason and stopped,
immutable checkpoint paths. This example describes a reaction cohort; it is
an illustration, not a claim that it has been selected or improved BSD68:

```json
{
  "sigma": 25,
  "reason": "Document the validation-only selection and stopping decision.",
  "models": [
    {"name": "historical", "role": "baseline", "kind": "historical"},
    {
      "name": "continued-control", "role": "control",
      "implementation": "beltrami_operator",
      "checkpoint": "../runs/continued-control-001/best.pt",
      "config": "../configs/pilot-finetune-frozen.json"
    },
    {
      "name": "reaction", "role": "candidate",
      "implementation": "beltrami_operator",
      "checkpoint": "../runs/continued-reaction-001/best.pt",
      "config": "../configs/pilot-reaction-frozen.json"
    }
  ]
}
```

For high-precision depth, use `beltrami_precision` for both capacity variant and
five-stage control, with `metric_precision: float64` in their configs. A wider
legacy model needs its legacy positive-metric control; the reaction variant
needs its frozen-metric control. The source namespaces and arithmetic settings
are part of the treatment, rather than labels that can be switched on a saved
checkpoint.

```bash
# From research/, after all writers of the selected checkpoints stop:
python tools/evaluate_finalists.py freeze --spec evaluations/spec.json \
  --selection evaluations/selection.json --purpose bsd68 \
  --model-dir "$MODEL_DIR" --training-manifest data/train400.json \
  --validation-manifest data/validation32.json
```

Freeze uses CPU only and opens no BSD68 labels. It checks original model pins,
checkpoint tensors, exact config/source/data hashes and the archived intensity
offset. A selection file and separate SHA256 receipt are created exclusively.
The evaluator's source hash, implementation hashes, checkpoint hashes, timing
settings and bootstrap parameters are frozen. Existing selections cannot be
overwritten. Keep raw selections private because they contain local paths.

Prepare the public original benchmark through the unchanged inference package:

```bash
deep-patch-beltrami prepare-bsd68 --cache-dir /path/to/benchmark-cache
```

The following evaluation is charged against the same ledger. A 0.25-hour lease
is only an example; verify that sufficient aggregate allowance remains:

```bash
python tools/lease_runner.py --budget-file BUDGET.json --run-id final-bsd68-001 \
  --hours 0.25 --log evaluations/final-bsd68-001.console.log -- \
  python tools/evaluate_finalists.py evaluate --selection evaluations/selection.json \
    --output evaluations/final-bsd68-001 --gpu-lock "$PWD/gpu.lock" \
    --device cuda:0 --max-cuda-gib 10 --cache-dir /path/to/benchmark-cache \
    --clean-dir /path/to/benchmark-cache/bsd68-clean \
    --asset-dir /path/to/benchmark-cache/downloads
```

`PublicBSD68` rechecks the public images, clean-label hashes, released noise
residuals and every reconstructed original float32 noisy-input hash. All 68
images are required. Inputs are native N,C,W,H, unrounded, unclipped and use the
fixed archived offset. Raw predictions must be finite before offset removal
and final float clipping. PSNR uses full-image float64 MSE, peak 255 and mean
per-image dB, without cropping, resizing or augmentation.

The results retain per-image input/label hashes, MSE, PSNR, clipping counts and
synchronized forward timings. Timing excludes transfers, clipping and scoring;
report shapes, warmups/repeats, software and device. Whole-image desktop
latency does not establish phone or ANE performance. Paired comparisons include
every image delta and a deterministic 95% percentile bootstrap interval of the
mean. That interval describes the fixed benchmark image/noise sample, rather
than training/selection uncertainty.

Only `complete: true` with all images and terminal healthy status establishes
finished evaluation. Partial output, exit code zero of a budgeted training run
or a stopped lease is insufficient. Preserve errors and diagnose them before
recovery under a unique new lease/output; never erase the original evidence.

## Additional validation noise

Before the final decision, the same finalist specification can be frozen for
`--purpose validation32 --validation-seed 1742913`, then evaluated with
`--validation-root data/validation32` instead of benchmark cache arguments.
The evaluator verifies every manifest entry and shares the declared noise
across models. This is a second validation draw, never a replacement for the
primary selection draw or a held-out BSD68 result.

## Verify the tools

```bash
python -m pytest
python tools/evaluate_finalists.py self-test
python tools/summarize_sweep.py --self-test
```

Synthetic CPU checks cover axes, offsets, clipping/non-finite order, float64
metrics, source/config/artifact guards, selection immutability, paired
uncertainty and nonblocking locks. Optional tests with
`BELTRAMI_MODEL_DIR=/path/to/models` check original graph parity and
function-preserving initialization against the public archived checkpoints.
The numerical tests do not need private data or original model downloads in CI.

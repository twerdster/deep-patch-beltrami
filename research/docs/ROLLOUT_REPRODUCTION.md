# Reproducing the long-rollout study

The 5-, 32- and 64-stage study is complete and separate from the immutable
v1.2.0 operator release. Selection was frozen before scoring, and a separately
leased tensor-only replay matched all 272 input/label identities and quality
records and all six paired summaries exactly. See the
[results](results/rollout-sigma25/REPORT.md) and
[reproduction receipt](rollout-released-weight-reproduction.json).

The maintainer procedures below document the completed author workflow. Do not
relaunch its finished jobs or overwrite its frozen selections and outputs.
For another study, complete training and independent validation before its own
selection/export/benchmark steps. The public replay uses released safe weights
without private checkpoints or the maintainer's report builder.

Run the commands in this document from the repository root, where `research/`
is a subdirectory. If you followed the training instructions into `research/`,
return to the root with `cd ..` first. Both packages must already be installed.

BSD68 has already been observed in the previous round. New scores are a
**reused benchmark**, not newly untouched held-out evidence. No checkpoint,
architecture, optimizer, stopping rule or release subset may be selected using
these benchmark scores. The fixed cohort is rollout-5, rollout-32, rollout-64
and the exact v1.2.0 released reaction52000. The six fixed pairs are 32−5,
64−5, 64−32, and each new arm minus released reaction.

## Author evidence and draft bundle

`tools/build_rollout_report.py` is a CPU-only maintainer tool. It accepts an
explicit, trusted `--private-evaluator` and requires its exact SHA256 to match
the frozen final selection. The private verifier must be installed beside its
frozen `evaluate_finalists.py` helper with the frozen numerical packages. Run
it on the author host where all frozen checkpoint/config/data/validation
metadata paths resolve. It revalidates the selection, complete terminal
status, input/label hashes and shapes against packaged archived descriptors,
MSE/PSNR, timings, all paired bootstrap summaries and the exported tensor bytes.
It reads descriptor JSON but never opens benchmark photographs or noise arrays.

The three rollout exports must already have passed the trusted CPU exporter
and independent raw/safe prediction checks. Supply an export-spec JSON in
selection order. The paths are private inputs, never copied into public output:

After joint terminal selection is frozen, verify each trusted raw/safe pair on
a hash-verified validation image, without scoring quality:

```bash
CUDA_VISIBLE_DEVICES= OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 \
  python research/tools/verify_rollout_export_cpu.py \
  --checkpoint RUN/best.pt --expected-checkpoint-sha256 SELECTED_RAW_SHA256 \
  --safe SELECTED_SAFE.pt --expected-safe-sha256 INDEPENDENT_SAFE_SHA256 \
  --validation-root VALIDATION_IMAGES --validation-manifest validation32.json \
  --report NEW_CPU_ACCEPTANCE.json
```

This compares every learned tensor and exact raw/safe/adapter CPU prediction
bytes on a fixed nonsquare crop (rows 7:24, columns 11:30) from the first
manifest image, singleton geometries and derived RGB. The source PNG's file,
decoded pixels and dimensions must match the pinned manifest. Receipts include
hashes and crop coordinates, without image pixels or private paths. Synthetic
tests establish tool behavior; each actual selected asset still needs its own
acceptance receipt. Raw pickle loading is restricted to the explicitly pinned,
trusted local checkpoint.

```json
{
  "schema": 1,
  "assets": [
    {"model": "rollout-5", "path": "SELECTED_CONTROL.pt", "filename": "RELEASE_CONTROL.pt", "url": "https://github.com/OWNER/REPO/releases/download/TAG/RELEASE_CONTROL.pt"},
    {"model": "rollout-32", "path": "SELECTED_32.pt", "filename": "RELEASE_32.pt", "url": "https://github.com/OWNER/REPO/releases/download/TAG/RELEASE_32.pt"},
    {"model": "rollout-64", "path": "SELECTED_64.pt", "filename": "RELEASE_64.pt", "url": "https://github.com/OWNER/REPO/releases/download/TAG/RELEASE_64.pt"}
  ]
}
```

```bash
python research/tools/build_rollout_report.py \
  --private-evaluator TRUSTED_EVALUATE_ROLLOUTS.py \
  --selection FROZEN_SELECTION.json \
  --results PRIVATE_COMPLETE_RESULTS.json \
  --results-sha256 INDEPENDENT_RESULTS_SHA256 \
  --status PRIVATE_COMPLETE_STATUS.json \
  --exports EXPORT_SPEC.json --output NEW_DRAFT_DIRECTORY
```

Omit `--budget` for this draft. Outputs include a path-free `models.json`,
`reference-results.json`, per-image/paired/timing CSVs, `REPORT.md`, aggregate
`results.json` and a file-hash receipt. The manifest contains all four safe
assets and all six pairs, their source/tensor/data provenance and the precise
released-reaction pin. It does not contain raw checkpoints, optimizer/RNG,
private paths, hostnames or photographs. Model exposure reports selected and
terminal steps, per-depth/full-target updates, eligible validation count,
charged process seconds, effective batch and 180px crop size. No control is
silently omitted after seeing a score.

## Separately leased public-weight replay

`tools/evaluate_rollout_released.py evaluate` is independent of private raw
checkpoints and the private evaluator. It loads only public tensor-only assets
with `weights_only=True`, the installed frozen numerical namespaces, and the
unchanged public input/scoring helper. The manifest requires an independent
SHA256 pin. The three rollout assets must be alongside the released reaction
asset in `--weights-dir`; their sizes, file/tensor hashes, source/configuration,
full-depth eligibility, metadata and protocol are checked before image access.

On the author Linux host, run the evaluator as the direct child of a NEW
bounded existing `lease_runner.py` lease, using the original live aggregate
ledger and the original existing shared GPU lock. Do not use a shell wrapper,
reset accounting, create a second ledger/lock, reuse a training/final-score
lease, or enlarge the 12-hour aggregate authorization. The child checks actual
parent/child/run/starttime/command identity and a maximum .5h lease. A missing
child PID may wait at most 9 charged CPU seconds for the existing supervisor's
initial ledger publication; a present wrong identity fails immediately.

```bash
python EXISTING_LEASE_RUNNER.py \
  --budget-file LIVE_BUDGET.json --run-id NEW_REPLAY_ID --hours .25 \
  --log NEW_REPLAY_LOG -- \
  python ABSOLUTE_RESEARCH_TOOLS/evaluate_rollout_released.py evaluate \
  --manifest ABSOLUTE_DRAFT/models.json --manifest-sha256 INDEPENDENT_MANIFEST_SHA256 \
  --weights-dir ABSOLUTE_SAFE_WEIGHTS \
  --budget-file ABSOLUTE_LIVE_BUDGET.json --lease-id NEW_REPLAY_ID \
  --gpu-lock ABSOLUTE_EXISTING_SHARED_GPU_LOCK \
  --clean-dir EXISTING_PINNED_BSD68_CLEAN_DIRECTORY \
  --asset-dir EXISTING_PINNED_NOISE_ASSET_DIRECTORY \
  --cache-dir EXISTING_PINNED_CACHE --device cuda:0 --max-cuda-gib 10 \
  --output ABSOLUTE_NEW_REPLAY_DIRECTORY
```

Use actual existing verified archive paths, not guessed replacement datasets.
The evaluator requires Linux CUDA, caps its CUDA allocator at 10GiB and
serializes against the shared lock. All startup, lock waits, failures and
scoring remain charged by the external exclusive supervisor. Separate bounded
service deadlines/global guard and the final reserve still apply. Preserve
partial/failed evidence and diagnose failures before any replacement run.

Native inputs remain archived unclipped float32 N,C,W,H with fixed +25; there
is no crop/TTA or input rounding/clipping. Raw outputs must be finite before
subtracting 25 and float clipping to [0,255]. Quality uses full-image float64
MSE and mean per-image PSNR. Paired percentile bootstrap settings, two warmups
and three timed repetitions are pinned in the manifest. Timing measures
synchronized whole-image forward only, excluding transfers/clipping/scoring.
This replay establishes reproduction; it must not trigger new model selection.

## Exact comparison and final accounting

After both private scoring and safe-weight replay are verified finished:

```bash
python research/tools/evaluate_rollout_released.py compare \
  --reference DRAFT/reference-results.json --reference-sha256 INDEPENDENT_REFERENCE_SHA256 \
  --replay REPLAY/results.json --replay-sha256 INDEPENDENT_REPLAY_SHA256 \
  --output NEW_REPRODUCTION_RECEIPT.json
```

The CPU comparator requires exact model identity, all **272** input/label
hashes, shapes, MSE/PSNR/clipping records, mean scores and all six paired
statistics. It does not compare prediction pixel arrays or require identical
runtime. Supply independently verified hashes; a mismatching record is failure,
not permission to relax tolerances or alter the frozen numerical source.

The publication destination is `research/docs/results/rollout-sigma25/`; use a
separate draft evidence directory, and create that final destination only once.

Completed validation curves have a separate CPU review tool. It refuses a live
queue, verifies eligible maxima and phase/exposure counts, and writes sanitized
JSON, Markdown and a plot without opening BSD68. After completion, the author
can run `report_rollout_validation.py --project-root AUTHOR_PROJECT --output
NEW_DIRECTORY`. A published validation JSON can regenerate the figure without
private data:

```bash
python research/tools/report_rollout_validation.py \
  --report-json research/docs/validation-rollouts.json \
  --output /tmp/rollout-validation-replot
```

Matplotlib is an optional plotting dependency. These are validation-selection
curves, not benchmark results or evidence of a convergence theorem.

Rebuild the author report into a new final directory with the unchanged
completed idle live ledger, adding `--budget LIVE_BUDGET.json`. Assert the final
`models.json`, `reference-results.json` and all three scientific CSV files are
byte-identical to the tested draft. Accounting is excluded from the stable
reference; only aggregate final results/report/receipts should reflect the
completed replay's extra charge. The tool refuses an active/enlarged ledger.
Publish clean code, tensor-only assets, actual results and verification receipts
only after final export/replay/release/CI checks are complete. Keep the monitor
active until all watched work is verified finished and actual outcomes reported.

## Interpretation and CPU preparation tests

The combined recipe changes depth, trainable positive metrics, identity
refinements, depth curriculum, AdamW schedule and intermediate supervision.
Equal charged caps do not imply equal full-depth updates or selection
opportunities. One seed/fixed noise, the shared selected warmstart, previous
benchmark exposure and 180px crop context limit conclusions. Conditional
image-bootstrap intervals omit training/selection uncertainty. Positive metrics
do not establish convergence; nonzero nominal stages do not prove useful
refinements. New results do not establish modern state of the art or new iOS
performance. Existing releases and mobile models remain unchanged.

```bash
PYTHONPATH=research/src:src CUDA_VISIBLE_DEVICES= OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 \
  python -m pytest research/tests/test_rollout_released_evaluation.py \
  research/tests/test_rollout_public_report.py -q
```

These synthetic CPU tests validate manifest/lease/provenance/score corruption,
bounded PID publication, exact replay comparison, output privacy, explicit
four-model/six-pair accounting and stable draft/final bytes. They open no image
pixels or noise archives, launch no GPU work and establish no accuracy result.

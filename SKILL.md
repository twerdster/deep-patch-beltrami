# Development and reproduction

Read `SUMMARY.md`, `README.md`, `docs/REPRODUCIBILITY.md`,
`docs/reproduction.md` and `THIRD_PARTY_NOTICES.md` before changing numerical
behavior or release workflows.

## Long-rollout workflow

Read `research/docs/LONG_ROLLOUTS.md` and
`research/docs/rollout-source-hashes.json` before working with
`beltrami_rollout`. This third numerical namespace and the two previous
namespaces are byte-frozen independently. Support/CLI/export/evaluation code
belongs outside them. Preserve previous release/result/asset bytes.

`beltrami-rollout-train` or `python -m beltrami_rollout.train` accepts the
new `research/configs/rollout-{5,32,64}-001.json` configurations and an explicit
SHA-pinned reaction safe-weight warmstart. Use the dedicated
`research/tools/export_rollout.py`, `beltrami-rollout-denoise` and
`research/tools/evaluate_rollout_released.py` for the new format; retain v1.2.0
tool implementation guards. Read `research/docs/ROLLOUT_REPRODUCTION.md` for
draft/replay/final-report order. Never publish raw checkpoints containing
optimizer/RNG/local-path metadata.

Run the new synthetic CPU checks with `python -m pytest
research/tests/test_rollout_model.py research/tests/test_rollout_training.py`.
Set `BELTRAMI_REACTION_SAFE` to the pinned local reaction tensor-only asset for
the optional actual-author-weight 64-stage CPU integration. Check the installed
training CLI and wheel namespace/source hashes. No GPU or benchmark is required
for packaging preparation; these checks do not establish trained quality.

Use an existing aggregate budget/unique lease and shared GPU lock; never reset
accounting. Reserve final evaluation and shutdown time, preflight actual target
depth and full-image validation/save latency, then monitor actual identities,
progress, checkpoints, full validation, finite states and machine resources.
Keep the recurring monitor through failures/recovery and final evaluation.
Selection requires at least 20 full-target-depth updates and independent
validation32, earliest exact tie. Record curriculum/full-depth counts and
actual exposure. Freeze all selection identities before benchmark scoring.
BSD68 was already observed in v1.2.0: new scores are a reused benchmark, not
a new untouched holdout. Report the training-protocol confound and 180px crop
context limitation. The fixed training queue is complete and joint selection is
frozen. All three selected exports and the four-model benchmark completed.
The actual safe-weight replay and exact comparator now pass all 272 records
and six paired summaries. Existing iOS models remain unchanged. Update
SUMMARY/SKILL when publication state changes. Source/tag CI and fresh release
downloads are now verified; no additional training or scoring is planned.

The initial preparation receipt `research/docs/rollout-packaging-verification.json`
records 200 combined CPU passes/17 optional skips, actual-author-weight
integration, installed noneditable source/README checks and all five command
helps. Dependencies were reused from an existing PyTorch 2.5.1 environment;
do not describe it as a fresh dependency install or release acceptance. Rebuild
the wheels after subsequent README/source changes before final release checks.
Preparation passed 341 CPU tests with 17 optional skips before the later crop
acceptance tool was added.
All three actual selected exports and CPU crop/adapter identity checks pass;
receipts are `research/docs/rollout-{5,32,64}-{export,crop}-verification.json`.
The four-model reused-benchmark evaluation passed complete/finite guards.
Public replay and release acceptance are complete. Released versions are
root 1.3.0/research 0.2.0; preserve the v1.3.0 source and asset bytes.

The fixed reused-benchmark means are 28.915269/28.922632/28.920873 dB for
5/32/64, versus 28.908215 for the unchanged released reaction. The 32−5
mean gain is +0.007363 dB (52/68 wins) at about 6.4× forward cost. The 64-stage
model loses to 32 on 58/68 images at about twice the cost. Preserve every arm
and all six comparisons; these scores do not authorize tuning or further
training. The new control's +0.007054 dB gain over released reaction is part
of the 32-stage total +0.014417 dB, not evidence solely for extra depth.

The new manifest SHA256 is
`93813fe42da42a749bcbf10e6c4e2cc1fef7dbea119171401f6553e83d236bba`;
stable reference SHA256 is
`fb4e8fb8491373e117f038ea4578c6e39d7f09dbad88e48f721c36d080a02a54`.
Read `research/docs/results/rollout-sigma25/` with
`research/docs/ROLLOUT_REPRODUCTION.md`. Final accounting was built only
after the replay completed and the live ledger became idle. Manifest/reference
and all three scientific CSV files remain byte-identical to the replayed
draft. `research/docs/rollout-released-weight-reproduction.json` records exact
comparison of 272 input/label hashes, shapes, MSE/PSNR, clipping and six paired
summaries. It does not establish prediction-array or timing identity. Preserve
these scientific bytes; do not repeat scoring to refresh bookkeeping.

The completed aggregate ledger contains 29 leases, three historical failures,
and 38,860.53637930413 charged seconds / 10.7945934387 hours of the original
12-hour ceiling. It includes the 390.749679-second private final evaluation
and 395.106899-second public replay. This is process wall time, not GPU busy
time. No training/evaluation remains running and no additional GPU work is
planned. Replay result SHA256:
`d5799050a548241eecbaff59827d9bce38d6f5f88567b883f145ea95ff9e30ea`;
completed ledger SHA256:
`ece58fba22c88a8f165f6b5bd5e9a52382e38f20fcbac05f73ff974e92229afb`.

The prepared Linux wheels reuse PyTorch 2.7.1 and verify six console commands,
the root module command, four support-tool helps, all installed numerical
source bytes and README metadata outside the checkout. See
`research/docs/rollout-linux-wheel-preparation.json`. This supersedes the
earlier Mac packaging receipt for the prepared versions; neither is final
release acceptance. Rebuild after final documentation changes.

Final Linux verification now passes 365 CPU tests with 17 optional skips and
five legacy helper checks. Both wheels were rebuilt after the final packaged
README edits and installed noneditably into a separate environment, explicitly
reusing PyTorch 2.7.1. Installed file sets, source bytes, numerical hashes and
README metadata match; six console commands, the root module and five support
tools pass outside the checkout with `PYTHONPATH` unset and CUDA hidden.
`research/docs/rollout-final-wheel-verification.json` records packaging only;
the separate actual replay receipt establishes numerical reproduction. No
numerical rerun is needed for later context-only publication status updates.

Release v1.3.0 is public at exact source commit
`d42538bb862105b02617c774da11e74fa4e7252d`. Source CI `43471635006` and tag
CI `43472077285` passed all six jobs. Fresh downloads of all nine assets match
their recorded hashes and sizes; anonymous manifest/tagged README checks pass.
Later context-only commits do not move this tag or alter its scientific files.
The shipped weights already passed CPU identity and exact numerical replay;
do not repeat inference merely to verify downloads or update accounting.

After joint terminal selection is frozen, run
`research/tools/verify_rollout_export_cpu.py` with independently pinned raw
and safe SHA256 values, the validation root/manifest and a new receipt path.
It checks every tensor and exact raw/safe/adapter predictions on the first
verified validation image's fixed 17×19 crop, singleton geometries and derived
RGB. Its 24 synthetic tests pass on Linux; require a separate actual selected
author-weight receipt for each of the three assets. This CPU identity check
does not score quality.

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

## Optional learned-operator research workflow

Read `research/README.md`, `research/docs/DATA.md`,
`research/docs/EVALUATION.md` and `research/docs/source-hashes.json` before
experimental changes. `research/` installs separately; the original inference
package and release pins remain the historical reproduction target. Keep
numerical namespaces byte-identical while they identify trained checkpoints.
Do not place export/CLI additions inside either frozen namespace; support
code belongs in `beltrami_research` or `research/tools`.

Train under one exclusive budget ledger/GPU lock with explicit wall-time
allowances and recurring progress/health checks. Preserve failure evidence;
exit-zero lease completion may accompany `stopped(reason=budget)` training.
Resume the exact config/source/data from `last.pt` under a unique new lease,
extending the cumulative inner allowance. Carry the best checkpoint forward,
reserve final-evaluation time, and report actual updates/images/time.
Use `research/tools/promote_run.py` for its documented standard data layout,
or `research/tools/resume_training.py` for explicit custom trainer arguments.
The separate resume wrapper validates current source, original model bytes,
config/data and finite model/optimizer state without modifying frozen trainer
source. Raw trainer resume checks only config/manifests and is not the supported
guarded continuation path.

For several prepared runs, read `research/tools/PROMOTION_QUEUE.md`. Validate
the complete plan before execution; keep the exact preparation interpreter,
model directory, GPU lock, frozen artifacts and reserve. Do not delete queue
status or execution evidence to retry a failed plan. Preserve evidence and
review a new plan/output/lease only after diagnosis.

Read `research/tools/SUMMARIZING.md` for promoted-run reports. Scope control
matching deliberately, verify ancestry and inherited best/step0, and distinguish
cumulative from continuation updates/pixels/time. Moved snapshots need explicit
`--recorded-root`; keep strict path containment and SHA checks. Live trends
are provisional until planned completion and never authorize more compute.

Select using validation only. Positive capacity variants need their positive
control; precision depth needs the precision five-stage control; fidelity
reaction needs frozen metric control. Freeze architecture/checkpoint/stopping
decision and source hashes before final BSD68 labels. Never present a
validation gain as a test result. Existing iOS halo/model assumptions do not
automatically support deeper models.

Run `python -m pytest research/tests`, evaluator `self-test`, summarizer
`--self-test`, `python research/tools/test_summarize_lineage.py` and research
command help after relevant changes. Focused support checks are
`python -m pytest research/tests/test_promotion_queue.py research/tools/test_summarize_lineage.py`.
CPU CI needs
no private data/model downloads. Optional archival integrations require an
explicit `BELTRAMI_MODEL_DIR`. Run affected original tests too. Build/install
the separate wheel and verify source hashes survive packaging.

Public model assets must contain tensor-only state and plain sanitized
metadata, preserve tensor digests and load with `weights_only=True`. Use the
explicit trusted-local-checkpoint exporter, then inspect the resulting asset
and provenance before release. Never publish raw run folders, optimizer/RNG
state, host/path metadata, dataset photos or private manuscript material.
Archival manifests do not grant image rights or provide a byte-exact training
crop reconstruction recipe. Update this file and `SUMMARY.md` before commit
when results/status/commands/assets change. Current result/asset pins are in
research/docs/results/operator-sigma25/ and the documented release manifest.

For each final asset, compare every exported tensor payload byte against the
trusted checkpoint and compare safe-reader/raw-operator predictions exactly on
a small hash-verified validation image crop. Check finite output, image-adapter
axes/offset/clipping and absence of private metadata. The recorded reaction
pilot integration in `research/docs/export-integration.json` verifies the
pipeline; it does not replace checks of the actual selected release tensors
or establish held-out quality. Keep raw checkpoints and scratch exports outside
tracked files; public evidence should contain hashes/results, never local paths.

Read `research/docs/PUBLICATION.md` before preparing final results. Use
`research/tools/build_public_report.py` only on complete, frozen joint BSD68
evidence and all explicitly selected safe exports; preserve original private
selection bytes and resolve their original artifact paths. Require the external
result SHA, terminal status, exact tensor identity and declared matched control
contrasts. Generated README/report files are reviewable output, not automatic
edits or uploads. Keep conditional image-bootstrap intervals distinct from
training/noise-seed and selection uncertainty. Optional completed budget input
publishes aggregate totals/hash only, including failures.
Build the draft without a budget before public replay; afterward rebuild into
a new directory with the completed ledger. `models.json.expected_results`
pins `reference-results.json`, the scientific payload without accounting.
Assert both files are byte-identical between draft and final. Only
`results.json`, report text and file receipts may change for accounting;
do not repeat BSD68 merely to incorporate bookkeeping.

Use the generated public `models.json` with
`research/tools/evaluate_released.py`; read `research/docs/RELEASED_EVALUATION.md`.
Its public path uses safe tensor-only weights and original public inputs,
requiring no raw training state/images. Actual full released-weight BSD68
acceptance must be run under the remaining aggregate lease after final freeze;
a CPU validation crop or synthetic fixture cannot establish that result.
For focused publication checks, run `python -m pytest` on
`research/tests/test_public_report.py` and `research/tests/test_released_evaluation.py`,
plus both tools' `--help`.

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

## Documentation maintenance

Read SUMMARY.md, then this file, README, linked setup/validation records and
relevant source before revising claims. Write concrete inputs, operations and
outputs; keep current implementation, historical aims and recorded evidence
separate. Label synthetic/untrained figures and schematics. Keep metric units,
implementation, hardware, sample count and measurement scope with the claim.
Source inspection or file export does not establish numerical execution or
training. Preserve original attribution, licenses and validation limitations.

Rebuild distributions when README metadata or bundled guides change; compare
installed documentation with source. For substantial prose revisions, request
an independent review if directed, verify its factual suggestions and review
the revision again. Never claim a review or runtime check that did not complete.
Review and synchronize both context files before every commit.

## Verified v1.2.0 operator-study release

Read `research/README.md`, `research/docs/RESULTS.md` and the pinned bundle at
`research/docs/results/operator-sigma25/` before interpreting new weights.
All training and both full evaluations are complete; no further tuning follows
this held-out study. Preserve the five frozen entries and all six comparisons.
The public weights reproduce all 340 per-image input/label hashes, MSE, PSNR,
clipping and paired statistics exactly. Do not infer pixel-array byte identity
or matching timing from those checks. Keep the conditional image-bootstrap,
one-seed, selection, training exposure and modern-baseline limitations explicit.

Current model-manifest SHA256:
`20a3ea6011dfde4facf8bbf94aad5ad71aff3bed006328a56c14e8ed2e83c40b`.
Released assets are tensor-only and retain the source hashes from training.
That release contains research package 0.1.0 within repository version 1.2.0; historical
model/noise/iOS release URLs and existing model assumptions stay unchanged.
The combined CPU suite is 174 passed/17 optional archival skips. Noneditable
wheel checks explicitly reused system PyTorch 2.7.1. Full released-weight numerical
acceptance is recorded separately in released-weight-reproduction.json.
Before another release, recheck the public commit/CI, exact asset downloads,
README commands, source guards and current SUMMARY/SKILL.

Protocol-generated result CSVs retain their original CRLF bytes and SHA256
receipts. `.gitattributes` disables line-ending conversion and recognizes
those CSV terminators; do not normalize a frozen result file for formatting.

The operator release incorporates the concurrent upstream documentation update,
preserving its image-I/O, measurement-scope and limitation clarifications. No
numerical source or result identity changed during that integration.

Both distributions were rebuilt after the documentation integration. Installed
root/research wheel README metadata matches the current source exactly; source
hashes and four installed CLI checks pass with explicitly reused PyTorch. The
path-free record is `research/docs/final-wheel-verification.json`.

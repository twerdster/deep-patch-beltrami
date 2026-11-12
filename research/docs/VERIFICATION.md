# Verification

## Completed frozen evaluation and safe-weight reproduction

The five-model BSD68 comparison completed after joint architecture/checkpoint/
stopping/source/data/evaluator selection was frozen. All 68 images per model
were finite under the fixed protocol. A separately leased run with the four
safe exported assets and historical baseline reproduced all 340 input/label
hashes, MSE, PSNR, clipping counts and all six paired statistics exactly.
Prediction pixel arrays were not compared in that full replay; timing samples
are hardware/load-dependent and were checked separately.

[Actual results](results/operator-sigma25/REPORT.md),
[replay evidence](released-weight-reproduction.json), and
[all four export checks](linux-finalist-export-checks.json) retain the relevant
hashes and acceptance scope. Refreshing the complete idle ledger preserved the
tested manifest, stable numerical reference and all scoring/timing CSV bytes.
Total process-wall charge was 6.890961199 hours within 12, including failed
leases, startup, training and both final evaluations. Source hashes stayed
unchanged. Existing original iOS models are separate from these research assets.

## Preparation record

The entries below record checks before the final freeze on 2026-11-12. Their
pending-status descriptions refer to that earlier preparation stage. CPU
fixtures establish implementation/packaging behavior, not denoising quality.

- Research CPU suite: 84 passed, 17 optional archival integrations skipped.
  The skipped cases need original released checkpoints or the separately
  preserved actual failure fixture; ordinary CI needs neither artifact.
- Added prepared-promotion queue: 21 focused CPU tests passed.
- Added independent promotion-report lineage regressions: 18 tests passed,
  with 27 subtests. Together the focused queue/report run passed 39 tests.
- Unchanged parent inference suite: eight passed.
- Finalist evaluator: ten synthetic CPU protocol checks passed.
- Validation summarizer: synthetic self-test passed, including paired image
  identity, finite gates, budget/exposure accounting and matched precision
  cohorts.
  It also passes after integrating strict recorded-root mapping, verified
  promotion ancestry, inherited best/step0, cumulative versus continuation
  accounting and provisional live-trend labels. The public default recorded
  root is the actual checkout; moved snapshots require an explicit origin.
- Original and research wheels built and installed noneditably in an isolated
  environment. Imports resolved to the installed wheels, all three research
  command entry points passed `--help`, and both numerical namespace hashes
  matched `source-hashes.json` after installation.
- Tensor-only export preserved every synthetic fixture tensor exactly, removed
  private optimizer/RNG/local metadata and reloaded using `weights_only=True`.
  Source/tensor tampering and non-finite tensors were rejected. Image adapters
  passed nonsquare, singleton-axis, grayscale offset and RGB channel checks.
- All 32 independent validation PNGs were reconstructed from the pinned cached
  upstream JPEGs with OpenCV 4.10.0 and Pillow 13.0.0. Both decoded-pixel and
  encoded-file hashes matched the frozen manifest exactly. This used no GPU
  and did not evaluate BSD68.

## Actual trained-checkpoint integration

A real 10,500-step reaction pilot checkpoint, SHA256
`1d91001f99a997345812cc4b19b8b76e8fb3aec9c149c9d0bccb25a5ccec7b50`,
was exported and loaded through the installed public research wheel on CPU.
All 40 learned tensors (3,195,605 parameters) retained their exact payload
bytes. The tensor-content SHA256 is
`0dfb5949a17e2e9b93352d6cfcfdb7a5767e5bd7bc5542e131ff714dd0330e6a`.
The 12,795,294-byte tensor-only export omitted optimizer/RNG/local metadata.

The safe `weights_only=True` reader and directly loaded frozen operator gave
bit-identical finite predictions on a 19×17 crop of a hash-verified validation32
image: maximum raw absolute difference 0.0 pixels. The public image adapter's
offset removal, final clipping and physical axes also matched exactly. CUDA
remained uninitialized, BSD68 was not opened and no denoising accuracy was
scored. [Sanitized integration evidence](export-integration.json) records the
artifact, config, source and tensor hashes without local paths.

This pilot artifact verifies the export/inference pipeline; it is not a final
model selection, released-weight pin or held-out quality claim. Repeat the
same checks on each final selected model before publishing its asset.

With the original public checkpoint cache explicitly supplied, the model and
precision integration tests passed 20 cases; the single actual-failure fixture
test remained optional and skipped. The three original float32 graph fixtures
matched the archived graph exactly; wider/deeper/reaction/positive warm starts
and the declared precision tolerances passed.
The complete research suite with that original-model cache passed 100 tests,
with only the optional actual-failure fixture test skipped, before adding the
21 separately verified queue cases. The focused support changes required no
numerical-source, model or dataset changes.

CPU/package verification used Python 3.11, PyTorch 2.5.1, NumPy 1.26.4 and
Pillow 13.0.0. The isolated wheel check reused an existing Torch installation;
it checked installed package/source/resource resolution without claiming a
fresh dependency installation. CI declares fresh CPU installations for
Python 3.10 and 3.13. Remote CI execution is pending publication.

The two original numerical namespaces were copied byte-identically from the
recorded experiment source. Existing inference/iOS source and all release
asset pins are unchanged. Only the optional research area, its CI job,
documentation links and synchronized project context were added.

## Locked reaction finalist export

The validation-selected reaction arm is frozen at update 52,000 for the final
comparison. Its trusted training checkpoint SHA256 is
`ffeeab6e16fcde8bf1e2e56bd69b897168b11d2250dd89cd45056c102787d769`.
The selected tensor-only artifact has 40 tensors and 3,195,605 parameters;
all tensor payloads were preserved byte-for-byte. Its tensor-content SHA256 is
`e70f948526b89fe7aeb569ef68b7f5721497868dccdbb8ad8a15a25a031d6264`.
The 12,795,294-byte export SHA256 is
`9a085c0745136eaee86bfaf0b2e1eb31926eb3bb0526169134fa3a89cf2cfed5`.

The installed tensor-only reader, direct frozen model and image adapter
produced exactly matching finite outputs on a hash-verified 19×17 validation32
crop. Odd RGB geometry and both singleton axes also matched the direct model
exactly. The first reaction scalar remains zero because the first stage starts
from the noisy input; the other four stored stage scalars are active.
Optimizer, RNG and local metadata are absent from the export, and it loads
with `weights_only=True`. CUDA remained uninitialized. The installed wheel
check reused the existing CPU PyTorch/NumPy installation without modifying
numerical source. [Sanitized finalist export evidence](reaction-finalist-export.json)
records the full provenance hashes and reaction scalars.

These checks verify the actual selected reaction tensors and their portable
inference path. No BSD68 data or quality score was accessed. The matched
continued control, remaining cohort and joint final held-out evaluation are
still pending; this export establishes no held-out accuracy claim.

## Publication and released-weight support

The publication builder's 30 synthetic CPU tests pass. They cover complete
output and rejection of partial/non-BSD68 results, wrong selection/result/source
identity, missing or changed inputs/models, inconsistent scores/timings and
bootstrap settings, absent or incompatible controls, altered tensors with a
recomputed self-digest, extra private export metadata and invalid aggregate
accounting. No synthetic result is committed as an example measurement.

Independent review identified missing direct checkpoint/export tensor proof
and matched-control enforcement; both were fixed and regression tested before
handoff. The public builder now verifies actual trusted frozen tensor content,
uses the released evaluator's contrast rules and reconstructs allowlisted
public metadata. It never selects models, runs inference or edits a README.

The new tensor-only released evaluator's 13 focused CPU tests and CLI help
passed. The actual final reaction asset also passed its safe metadata/source/
parameter preflight and exact raw-model forward parity on a hash-verified
validation crop (native shape 1×1×17×19), with finite predictions, maximum
difference 0.0 and no CUDA initialization, BSD68 access or accuracy scoring.
The [sanitized preflight record](released-reaction-preflight.json) pins the
actual asset and evaluator/protocol source hashes.
These checks cover the locked reaction export, rather than the pending full
joint frozen BSD68 comparison or complete public release rerun.

The final focused reporter/released-evaluator run passed all 43 tests together.
The successful synthetic report also passed the released evaluator's manifest
and asset preflight contract. Both tools' help commands passed. The frozen
numerical namespaces and original evaluator bytes remain unchanged.

An additional publication integration builds the same frozen scientific
evidence with no budget and with two different completed ledgers/output
directories. `results.json` correctly changes accounting, while `models.json`,
its pinned `reference-results.json` and every score/paired/runtime CSV remain
byte-identical. File receipts verify all generated outputs. No current
timestamps or output paths enter the stable reference. The tested manifest can
therefore remain the exact published manifest after the replay charge is added.

## Linux finalization preparation

An isolated Linux environment using Python3.13.5 and the existing
PyTorch2.7.1+cu126 installation passed43 focused reporter/released-evaluator
CPU tests. This includes the stable-reference contract: after public replay,
updating the completed compute ledger changes final accounting without changing
models.json, reference-results.json or scoring-table bytes. Both numerical
namespaces and the frozen evaluation helper remain byte-identical.

The actual locked reaction export also passed the public asset preflight on
Linux: all six historical model/graph pins verified, safe-reader/raw-checkpoint
outputs bit-identical on a hash-verified validation32 crop, finite output, CUDA
uninitialized, no BSD68 access or accuracy scoring. Sanitized evidence is in
[linux-reaction-preflight.json](linux-reaction-preflight.json). This checks the
release/loading path; full held-out and public released-weight reproduction
remain pending. Dependencies were reused explicitly; this was not a fresh
dependency install. The new environment used its own supported build backend
without changing the existing runtime or training environment.
# Release preparation on Linux

The version 1.2.0 root package and 0.1.0 optional research package were built as
wheels and installed noneditably into a separate environment, with PYTHONPATH
unset and the working directory outside the checkout. Imports resolved to that
environment; both metadata versions, four installed CLI help checks and both frozen
numerical source hashes passed. The environment explicitly reused existing
system PyTorch2.7.1+cu126 under Python3.13.5. CUDA remained uninitialized and
BSD68 unopened. This checks packaging, not a fresh dependency installation.

The combined original/research/lineage CPU suite passed 174 tests, with 17
optional archival integrations skipped, in 3.75seconds. Actual frozen-control
and precision-depth export evidence is recorded separately; no final joint
held-out quality or public release acceptance follows from these CPU checks.

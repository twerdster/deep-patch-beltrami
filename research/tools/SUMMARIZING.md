# Validation32 sweep reporting

`summarize_sweep.py` reads copied run JSON/JSONL and an aggregate-budget snapshot.
It is CPU-only, uses the standard library for tables/statistics, never loads a
checkpoint or imports PyTorch, and never promotes, trains, or evaluates a model.
The optional static PNG uses an already installed Matplotlib with the Agg backend.
Without Matplotlib the report/CSV/JSON are still produced and the skipped plot is
clearly recorded. The execution host's live ledger remains the budget authority.
This is validation-only reporting; it does not replace process monitoring or
the final frozen-model evaluation. Reporting does not establish device health
or authorize a run. See the [research workflow](../README.md) and
[final evaluation protocol](../docs/EVALUATION.md).

Run from `research/`:

```sh
python3 tools/summarize_sweep.py --self-test
python3 tools/summarize_sweep.py --no-plot
```

To include the optional plot, install the `plots` extra in your environment:

```sh
python -m pip install -e '.[plots]'
python tools/summarize_sweep.py
```

For a separate snapshot or output directory:

```sh
python3 tools/summarize_sweep.py --root /path/to/snapshot \
  --recorded-root /path/to/original/research --output /path/to/report --no-plot
```

The snapshot needs `BUDGET.json` and `runs/pilot-*/metadata.json`, `status.json`,
`events.jsonl`, and `validation-*.json`. Original data and checkpoint tensors are
unnecessary. If available, local manifest files and numerical `src` files are
hash-checked against the run metadata. `--run-glob` can select other run names,
but this report checks the primary sigma25/+25/seed742913 validation protocol.
If multiple possible controls exist, control deltas are unavailable rather than
choosing the control with the best score. Multiple implementation/precision
cohorts can coexist in a snapshot; use a deliberately scoped snapshot when
multiple otherwise matched controls would make the reference ambiguous.
Do not mix primary and second-noise evaluations.

Outputs default to `reports/validation32-snapshot/`:

- `REPORT.md`: readable trial table, comparison definitions and evidence checks.
- `trials.csv`: parameter counts, metric architecture, run state, actual charged
  lease time, updates and training patch exposure/throughput, selected
  step/scores/deltas, implementation/metric precision, initial arithmetic shifts,
  and latest common evaluated-step comparisons.
- `paired-per-image.csv`: exact selected-checkpoint image pairs and differences.
- `summary.json`: all numerical summaries, deterministic bootstrap seeds,
  status/config/source metadata and SHA256 of each source validation JSON.
- `validation-curves.png`: static curves versus recorded inner elapsed time and
  optimizer updates, with each selected maximum marked. Inner time excludes
  startup; CSV charged lease time includes startup and associated evaluation.

Every output explicitly labels **validation32 model selection**, never final
BSD68 accuracy. The six original expected pilots and planned `configs/pilot-*`
IDs absent from a partial snapshot are listed as not started. Active runs are
marked ongoing; a normal pilot budget stop is
marked stopped and is not called completion of the60000-step ceiling. Snapshot
timestamps cannot establish current remote health. Partial final JSONL lines
are reported and ignored; malformed/non-finite validation evidence is reported
and excluded from score selection. Inspect issues before using a result.

Best means the highest observed primary-validation mean per run, earliest step
on ties. Initial means that run's step0 evaluation; a resumed-only snapshot
without verified ancestry supplying step0 cannot supply an initial delta. Deeper/wider models compare to
the5×80 same-metric control, reaction to frozen, legacy positive metric to frozen,
and100-pixel patches to160. Matching also checks implementation namespace,
metric precision, source/data/checkpoint hashes, sigma/offset, seeds, batch size
and recorded optimizer settings. Both models
are compared at their independently selected best, so these are neither
equal-update nor exact equal-exposure comparisons. Faster runs may have more
checkpoint selection opportunities. No automatic promotion follows.

A separate endpoint table pairs each candidate and control at their largest
mutually evaluated step, chosen by availability without score selection. This
exposes equal-update behavior alongside the elapsed-time curves and
best-vs-best summaries. Equal updates do not imply equal compute; different
patch sizes also differ in pixel exposure. Charged-second throughput includes
setup/evaluation; live ledger and status snapshots may be several seconds apart.

## Precision cohorts

Legacy metadata without `implementation` means `beltrami_operator`, with float32
metric arithmetic. New metadata declares `implementation='beltrami_precision'`;
`config.model.metric_precision` selects `float64` or diagnostic `float32`, with
the namespace's default being float64. Numerical source verification hashes
only `src/<declared implementation>/*.py`, matching the training runner's method.
Only these two recognized namespaces are read; no numerical module is imported.
Any config/numerics/initialization precision disagreement is reported.

Implementation and metric precision must both match before a control can be
paired, even when source hashes are coincidentally equal. Thus legacy float32
and precision float64 controls cannot mix in a shared `pilot-*` snapshot. The
precision5×80 base operator is its cohort's continued self-control; the8×80
candidate compares to that same-implementation/precision5×80 operator. Model
shape, trainability, parameter formula and control role come from metadata;
run-name words are not used to infer the architecture or role. Report and plot
labels display the actual namespace, width/stages and metric precision.

The precision variant intentionally changes floating-point evaluation while
preserving the mathematical initialization. Its step0 legacy score shift is
recorded against the original fixed validation32 mean, but is not treated as a
training gain or required to be zero. Precision candidate warmstart checks use
its matched precision control's step0 image-score pairs instead. The report
records their mean and maximum absolute per-image PSNR difference; the check
uses1e-4dB tolerance and cannot establish pixelwise or byte parity. A precision
base's self-comparison is labeled as a base record, not independent parity
evidence. No step0 control means this warmstart assessment is unavailable.

`cohort_initial` rows in `paired-per-image.csv` and corresponding JSON fields
preserve these initial comparisons. The known legacy initial mean and each
recorded initial score shift appear in CSV/JSON. Intentional precision changes
therefore remain visible separately from post-training score deltas.

Paired mean differences use all32 matching image names AND decoded hashes.
The percentile95% bootstrap resamples paired images with replacement
(default20000replicates; base seed20261112, explicit per-comparison seeds in
JSON). Intervals condition on selected checkpoints and one fixed noise/training
seed. They do not correct checkpoint/trial maximum selection, estimate
training-seed or noise-draw variation, or establish unbiased generalization.

Finite evidence checks logged losses/gradient norms, status/event numbers and
every saved per-image PSNR. It does not inspect unlogged activations, raw
predictions, model tensors or replay runtime finite gates. Legacy warmstart checks a
preserving-initialization declaration and step0 mean within1e-4dB of the known
initial validation32 mean; this is weaker than independent pixelwise parity.
Data checks compare the audited manifest SHA/counts and any available local
manifests, including saved validation image names and decoded hashes. Stored
parameter counts are checked against the declared architecture formula.
Source-identity exclusion remains the separate recorded data audit.

`--self-test` uses temporary synthetic files and verifies deterministic paired
bootstrap, exact constant shifts, rejection of mismatched image sets and
non-finite numbers, earliest equal maxima, partial final JSONL, ongoing status,
exposure/lease accounting, self-control pairing and report-file integration.
The latest common-step endpoint is checked separately from score maxima.
Additional synthetic fixtures verify absent-metadata legacy defaults, namespace
source-hash isolation, rejection of cross-namespace and cross-precision controls,
metadata-driven roles with arbitrary names, non-finite failure-message detection,
and precision cohort step0 matching despite a deliberate nonzero legacy shift.
It runs without NumPy, Matplotlib, PyTorch, network access or GPUs.

## Strict manifest mapping

Metadata manifest paths may be project-relative or absolute under the explicit
recorded project root, which defaults to this `research/` checkout. That prefix
maps directly into the supplied `--root` snapshot. For a snapshot copied from
another project path, supply `--recorded-root /absolute/original/project`.
There is no basename search
or fallback outside the snapshot. Parent traversal, sibling prefixes, arbitrary
absolute paths and symlink escapes are rejected before evidence is read. JSON
retains the declared path, mapping root, actual local path and recorded/local
SHA256 values, including missing or rejected evidence. A SHA mismatch is a
data-check failure; missing files are not called successful verification.

## Continuation ancestry and resource allocation

For promotions, use a deliberately scoped report so only promoted controls
participate in control matching:

```sh
python3 tools/summarize_sweep.py --run-glob 'promote-*' \
  --output reports/promotions-validation32-snapshot --no-plot
python3 tools/test_summarize_lineage.py
```

The report follows `promotion.json` only to a different, direct child directory
of this snapshot's `runs` root. It maps the explicit recorded project prefix
and rejects traversal, external/symlink escapes, cycles and overlong ancestry.
A trusted link requires original parent metadata/status and their exact
retained child copies, a clean terminal parent status, matching parent-last
step, identical model/training config and data hashes, original initialization
hash, numerical source, implementation and metric precision. The child's
resume path must identify its own retained `parent-last.pt`. Recorded checkpoint
hashes must be valid SHA256 tokens. Available parent checkpoint binaries are
hash-checked without importing PyTorch or parsing tensors. Missing binaries are
explicitly unavailable and do not invalidate otherwise verified JSON linkage.
A retained promotion-decision file, when present, must match its pinned hash.

This checks copied JSON linkage and available binary identities. The promotion
plan does not pin a hash of every parent validation/event record. Curves are
recorded JSON evidence with individual file hashes, not cryptographic proof
that tensor bytes produced each prediction. The supervising process separately
verifies actual run/checkpoint health.

Ancestor validation curves and step0 are inherited. A child's repeated
validation at the same global step overrides the parent curve point and keeps
its own event timestamp. A difference exceeding 1e-4dB is flagged. Unique
plotted checkpoint count and actual validation-call count are separate.
Best-so-far is selected over the complete recorded ancestry, so an inherited
parent best remains visible if the continuation never improves. Its originating
run is recorded. No parent file is changed.

Global updates/image/pixel exposure already include updates before resume.
The main table's lease time and throughput sum each unique ancestor's actual
lease plus the child's lease once. Original historical pretraining is outside
this step0-based accounting. Separate continuation columns subtract the frozen
**last** step and divide only new exposure by the child's own lease. Every
ancestor and child needs exactly one finite, nonnegative charged lease entry.
Absent/invalid accounting makes cumulative throughput unavailable rather than
treating missing charges as zero. The trainer's `gpu_seconds` already restores
cumulative inner time and is never summed across ancestors. Parent terminal
status elapsed may differ slightly from checkpoint elapsed because saving
takes time; no exact continuation-inner-time value is invented from it.

If linkage is incomplete or inconsistent, issues are explicit; cumulative
lease/throughput and continuation exposure are unavailable, and control
comparisons are disabled for that row. A prepared promotion lacking child
metadata is labeled not started in the snapshot. Loading a parent as evidence
does not add it as a control candidate. Selecting pilot and promoted rows
together can yield multiple controls; comparisons remain unavailable until
the scope is made explicit.

The predeclared continuation resource trend uses only unique child validation
global steps strictly greater than its frozen parent-last step. It excludes
inherited validation and resumed initialization. Sort by global step; with
`n` records use `q=max(1,n//4)` records from each end. Subtract the median PSNR
of the first `q` from the last `q`. With at least four new evaluations, an
increase below 0.002dB sets `no_further_extension=true`; fewer records are
unavailable. Both medians, count, rounding, steps and threshold are written to
JSON/CSV/Markdown. This is a compute-allocation heuristic, not a significance
test. A positive trend neither authorizes extra hours nor requires more
training. Further candidate allowance requires its matched control and must
fit the existing aggregate budget and final reserve.
Live-snapshot trends are marked provisional; this extension criterion is
assessed when the planned continuation ends, rather than used to interrupt
the currently authorized run based on a short early window.

`tools/test_summarize_lineage.py` adds independent temporary-file tests for
inherited best/step0 and global curves, child override at the resume boundary,
global versus continuation exposure, cumulative versus own lease charges,
missing/invalid accounting, lineage/config/source mismatches, strict path and
individual-file symlink containment, exact manifest SHA evidence and the
predeclared quarter rounding/threshold. No model tensor is loaded.

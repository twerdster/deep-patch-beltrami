# Build a public result bundle after the final evaluation

`tools/build_public_report.py` turns completed frozen BSD68 evidence and the
corresponding tensor-only exports into publication files. It performs no
inference, training, model selection, upload or README mutation. It reads no
photographs or noise arrays. Use it only after every finalist/stopping decision
is frozen and the full joint final evaluation has finished.

The completed study's [result bundle](results/operator-sigma25/REPORT.md) and
[released-weight checks](released-weight-reproduction.json) record actual
measurements. This page documents the build workflow. Unit tests use temporary
synthetic fixtures rather than research measurements.

## Inputs and guards

Run on the host where the frozen selection's original artifact paths resolve.
Do not rewrite a frozen selection to relocate paths. Required inputs are:

- The original selection JSON and its separate `.sha256` receipt.
- Completed evaluator `results.json` and terminal `status.json`, with an
  explicitly supplied SHA256 of the results file.
- The selected trusted training checkpoints, configs, manifests and original
  model files referenced by that selection. The tool verifies their hashes.
- A specification of **every** frozen non-historical model's tensor-only export,
  in selection order, with portable release filenames and intended HTTPS GitHub
  release URLs. Export files may live outside the repository.
- Optionally, the completed aggregate budget ledger. Only allowance, consumed
  wall time, lease/failure counts and its hash are retained publicly.

The report builder rejects partial/failing/non-BSD68 results, wrong selection
or evaluator/source/file hashes, changed artifact identities, missing models
or images, altered image order/input/label hashes/shapes, inconsistent PSNR/MSE,
timing/mean summaries and paired bootstrap statistics. Every candidate needs
its explicitly declared matched control and historical comparison; the same
rules are shared with [released-weight evaluation](RELEASED_EVALUATION.md).
Timing needs at least two warmups/three repeats, and bootstrap intervals at
least 1,000 resamples; these settings must already be frozen.

The exporter metadata is not accepted as proof of tensor identity: after the
frozen checkpoint SHA is verified, the builder reads that **trusted local**
training state and compares its full tensor-content digest with the safe export.
This maintainer operation uses trusted training serialization. Public inference
and public result replication read tensor-only assets with `weights_only=True`
and require no author training images or raw training checkpoints.

## Export specification

Create this private specification after all actual exports are verified. The
following filenames/URLs illustrate the four declared arms; replace the owner,
repository, tag and paths with the reviewed release destination. The tool never
uploads files or checks whether a future URL is live.

```json
{
  "schema": 1,
  "assets": [
    {"model": "frozen-control", "path": "/path/to/exports/frozen-control.pt", "filename": "frozen-control.pt", "url": "https://github.com/OWNER/REPOSITORY/releases/download/TAG/frozen-control.pt"},
    {"model": "reaction", "path": "/path/to/exports/reaction.pt", "filename": "reaction.pt", "url": "https://github.com/OWNER/REPOSITORY/releases/download/TAG/reaction.pt"},
    {"model": "precision-control", "path": "/path/to/exports/precision-control.pt", "filename": "precision-control.pt", "url": "https://github.com/OWNER/REPOSITORY/releases/download/TAG/precision-control.pt"},
    {"model": "precision-depth", "path": "/path/to/exports/precision-depth.pt", "filename": "precision-depth.pt", "url": "https://github.com/OWNER/REPOSITORY/releases/download/TAG/precision-depth.pt"}
  ]
}
```

Names, roles, model settings and contrast pairs come from the frozen selection,
rather than these examples. Model configs are canonicalized through the
declared implementation's `ModelConfig`, so omitted defaults such as
`reaction: false` compare correctly with full exported metadata.

```bash
# From research/, with the original private paths still available:
RESULTS_SHA="$(python -c 'import hashlib; from pathlib import Path; print(hashlib.sha256(Path("/path/to/final/results.json").read_bytes()).hexdigest())')"
python tools/build_public_report.py \
  --selection /path/to/final/selection.json \
  --results /path/to/final/results.json --results-sha256 "$RESULTS_SHA" \
  --status /path/to/final/status.json --exports /path/to/exports.json \
  --output results/operator-sigma25-draft
```

This first build intentionally omits the budget: it creates the exact manifest
needed for the leased public replay. After that replay finishes, rerun the same
command with `--budget /path/to/completed/BUDGET.json` and a new
`--output results/operator-sigma25`. Keep the builder, frozen evaluation,
exports/specification and environment unchanged. Only accounting and the output
directory change. Assert the tested manifest and scoring reference are identical:

```bash
python - <<'PY'
from pathlib import Path
draft, final = Path("results/operator-sigma25-draft"), Path("results/operator-sigma25")
for name in ("models.json", "reference-results.json"):
    assert (draft / name).read_bytes() == (final / name).read_bytes(), name
print("Published manifest and scoring reference match the tested bytes")
PY
```

`reference-results.json` contains the deterministic scoring/provenance payload
without `compute_accounting`; `models.json.expected_results` pins this stable
file. `results.json` retains the complete optional accounting, including the
public replay. Thus updating the final ledger changes the report/accounting
receipts without changing the manifest that was actually tested. If the byte
assertion fails, investigate changed inputs/code before publishing; accounting
alone must not require another BSD68 run.

Output creation is exclusive. All validation happens before creating the new
directory. Keep failed evidence and choose a new reviewed output after diagnosis;
never overwrite an existing report silently.

## Reviewable outputs

| File | Content |
| --- | --- |
| `results.json` | Sanitized model roles/configs/steps/exposure, per-image scores/timings, paired statistics, provenance and optional complete accounting |
| `reference-results.json` | The same deterministic scientific payload without compute accounting; stable across final ledger updates |
| `models.json` | Tensor-only release URLs, byte/tensor/source/provenance pins, original baseline files, all declared comparisons and the stable reference-result hash |
| `per-image.csv` | All model/image MSE, PSNR, input/label hashes and median forward times |
| `paired-per-image.csv` | Explicit candidate/reference roles, paired scores and signed image deltas |
| `runtime.csv` | Every forward-time repeat with physical width/height |
| `REPORT.md` | Model/comparison tables, uncertainty/runtime limits, citation and license |
| `README-snippet.md` | The same factual tables for deliberate README insertion |
| `files.json` | Hashes of the other generated publication files |

No private selection reason, host identifier, PID, original run/asset path,
optimizer/RNG state, raw command or photograph is copied. Numerical hashes
preserve links back to the unchanged private evidence. The public environment
retains software versions and the device model needed to assess latency.

Read the report without ranking or discarding arms after seeing test scores.
Small and negative effects remain at their actual scale. The 95% percentile
bootstrap intervals condition on the frozen checkpoints, one fixed noise draw
and one training seed; they exclude training/noise-seed and model-selection
uncertainty. Per-checkpoint inner seconds stop at the selected weight snapshot;
the optional ledger records the complete aggregate process wall time including
discarded trials/failures. Neither quantity measures GPU busy time.

Before publication, build the draft bundle, review the new README snippet and
report, upload the exact verified tensors under the intended filenames, verify
downloads against `models.json`, and run the
[released-weight full BSD68 command](RELEASED_EVALUATION.md) under the same
remaining aggregate allowance. Refresh the final report with the completed
ledger and verify manifest/reference byte identity as above. Keep original inference/iOS
checkpoints and existing release bytes intact. Only actual complete results
justify updating the README's pending status or publishing experimental models.

```bash
python -m pytest tests/test_public_report.py tests/test_released_evaluation.py
python tools/build_public_report.py --help
python tools/evaluate_released.py --help
```

The synthetic reporter tests exercise both valid path-free output and deliberate
selection/result/input/tensor/contrast/bootstrap/timing/accounting corruption.
They never stand in for the final real-artifact/public-replication checks.

# Research release checklist

The final held-out evaluation and public-weight numerical reproduction are
complete. Actual measurements and acceptance evidence are linked from
[RESULTS.md](RESULTS.md). This maintainer checklist covers publication and
future releases; it is not a substitute for those numerical records.

## Repository integration

Readiness checked on 2026-11-12: the proposed `v1.2.0` tag/release is unused.
The public default branch is `main`, at
`205e4e7941453f565ae00af6e38caef8c431281b`; its Tests workflow passed. The
prepared research branch incorporates that main commit. Recheck these
remote facts before publication.

- Preserve the newer upstream README badges and the README-presentation
  additions to `SUMMARY.md` and `SKILL.md` when integrating the prepared work.
  Preserve unrelated changes if the remote advances again. Use a normal
  descendant of current `main`; do not force-push over it.
- The separate `main` worktree and the prepared research worktree share Git
  history. Keep publication on the research branch until integration is ready;
  do not check out `main` in two worktrees.
- For a repository release numbered `1.2.0`, update the root `pyproject.toml`,
  root `CITATION.cff` and current package-version statement in `SUMMARY.md`.
  The new research package can retain its first version `0.1.0`.
- Preserve historical asset pins, the original `v1.0.0` model/noise release,
  `v1.1.0` Core ML assets and iOS app version. Research weights do not update
  the existing mobile app.

## Frozen scientific evidence

- Verify every planned training lease is terminal and the selected checkpoints
  are finite. Review all pilot/continuation results, including failed, weak and
  negative trials, with matched controls and actual exposure/time.
- Freeze all five checkpoint/config/source/data/evaluator identities and the
  stopping decision before final BSD68 scoring. Preserve the selection receipt.
- Complete the private final evaluation under a new bounded lease. Report all
  declared comparisons, paired uncertainty and measured forward time.
- Export all four selected experimental models without optimizer/RNG/private
  metadata. Compare every tensor and small validation-image predictions against
  the trusted raw checkpoints. Never publish raw run folders or photographs.
- Follow [the publication pipeline](PUBLICATION.md): build the draft without
  final accounting, replay the safe released assets under a new lease, compare
  every per-image input/label hash and quality result, then rebuild with the
  complete idle ledger. The tested `models.json` and `reference-results.json`
  must remain byte-identical.

## Public documentation and acceptance

- Replace current pending-status prose with actual final results after verified
  completion. Keep validation screening distinct from held-out accuracy and
  explain one-seed, data/exposure, arithmetic and timing limitations.
- Add the generated model manifest, stable reference, final report and scoring
  tables at deliberate tracked documentation paths. Preserve original demos.
- Publish exact asset names, intended versioned release URLs and the independent
  manifest SHA256 next to [the reproduction command](RELEASED_EVALUATION.md).
  The evaluator requires local weights and does not fetch research assets.
- Verify the documented CPU/default image-inference commands and the optional
  bounded GPU path. A first independent bounded experiment needs a new ledger;
  continuing experiments must retain theirs.
- Run the affected CPU tests, source-hash/packaging checks and documentation
  diff checks. Review `SUMMARY.md` and `SKILL.md` again before committing.
- Review the staged file list for private paths, credentials, model state,
  dataset photographs and unintended generated files. The validation plot is
  an aggregate chart; it contains no benchmark photograph.
- Push the final source commit, verify the corresponding public CI, create the
  new tag at that exact commit and upload the verified inference assets without
  replacing older release bytes. Download the published assets into a new
  location and compare bytes/hashes against the tested manifest.
- Report actual results and release links. Stop monitoring only after all
  watched training and final/public evaluations are verified finished and their
  results have been reported.

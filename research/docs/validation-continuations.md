# Completed validation-only continuations

The four predeclared continuations finished at their time allowances. **These
are validation32 model-selection results, not BSD68 results.** They complete
the [pilot screen](validation-screening.md); no held-out score was used to
rank, extend or replace these arms. Full terminal training exposure is given
here because time stored in a selected checkpoint ends before the run stops.

## Selected checkpoints and terminal exposure

All runs use sigma 25, offset +25, 160×160 training patches, batch 2, Adam at 1e-5,
gradient clipping at 1, training seed 20261112 and primary validation-noise seed 742913.
Noise is fresh and unclipped, with random spatial crops and D4 augmentation.
The historical warmstart's earlier pretraining cost is excluded below.

| Arm | Stages×width / metric | Parameters | Best step / validation dB | Terminal updates | Validation opportunities¹ |
|---|---|---:|---|---:|---:|
| Frozen control | 5×80 / float32 frozen | 3,195,600 | 52,000 / 28.2460935032 | 53,660 | 109 |
| Reaction | 5×80 / float32 frozen + fidelity | 3,195,605 | 52,000 / 28.2556934783 | 57,165 | 116 |
| Precision control | 5×80 / float64 positive | 3,195,600 | 25,000 / 28.2393090314 | 26,204 | 54 |
| Precision depth | 8×80 / float64 positive | 5,112,960 | 14,500 / 28.2430307761 | 16,004 | 34 |

¹ Unique evaluated global steps across verified parent and child, including
initialization. The maximum primary-validation mean over this complete lineage
selects the checkpoint; earliest step wins a tie. All four maxima are unique.
The precision cohort uses float64 metric/flux arithmetic and float32 dense
operations/parameters. It has its own matched control. Reaction adds five
stored scalars, four effective, to the frozen-metric operator.

| Arm | Parent / new updates | Parent / new / total charged seconds | Terminal patches / pixels |
|---|---|---|---|
| Frozen control | 10,785 / 42,875 | 1225.761 / 5329.623 / 6555.384 | 107,320 / 2,747,392,000 |
| Reaction | 10,690 / 46,475 | 1225.799 / 5329.548 / 6555.347 | 114,330 / 2,926,848,000 |
| Precision control | 8,278 / 17,926 | 1225.932 / 2629.630 / 3855.562 | 52,408 / 1,341,644,800 |
| Precision depth | 4,896 / 11,108 | 1226.027 / 2629.757 / 3855.784 | 32,008 / 819,404,800 |

Charged lease time includes startup and evaluation. Parent time is counted
once; restored cumulative inner trainer time is not summed across ancestry.
The new continuation allowances were 1.5 hours each for reaction/frozen control
and 0.75 hours each for precision depth/control. All stopped cleanly by budget
before the 60,000-update ceiling. Patch/pixel counts are repeated exposure,
not distinct photographs. The [JSON record](validation-continuations.json)
separates parent, continuation, total and selected-checkpoint exposure exactly,
including full-precision charge values and configuration/data/source receipts.

## Matched comparisons

| Candidate minus proper control | Selected-best delta dB | Improved images | Conditional paired 95% interval | Latest common update / delta dB |
|---|---:|---:|---|---|
| Reaction − frozen control | +0.0095999751 | 31/32 | [+0.0071861710, +0.0124362528] | 53,500 / +0.0103533022 |
| Precision depth − precision control | +0.0037217447 | 18/32 | [−0.0001469844, +0.0084829310] | 16,000 / +0.0104105677 |

Reaction and frozen control both select 52,000 updates, but the runs had
116 versus 109 selection opportunities. Their actual terminal lineage charges
are nearly equal. Median paired change is +0.0078875728 dB and the mean after
dropping three highest and three lowest changes is +0.0087239269 dB. All 86 shared
new evaluated steps from 11,000 through 53,500 favor reaction in mean. This is
a small, consistent validation effect, not a held-out result.

Precision depth's selected 14,500-step checkpoint compares with the control's
25,000-step checkpoint under nearly equal actual terminal lease exposure.
Its median change is only +0.0000742339 dB; the trimmed mean is +0.0014031286 dB,
and two largest changes account for 77.0% of the net mean. The interval includes
zero. At equal 16,000 updates all 32 images improve, but deeper updates cost more
compute. The weak selected-best, comparable-time result remains exploratory.

Common-update endpoints use the largest mutually evaluated step, chosen by
availability without looking at scores. Equal updates do not imply equal
compute. Full paired means, medians, trimmed summaries, bootstrap seeds and
intervals for both endpoint definitions are retained in the JSON.

## Fixed stopping rule

The decision at 16:06 UTC on November 12, 2026 fixed ending training after these
four continuations and including both candidates, both controls and the
historical model in the final comparison, regardless of later validation rank.
No further training or sigma/hyperparameter sweep followed the queue.

The predeclared resource trend counts only unique child validations strictly
above parent-last; resumed initialization is excluded. Each end uses
q=max(1,n//4), integer floor, then subtracts its first-quarter median from its
last-quarter median:

| Arm | New observations n | Quarter q | Last−first median dB |
|---|---:|---:|---:|
| Frozen control | 86 | 21 | +0.0130752176 |
| Reaction | 93 | 23 | +0.0141079588 |
| Precision control | 36 | 9 | +0.0064858838 |
| Precision depth | 23 | 5 | +0.0074781409 |

All exceed the 0.002 dB extension-exclusion threshold. That permits consideration;
it does not require more compute, demonstrate convergence or alter the earlier
fixed stop. The aggregate 12-hour authorization was a ceiling, not a target.

Intervals use 20,000 paired-image bootstrap resamples and condition on selected
checkpoints, one training seed and one fixed noise draw. They do not correct
trial/checkpoint selection or estimate seed variation. Further-training gains
over initialization also include the new sampling/optimizer recipe. Source
photographs are disjoint from supplied training/BSD68 identities, but historical
development exposure and near-duplicate scenes cannot be ruled out. No global
stability, modern state-of-the-art or new mobile-performance claim follows.

The JSON pins the reviewed validation summary and independent review by SHA256;
it contains no photographs or private run state. See [data provenance](DATA.md)
and [numerical source hashes](source-hashes.json).

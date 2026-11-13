# Long-rollout validation review

validation32 only; no BSD68 access; no final selection freeze.
Selection review: maximum eligible validation32 mean, earliest exact tie; at least20 full-depth updates.
This report does not freeze or change the final evaluation cohort.

| Arm | Terminal updates | Full-depth updates | Selected update | Validation dB | Eligible observations | Charged seconds |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| rollout-5 | 16286 | 16286 | 12750 | 28.261453959318 | 63 | 4384.321584 |
| rollout-32 | 2959 | 2209 | 2959 | 28.267359464781 | 9 | 4401.128444 |
| rollout-64 | 1803 | 1002 | 1803 | 28.265857357155 | 4 | 4421.255824 |

Validation comparisons below condition on separately selected checkpoints. They are not test results.

| Candidate − reference | Mean dB | Median dB | Image wins/losses | Two largest changes / positive net |
| --- | ---: | ---: | ---: | ---: |
| rollout-32 − rollout-5 | +0.005905505463 | +0.001336505657 | 21/11 | 70.71% |
| rollout-64 − rollout-5 | +0.004403397837 | +0.001061076367 | 17/15 | 93.23% |
| rollout-64 − rollout-32 | -0.001502107626 | -0.001723697059 | 4/28 | net nonpositive |

The JSON includes every validation point, eligibility, curriculum phases, selected/terminal tensor pins, actual charges, descriptive common-wall comparisons and common-update paired comparisons.

Plot time uses recorded UTC differences from each lease start. The ledger reports total conservative supervisor wall time; neither quantity is GPU busy time. All saved BEST/LAST model and optimizer states and all logged losses/gradients were finite. All 400 training and 32 validation files, pixels and shapes were checked.

Limitations:

- one training seed and fixed validation-noise realization; model selection uncertainty excluded.
- depth, curriculum, refinements and auxiliary endpoints form a combined recipe; no isolated causal depth effect.
- equal maximum wall allowances yield unequal updates, phase exposure and selection opportunities.
- 180px training crops do not provide full nominal spatial context at 32 or 64 stages.
- positive metrics and finite training do not prove global stability or convergence.
- BSD68 is already exposed; subsequent scores are a reused benchmark, not untouched held-out evidence.

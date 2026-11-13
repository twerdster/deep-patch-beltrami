# Deep Beltrami long-rollout comparison

Reused BSD68 benchmark, already exposed in the previous research round; not untouched held-out evidence.

| Model | Steps | Selected updates | Full-depth updates | PSNR (dB) | Median forward (ms) |
| --- | ---: | ---: | ---: | ---: | ---: |
| rollout-5 | 5 | 12750 | 12750 | 28.915269 | 87.071 |
| rollout-32 | 32 | 2959 | 2209 | 28.922632 | 557.607 |
| rollout-64 | 64 | 1803 | 1002 | 28.920873 | 1110.835 |
| released-reaction | 5 | 52000 | previous study | 28.908215 | 59.333 |

| Declared pair | Mean ΔdB | Conditional 95% interval | Improved / tied / worse |
| --- | ---: | --- | --- |
| rollout-32 vs rollout-5 | +0.007363 | [+0.003968, +0.011675] | 52 / 0 / 16 |
| rollout-64 vs rollout-5 | +0.005604 | [+0.002271, +0.009852] | 40 / 0 / 28 |
| rollout-64 vs rollout-32 | -0.001759 | [-0.002279, -0.001246] | 10 / 0 / 58 |
| rollout-5 vs released-reaction | +0.007054 | [+0.005224, +0.009070] | 56 / 0 / 12 |
| rollout-32 vs released-reaction | +0.014417 | [+0.009744, +0.020301] | 61 / 0 / 7 |
| rollout-64 vs released-reaction | +0.012658 | [+0.008042, +0.018500] | 57 / 0 / 11 |

95% paired-image percentile bootstrap intervals condition on the frozen checkpoints, one fixed noise realization and one training seed. They exclude training-seed, noise-draw, model-selection and prior benchmark-exposure uncertainty.

Depth, positive trainable metrics, identity refinements, depth curriculum, optimizer schedule and intermediate loss form one combined recipe. This is not an isolated causal depth comparison. The 180px crops lack full nominal context at32/64steps; positive metrics do not prove convergence. Nominal stages do not establish useful denoising stages.

Equal charged caps do not imply equal updates, full-depth exposure or checkpoint-selection opportunities. The recorded exposure and eligible validation counts accompany each model. The shared reaction warmstart and previously observed benchmark constrain interpretation.

Runtime is synchronized whole-image forward work on the recorded shared GPU; transfers, clipping and scoring are excluded. This is not a modern state-of-the-art, stability guarantee or mobile-deployment claim. Prior releases and iOS models remain unchanged.

Aggregate charged process wall time: 10.794593 / 12.000000 hours, including startup, failures and evaluation; not GPU busy time.

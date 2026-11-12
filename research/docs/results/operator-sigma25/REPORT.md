# Learned Beltrami operator comparison

## Frozen BSD68 comparison (σ = 25)

| Model | Role | Parameters | New updates | Mean PSNR (dB) | Median forward (ms) |
| --- | --- | ---: | ---: | ---: | ---: |
| historical | baseline | 3,195,600 | historical 40,000 | 28.865240 | 63.904 |
| frozen-control | control | 3,195,600 | 52000 | 28.897944 | 58.891 |
| reaction | candidate | 3,195,605 | 52000 | 28.908215 | 58.964 |
| precision-control | control | 3,195,600 | 25000 | 28.891054 | 87.685 |
| precision-depth | candidate | 5,112,960 | 14500 | 28.894934 | 139.432 |

| Declared comparison | Mean ΔPSNR (dB) | Conditional 95% interval | Improved / tied / worse |
| --- | ---: | --- | --- |
| frozen-control (control) vs historical (baseline) | +0.032704 | [+0.027555, +0.038027] | 65 / 0 / 3 |
| reaction (candidate) vs historical (baseline) | +0.042975 | [+0.038241, +0.047804] | 66 / 0 / 2 |
| reaction (candidate) vs frozen-control (control) | +0.010271 | [+0.008306, +0.012338] | 66 / 0 / 2 |
| precision-control (control) vs historical (baseline) | +0.025814 | [+0.020777, +0.030680] | 64 / 0 / 4 |
| precision-depth (candidate) vs historical (baseline) | +0.029695 | [+0.024900, +0.034870] | 66 / 0 / 2 |
| precision-depth (candidate) vs precision-control (control) | +0.003881 | [+0.000096, +0.008672] | 32 / 0 / 36 |

95% paired-image percentile bootstrap intervals condition on the frozen checkpoints, one fixed noise realization and one training seed. They exclude training-seed, noise-draw and model-selection uncertainty.

Runtime measures synchronized whole-image forward passes; transfers, clipping and scoring are excluded. It does not establish phone latency. Original reproduction/iOS checkpoints remain unchanged.

Measured on NVIDIA GeForce RTX 4090 with PyTorch 2.7.1+cu126; 2 warmups and 3 timed repeats per image. New updates exclude the preserved 40,000-step pretraining; checkpoint time is not total job accounting.

All models and comparison pairs follow the frozen selection; no post-test model ranking or selection is performed.

Aggregate recorded process wall time: 6.890961 hours within 12.000000 authorized hours, including startup, evaluation and failures. This is not GPU busy time.

The selection/checkpoint/evaluator/input hashes are retained in `results.json`. Per-image scores, paired changes and timing repeats accompany this report. The code and author-trained parameters are MIT licensed; dataset photographs retain separate terms. Cite Aaron Wetzler, *Topics in Computational Geometric Vision*, Technion PhD thesis, 2017, Chapter 4 §4.5.

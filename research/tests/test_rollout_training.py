"""CPU checks for progressive-depth optimization and selection boundaries."""
import json
import sys

import numpy as np
import pytest
import torch
from torch import nn

from beltrami_rollout.model import BeltramiOperator, ModelConfig
from beltrami_rollout import train


def configuration(stages=64):
    curriculum = ([{"until_fraction": 1.0, "stages": 5}] if stages == 5 else
                  [{"until_fraction": 0.05, "stages": 5},
                   {"until_fraction": 0.15, "stages": 16},
                   {"until_fraction": 0.3 if stages == 64 else 1.0, "stages": 32}])
    if stages == 64:
        curriculum.append({"until_fraction": 1.0, "stages": 64})
    return {"model": {"stages": stages}, "patch_size": 180, "microbatch": 1,
            "accumulation": 2, "warmup_fraction": 0.05, "lr_floor_fraction": 0.1,
            "auxiliary_weight": 0.05, "auxiliary_until_fraction": 0.3,
            "curriculum": curriculum, "learning_rate": 1e-5,
            "refinement_lr_multiplier": 5, "weight_decay": 0.01}


@pytest.mark.parametrize("stages", [5, 32, 64])
def test_curriculum_boundaries_reach_full_depth_and_anneal(stages):
    config = configuration(stages)
    train.validate_config(config)
    factor, active, auxiliary = train.schedule(config, -1)
    assert factor == 0.05 and active == 5 and auxiliary == 0.05
    peak_factor, _, _ = train.schedule(config, 0.05)
    assert peak_factor == 1
    assert train.schedule(config, 0.3)[2] == 0
    for row in config["curriculum"]:
        before = max(0, row["until_fraction"] - 1e-9)
        assert train.schedule(config, before)[1] == row["stages"]
    for progress in (1, 1.5):
        factor, active, auxiliary = train.schedule(config, progress)
        assert factor == pytest.approx(0.1)
        assert active == stages and auxiliary == 0
    factors = [train.schedule(config, float(p))[0] for p in np.linspace(0.05, 1, 100)]
    assert all(left >= right for left, right in zip(factors, factors[1:]))


@pytest.mark.parametrize("change", [
    {"patch_size": 160}, {"microbatch": 2}, {"accumulation": 1},
    {"warmup_fraction": 0}, {"warmup_fraction": 1},
    {"curriculum": [{"until_fraction": 0.2, "stages": 16},
                    {"until_fraction": 0.1, "stages": 64}]},
    {"curriculum": [{"until_fraction": 1, "stages": 32}]},
])
def test_invalid_cohort_config_rejected(change):
    config = configuration()
    config.update(change)
    with pytest.raises(ValueError):
        train.validate_config(config)


def test_optimizer_covers_every_trainable_parameter_once_with_expected_scales():
    model = BeltramiOperator(ModelConfig(stages=8))
    model.stages[0].dx.weight.requires_grad_(False)
    config = configuration()
    groups = train.optimizer_groups(model, config)
    membership = {id(parameter): group for group in groups for parameter in group["params"]}
    members = [id(parameter) for group in groups for parameter in group["params"]]
    expected = {id(parameter) for parameter in model.parameters() if parameter.requires_grad}
    assert len(members) == len(set(members))
    assert set(members) == expected
    assert id(model.stages[0].dx.weight) not in membership
    for name, parameter in model.named_parameters():
        if not parameter.requires_grad:
            continue
        group = membership[id(parameter)]
        index = int(name.split(".")[1])
        expected_lr = config["learning_rate"] * (5 if index >= 5 else 1)
        assert group["peak_lr"] == group["lr"] == expected_lr
        if "metric_" in name or name.endswith("reaction"):
            assert group["weight_decay"] == 0
        else:
            assert group["weight_decay"] == 0.01
    # The actual optimizer accepts these custom per-group peak LR fields and
    # applies no implicit decay to a metric/reaction parameter group.
    optimizer = torch.optim.AdamW(groups)
    assert sum(len(group["params"]) for group in optimizer.param_groups) == len(expected)


def test_finite_state_rejects_nested_failure_values():
    assert train.finite_state({"state": [torch.tensor([1.0]), None, 4.0], "step": 1})
    assert not train.finite_state({"optimizer": {"exp_avg": torch.tensor([float("nan")])}})
    assert not train.finite_state([{"loss": float("inf")}])


def test_inactive_curriculum_stages_remain_untouched_even_with_adamw_decay():
    model = BeltramiOperator(ModelConfig(stages=8, checkpoint_blocks=4))
    with torch.no_grad():
        for stage in model.stages:
            stage.projection.weight.zero_()
    inactive_before = {name: tensor.clone() for name, tensor in model.stages[5:].state_dict().items()}
    optimizer = torch.optim.AdamW(train.optimizer_groups(model, configuration()))
    noisy = torch.randn(1, 1, 3, 5)
    loss = (model(noisy, active_stages=5) - noisy - 0.3).square().mean()
    loss.backward()
    optimizer.step()
    for name, tensor in model.stages[5:].state_dict().items():
        torch.testing.assert_close(tensor, inactive_before[name], rtol=0, atol=0)
    assert all(parameter.grad is None for stage in model.stages[5:] for parameter in stage.parameters())
    assert model.stages[0].projection.weight.abs().sum() > 0


class TinyStage(nn.Module):
    def __init__(self):
        super().__init__()
        self.weight = nn.Parameter(torch.zeros(()))
        self.reaction = nn.Parameter(torch.zeros(()))


class ProbeOperator(nn.Module):
    """Replace only costly convolutions, retaining the real training loop."""
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.stages = nn.ModuleList(TinyStage() for _ in range(config.stages))

    def initialize_from_safe(self, path, expected_sha256):
        return {"source": "CPU-loop-fixture"}

    def forward(self, value, active_stages=None, return_intermediates=False):
        active = active_stages or self.config.stages
        result = value + sum(stage.weight + stage.reaction for stage in self.stages[:active])
        return (result, {active: result}) if return_intermediates else result

    def project_constraints(self):
        return None


class TinySampler:
    def __init__(self, images, patch_size, seed):
        self.rng = np.random.default_rng(seed)

    def batch(self, microbatch, sigma, offset):
        value = torch.full((microbatch, 1, 3, 5), 125.0)
        return value, value - 1


@pytest.mark.parametrize("minimum_updates,eligible", [(2, True), (3, False)])
def test_real_loop_excludes_initialization_and_insufficient_full_depth_updates(
        monkeypatch, tmp_path, minimum_updates, eligible):
    config = configuration(5)
    config.update({"source_pins": train.source_pins(), "warmstart_sha256": "CPU-fixture",
                   "sigma": 25, "offset": 25, "seed": 19, "validation_seed": 7,
                   "steps": 2, "max_cuda_gib": 10, "minimum_full_depth_updates": minimum_updates,
                   "gradient_clip": 1, "log_every": 1, "evaluate_every": 100,
                   "checkpoint_every": 1, "minimum_disk_gib": 0, "cpu_threads": 1})
    config_path = tmp_path / "config.json"
    config_path.write_text(json.dumps(config))
    manifest = tmp_path / "train.json"
    manifest.write_text("{}")
    validation = tmp_path / "validation.json"
    validation.write_text("{}")
    output = tmp_path / "output"
    monkeypatch.setattr(train, "BeltramiOperator", ProbeOperator)
    monkeypatch.setattr(train, "PatchSampler", TinySampler)
    monkeypatch.setattr(train.signal, "signal", lambda *args: None)

    def load_images(root, path, split):
        return [np.zeros((3, 5), np.uint8)], [{"pixels_sha256": split}], {}

    def pairs(images, records, sigma, seed, offset):
        yield (torch.full((1, 1, 3, 5), 125.0), np.full((1, 1, 3, 5), 100.0),
               {"path": "cpu-fixture.png", "pixels_sha256": "validation"})

    monkeypatch.setattr(train, "load_images", load_images)
    monkeypatch.setattr(train, "validation_pairs", pairs)
    monkeypatch.setattr(sys, "argv", ["train", "--config", str(config_path),
        "--data-root", str(tmp_path), "--manifest", str(manifest),
        "--validation-root", str(tmp_path), "--validation-manifest", str(validation),
        "--warmstart", str(tmp_path / "unused.pt"), "--output", str(output),
        "--gpu-lock", str(tmp_path / "lock"), "--max-gpu-hours", "0.01", "--device", "cpu"])
    train.main()
    initial = json.loads((output / "validation-0000000.json").read_text())
    terminal = json.loads((output / "validation-0000002.json").read_text())
    status = json.loads((output / "status.json").read_text())
    assert initial["selection_eligible"] is False
    assert initial["full_depth_updates"] == 0
    assert terminal["selection_eligible"] is eligible
    assert terminal["full_depth_updates"] == 2
    assert terminal["stages"] == 5
    assert status["state"] == "complete" and status["finite_optimizer"]
    assert (output / "best.pt").exists() is eligible
    if eligible:
        best = torch.load(output / "best.pt", map_location="cpu", weights_only=False)
        assert best["step"] == best["best_step"] == 2
    else:
        assert status["best_psnr"] is None and status["best_step"] is None

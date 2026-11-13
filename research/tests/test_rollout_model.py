"""CPU checks for long-rollout initialization, constraints and recomputation."""
from dataclasses import asdict
import hashlib
import os
from pathlib import Path

import pytest
import torch

from beltrami_precision.model import BeltramiOperator as PrecisionOperator
from beltrami_precision.model import ModelConfig as PrecisionConfig
from beltrami_rollout.model import BeltramiOperator, ModelConfig, source_hash


@pytest.fixture(scope="module", autouse=True)
def threads():
    previous = torch.get_num_threads()
    torch.set_num_threads(1)
    yield
    torch.set_num_threads(previous)


@pytest.fixture(scope="module")
def safe(tmp_path_factory):
    torch.manual_seed(17)
    source = PrecisionOperator(PrecisionConfig(metric="frozen", reaction=True))
    with torch.no_grad():
        for stage in source.stages:
            stage.dx.weight.normal_(0, 0.001)
            stage.dy.weight.normal_(0, 0.001)
            stage.divx.weight.normal_(0, 0.001)
            stage.divy.weight.normal_(0, 0.001)
            stage.projection.weight.normal_(0, 0.00001)
            stage.metric_weight.fill_(0.3)
            stage.metric_bias.fill_(1)
            stage.reaction.fill_(0.02)
    payload = {"format": "beltrami-operator-tensors-v1", "implementation": "beltrami_operator",
               "model_config": {"width": 80, "stages": 5, "metric": "frozen", "zero_dc": False,
                                "step_scale": 1000.0, "reaction": True},
               "sigma": 25, "grayscale_offset": 25, "training_step": 52000,
               "state_dict": source.state_dict(), "tensor_sha256": "synthetic-fixture"}
    path = tmp_path_factory.mktemp("safe") / "safe.pt"
    torch.save(payload, path)
    return path, hashlib.sha256(path.read_bytes()).hexdigest(), source


@pytest.mark.parametrize("stages,expected", [(5, 3195605), (32, 20451872), (64, 40903744)])
def test_counts_and_independent_storage(stages, expected):
    model = BeltramiOperator(ModelConfig(stages=stages))
    assert sum(parameter.numel() for parameter in model.parameters()) == expected
    assert len({stage.dx.weight.data_ptr() for stage in model.stages}) == stages
    assert model.config.refinement_scale == 1 / max(1, stages - 5)


@pytest.mark.parametrize("stages", [32, 64])
def test_long_rollout_is_exact_identity_extension(safe, stages):
    path, digest, _ = safe
    model = BeltramiOperator(ModelConfig(stages=stages)).eval()
    metadata = model.initialize_from_safe(path, digest)
    value = torch.randn(1, 1, 7, 9) * 25 + 130
    with torch.no_grad():
        expected = model(value, active_stages=5)
        actual, intermediates = model(value, return_intermediates=True)
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)
    assert set(intermediates) == ({5, 16, 32} if stages == 32 else {5, 16, 32, 64})
    assert all(torch.equal(result, expected) for result in intermediates.values())
    assert metadata["global_stability_guarantee"] is False
    assert metadata["source_namespace_sha256"] == source_hash()
    assert all(torch.isfinite(parameter).all() for parameter in model.parameters())
    assert all((tensor > 0).all() for stage in model.stages for tensor in stage.metric_parameters())


def test_precision_control_prefix_matches_existing_equation(safe):
    path, digest, frozen = safe
    model = BeltramiOperator(ModelConfig(stages=5)).eval()
    model.initialize_from_safe(path, digest)
    # Use the same positive roundtrip in both implementations; no legacy-FP32
    # bit-parity claim is made by this independently versioned branch.
    reference = PrecisionOperator(PrecisionConfig(metric="positive", reaction=True)).eval()
    reference.load_state_dict(model.state_dict())
    value = torch.randn(2, 1, 7, 11) * 20 + 120
    with torch.no_grad():
        torch.testing.assert_close(model(value), reference(value), rtol=0, atol=0)
        assert torch.isfinite(frozen(value)).all()


def test_checkpoint_recompute_outputs_and_gradients_are_exact(safe):
    path, digest, _ = safe
    plain = BeltramiOperator(ModelConfig(stages=8, checkpoint_blocks=0)).train()
    plain.initialize_from_safe(path, digest)
    recomputed = BeltramiOperator(ModelConfig(stages=8, checkpoint_blocks=4)).train()
    recomputed.load_state_dict(plain.state_dict())
    value = torch.randn(1, 1, 7, 9) * 20 + 120
    target = value - 0.3
    outputs = []
    for model in (plain, recomputed):
        output, intermediates = model(value, return_intermediates=True)
        loss = (output - target).square().mean() + 0.05 * (intermediates[5] - target).square().mean()
        loss.backward()
        assert torch.isfinite(loss)
        assert all(p.grad is not None and torch.isfinite(p.grad).all() for p in model.parameters())
        for stage in model.stages[5:]:
            assert stage.projection.weight.grad.abs().sum() > 0
            assert stage.reaction.grad.abs() > 0
        outputs.append(output.detach())
    torch.testing.assert_close(*outputs, rtol=0, atol=0)
    for left, right in zip(plain.parameters(), recomputed.parameters()):
        torch.testing.assert_close(left.grad, right.grad, rtol=0, atol=0)


def test_curriculum_only_activates_requested_stages(safe):
    path, digest, _ = safe
    model = BeltramiOperator(ModelConfig(stages=8, checkpoint_blocks=0))
    model.initialize_from_safe(path, digest)
    value = torch.randn(1, 1, 7, 9) + 120
    model(value, active_stages=5).mean().backward()
    assert all(parameter.grad is None for stage in model.stages[5:] for parameter in stage.parameters())
    for invalid in (0, 4, 9, 5.5, "5"):
        with pytest.raises(ValueError, match="active_stages"):
            model(value, active_stages=invalid)


def test_pin_and_config_rejected(safe):
    path, digest, _ = safe
    with pytest.raises(ValueError, match="SHA256"):
        BeltramiOperator(ModelConfig(stages=5)).initialize_from_safe(path, "wrong")
    for values in ({"metric_precision": "float32"}, {"metric": "frozen"},
                   {"width": 40}, {"zero_dc": True}, {"checkpoint_blocks": -1},
                   {"refinement_scale": 0}):
        with pytest.raises(ValueError):
            ModelConfig(**values)


def test_reaction_projection(safe):
    path, digest, _ = safe
    model = BeltramiOperator(ModelConfig(stages=5))
    model.initialize_from_safe(path, digest)
    with torch.no_grad():
        model.stages[0].reaction.fill_(-1)
        model.stages[1].reaction.fill_(2)
    model.project_constraints()
    assert model.stages[0].reaction == 0
    assert model.stages[1].reaction == 1


def test_actual_pinned_reaction64_identity_and_finite_gradient():
    configured = os.environ.get("BELTRAMI_REACTION_SAFE")
    if not configured:
        pytest.skip("Set BELTRAMI_REACTION_SAFE for actual author-weight integration")
    path = Path(configured)
    model = BeltramiOperator(ModelConfig(stages=64, checkpoint_blocks=4))
    model.initialize_from_safe(path, "9a085c0745136eaee86bfaf0b2e1eb31926eb3bb0526169134fa3a89cf2cfed5")
    value = torch.randn(1, 1, 7, 9) * 25 + 130
    with torch.no_grad():
        expected = model(value, active_stages=5)
    prediction = model(value)
    torch.testing.assert_close(prediction, expected, rtol=0, atol=0)
    (prediction - value).square().mean().backward()
    assert all(p.grad is not None and torch.isfinite(p.grad).all() for p in model.parameters())
    assert model.stages[-1].projection.weight.grad.abs().sum() > 0

import os
from pathlib import Path
import sys
import numpy as np
import pytest
import torch
from beltrami_operator.model import BeltramiOperator, ModelConfig


@pytest.fixture(scope="module")
def original():
    directory = os.environ.get("BELTRAMI_MODEL_DIR")
    if not directory:
        pytest.skip("Set BELTRAMI_MODEL_DIR for archived checkpoint parity")
    return Path(directory)


@pytest.mark.parametrize("shape", [(1, 1, 17, 23), (1, 1, 1, 19), (2, 1, 11, 7)])
def test_original_graph_parity(original, shape):
    from deep_patch_beltrami.inference import DeepBeltrami
    torch.set_num_threads(2)
    rng = np.random.default_rng(141)
    value = rng.normal(130, 60, shape).astype(np.float32)
    reference = DeepBeltrami(original, 25)
    expected = torch.from_numpy(value.copy())
    for graph in reference.graphs:
        expected = graph(expected, debug=True)
    operator = BeltramiOperator().eval()
    operator.initialize_original(original)
    with torch.no_grad():
        actual = operator(torch.from_numpy(value))
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)


@pytest.mark.parametrize("config", [ModelConfig(width=112), ModelConfig(stages=8),
                                    ModelConfig(reaction=True), ModelConfig(metric="positive")])
def test_warmstart_preserves_function(original, config):
    torch.set_num_threads(2)
    torch.manual_seed(81)
    value = torch.randn(1, 1, 23, 17) * 50 + 130
    reference = BeltramiOperator().eval()
    reference.initialize_original(original)
    operator = BeltramiOperator(config).eval()
    operator.initialize_original(original)
    with torch.no_grad():
        expected, actual = reference(value), operator(value)
    torch.testing.assert_close(actual, expected, rtol=0, atol=3e-4)


def test_positive_metric_and_backward(original):
    torch.set_num_threads(2)
    operator = BeltramiOperator(ModelConfig(metric="positive", reaction=True))
    operator.initialize_original(original)
    value = torch.randn(2, 1, 11, 13) * 25 + 130
    result = operator(value)
    result.square().mean().backward()
    assert torch.isfinite(result).all()
    for name, parameter in operator.named_parameters():
        if parameter.requires_grad:
            assert parameter.grad is not None, name
            assert torch.isfinite(parameter.grad).all(), name
    for stage in operator.stages:
        weight, bias = stage.metric_parameters()
        assert weight.min() > 0 and bias.min() > 0


def test_reaction_projection():
    operator = BeltramiOperator(ModelConfig(reaction=True))
    with torch.no_grad():
        operator.stages[0].reaction.fill_(-1)
        operator.stages[1].reaction.fill_(2)
    operator.project_constraints()
    assert operator.stages[0].reaction == 0
    assert operator.stages[1].reaction == 1


def test_new_wide_features_receive_gradient(original):
    torch.set_num_threads(2)
    operator = BeltramiOperator(ModelConfig(width=112))
    operator.initialize_original(original)
    value = torch.randn(1, 1, 13, 11) * 25 + 130
    operator(value).square().mean().backward()
    assert operator.stages[0].dx.weight.grad[80:].abs().sum() > 0

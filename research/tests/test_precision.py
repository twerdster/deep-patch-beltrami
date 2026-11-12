"""CPU protocol checks for the independently versioned precision experiment."""
import hashlib
import ast
import os
from pathlib import Path

import numpy as np
import pytest
import torch

from beltrami_operator.model import BeltramiOperator as HistoricalOperator
from beltrami_operator.model import ModelConfig as HistoricalConfig
from beltrami_precision.data import PatchSampler, load_images
from beltrami_precision.model import BeltramiOperator, ModelConfig


ROOT = Path(__file__).resolve().parents[1]
LEGACY_SHA256 = {
    "__init__.py": "b4f6cb7b42e8d6261cd2f9c2c6cebc329930608d87cc83078c1a88a80d45c187",
    "data.py": "8bc40e33f0b8b208d88c2aab5f09e618c0c40437635f21925b0dbcd849f24f24",
    "model.py": "f257ede9e6e5b79bbd31df49a12247f1987681f40d66a51599e2898686bdda8d",
    "train.py": "f87c34123d11c7a2b9a43076e09120c0d90b560c84727935f529ee83703bfff9",
}


@pytest.fixture(scope="module")
def original():
    value = os.environ.get("BELTRAMI_MODEL_DIR")
    if not value:
        pytest.skip("Set BELTRAMI_MODEL_DIR for original-checkpoint integration")
    torch.set_num_threads(2)
    return Path(value)


def test_legacy_numerical_source_is_unchanged():
    for name, expected in LEGACY_SHA256.items():
        assert hashlib.sha256((ROOT / "src/beltrami_operator" / name).read_bytes()).hexdigest() == expected


def test_data_and_training_protocol_are_copied_exactly():
    assert (ROOT / "src/beltrami_operator/data.py").read_bytes() == (
        ROOT / "src/beltrami_precision/data.py").read_bytes()
    historical = ast.parse((ROOT / "src/beltrami_operator/train.py").read_text())
    precision = ast.parse((ROOT / "src/beltrami_precision/train.py").read_text())
    for node in ast.walk(precision):
        if isinstance(node, ast.Assign) and any(
            isinstance(target, ast.Name) and target.id == "metadata" for target in node.targets):
            retained = [(key, value) for key, value in zip(node.value.keys, node.value.values)
                        if key.value not in ("implementation", "numerics")]
            node.value.keys = [item[0] for item in retained]
            node.value.values = [item[1] for item in retained]
    assert ast.dump(historical) == ast.dump(precision)


@pytest.mark.parametrize("shape", [(1, 1, 17, 23), (1, 1, 1, 19), (2, 1, 11, 7)])
def test_float32_arithmetic_matches_legacy_exactly(original, shape):
    value = torch.from_numpy(np.random.default_rng(141).normal(130, 60, shape).astype(np.float32))
    historical = HistoricalOperator(HistoricalConfig(metric="positive"))
    historical.initialize_original(original)
    precision = BeltramiOperator(ModelConfig(metric="positive", metric_precision="float32"))
    precision.initialize_original(original)
    with torch.no_grad():
        torch.testing.assert_close(precision(value), historical(value), rtol=0, atol=0)


@pytest.mark.parametrize("shape", [(1, 1, 17, 23), (1, 1, 1, 19), (2, 1, 11, 7)])
def test_double_metric_keeps_native_shape_and_small_initial_rounding(original, shape):
    value = torch.from_numpy(np.random.default_rng(141).normal(130, 60, shape).astype(np.float32))
    historical = HistoricalOperator(HistoricalConfig(metric="positive"))
    historical.initialize_original(original)
    precision = BeltramiOperator(ModelConfig(metric="positive"))
    metadata = precision.initialize_original(original)
    with torch.no_grad():
        expected, actual = historical(value), precision(value)
    assert actual.shape == value.shape and actual.dtype == torch.float32
    assert all(parameter.dtype == torch.float32 for parameter in precision.parameters())
    assert torch.isfinite(actual).all()
    assert metadata["metric_precision"] == "float64"
    maximum_difference = float((actual - expected).abs().max())
    print(f"precision fixture {shape}: maximum legacy difference {maximum_difference:.12g} pixels")
    # The singleton axis is unusually ill conditioned after repeated boundaries;
    # its measured rounding change is about0.014pixels, so bit parity is false.
    tolerance = 0.02 if min(shape[-2:]) == 1 else 3e-4
    torch.testing.assert_close(actual, expected, rtol=0, atol=tolerance)


def test_precision_depth_identity_warmstart_and_finite_backward(original):
    value = torch.randn(2, 1, 13, 11) * 25 + 130
    control = BeltramiOperator(ModelConfig(metric="positive"))
    control.initialize_original(original)
    deeper = BeltramiOperator(ModelConfig(metric="positive", stages=8))
    deeper.initialize_original(original)
    actual = deeper(value)
    with torch.no_grad():
        torch.testing.assert_close(actual, control(value), rtol=0, atol=0)
    actual.square().mean().backward()
    assert all(parameter.grad is not None and torch.isfinite(parameter.grad).all()
               for parameter in deeper.parameters())


def test_invalid_precision_is_rejected():
    with pytest.raises(ValueError, match="metric_precision"):
        BeltramiOperator(ModelConfig(metric_precision="float16"))


def test_actual_depth_failure_batch_recovers_finite_forward_and_backward():
    configured = os.environ.get("BELTRAMI_DEPTH_FAILURE_DIR")
    if not configured:
        pytest.skip("Set BELTRAMI_DEPTH_FAILURE_DIR for trusted actual-failure integration")
    project = Path(configured)
    path = project / "runs/pilot-deeper-positive-001/failure.pt"
    assert hashlib.sha256(path.read_bytes()).hexdigest() == (
        "284d834a3c360f11b59b7d7bb9dd00817ee27f8a672d816514489690015ef0c9")
    checkpoint = torch.load(path, map_location="cpu", weights_only=False)
    config = checkpoint["config"]
    images, _, _ = load_images(project / "data/original400", project / "data/train400.json", "train")
    sampler = PatchSampler(images, config["patch_size"], config["seed"])
    for _ in range(checkpoint["step"] + 1):
        noisy, clean = sampler.batch(config["batch_size"], config["sigma"], config["offset"])
    assert sampler.rng.bit_generator.state == checkpoint["sampler_rng"]
    assert hashlib.sha256(noisy.numpy().tobytes()).hexdigest() == (
        "abb2203b78be9b4cfa68b0809b5774f3c421627385d162dec557d4e3382119e4")
    torch.set_num_threads(4)
    historical = HistoricalOperator(HistoricalConfig(**config["model"]))
    historical.load_state_dict(checkpoint["model"])
    with torch.no_grad():
        assert not torch.isfinite(historical(noisy)).all()
    precision = BeltramiOperator(ModelConfig(**config["model"], metric_precision="float64"))
    precision.load_state_dict(checkpoint["model"])
    prediction = precision(noisy)
    assert torch.isfinite(prediction).all()
    loss = (prediction - clean).square().mean() / 255.0 ** 2
    loss.backward()
    assert torch.isfinite(loss)
    assert all(parameter.grad is not None and torch.isfinite(parameter.grad).all()
               for parameter in precision.parameters())
    norm = torch.nn.utils.clip_grad_norm_(precision.parameters(), 1.0)
    assert torch.isfinite(norm)

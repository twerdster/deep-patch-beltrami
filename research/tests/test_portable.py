"""Portable artifacts preserve tensor bytes and use the declared image protocol."""
import hashlib
import importlib.util
import json
from pathlib import Path

import numpy as np
import pytest
import torch

from beltrami_operator.model import BeltramiOperator, ModelConfig
from beltrami_precision.model import BeltramiOperator as PreciseOperator
from beltrami_precision.model import ModelConfig as PreciseConfig
from beltrami_research import artifacts
from beltrami_research.cli import denoise, image_pixels
from deep_patch_beltrami.assets import metadata
from PIL import Image


ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture(autouse=True)
def threads():
    torch.set_num_threads(2)


@pytest.fixture
def model():
    torch.manual_seed(19)
    result = BeltramiOperator(ModelConfig(metric="positive", reaction=True))
    with torch.no_grad():
        for stage in result.stages:
            stage.dx.weight.mul_(1e-3)
            stage.dy.weight.mul_(1e-3)
            stage.divx.weight.mul_(1e-3)
            stage.divy.weight.mul_(1e-3)
            stage.projection.weight.mul_(1e-3)
    return result


@pytest.fixture
def checkpoint(tmp_path, model):
    config = {"sigma": 25, "offset": 25, "model": {
        "width": 80, "stages": 5, "metric": "positive", "reaction": True}}
    path = tmp_path / "config.json"
    path.write_text(json.dumps(config))
    name = "PB_25_80_5__iter_40000.caffemodel"
    original = next(item for item in metadata("models.json")["files"] if item["name"] == name)
    value = {"model": model.state_dict(), "config": config,
             "config_sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
             "source_sha256": artifacts.source_hash("beltrami_operator"), "step": 43,
             "gpu_seconds": 1.5,
             "best_psnr": 28.0,
             "manifest_sha256": "a" * 64, "validation_manifest_sha256": "b" * 64,
             "initialization": {"source_name": name, "source_sha256": original["sha256"], "source_sigma": 25},
             "optimizer": {"private": "not-for-export"}, "sampler_rng": "not-for-export",
             "host": "not-for-export", "paths": {"local": "not-for-export"}}
    trained = tmp_path / "trusted.pt"
    torch.save(value, trained)
    return trained, path


def test_public_sources_match_frozen_training_namespaces():
    pins = json.loads((ROOT / "docs/source-hashes.json").read_text())
    for implementation, record in pins.items():
        assert artifacts.source_hash(implementation) == record["sha256"]
        for name, expected in record["files"].items():
            assert hashlib.sha256((ROOT / "src" / implementation / name).read_bytes()).hexdigest() == expected


def test_tensor_only_export_is_exact_and_strips_local_state(tmp_path, checkpoint, model):
    original, config = checkpoint
    output = tmp_path / "operator.pt"
    artifacts.export(original, config, "beltrami_operator", output)
    public = torch.load(output, weights_only=True)
    assert not {"optimizer", "sampler_rng", "host", "paths"} & public.keys()
    assert b"not-for-export" not in output.read_bytes()
    actual, values = artifacts.load(output)
    assert values["training_step"] == 43
    assert artifacts.tensor_hash(actual.state_dict()) == artifacts.tensor_hash(model.state_dict())
    for name, tensor in model.state_dict().items():
        assert torch.equal(tensor, actual.state_dict()[name])
    with pytest.raises(FileExistsError):
        artifacts.export(original, config, "beltrami_operator", output)


@pytest.mark.parametrize("field,value,error", [
    ("source_sha256", "0" * 64, "source"),
    ("tensor_sha256", "0" * 64, "digest"),
    ("implementation", "unsupported", "Unsupported")])
def test_inference_rejects_changed_provenance(tmp_path, checkpoint, field, value, error):
    original, config = checkpoint
    path = tmp_path / "model.pt"
    artifacts.export(original, config, "beltrami_operator", path)
    public = torch.load(path, weights_only=True)
    public[field] = value
    torch.save(public, path)
    with pytest.raises(ValueError, match=error):
        artifacts.load(path)


def test_inference_rejects_nonfinite_weights(tmp_path, checkpoint):
    original, config = checkpoint
    path = tmp_path / "model.pt"
    artifacts.export(original, config, "beltrami_operator", path)
    public = torch.load(path, weights_only=True)
    next(iter(public["state_dict"].values())).fill_(float("inf"))
    torch.save(public, path)
    with pytest.raises(FloatingPointError):
        artifacts.load(path)


@pytest.mark.parametrize("shape", [(11, 17), (1, 19), (13, 9, 3)])
def test_image_adapter_preserves_physical_axes_channels_and_offset(shape):
    class Echo(torch.nn.Module):
        def forward(self, value):
            self.seen = value.clone()
            return value
    model = Echo()
    image = np.arange(np.prod(shape), dtype=np.float32).reshape(shape) % 256
    output = denoise(model, image, 25, "cpu")
    np.testing.assert_array_equal(output, image)
    if len(shape) == 2:
        np.testing.assert_array_equal(model.seen[0, 0].numpy(), image.T + 25)
    else:
        np.testing.assert_array_equal(model.seen[:, 0].numpy(), image.transpose(2, 1, 0))


def test_finite_gate_precedes_prediction_clipping():
    class Bad(torch.nn.Module):
        def forward(self, value):
            return value * float("inf")
    with pytest.raises(FloatingPointError, match="before clipping"):
        denoise(Bad(), np.ones((7, 9), dtype=np.float32), 25, "cpu")


def test_precise_operator_matches_float32_on_well_conditioned_synthetic_fixture(model):
    precise = PreciseOperator(PreciseConfig(metric="positive", reaction=True))
    precise.load_state_dict(model.state_dict(), strict=True)
    image = torch.randn(2, 1, 17, 11) * 25 + 130
    prediction = precise(image)
    torch.testing.assert_close(prediction, model(image), atol=2e-5, rtol=0)
    prediction.square().mean().backward()
    assert torch.isfinite(prediction).all()
    assert all(parameter.grad is not None and torch.isfinite(parameter.grad).all()
               for parameter in precise.parameters())


def test_dataset_source_identity_exclusion():
    evidence = json.loads((ROOT / "manifests/source-identities.json").read_text())
    train = {Path(item["source"]).stem for item in evidence["training_sources"]}
    bsd68 = {Path(item["source"]).stem for item in evidence["bsd68_matches"]}
    validation = {Path(item["source"]).stem for item in evidence["validation"]}
    assert len(train) == 400 and len(bsd68) == 68 and len(validation) == 32
    assert not train & validation and not train & bsd68 and not bsd68 & validation


def test_new_budget_never_overwrites_old_accounting(tmp_path):
    spec = importlib.util.spec_from_file_location("create_budget", ROOT / "tools/create_budget.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    output = tmp_path / "BUDGET.json"
    created = module.create(output, 2)
    assert created["authorized_gpu_seconds"] == 7200
    with pytest.raises(FileExistsError):
        module.create(output, 12)
    with pytest.raises(ValueError):
        module.create(tmp_path / "invalid.json", 13)


@pytest.mark.parametrize("mode", ["1", "L", "I", "I;16", "F", "RGB"])
def test_file_decoder_keeps_grayscale_modes_and_historical_conversion(tmp_path, mode):
    image = Image.fromarray(np.arange(9 * 13, dtype=np.uint8).reshape(9, 13)).convert(mode)
    path = tmp_path / "fixture.tiff"
    image.save(path)
    actual = image_pixels(path)
    gray = mode != "RGB"
    assert actual.shape == ((9, 13) if gray else (9, 13, 3))
    np.testing.assert_array_equal(actual, np.asarray(image.convert("L" if gray else "RGB"), dtype=np.float32))


def test_resume_preflight_guards_custom_manifest_and_optimizer(tmp_path, checkpoint, monkeypatch):
    from deep_patch_beltrami import assets
    trusted, config = checkpoint
    value = torch.load(trusted, weights_only=False)
    directory = tmp_path / "original"
    directory.mkdir()
    original = directory / value["initialization"]["source_name"]
    original.write_bytes(b"synthetic original model fixture")
    original_sha = hashlib.sha256(original.read_bytes()).hexdigest()
    value["initialization"]["source_sha256"] = original_sha
    original_metadata = assets.metadata
    def fixture_metadata(name):
        result = original_metadata(name)
        if name == "models.json":
            for item in result["files"]:
                if item["name"] == original.name:
                    item["sha256"] = original_sha
        return result
    monkeypatch.setattr(assets, "metadata", fixture_metadata)
    train, validation = tmp_path / "train.json", tmp_path / "validation.json"
    train.write_text('{"images": []}')
    validation.write_text('{"images": [], "split": "validation"}')
    value["manifest_sha256"] = hashlib.sha256(train.read_bytes()).hexdigest()
    value["validation_manifest_sha256"] = hashlib.sha256(validation.read_bytes()).hexdigest()
    torch.save(value, trusted)
    assert artifacts.validate_resume(trusted, config, "beltrami_operator", train, validation, directory)["step"] == 43
    train.write_text("changed manifest")
    with pytest.raises(ValueError, match="manifest_sha256"):
        artifacts.validate_resume(trusted, config, "beltrami_operator", train, validation, directory)
    train.write_text('{"images": []}')
    value["optimizer"]["moment"] = torch.tensor([float("nan")])
    torch.save(value, trusted)
    with pytest.raises(FloatingPointError, match="optimizer"):
        artifacts.validate_resume(trusted, config, "beltrami_operator", train, validation, directory)
    value["optimizer"].pop("moment")
    torch.save(value, trusted)
    original.write_bytes(b"changed original checkpoint")
    with pytest.raises(ValueError, match="original model bytes"):
        artifacts.validate_resume(trusted, config, "beltrami_operator", train, validation, directory)

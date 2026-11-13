"""Synthetic tensor-only long-rollout loading and physical image adapters."""
from dataclasses import asdict
import json
from pathlib import Path
import subprocess
import sys

import numpy as np
from PIL import Image
import pytest
import torch

from beltrami_research import rollout_artifacts as artifacts
from beltrami_research import rollout_cli as cli
from beltrami_rollout.model import BeltramiOperator, ModelConfig


@pytest.fixture(scope="module", autouse=True)
def threads():
    previous = torch.get_num_threads()
    torch.set_num_threads(1)
    yield
    torch.set_num_threads(previous)


@pytest.fixture(scope="module")
def safe(tmp_path_factory):
    root = tmp_path_factory.mktemp("safe_rollout")
    model = BeltramiOperator(ModelConfig(stages=5)).eval()
    with torch.no_grad():
        for stage in model.stages:
            stage.dx.weight.mul_(0.01)
            stage.dy.weight.mul_(0.01)
            stage.projection.weight.zero_()
            stage.reaction.zero_()
    state = model.state_dict()
    provenance = {key: "a" * 64 for key in artifacts._PROVENANCE_HASHES}
    provenance.update(warmstart_sha256=artifacts.WARMSTART_SHA256,
                      license="MIT", citation=artifacts.CITATION)
    value = {"format": artifacts.FORMAT, "implementation": "beltrami_rollout",
             "source_pins": artifacts.source_pins(), "model_config": asdict(model.config),
             "numerics": artifacts.numerical_metadata(model.config),
             "sigma": 25, "grayscale_offset": 25, "state_dict": state,
             "tensor_sha256": artifacts.tensor_hash(state),
             "parameters": sum(parameter.numel() for parameter in model.parameters()),
             "training_step": 20, "full_depth_updates": 20, "phase_updates": {"5": 20},
             "provenance": provenance}
    path = root / "synthetic-safe.pt"
    torch.save(value, path)
    return path, value, model


def test_reader_uses_only_weights_and_preserves_tensor_bytes(safe, monkeypatch):
    path, expected, reference = safe
    original = torch.load
    calls = []
    def read(*args, **kwargs):
        calls.append(kwargs)
        return original(*args, **kwargs)
    monkeypatch.setattr(artifacts.torch, "load", read)
    model, metadata = artifacts.load(path, expected_sha256=artifacts.file_sha(path))
    assert calls == [{"map_location": "cpu", "weights_only": True}]
    assert model.training is False
    for name, tensor in reference.state_dict().items():
        assert tensor.numpy().tobytes() == model.state_dict()[name].numpy().tobytes()
    native = torch.randn(1, 1, 9, 13) + 130
    with torch.no_grad():
        assert torch.equal(model(native), reference(native))
    assert metadata["tensor_sha256"] == expected["tensor_sha256"]
    public = json.dumps({key: value for key, value in metadata.items() if key != "state_dict"})
    assert not any(token in public for token in ("/home/", "/Users/", "optimizer", "rng"))


def test_full_file_pin_precedes_deserialization(safe, monkeypatch):
    def forbidden(*args, **kwargs):
        raise AssertionError("No deserialization on wrong external hash")
    monkeypatch.setattr(artifacts.torch, "load", forbidden)
    with pytest.raises(ValueError, match="SHA256"):
        artifacts.load_safe(safe[0], expected_sha256="0" * 64)


def test_installed_source_is_pinned_independently_of_payload(safe, monkeypatch):
    assert artifacts.source_pins() == artifacts.SOURCE_PINS
    monkeypatch.setattr(artifacts, "source_pins", lambda: {"beltrami_rollout": "b" * 64})
    with pytest.raises(ValueError, match="frozen numerical"):
        artifacts.load_safe(safe[0])


@pytest.mark.parametrize("field,value,message", [
    ("format", "raw-training-checkpoint", "format"),
    ("source_pins", {}, "source pins"),
    ("private_path", "/home/private", "format"),
    ("sigma", 50, "sigma"), ("grayscale_offset", 0, "offset"),
    ("full_depth_updates", 19, "full-depth"),
    ("training_step", True, "full-depth"),
    ("phase_updates", {"5": 19}, "chronology"),
    ("phase_updates", {"5": 20, "private": 0}, "chronology"),
    ("phase_updates", {"5": True}, "chronology"),
    ("tensor_sha256", "bad", "digest"),
    ("parameters", 1, "parameter count"),
])
def test_reader_rejects_changed_protocol_and_identity(safe, monkeypatch, field, value, message):
    payload = dict(safe[1], **{field: value})
    monkeypatch.setattr(artifacts.torch, "load", lambda *args, **kwargs: payload)
    with pytest.raises(ValueError, match=message):
        artifacts.load_safe(safe[0])


@pytest.mark.parametrize("field,value", [
    ("checkpoint_blocks", 0), ("stages", 6), ("refinement_scale", 0.25),
    ("reaction", 1), ("zero_dc", 0), ("width", 80.0),
    ("metric_precision", "float32"), ("private_path", "/Users/private"),
])
def test_configuration_must_match_frozen_cohort(safe, monkeypatch, field, value):
    config = dict(safe[1]["model_config"], **{field: value})
    payload = dict(safe[1], model_config=config)
    monkeypatch.setattr(artifacts.torch, "load", lambda *args, **kwargs: payload)
    with pytest.raises(ValueError):
        artifacts.load_safe(safe[0])


@pytest.mark.parametrize("field,value", [
    ("private_path", "/Users/private"), ("training_config_sha256", "not-a-hash"),
    ("license", "private"), ("warmstart_sha256", "b" * 64),
    ("citation", "/home/private/thesis"),
])
def test_provenance_is_whitelisted_and_sanitized(safe, monkeypatch, field, value):
    payload = dict(safe[1], provenance=dict(safe[1]["provenance"], **{field: value}))
    monkeypatch.setattr(artifacts.torch, "load", lambda *args, **kwargs: payload)
    with pytest.raises(ValueError, match="provenance"):
        artifacts.load_safe(safe[0])


def test_numerical_metadata_cannot_claim_clamps_or_stability(safe, monkeypatch):
    numerics = dict(safe[1]["numerics"], determinant_clipping=True)
    monkeypatch.setattr(artifacts.torch, "load", lambda *args, **kwargs: dict(safe[1], numerics=numerics))
    with pytest.raises(ValueError, match="numerical"):
        artifacts.load_safe(safe[0])


@pytest.mark.parametrize("change,message", [("shape", "shapes"), ("name", "names"),
                                            ("dtype", "float32"), ("nan", "float32")])
def test_tensors_must_have_exact_names_shapes_dtype_and_finite_values(safe, monkeypatch, change, message):
    state = dict(safe[1]["state_dict"])
    name = next(iter(state))
    if change == "shape":
        state[name] = state[name][..., :-1].clone()
    elif change == "name":
        state["private"] = state.pop(name)
    elif change == "dtype":
        state[name] = state[name].half()
    else:
        state[name] = state[name].clone().fill_(float("nan"))
    payload = dict(safe[1], state_dict=state, tensor_sha256=artifacts.tensor_hash(state))
    monkeypatch.setattr(artifacts.torch, "load", lambda *args, **kwargs: payload)
    with pytest.raises((ValueError, FloatingPointError), match=message):
        artifacts.load_safe(safe[0])


class UnsafePayload:
    pass


def test_reader_refuses_raw_python_objects(tmp_path):
    path = tmp_path / "unsafe.pt"
    torch.save(UnsafePayload(), path)
    with pytest.raises(Exception, match="Weights only load failed"):
        artifacts.load_safe(path)


@pytest.mark.parametrize("shape", [(7, 11), (1, 13), (17, 1), (9, 13, 3), (1, 11, 3), (13, 1, 3)])
def test_adapter_applies_offset_per_channel_without_axis_mixup(shape):
    class Echo(torch.nn.Module):
        def forward(self, value):
            self.seen = value.clone()
            return value
    pixels = (np.arange(np.prod(shape), dtype=np.float32).reshape(shape) - 35) * 1.25
    model = Echo()
    output = cli.denoise(model, pixels)
    assert output.shape == pixels.shape and output.dtype == np.float32
    assert np.array_equal(output, pixels.clip(0, 255))
    channels = pixels[None] if pixels.ndim == 2 else pixels.transpose(2, 0, 1)
    assert np.array_equal(model.seen.numpy(), channels.transpose(0, 2, 1)[:, None] + 25)
    assert model.seen.dtype == torch.float32
    assert model.seen.min() < 25  # No clipping of negative noisy inputs.


def test_actual_safe_model_adapter_matches_independent_rgb_passes(safe):
    model, _ = artifacts.load_safe(safe[0])
    pixels = np.arange(5 * 7 * 3, dtype=np.float32).reshape(5, 7, 3)
    output = cli.denoise(model, pixels)
    independent = np.stack([cli.denoise(model, pixels[:, :, channel]) for channel in range(3)], axis=-1)
    assert np.array_equal(output, independent)


@pytest.mark.parametrize("stages", [32, 64])
def test_safe_reader_runs_actual_supported_long_depths_without_disk_assets(safe, monkeypatch, stages):
    reference = BeltramiOperator(ModelConfig(stages=stages)).eval()
    with torch.no_grad():
        for stage in reference.stages:
            stage.dx.weight.mul_(0.01)
            stage.dy.weight.mul_(0.01)
            stage.projection.weight.zero_()
            stage.reaction.zero_()
    state = reference.state_dict()
    payload = dict(safe[1], state_dict=state, tensor_sha256=artifacts.tensor_hash(state),
                   model_config=asdict(reference.config),
                   numerics=artifacts.numerical_metadata(reference.config),
                   parameters=sum(parameter.numel() for parameter in reference.parameters()),
                   phase_updates={str(stages): 20})
    monkeypatch.setattr(artifacts.torch, "load", lambda *args, **kwargs: payload)
    model, metadata = artifacts.load_safe(safe[0])
    pixels = np.arange(3 * 5, dtype=np.float32).reshape(3, 5)
    assert np.array_equal(cli.denoise(model, pixels), pixels)
    assert metadata["model_config"]["stages"] == stages


@pytest.mark.parametrize("pixels", [np.zeros((0, 4)), np.zeros((3, 4, 1)),
                                    np.zeros((3, 4, 4)), np.zeros((2,)),
                                    np.full((3, 4), "a"), np.zeros((3, 4), dtype=bool)])
def test_adapter_rejects_ambiguous_geometry_or_channels(pixels):
    with pytest.raises(ValueError, match="grayscale"):
        cli.denoise(torch.nn.Identity(), pixels)


def test_nonfinite_values_are_rejected_before_output_clipping():
    with pytest.raises(FloatingPointError, match="input"):
        cli.denoise(torch.nn.Identity(), np.array([[np.nan]], dtype=np.float32))
    class Bad(torch.nn.Module):
        def forward(self, value):
            return value * float("inf")
    with pytest.raises(FloatingPointError, match="before clipping"):
        cli.denoise(Bad(), np.ones((3, 4), dtype=np.float32))
    with pytest.raises(ValueError, match="offset"):
        cli.denoise(torch.nn.Identity(), np.ones((3, 4), dtype=np.float32), offset=0)


@pytest.mark.parametrize("shape", [(7, 11), (5, 9, 3)])
def test_actual_safe_cli_end_to_end_png(safe, tmp_path, shape):
    pixels = (np.arange(np.prod(shape)).reshape(shape) % 256).astype(np.uint8)
    source, output = tmp_path / "input.png", tmp_path / "output.png"
    Image.fromarray(pixels).save(source)
    completed = subprocess.run([sys.executable, "-m", "beltrami_research.rollout_cli",
                                str(source), str(output), "--checkpoint", str(safe[0]),
                                "--expected-sha256", artifacts.file_sha(safe[0])],
                               capture_output=True, text=True, check=True)
    result = json.loads(completed.stdout)
    assert result["shape"] == list(shape)
    assert result["per_channel_offset"] == 25
    assert np.array_equal(np.asarray(Image.open(output)), pixels)
    with pytest.raises(FileExistsError):
        cli.main([str(source), str(output), "--checkpoint", str(safe[0])])
    assert np.array_equal(np.asarray(Image.open(output)), pixels)


def test_cli_preserves_existing_output_before_model_load(safe, tmp_path, monkeypatch):
    output = tmp_path / "existing.png"
    output.write_bytes(b"existing evidence")
    def forbidden(*args, **kwargs):
        raise AssertionError("Existing output must be rejected first")
    monkeypatch.setattr(cli, "load_safe", forbidden)
    with pytest.raises(FileExistsError):
        cli.main(["absent.png", str(output), "--checkpoint", str(safe[0])])
    assert output.read_bytes() == b"existing evidence"

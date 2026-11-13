"""Actual CPU checkpoint/export/crop identity using only synthetic PNGs."""
import hashlib
import importlib.util
import json
from pathlib import Path
import subprocess
import sys

import numpy as np
from PIL import Image
import pytest
import torch


ROOT = Path(__file__).resolve().parents[1]


def module(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    value = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(value)
    return value


subject = module("rollout_cpu_verifier_subject", ROOT / "tools/verify_rollout_export_cpu.py")
base = module("rollout_cpu_verifier_fixtures", ROOT / "tests/test_rollout_export.py")


@pytest.fixture(scope="module")
def prepared(tmp_path_factory):
    torch.set_num_threads(1)
    value = base.prepared.__wrapped__(tmp_path_factory)
    pixels = (np.arange(33 * 41).reshape(33, 41) % 256).astype(np.uint8)
    pixels[7, 11:13] = [0, 255]
    image = value["root"] / "fixture.png"
    Image.fromarray(pixels).save(image)
    records = json.loads(value["validation_manifest"].read_text())["images"]
    records[0] = {"path": image.name, "split": "validation", "shape_hw": list(pixels.shape),
                  "file_sha256": subject.artifacts.file_sha(image),
                  "pixels_sha256": hashlib.sha256(pixels.tobytes()).hexdigest()}
    value["validation_manifest"].write_text(json.dumps({"images": records}))
    validation = value["validation"]
    validation["images"] = [{"image": row["path"], "pixels_sha256": row["pixels_sha256"],
                              "mse": 103.058, "psnr": 28.0} for row in records]
    value["validation_path"].write_text(json.dumps(validation))
    value["checkpoint"]["validation_manifest_sha256"] = subject.artifacts.file_sha(value["validation_manifest"])
    torch.save(value["checkpoint"], value["checkpoint_path"])
    value["expected_sha"] = subject.artifacts.file_sha(value["checkpoint_path"])
    safe = value["root"] / "synthetic-safe.pt"
    base.tool.export(*base.trusted_args(value), safe, value["root"] / "synthetic-export.json")
    value.update(safe=safe, expected_safe_sha=subject.artifacts.file_sha(safe), pixels=pixels)
    return value


def arguments(prepared, report):
    return (prepared["checkpoint_path"], prepared["expected_sha"], prepared["safe"],
            prepared["expected_safe_sha"], prepared["root"], prepared["validation_manifest"], report)


def test_actual_raw_safe_crop_and_public_adapter_bytes_exact(prepared, tmp_path):
    report = tmp_path / "verified.json"
    receipt = subject.verify(*arguments(prepared, report))
    assert json.loads(report.read_text()) == receipt
    assert receipt["complete"] and receipt["all_tensor_bytes_preserved"]
    assert receipt["tensor_count"] == 40
    assert [row["pixel_shape"] for row in receipt["checks"]] == [[17, 19], [1, 19], [17, 1], [17, 19, 3]]
    assert all(row["prediction_bytes_exact"] and row["adapter_bytes_exact"] for row in receipt["checks"])
    assert all(row["per_channel_offset"] == 25 for row in receipt["checks"])
    assert receipt["fixture"]["crop"] == subject.CROP
    assert receipt["fixture"]["crop_uint8_sha256"] == hashlib.sha256(prepared["pixels"][7:24, 11:30].tobytes()).hexdigest()
    assert not receipt["cuda_initialized"] and not receipt["bsd68_accessed"] and not receipt["accuracy_scored"]
    content = report.read_text()
    assert all(token not in content for token in (str(prepared["root"]), "fixture.png", "/home/", "/Users/", "optimizer_state"))
    with pytest.raises(FileExistsError):
        subject.verify(*arguments(prepared, report))


@pytest.mark.parametrize("index", [1, 3])
def test_explicit_raw_and_safe_file_pins_precede_deserialization(prepared, tmp_path, monkeypatch, index):
    args = list(arguments(prepared, tmp_path / "not-created.json"))
    args[index] = "0" * 64
    def forbidden(*args, **kwargs):
        raise AssertionError("No model loading before the file pin matches")
    monkeypatch.setattr(subject.torch, "load", forbidden)
    with pytest.raises(ValueError, match="SHA256"):
        subject.verify(*args)
    assert not args[-1].exists()


@pytest.mark.parametrize("key,value", [
    ("step", 19), ("best_step", 19), ("full_depth_updates", 19),
    ("phase_updates", {"5": 19}), ("source_pins", {}), ("config_sha256", "bad"),
    ("manifest_sha256", "bad"), ("validation_manifest_sha256", "bad"),
    ("initialization", {"source_sha256": "bad"}),
    ("optimizer", {"exp_avg": torch.tensor([float("nan")])}),
    ("last_evaluation", {"step": 19}),
])
def test_raw_checkpoint_identity_and_finite_guards(prepared, monkeypatch, key, value):
    safe = torch.load(prepared["safe"], map_location="cpu", weights_only=True)
    changed = dict(prepared["checkpoint"], **{key: value})
    monkeypatch.setattr(subject.torch, "load", lambda *args, **kwargs: changed)
    with pytest.raises(ValueError):
        subject.trusted_raw(prepared["checkpoint_path"], prepared["expected_sha"], safe)


@pytest.mark.parametrize("kind", ["manifest", "file", "pixels", "shape", "escape", "small"])
def test_crop_requires_hash_verified_manifest_file_pixels_and_fixed_geometry(prepared, tmp_path, kind):
    manifest = json.loads(prepared["validation_manifest"].read_text())
    first = manifest["images"][0]
    Image.fromarray(prepared["pixels"]).save(tmp_path / "fixture.png")
    path = tmp_path / "manifest.json"
    if kind == "file": first["file_sha256"] = "0" * 64
    elif kind == "pixels": first["pixels_sha256"] = "0" * 64
    elif kind == "shape": first["shape_hw"] = [41, 33]
    elif kind == "escape": first["path"] = "../fixture.png"
    elif kind == "small":
        pixels = np.ones((20, 20), dtype=np.uint8)
        Image.fromarray(pixels).save(tmp_path / "fixture.png")
        first.update(file_sha256=subject.artifacts.file_sha(tmp_path / "fixture.png"),
                     pixels_sha256=hashlib.sha256(pixels.tobytes()).hexdigest(), shape_hw=[20, 20])
    path.write_text(json.dumps(manifest))
    pin = "0" * 64 if kind == "manifest" else subject.artifacts.file_sha(path)
    with pytest.raises(ValueError):
        subject.crop_pixels(tmp_path, path, pin)


def test_adapter_check_tests_actual_clipping_and_rejects_nonfinite_output(monkeypatch):
    class Scale(torch.nn.Module):
        def forward(self, value):
            return value * 2
    pixels = np.arange(17 * 19, dtype=np.float32).reshape(17, 19)
    result = subject.adapter_check(Scale(), Scale(), pixels)
    assert result["clipped_pixels"] > 0
    class Nonfinite(torch.nn.Module):
        def forward(self, value):
            return value * float("nan")
    with pytest.raises(ValueError, match="Nonfinite"):
        subject.adapter_check(Nonfinite(), Nonfinite(), pixels)
    monkeypatch.setattr(subject, "denoise", lambda *args, **kwargs: np.zeros_like(pixels))
    with pytest.raises(ValueError, match="adapter"):
        subject.adapter_check(Scale(), Scale(), pixels)


def test_nonidentical_raw_safe_prediction_is_not_export_parity():
    class Shift(torch.nn.Module):
        def forward(self, value):
            return value + 1
    with pytest.raises(ValueError, match="prediction bytes"):
        subject.adapter_check(torch.nn.Identity(), Shift(), np.zeros((17, 19), dtype=np.float32))


def test_raw_safe_tensor_payload_mismatch_fails_before_report(prepared, tmp_path, monkeypatch):
    raw = dict(prepared["checkpoint"], model=dict(prepared["checkpoint"]["model"]))
    name = "stages.0.projection.weight"
    raw["model"][name] = raw["model"][name].clone() + 1
    reader = torch.load
    def load(*args, **kwargs):
        return raw if kwargs.get("weights_only") is False else reader(*args, **kwargs)
    monkeypatch.setattr(subject.torch, "load", load)
    report = tmp_path / "not-created.json"
    with pytest.raises(ValueError, match="tensor bytes"):
        subject.verify(*arguments(prepared, report))
    assert not report.exists()


def test_actual_cpu_verifier_cli(prepared, tmp_path):
    report = tmp_path / "cli-report.json"
    completed = subprocess.run([sys.executable, str(ROOT / "tools/verify_rollout_export_cpu.py"),
        "--checkpoint", str(prepared["checkpoint_path"]),
        "--expected-checkpoint-sha256", prepared["expected_sha"],
        "--safe", str(prepared["safe"]), "--expected-safe-sha256", prepared["expected_safe_sha"],
        "--validation-root", str(prepared["root"]), "--validation-manifest", str(prepared["validation_manifest"]),
        "--report", str(report)], capture_output=True, text=True, check=True)
    assert json.loads(completed.stdout) == json.loads(report.read_text())
    assert json.loads(completed.stdout)["complete"] is True

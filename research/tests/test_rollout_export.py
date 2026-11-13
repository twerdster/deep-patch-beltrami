"""CPU export guards, exact tensor preservation and safe-reader checks."""
from dataclasses import asdict
import hashlib
import importlib.util
import json
from pathlib import Path

import pytest
import torch

from beltrami_rollout.model import BeltramiOperator, ModelConfig


SPEC = importlib.util.spec_from_file_location(
    "export_rollout_tool", Path(__file__).resolve().parents[1] / "tools/export_rollout.py")
tool = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(tool)


@pytest.fixture(scope="module", autouse=True)
def threads():
    previous = torch.get_num_threads()
    torch.set_num_threads(1)
    yield
    torch.set_num_threads(previous)


@pytest.fixture(scope="module")
def prepared(tmp_path_factory):
    root = tmp_path_factory.mktemp("rollout_export")
    model = BeltramiOperator(ModelConfig(stages=5))
    with torch.no_grad():
        for stage in model.stages:
            stage.dx.weight.normal_(0, 0.001)
            stage.dy.weight.normal_(0, 0.001)
            stage.divx.weight.normal_(0, 0.001)
            stage.divy.weight.normal_(0, 0.001)
            stage.projection.weight.normal_(0, 0.000001)
            stage.reaction.fill_(0.02)
    config = {"model": asdict(model.config), "source_pins": tool.source_pins(),
              "sigma": 25, "offset": 25, "minimum_full_depth_updates": 20,
              "warmstart_sha256": tool.WARMSTART_SHA256,
              "private_training_path": "/home/private/training"}
    config_path = root / "config.json"
    config_path.write_text(json.dumps(config))
    manifest = root / "train.json"
    manifest.write_text(json.dumps({"images": [{"path": "train.png", "split": "train"}]}))
    records = [{"path": f"image{i:02d}.png", "split": "validation",
                "pixels_sha256": hashlib.sha256(str(i).encode()).hexdigest()} for i in range(32)]
    val_manifest = root / "validation.json"
    val_manifest.write_text(json.dumps({"images": records}))
    validation = {"step": 20, "stages": 5, "full_depth_updates": 20,
                  "selection_eligible": True, "smoke_one_image": False, "mean_psnr": 28.0,
                  "images": [{"image": record["path"], "pixels_sha256": record["pixels_sha256"],
                              "mse": 103.058, "psnr": 28.0} for record in records]}
    validation_path = root / "validation-0000020.json"
    validation_path.write_text(json.dumps(validation))
    checkpoint = {"model": model.state_dict(), "optimizer": {"exp_avg": torch.tensor([0.1])},
                  "step": 20, "best_step": 20, "full_depth_updates": 20, "phase_updates": {"5": 20},
                  "config": config, "config_sha256": tool.sha(config_path),
                  "manifest_sha256": tool.sha(manifest), "source_pins": tool.source_pins(),
                  "validation_manifest_sha256": tool.sha(val_manifest), "best_psnr": 28.0,
                  "last_evaluation": {"step": 20, "stages": 5, "mean_psnr": 28.0},
                  "initialization": {"source_sha256": tool.WARMSTART_SHA256},
                  "sampler_rng": "not for export", "private_path": "/Users/private/notpublic"}
    checkpoint_path = root / "best.pt"
    torch.save(checkpoint, checkpoint_path)
    return {"root": root, "checkpoint_path": checkpoint_path, "checkpoint": checkpoint,
            "config_path": config_path, "config": config, "manifest": manifest,
            "validation_manifest": val_manifest, "validation_path": validation_path,
            "validation": validation, "expected_sha": tool.sha(checkpoint_path)}


def trusted_args(prepared):
    return (prepared["checkpoint_path"], prepared["config_path"], prepared["manifest"],
            prepared["validation_manifest"], prepared["expected_sha"], 20)


@pytest.fixture(scope="module")
def exported(prepared):
    output = prepared["root"] / "safe.pt"
    report = prepared["root"] / "report.json"
    result = tool.export(*trusted_args(prepared), output, report)
    return output, report, result


def test_export_preserves_every_tensor_byte_and_strips_private_state(prepared, exported):
    output, report_path, report = exported
    model, safe = tool.load_safe(output)
    assert safe["format"] == tool.FORMAT
    assert safe["training_step"] == 20 and safe["full_depth_updates"] == 20
    assert safe["tensor_sha256"] == tool.tensor_hash(prepared["checkpoint"]["model"])
    for name, tensor in prepared["checkpoint"]["model"].items():
        exported_tensor = safe["state_dict"][name]
        assert tensor.dtype == exported_tensor.dtype and tensor.shape == exported_tensor.shape
        assert tensor.numpy().tobytes() == exported_tensor.numpy().tobytes()
    assert all(check["prediction_bytes_exact"] for check in report["cpu_geometry_checks"])
    assert report["all_tensor_bytes_preserved"] and report["raw_model_and_optimizer_finite"]
    assert report["export_sha256"] == tool.sha(output)
    assert report["export_bytes"] == output.stat().st_size
    assert json.loads(report_path.read_text()) == report
    assert report["cuda_initialized"] is False and report["bsd68_accessed"] is False
    metadata = json.dumps({name: value for name, value in safe.items() if name != "state_dict"})
    assert all(token not in metadata for token in ("/home/", "/Users/", "optimizer", "sampler_rng"))
    assert safe["numerics"]["global_stability_guarantee"] is False


@pytest.mark.parametrize("key,value,message", [
    ("step", 19, "step"), ("best_step", 19, "step"),
    ("full_depth_updates", 19, "insufficient"),
    ("phase_updates", {"5": 19}, "chronology"),
    ("source_pins", {}, "source_pins"), ("config_sha256", "bad", "config_sha256"),
    ("manifest_sha256", "bad", "manifest_sha256"),
    ("validation_manifest_sha256", "bad", "validation_manifest_sha256"),
    ("initialization", {"source_sha256": "bad"}, "warmstart"),
    ("last_evaluation", {"step": 0}, "eligible"),
    ("optimizer", {"exp_avg": torch.tensor([float("nan")])}, "nonfinite"),
])
def test_trusted_best_guard_rejects_changed_identity_before_export(prepared, monkeypatch,
                                                                key, value, message):
    changed = dict(prepared["checkpoint"])
    changed[key] = value
    monkeypatch.setattr(tool.torch, "load", lambda *args, **kwargs: changed)
    with pytest.raises((ValueError, FloatingPointError), match=message):
        tool.trusted_best(*trusted_args(prepared))


def test_explicit_checkpoint_pin_checked_before_pickle(prepared, monkeypatch):
    def forbidden(*args, **kwargs):
        raise AssertionError("Untrusted pickle must not be opened")
    monkeypatch.setattr(tool.torch, "load", forbidden)
    args = list(trusted_args(prepared))
    args[4] = "bad"
    with pytest.raises(ValueError, match="SHA256"):
        tool.trusted_best(*args)
    args[0] = prepared["root"] / "last.pt"
    with pytest.raises(ValueError, match="best.pt"):
        tool.trusted_best(*args)


def test_validation_requires_all32_exact_images(prepared, monkeypatch):
    original = prepared["validation_path"].read_text()
    invalid = dict(prepared["validation"])
    invalid["images"] = invalid["images"][:-1]
    prepared["validation_path"].write_text(json.dumps(invalid))
    try:
        with pytest.raises(ValueError, match="exact32"):
            tool.trusted_best(*trusted_args(prepared))
    finally:
        prepared["validation_path"].write_text(original)


@pytest.mark.parametrize("tamper,message", [("digest", "digest"), ("numerics", "numerical"),
                                            ("parameters", "parameter count"), ("extra_key", "format")])
def test_safe_reader_rejects_tampering(exported, monkeypatch, tamper, message):
    output, _, _ = exported
    payload = torch.load(output, map_location="cpu", weights_only=True)
    if tamper == "digest":
        payload["tensor_sha256"] = "bad"
    elif tamper == "numerics":
        payload["numerics"] = dict(payload["numerics"], determinant_clipping=True)
    elif tamper == "parameters":
        payload["parameters"] += 1
    else:
        payload["private_path"] = "/home/private"
    calls = []
    def reader(*args, **kwargs):
        calls.append(kwargs)
        return payload
    monkeypatch.setattr(tool.torch, "load", reader)
    with pytest.raises(ValueError, match=message):
        tool.load_safe(output)
    assert calls == [{"map_location": "cpu", "weights_only": True}]


def test_export_refuses_to_overwrite_existing_evidence(prepared, exported):
    output, report, _ = exported
    before = tool.sha(output), tool.sha(report)
    with pytest.raises(FileExistsError):
        tool.export(*trusted_args(prepared), output, report)
    assert (tool.sha(output), tool.sha(report)) == before


def test_stage_activity_distinguishes_nominal_depth_from_nonidentity_updates():
    model = BeltramiOperator(ModelConfig(stages=32)).eval()
    with torch.no_grad():
        for stage in model.stages:
            stage.projection.weight.zero_()
            stage.reaction.zero_()
        model.stages[0].projection.weight.fill_(1e-8)
    fixture = torch.randn(1, 1, 3, 5) + 120
    rows = tool.stage_activity(model, fixture)
    assert len(rows) == 32
    assert rows[0]["nonzero_projection_channels"] == 80
    assert rows[-1]["nonzero_projection_channels"] == 0
    assert rows[-1]["projection_l2_norm"] == 0
    assert rows[-1]["reaction"] == 0
    assert rows[-1]["fixture_update_rms_pixels"] == 0
    assert rows[-1]["exact_identity_on_this_fixture"] is True
    assert rows[-1]["fixed_update_scale"] == pytest.approx(1 / 27)

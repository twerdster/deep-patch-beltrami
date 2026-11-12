"""Synthetic CPU checks; no BSD68 labels, archived noise or accuracy are opened."""
from argparse import Namespace
import copy
from dataclasses import asdict
import hashlib
import importlib.util
import json
from pathlib import Path

import numpy as np
import pytest
import torch
from beltrami_operator.model import BeltramiOperator, ModelConfig
from beltrami_research import artifacts


TOOL = Path(__file__).resolve().parents[1] / "tools" / "evaluate_released.py"
SPEC = importlib.util.spec_from_file_location("released_eval_tests_tool", TOOL)
evaluator = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(evaluator)
SHA = "1" * 64


def write_json(path, value):
    path.write_text(json.dumps(value, indent=2) + "\n")
    return evaluator.protocol.file_hash(path)


@pytest.fixture(scope="module")
def synthetic(tmp_path_factory):
    root = tmp_path_factory.mktemp("released-fixture")
    entries = []
    for name, reaction in (("control", False), ("reaction", True)):
        cfg = ModelConfig(reaction=reaction)
        model = BeltramiOperator(cfg)
        with torch.no_grad():
            for parameter in model.parameters():
                parameter.zero_()
            for stage in model.stages:
                stage.metric_bias.fill_(1)
        state = model.state_dict()
        value = {"format": "beltrami-operator-tensors-v1", "implementation": "beltrami_operator",
            "source_sha256": artifacts.source_hash("beltrami_operator"), "model_config": asdict(cfg),
            "sigma": 25, "grayscale_offset": 25, "tensor_sha256": artifacts.tensor_hash(state),
            "parameters": sum(p.numel() for p in model.parameters()), "training_step": 500,
            "state_dict": state, "provenance": {key: SHA for key in evaluator._PROVENANCE_KEYS
                                                if key not in ("license", "citation")}}
        value["provenance"].update(license="MIT", citation="Wetzler, thesis, 2017")
        path = root / f"{name}.pt"
        torch.save(value, path)
        entries.append({"model": name, "role": "candidate" if reaction else "control",
                        "filename": path.name, "url": f"https://example.org/{path.name}",
                        "bytes": path.stat().st_size, "sha256": evaluator.protocol.file_hash(path),
                        **{key: item for key, item in value.items() if key not in ("state_dict", "format")}})
    original = root / "original.caffemodel"
    original.write_bytes(b"synthetic original asset; not a Caffe graph")
    originals = [{"path": str(original), "sha256": evaluator.protocol.file_hash(original)}]
    # Synthetic original pin is the actual fixture hash, never the real release hash.
    for entry in entries:
        path = root / entry["filename"]
        value = torch.load(path, map_location="cpu", weights_only=True)
        value["provenance"]["original_checkpoint_sha256"] = originals[0]["sha256"]
        entry["provenance"] = value["provenance"]
        torch.save(value, path)
        entry.update(bytes=path.stat().st_size, sha256=evaluator.protocol.file_hash(path))
    manifest = {"schema": 1, "purpose": "bsd68", "sigma": 25, "offset": 25,
        "selection_sha256": SHA, "results_sha256": SHA,
        "evaluator_sha256": evaluator.protocol.file_hash(evaluator._HELPER_PATH),
        "baseline": {"name": "historical", "role": "baseline", "kind": "historical",
            "parameters": 3195600, "training_step": 40000,
            "files": [{"name": original.name, "sha256": originals[0]["sha256"]}]},
        "assets": entries,
        "comparisons": [{"candidate": "control", "reference": "historical"},
            {"candidate": "reaction", "reference": "historical"},
            {"candidate": "reaction", "reference": "control"}],
        "timing_warmup": 2, "timing_repeats": 3, "bootstrap_seed": 71,
        "bootstrap_samples": 20000}
    return root, manifest, originals


def save_manifest(tmp_path, manifest):
    path = tmp_path / "models.json"
    digest = write_json(path, manifest)
    return path, digest


def test_manifest_pin_and_protocol_guard(tmp_path, synthetic):
    _, manifest, _ = synthetic
    path, digest = save_manifest(tmp_path, manifest)
    assert evaluator.read_manifest(path, digest) == manifest
    with pytest.raises(ValueError, match="SHA256 differs"):
        evaluator.read_manifest(path, "0" * 64)
    for field, value in (("offset", 0), ("evaluator_sha256", "0" * 64),
                         ("timing_warmup", 0), ("timing_repeats", 1)):
        changed = copy.deepcopy(manifest)
        changed[field] = value
        digest = write_json(path, changed)
        with pytest.raises(ValueError):
            evaluator.read_manifest(path, digest)


def test_matched_control_roles_and_comparison_coverage(synthetic):
    _, manifest, _ = synthetic
    evaluator.validate_comparisons(manifest)
    variants = []
    wrong = copy.deepcopy(manifest)
    wrong["assets"][0]["role"] = "candidate"
    variants.append(wrong)
    wrong = copy.deepcopy(manifest)
    wrong["comparisons"] = wrong["comparisons"][:-1]
    variants.append(wrong)
    wrong = copy.deepcopy(manifest)
    wrong["assets"][0]["implementation"] = "beltrami_precision"
    variants.append(wrong)
    wrong = copy.deepcopy(manifest)
    wrong["assets"][0]["model_config"]["metric"] = "positive"
    variants.append(wrong)
    wrong = copy.deepcopy(manifest)
    wrong["comparisons"].append(wrong["comparisons"][0])
    variants.append(wrong)
    for wrong in variants:
        with pytest.raises(ValueError):
            evaluator.validate_comparisons(wrong)


def test_precision_depth_requires_precision_five_stage_control(synthetic):
    _, manifest, _ = synthetic
    candidate, control = copy.deepcopy(manifest["assets"][1]), copy.deepcopy(manifest["assets"][0])
    for entry in (candidate, control):
        entry["implementation"] = "beltrami_precision"
        entry["model_config"].update(metric="positive", reaction=False, metric_precision="float64")
    candidate["model_config"]["stages"] = 8
    assert evaluator.matched_control(candidate, control)
    control["implementation"] = "beltrami_operator"
    assert not evaluator.matched_control(candidate, control)
    control["implementation"] = "beltrami_precision"
    control["model_config"]["stages"] = 6
    assert not evaluator.matched_control(candidate, control)


def test_contained_artifact_paths(tmp_path):
    assert evaluator.artifact_path(tmp_path, "weights/model.pt") == tmp_path / "weights/model.pt"
    for bad in ("../escape.pt", "/tmp/escape.pt", "a/../escape.pt", "a\\escape.pt", ""):
        with pytest.raises(ValueError):
            evaluator.artifact_path(tmp_path, bad)
    outside = tmp_path.parent / "outside-model.pt"
    outside.write_bytes(b"outside")
    (tmp_path / "link.pt").symlink_to(outside)
    with pytest.raises(ValueError):
        evaluator.artifact_path(tmp_path, "link.pt")


def test_safe_load_exact_tensor_metadata_and_source(synthetic, monkeypatch):
    root, manifest, originals = synthetic
    entry = copy.deepcopy(manifest["assets"][1])
    path = root / entry["filename"]
    model = evaluator.verify_asset(entry, path, 25, 25, originals[0]["sha256"])
    raw = torch.tensor([[[[-5., 0., 20.], [260., 120., 70.]]]])
    with torch.inference_mode():
        torch.testing.assert_close(model(raw), raw, rtol=0, atol=0)
    assert not torch.cuda.is_initialized()
    for field, value in (("source_sha256", "0" * 64), ("tensor_sha256", "0" * 64),
                         ("training_step", 600), ("parameters", 1)):
        changed = copy.deepcopy(entry)
        changed[field] = value
        with pytest.raises(ValueError, match="differs"):
            evaluator.verify_asset(changed, path, 25, 25, originals[0]["sha256"])
    # A changed artifact is rejected before the safe reader is invoked.
    changed = copy.deepcopy(entry)
    changed["sha256"] = "0" * 64
    monkeypatch.setattr(evaluator.artifacts, "load", lambda *a, **k: pytest.fail("loader must not run"))
    with pytest.raises(ValueError, match="bytes differ"):
        evaluator.verify_asset(changed, path, 25, 25, originals[0]["sha256"])


def test_preflight_rejects_historical_and_data_identity(synthetic, monkeypatch):
    root, manifest, originals = synthetic
    monkeypatch.setattr(evaluator.protocol, "original_files", lambda *a: originals)
    actual, paths = evaluator.preflight(manifest, root, root)
    assert actual == originals and set(paths) == {"reaction", "control"}
    changed = copy.deepcopy(manifest)
    changed["baseline"]["files"][0]["sha256"] = "0" * 64
    with pytest.raises(ValueError, match="Historical baseline"):
        evaluator.preflight(changed, root, root)


def synthetic_cases():
    cases = []
    for i in range(2):
        clean = np.arange(6, dtype=np.float32).reshape(1, 1, 2, 3) + 20 + i
        noisy = clean + np.float32(1 + i)
        cases.append({"name": str(i), "clean": clean, "noisy": noisy,
            "noisy_native_sha256": evaluator.protocol.array_hash(noisy),
            "label_native_sha256": evaluator.protocol.array_hash(clean)})
    return cases


def arguments(tmp_path, synthetic):
    root, manifest, _ = synthetic
    path, digest = save_manifest(tmp_path, manifest)
    return Namespace(manifest=path, manifest_sha256=digest, weights_dir=root, model_dir=root,
        clean_dir=root / "never-opened-clean", asset_dir=root / "never-opened-noise", cache_dir=None,
        output=tmp_path / "evidence", gpu_lock=tmp_path / "gpu.lock", device="cpu", threads=2,
        max_cuda_gib=10)


def test_complete_synthetic_pipeline_uses_shared_protocol(tmp_path, synthetic, monkeypatch):
    _, manifest, originals = synthetic
    args = arguments(tmp_path, synthetic)
    monkeypatch.setattr(evaluator.protocol, "original_files", lambda *a: originals)
    monkeypatch.setattr(evaluator.protocol, "HistoricalOperator", lambda *a: lambda x: x)
    monkeypatch.setattr(evaluator.protocol, "dataset_cases", lambda *a: (synthetic_cases(), {"synthetic": True}))
    report = evaluator.evaluate(args)
    assert report["complete"] and report["image_count"] == 2
    assert len(report["models"]) == 3 and len(report["comparisons"]) == 3
    assert all(row["complete"] for row in report["models"])
    assert all(len(image["seconds_forward"]) == 3 for row in report["models"] for image in row["images"])
    for row in report["models"]:
        for actual, case in zip(row["images"], synthetic_cases()):
            expected = evaluator.protocol.score_prediction(case["clean"], case["noisy"] + 25, 25)
            assert actual["mse"] == expected["mse"] and actual["psnr"] == expected["psnr"]
    assert all(pair["mean_delta_db"] == 0 for pair in report["comparisons"])
    assert json.loads((args.output / "status.json").read_text())["state"] == "complete"
    assert not torch.cuda.is_initialized()
    with pytest.raises(FileExistsError):
        evaluator.evaluate(args)


def test_bad_pin_preserves_failure_before_labels(tmp_path, synthetic, monkeypatch):
    args = arguments(tmp_path, synthetic)
    args.manifest_sha256 = "0" * 64
    monkeypatch.setattr(evaluator.protocol, "dataset_cases", lambda *a: pytest.fail("labels must stay closed"))
    with pytest.raises(ValueError, match="SHA256 differs"):
        evaluator.evaluate(args)
    report = json.loads((args.output / "results.json").read_text())
    assert report["complete"] is False and report["models"] == []
    assert report["failure"]["type"] == "ValueError"
    assert json.loads((args.output / "status.json").read_text())["state"] == "failed"


def test_nonfinite_prediction_preserves_partial_evidence(tmp_path, synthetic, monkeypatch):
    _, _, originals = synthetic
    args = arguments(tmp_path, synthetic)
    monkeypatch.setattr(evaluator.protocol, "original_files", lambda *a: originals)
    monkeypatch.setattr(evaluator.protocol, "dataset_cases", lambda *a: (synthetic_cases(), {"synthetic": True}))
    monkeypatch.setattr(evaluator.protocol, "HistoricalOperator", lambda *a: lambda x: x * float("nan"))
    with pytest.raises(FloatingPointError, match="warmup"):
        evaluator.evaluate(args)
    report = json.loads((args.output / "results.json").read_text())
    assert report["complete"] is False and report["models"][0]["complete"] is False
    assert report["failure"]["type"] == "FloatingPointError"


@pytest.mark.parametrize("mutation", ["source", "tensor", "private_state"])
def test_safe_artifact_internal_identity_cannot_be_relabelled(tmp_path, synthetic, mutation):
    root, manifest, originals = synthetic
    entry = copy.deepcopy(manifest["assets"][1])
    value = torch.load(root / entry["filename"], map_location="cpu", weights_only=True)
    if mutation == "source":
        value["source_sha256"] = entry["source_sha256"] = "0" * 64
    elif mutation == "tensor":
        value["tensor_sha256"] = entry["tensor_sha256"] = "0" * 64
    else:
        value["optimizer"] = {"private": "not part of a released artifact"}
    changed = tmp_path / "changed.pt"
    torch.save(value, changed)
    entry.update(bytes=changed.stat().st_size, sha256=evaluator.protocol.file_hash(changed))
    with pytest.raises(ValueError):
        evaluator.verify_asset(entry, changed, 25, 25, originals[0]["sha256"])


def test_malformed_metadata_cannot_open_labels(tmp_path, synthetic, monkeypatch):
    _, manifest, originals = synthetic
    args = arguments(tmp_path, synthetic)
    altered = copy.deepcopy(manifest)
    altered["assets"][1]["provenance"]["training_manifest_sha256"] = "0" * 64
    args.manifest_sha256 = write_json(args.manifest, altered)
    monkeypatch.setattr(evaluator.protocol, "original_files", lambda *a: originals)
    monkeypatch.setattr(evaluator.protocol, "dataset_cases", lambda *a: pytest.fail("labels must stay closed"))
    with pytest.raises(ValueError, match="provenance differs"):
        evaluator.evaluate(args)
    report = json.loads((args.output / "results.json").read_text())
    assert not report["complete"] and not report["models"]

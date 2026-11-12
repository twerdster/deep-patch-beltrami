"""Synthetic frozen-evidence fixtures exercise publication guards without images or GPU."""
import copy
from dataclasses import asdict
import hashlib
import importlib.util
import json
import math
from pathlib import Path
import subprocess
import sys

import pytest
import torch

from beltrami_operator.model import BeltramiOperator, ModelConfig
from beltrami_research import artifacts
from deep_patch_beltrami.assets import metadata


ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location("public_report_subject", ROOT / "tools/build_public_report.py")
subject = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(subject)


def put(path, value):
    path.write_text(json.dumps(value, indent=2) + "\n")


@pytest.fixture(scope="module")
def synthetic_exports(tmp_path_factory):
    root = tmp_path_factory.mktemp("synthetic-export-fixtures")
    torch.set_num_threads(2)
    torch.manual_seed(431)
    source = artifacts.source_hash("beltrami_operator")
    original_name = "PB_25_80_5__iter_40000.caffemodel"
    pins = {item["name"]: item["sha256"] for item in metadata("models.json")["files"]}
    for split in ("training", "validation"):
        put(root / (split + ".json"), {"split": split, "images": []})
    train_sha = subject.evaluation.file_hash(root / "training.json")
    validation_sha = subject.evaluation.file_hash(root / "validation.json")
    configs, models, assets = [], [], []
    for name, role, reaction in (("control", "control", False), ("reaction", "candidate", True)):
        config = {"sigma": 25, "offset": 25, "seed": 20261112, "batch_size": 2, "patch_size": 160,
                  "model": {"width": 80, "stages": 5, "metric": "frozen", "reaction": reaction}}
        config_path, checkpoint = root / (name + ".json"), root / (name + "-synthetic-checkpoint.pt")
        put(config_path, config)
        model = BeltramiOperator(ModelConfig(**config["model"]))
        state = {key: tensor.detach().clone() for key, tensor in model.state_dict().items()}
        torch.save({"model": state, "config": config, "config_sha256": subject.evaluation.file_hash(config_path),
                    "source_sha256": source, "step": 100, "gpu_seconds": 12.,
                    "manifest_sha256": train_sha, "validation_manifest_sha256": validation_sha,
                    "initialization": {"source_name": original_name, "source_sha256": pins[original_name], "source_sigma": 25}}, checkpoint)
        entry = {"name": name, "role": role, "kind": "checkpoint", "implementation": "beltrami_operator",
                 "checkpoint": str(checkpoint), "checkpoint_sha256": subject.evaluation.file_hash(checkpoint),
                 "config": str(config_path), "config_sha256": subject.evaluation.file_hash(config_path),
                 "model_config": config["model"], "parameters": sum(p.numel() for p in model.parameters()),
                 "training_step": 100, "training_gpu_seconds": 12.0,
                 "training_manifest_sha256": train_sha, "validation_manifest_sha256": validation_sha,
                 "training_source_sha256": source, "initialization": {"source_sha256": pins[original_name]}}
        value = {"format": "beltrami-operator-tensors-v1", "implementation": "beltrami_operator",
                 "source_sha256": source, "model_config": asdict(model.config), "sigma": 25,
                 "grayscale_offset": 25, "state_dict": state, "tensor_sha256": artifacts.tensor_hash(state),
                 "parameters": entry["parameters"], "training_step": 100,
                 "provenance": {"training_checkpoint_sha256": entry["checkpoint_sha256"],
                     "training_config_sha256": entry["config_sha256"], "training_manifest_sha256": train_sha,
                     "validation_manifest_sha256": validation_sha, "original_checkpoint_sha256": pins[original_name],
                     "license": "MIT", "citation": "Wetzler, Topics in Computational Geometric Vision, 2017, Chapter 4 Section 4.5"}}
        artifact = root / (name + ".pt")
        torch.save(value, artifact)
        models.append(entry)
        assets.append({"model": name, "path": str(artifact), "filename": name + ".pt",
                       "url": "https://github.com/example/research/releases/download/test/" + name + ".pt"})
        configs.append(config)
    return root, models, assets, pins


@pytest.fixture
def frozen(tmp_path, monkeypatch, synthetic_exports):
    root, checkpoints, assets, pins = synthetic_exports
    manifest = metadata("bsd68.json")
    names = [f"PB_25_80_5__iter_40000.caffemodel"] + [f"stage_{index:03d}_deploy.prototxt" for index in range(1, 6)]
    baseline = {"name": "historical", "role": "baseline", "kind": "historical", "parameters": 3195600,
                "training_step": 40000, "files": [{"path": str(root / name), "sha256": pins[name]} for name in names]}
    # No historical model download is needed: retain its real public descriptor pins,
    # while exercising the reused file verifier on every synthetic non-baseline file.
    def verify_artifacts(selection):
        subject.evaluation.verify_files([{"path": selection[f"{split}_manifest"], "sha256": selection[f"{split}_manifest_sha256"]}
                                         for split in ("training", "validation")])
        for entry in selection["models"]:
            if entry["kind"] == "checkpoint":
                subject.evaluation.verify_files([{"path": entry[key], "sha256": entry[key + "_sha256"]}
                                                 for key in ("checkpoint", "config")])
    monkeypatch.setattr(subject.evaluation, "verify_artifacts", verify_artifacts)
    selection = {"schema": 1, "purpose": "bsd68", "created_utc": "2026-01-01T10:00:00+00:00",
                 "reason": "PRIVATE_RESEARCH_NOTE", "sigma": 25, "offset": 25,
                 "models": [baseline, *copy.deepcopy(checkpoints)],
                 "comparisons": [{"candidate": "control", "reference": "historical"},
                                 {"candidate": "reaction", "reference": "historical"},
                                 {"candidate": "reaction", "reference": "control"}],
                 "implementation_source_sha256": {"beltrami_operator": artifacts.source_hash("beltrami_operator")},
                 "evaluator_sha256": subject.evaluation.file_hash(subject.evaluation.__file__),
                 "timing_warmup": 2, "timing_repeats": 3, "bootstrap_seed": 431, "bootstrap_samples": 1000,
                 "protocol": subject.PROTOCOL.copy()}
    for split in ("training", "validation"):
        path = root / (split + ".json")
        selection[split + "_manifest"] = str(path)
        selection[split + "_manifest_sha256"] = subject.evaluation.file_hash(path)
    selection_path = tmp_path / "selection.json"
    subject.evaluation.write_frozen_selection(selection_path, selection)
    images_by_name = {item["name"]: item for item in manifest["images"]}
    cases = {item["name"]: item for item in manifest["cases"] if item["sigma"] == 25}
    models = []
    for model_index, entry in enumerate(selection["models"]):
        images = []
        for index, name in enumerate(sorted(images_by_name)):
            psnr = 20.0 + index / 100 + model_index / 10
            images.append({"name": name, "shape_native": images_by_name[name]["shape"],
                           "noisy_native_sha256": cases[name]["noisy_native_sha256"],
                           "label_native_sha256": images_by_name[name]["label_native_sha256"],
                           "mse": 255 ** 2 / 10 ** (psnr / 10), "psnr": psnr, "perfect_prediction": False,
                           "clipped_pixels": 0, "seconds_forward": [.01, .02, .03], "median_seconds_forward": .02})
        models.append({"name": entry["name"], "role": entry["role"], "parameters": entry["parameters"],
                       "complete": True, "images": images, "mean_psnr": sum(x["psnr"] for x in images) / 68,
                       "median_seconds_forward": .02, "sum_seconds_forward": 68 * .06})
    by_name = {model["name"]: model for model in models}
    report = {"schema": 1, "dataset": "bsd68", "image_count": 68, "complete": True,
              "started_utc": "2026-01-01T11:00:00+00:00", "ended_utc": "2026-01-01T12:00:00+00:00",
              "selection": selection, "selection_sha256": subject.evaluation.file_hash(selection_path),
              "environment": {"host": "PRIVATE_HOST_MARKER", "python": "3.11.9 PRIVATE_BUILD_MARKER",
                              "torch": "2.5.1", "numpy": "1.26.4", "cuda": None,
                              "device": "cpu", "device_name": "cpu", "cpu_threads": 2},
              "dataset_provenance": {"loader": "deep_patch_beltrami.reproduce.PublicBSD68",
                  "public_bsd68_manifest_sha256": hashlib.sha256(json.dumps(manifest, sort_keys=True).encode()).hexdigest(),
                  "noise_asset_sha256": metadata("release-assets.json")["assets"]["bsd68_noise"]["sha256"]},
              "models": models, "comparisons": [{**pair, **subject.evaluation.paired_summary(
                  by_name[pair["candidate"]]["images"], by_name[pair["reference"]]["images"], 431, 1000)}
                  for pair in selection["comparisons"]]}
    results, status, exports = tmp_path / "results.json", tmp_path / "status.json", tmp_path / "exports.json"
    put(results, report)
    put(status, {"state": "complete", "models": 3, "images": 68, "pid": 12345})
    put(exports, {"schema": 1, "assets": copy.deepcopy(assets)})
    return {"selection": selection_path, "results": results, "status": status, "exports": exports,
            "output": tmp_path / "public", "report": report}


def build(frozen):
    return subject.build(frozen["selection"], frozen["results"], subject.evaluation.file_hash(frozen["results"]),
                         frozen["status"], frozen["exports"], frozen["output"])


def test_valid_bundle_preserves_roles_pairs_hashes_and_removes_private_metadata(frozen):
    report = build(frozen)
    output = frozen["output"]
    assert report["complete"] and report["image_count"] == 68
    assert [model["role"] for model in report["models"]] == ["baseline", "control", "candidate"]
    assert report["comparisons"][-1]["reference_role"] == "control"
    manifest = json.loads((output / "models.json").read_text())
    assert manifest["expected_results"]["filename"] == "reference-results.json"
    assert manifest["expected_results"]["sha256"] == subject.evaluation.file_hash(output / "reference-results.json")
    assert len(manifest["assets"]) == 2 and manifest["assets"][1]["parameters"] == 3195605
    released_spec = importlib.util.spec_from_file_location("released_manifest_contract", ROOT / "tools/evaluate_released.py")
    released = importlib.util.module_from_spec(released_spec)
    released_spec.loader.exec_module(released)
    released.read_manifest(output / "models.json", subject.evaluation.file_hash(output / "models.json"))
    asset_path = Path(json.loads(frozen["exports"].read_text())["assets"][0]["path"])
    original_sha = manifest["baseline"]["files"][0]["sha256"]
    released.verify_asset(manifest["assets"][0], asset_path, 25, 25, original_sha, "cpu")
    content = "\n".join(path.read_text() for path in output.iterdir())
    for private in ("PRIVATE_HOST_MARKER", "PRIVATE_RESEARCH_NOTE", "PRIVATE_BUILD_MARKER", str(frozen["output"].parent)):
        assert private not in content
    assert "one training seed" in content and "Conditional 95% interval" in content
    assert len((output / "per-image.csv").read_text().splitlines()) == 1 + 3 * 68
    assert len((output / "paired-per-image.csv").read_text().splitlines()) == 1 + 3 * 68
    assert len((output / "runtime.csv").read_text().splitlines()) == 1 + 3 * 68 * 3
    with pytest.raises(FileExistsError):
        build(frozen)


@pytest.mark.parametrize("mutation", ["partial", "wrong_selection", "wrong_role", "missing_image", "wrong_input_hash",
                                       "wrong_psnr", "wrong_mean", "wrong_timing", "wrong_pair", "wrong_bootstrap"])
def test_inconsistent_result_is_rejected_before_output_creation(frozen, mutation):
    report = frozen["report"]
    if mutation == "partial": report["complete"] = False
    elif mutation == "wrong_selection": report["selection_sha256"] = "0" * 64
    elif mutation == "wrong_role": report["models"][1]["role"] = "candidate"
    elif mutation == "missing_image": report["models"][0]["images"].pop()
    elif mutation == "wrong_input_hash": report["models"][1]["images"][0]["noisy_native_sha256"] = "0" * 64
    elif mutation == "wrong_psnr": report["models"][1]["images"][0]["psnr"] += 1
    elif mutation == "wrong_mean": report["models"][1]["mean_psnr"] += 1
    elif mutation == "wrong_timing": report["models"][1]["images"][0]["seconds_forward"][0] = -1
    elif mutation == "wrong_pair": report["comparisons"][-1]["reference"] = "historical"
    elif mutation == "wrong_bootstrap": report["comparisons"][-1]["bootstrap_mean_95_percentile_ci_db"][0] += 1
    put(frozen["results"], report)
    with pytest.raises(ValueError): build(frozen)
    assert not frozen["output"].exists()


def test_wrong_external_results_hash_and_changed_selection_receipt_fail(frozen):
    with pytest.raises(ValueError, match="Results file hash"):
        subject.build(frozen["selection"], frozen["results"], "0" * 64, frozen["status"], frozen["exports"], frozen["output"])
    frozen["selection"].write_text(frozen["selection"].read_text() + " ")
    with pytest.raises(ValueError, match="Selection digest"):
        build(frozen)


@pytest.mark.parametrize("mutation", ["purpose", "evaluator", "bootstrap_count", "timing_count", "status", "missing_export",
                                       "extra_export_metadata", "export_parent", "export_tensor"])
def test_source_status_and_export_guards(frozen, mutation):
    if mutation in ("purpose", "evaluator", "bootstrap_count", "timing_count"):
        selection = frozen["report"]["selection"]
        field, value = {"purpose": ("purpose", "validation32"), "evaluator": ("evaluator_sha256", "0" * 64),
                        "bootstrap_count": ("bootstrap_samples", 1), "timing_count": ("timing_repeats", True)}[mutation]
        selection[field] = value
        put(frozen["selection"], selection)
        frozen["selection"].with_name("selection.json.sha256").write_text(subject.evaluation.file_hash(frozen["selection"]) + "\n")
        frozen["report"]["selection_sha256"] = subject.evaluation.file_hash(frozen["selection"])
        put(frozen["results"], frozen["report"])
    elif mutation == "status": put(frozen["status"], {"state": "failed"})
    else:
        exports = json.loads(frozen["exports"].read_text())
        if mutation == "missing_export": exports["assets"].pop()
        else:
            item = exports["assets"][0]
            value = torch.load(item["path"], map_location="cpu", weights_only=True)
            if mutation == "extra_export_metadata": value["host"] = "PRIVATE_HOST_MARKER"
            elif mutation == "export_parent": value["provenance"]["training_checkpoint_sha256"] = "0" * 64
            else:
                next(iter(value["state_dict"].values())).flatten()[0] += .5
                value["tensor_sha256"] = artifacts.tensor_hash(value["state_dict"])
            changed = frozen["exports"].parent / "changed.pt"
            torch.save(value, changed)
            item["path"] = str(changed)
        put(frozen["exports"], exports)
    with pytest.raises(ValueError): build(frozen)
    assert not frozen["output"].exists()


@pytest.mark.parametrize("kind", ["active", "overspent", "wrong_total", "negative"])
def test_aggregate_ledger_rejects_unfinished_or_inconsistent_compute(frozen, kind):
    ledger = {"authorized_gpu_seconds": 100., "consumed_gpu_seconds": 10.,
              "entries": [{"state": "complete", "charged_seconds": 10., "host": "PRIVATE_HOST_MARKER"}]}
    if kind == "active": ledger["entries"][0]["state"] = "running"
    elif kind == "overspent": ledger["authorized_gpu_seconds"] = 5.
    elif kind == "wrong_total": ledger["consumed_gpu_seconds"] = 1.
    else: ledger["entries"][0]["charged_seconds"] = -1.
    path = frozen["results"].parent / "budget.json"
    put(path, ledger)
    with pytest.raises(ValueError): subject.compute_accounting(path)


def test_aggregate_ledger_only_publishes_totals_and_hash(frozen):
    path = frozen["results"].parent / "budget.json"
    put(path, {"authorized_gpu_seconds": 100., "consumed_gpu_seconds": 10.,
               "entries": [{"state": "failed", "charged_seconds": 10., "command": ["PRIVATE_COMMAND_MARKER"]}]})
    result = subject.compute_accounting(path)
    assert result["consumed_process_wall_hours"] == 10 / 3600
    assert result["failed_leases"] == 1 and "PRIVATE_COMMAND_MARKER" not in json.dumps(result)


def test_accounting_refresh_preserves_exact_replayed_manifest_and_reference(frozen):
    build(frozen)
    outputs = [frozen["output"]]
    for label, charge in (("before-replay", 10.), ("after-replay", 20.)):
        ledger = frozen["results"].parent / (label + "-budget.json")
        put(ledger, {"authorized_gpu_seconds": 100., "consumed_gpu_seconds": charge,
                     "entries": [{"state": "complete", "charged_seconds": charge}]})
        output = frozen["output"].with_name(label)
        subject.build(frozen["selection"], frozen["results"], subject.evaluation.file_hash(frozen["results"]),
                      frozen["status"], frozen["exports"], output, ledger)
        outputs.append(output)
    for filename in ("models.json", "reference-results.json", "per-image.csv", "paired-per-image.csv", "runtime.csv"):
        assert all((output / filename).read_bytes() == (outputs[0] / filename).read_bytes() for output in outputs[1:])
    reports = [json.loads((output / "results.json").read_text()) for output in outputs]
    assert reports[0]["compute_accounting"]["available"] is False
    assert [report["compute_accounting"]["consumed_process_wall_hours"] for report in reports[1:]] == [10 / 3600, 20 / 3600]
    assert len({(output / "results.json").read_bytes() for output in outputs}) == 3
    reference = json.loads((outputs[0] / "reference-results.json").read_text())
    assert "compute_accounting" not in reference
    assert all({key: value for key, value in report.items() if key != "compute_accounting"} == reference for report in reports)
    for output in outputs:
        receipt = json.loads((output / "files.json").read_text())
        assert "reference-results.json" in {item["name"] for item in receipt["files"]}
        assert all(subject.evaluation.file_hash(output / item["name"]) == item["sha256"] for item in receipt["files"])


def test_omitted_matched_control_comparison_is_rejected(frozen):
    selection = frozen["report"]["selection"]
    selection["comparisons"].pop()
    frozen["report"]["comparisons"].pop()
    put(frozen["selection"], selection)
    frozen["selection"].with_name("selection.json.sha256").write_text(subject.evaluation.file_hash(frozen["selection"]) + "\n")
    frozen["report"]["selection_sha256"] = subject.evaluation.file_hash(frozen["selection"])
    put(frozen["results"], frozen["report"])
    with pytest.raises(ValueError, match="matched continued control"):
        build(frozen)
    assert not frozen["output"].exists()


def test_cross_namespace_controls_are_rejected():
    baseline = {"name": "historical", "kind": "historical", "role": "baseline"}
    assets = [{"model": "control", "role": "control", "implementation": "beltrami_operator",
               "model_config": {"width": 80, "stages": 5, "metric": "positive", "reaction": False}},
              {"model": "depth", "role": "candidate", "implementation": "beltrami_precision",
               "model_config": {"width": 80, "stages": 8, "metric": "positive", "reaction": False}}]
    pairs = [{"candidate": "control", "reference": "historical"}, {"candidate": "depth", "reference": "historical"},
             {"candidate": "depth", "reference": "control"}]
    with pytest.raises(ValueError, match="matched continued control"):
        subject.validate_release_pairs(baseline, assets, pairs)


def test_cli_rejects_incomplete_synthetic_result(frozen):
    frozen["report"]["complete"] = False
    put(frozen["results"], frozen["report"])
    result = subprocess.run([sys.executable, str(ROOT / "tools/build_public_report.py"),
        "--selection", str(frozen["selection"]), "--results", str(frozen["results"]),
        "--results-sha256", subject.evaluation.file_hash(frozen["results"]),
        "--exports", str(frozen["exports"]), "--output", str(frozen["output"])], capture_output=True, text=True)
    assert result.returncode != 0 and "incomplete" in result.stderr
    assert not frozen["output"].exists()

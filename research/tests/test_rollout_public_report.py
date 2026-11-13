"""Synthetic publication records; metadata is read, never images/noise or CUDA."""
import copy
import importlib.util
import json
import math
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
from deep_patch_beltrami.assets import metadata

ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location("rollout_report_tests_subject", ROOT / "tools/build_rollout_report.py")
subject = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(subject)
SHA = "1" * 64


def put(path, value):
    path.write_text(json.dumps(value, indent=2) + "\n")
    return subject.protocol.file_hash(path)


@pytest.fixture
def frozen(tmp_path, monkeypatch):
    private = tmp_path / "private_evaluator.py"
    private.write_text("# trusted synthetic verifier fixture\n")
    selection = {"schema": "private-synthetic-schema", "purpose": "bsd68", "sigma": 25, "offset": 25,
        "created_utc": "2026-01-01T00:00:00+00:00", "evaluator_sha256": subject.protocol.file_hash(private),
        "protocol_helper_sha256": subject.protocol.file_hash(subject.replay.HELPER_PATH),
        "source_pins": subject.rollout_artifacts.source_pins(), "protocol": copy.deepcopy(subject.PRIVATE_PROTOCOL),
        "comparisons": copy.deepcopy(subject.replay.COMPARISONS), "timing_warmup": 2, "timing_repeats": 3,
        "bootstrap_samples": 1000, "bootstrap_seed": 7, "evaluation_scope": "reused benchmark; not untouched",
        "budget_at_freeze": {"existing_run_ids": ["earlier-training"]}, "models": [], "reason": "PRIVATE_SELECTION_NOTE"}
    assets = []
    for name, role, stages in zip(subject.replay.MODEL_NAMES, ("control", "candidate", "candidate", "benchmark"), (5, 32, 64, 5)):
        entry = {"name": name, "role": role, "kind": "safe-released" if name == "released-reaction" else "rollout-checkpoint",
            "parameters": 100, "training_step": 52000 if name == "released-reaction" else 300,
            "full_depth_updates": 100, "phase_updates": {str(stages): 100, "prefix": 200}, "eligible_validation_records": 2,
            "terminal_step": 350, "terminal_phase_updates": {str(stages): 150, "prefix": 200}, "charged_seconds": 100.,
            "training_gpu_seconds_at_best": 90., "checkpoint_sha256": SHA, "config_sha256": SHA,
            "model_config": {"stages": stages}}
        config = tmp_path / f"{name}.json"
        put(config, {"seed": 20261113, "microbatch": 1, "accumulation": 2, "patch_size": 180})
        entry["config"] = str(config)
        selection["models"].append(entry)
        assets.append({"model": name, "role": role, "filename": name + ".pt", "sha256": subject.replay.REACTION_SHA256 if name == "released-reaction" else SHA,
            "metadata": {"implementation": "beltrami_operator" if name == "released-reaction" else "beltrami_rollout",
                "model_config": {"stages": stages}, "sigma": 25, "grayscale_offset": 25, "full_depth_updates": 100,
                "training_step": entry["training_step"]}})
    path = tmp_path / "selection.json"
    selection_sha = put(path, selection)
    path.with_name(path.name + ".sha256").write_text(selection_sha + "\n")
    original_module = subject.module
    def fake_private(name, private_path):
        if name == "rollout_private_frozen_verifier":
            def verify(p):
                sha = subject.protocol.file_hash(p)
                if p.with_name(p.name + ".sha256").read_text().strip() != sha:
                    raise ValueError("Selection digest differs")
                return subject.read(p), sha
            return SimpleNamespace(SCHEMA="private-synthetic-schema", load_selection=verify)
        return original_module(name, private_path)
    monkeypatch.setattr(subject, "module", fake_private)
    monkeypatch.setattr(subject, "verify_exports", lambda spec, selected: copy.deepcopy(assets))
    published = metadata("bsd68.json")
    images = {x["name"]: x for x in published["images"]}
    cases = {x["name"]: x for x in published["cases"] if x["sigma"] == 25}
    models = []
    for m, entry in enumerate(selection["models"]):
        records = []
        for i, name in enumerate(sorted(images)):
            psnr = 20 + i / 100 + m / 10
            records.append({"name": name, "shape_native": images[name]["shape"], "noisy_native_sha256": cases[name]["noisy_native_sha256"],
                "label_native_sha256": images[name]["label_native_sha256"], "psnr": psnr, "mse": 255 ** 2 / 10 ** (psnr / 10),
                "perfect_prediction": False, "clipped_pixels": 0, "seconds_forward": [.01, .02, .03], "median_seconds_forward": .02,
                "private_extra": "PRIVATE_IMAGE_MARKER"})
        models.append({"name": entry["name"], "role": entry["role"], "parameters": 100, "complete": True, "images": records,
                       "mean_psnr": float(np.mean([x["psnr"] for x in records])), "median_seconds_forward": .02, "sum_seconds_forward": 68 * .06})
    by_name = {x["name"]: x for x in models}
    raw = {"schema": "private-synthetic-schema", "complete": True, "dataset": "bsd68", "image_count": 68,
        "selection": selection, "selection_sha256": selection_sha, "lease_id": "private-final-evaluation",
        "started_utc": "2026-02-01T00:00:00+00:00", "ended_utc": "2026-02-01T01:00:00+00:00", "models": models,
        "dataset_provenance": {"loader": "deep_patch_beltrami.reproduce.PublicBSD68",
            "public_bsd68_manifest_sha256": subject.hashlib.sha256(json.dumps(published, sort_keys=True).encode()).hexdigest(),
            "noise_asset_sha256": metadata("release-assets.json")["assets"]["bsd68_noise"]["sha256"]},
        "environment": {"host": "PRIVATE_HOST_MARKER", "python": "3.11.9 PRIVATE_BUILD_MARKER", "torch": "2.5.1",
            "numpy": "2.1.3", "cuda": "12.6", "device": "cuda:0", "device_name": "NVIDIA GeForce RTX 4090", "tf32": False},
        "comparisons": [{**p, **subject.protocol.paired_summary(by_name[p["candidate"]]["images"], by_name[p["reference"]]["images"], 7, 1000),
                         "private_extra": "PRIVATE_PAIR_MARKER"} for p in subject.replay.COMPARISONS]}
    results = tmp_path / "results.json"
    put(results, raw)
    status = tmp_path / "status.json"
    put(status, {"state": "complete", "images": 68, "models": 4})
    return {"selection": path, "results": results, "status": status, "exports": tmp_path / "ignored-spec", "private": private,
            "output": tmp_path / "output", "raw": raw, "assets": assets}


def build(frozen, output=None, budget=None):
    return subject.build(frozen["selection"], frozen["results"], subject.protocol.file_hash(frozen["results"]), frozen["status"],
        frozen["exports"], output or frozen["output"], frozen["private"], budget)


def test_synthetic_report_is_path_free_and_all_six_pairs_retained(frozen):
    result = build(frozen)
    assert len(result["models"]) == 4 and len(result["comparisons"]) == 6
    output = frozen["output"]
    content = "\n".join(p.read_text() for p in output.iterdir())
    assert str(output.parent) not in content
    assert "PRIVATE_" not in content
    assert "Reused BSD68" in content and "not untouched" in content
    manifest = subject.read(output / "models.json")
    assert subject.replay.read_manifest(output / "models.json", subject.protocol.file_hash(output / "models.json")) == manifest
    assert len((output / "per-image.csv").read_text().splitlines()) == 1 + 272
    assert len((output / "paired-per-image.csv").read_text().splitlines()) == 1 + 408
    assert len((output / "runtime.csv").read_text().splitlines()) == 1 + 816
    with pytest.raises(FileExistsError): build(frozen)


@pytest.mark.parametrize("mutation", ["partial", "status", "wrong_hash", "wrong_mean", "wrong_psnr", "wrong_timing", "wrong_pair", "wrong_bootstrap", "evaluator"])
def test_corrupt_evidence_fails_before_output(frozen, mutation):
    raw = frozen["raw"]
    if mutation == "partial": raw["complete"] = False
    elif mutation == "status": put(frozen["status"], {"state": "failed"})
    elif mutation == "wrong_hash": raw["models"][0]["images"][0]["noisy_native_sha256"] = "0" * 64
    elif mutation == "wrong_mean": raw["models"][0]["mean_psnr"] += 1
    elif mutation == "wrong_psnr": raw["models"][0]["images"][0]["psnr"] += 1
    elif mutation == "wrong_timing": raw["models"][0]["images"][0]["seconds_forward"][0] = -1
    elif mutation == "wrong_pair": raw["comparisons"].pop()
    elif mutation == "wrong_bootstrap": raw["comparisons"][0]["mean_delta_db"] += 1
    else: frozen["private"].write_text("# changed verifier\n")
    put(frozen["results"], raw)
    with pytest.raises(ValueError): build(frozen)
    assert not frozen["output"].exists()


def test_accounting_refresh_changes_no_replayed_scientific_bytes(frozen):
    build(frozen)
    ledger = frozen["results"].parent / "ledger.json"
    put(ledger, {"authorized_gpu_seconds": 43200, "status": "idle", "consumed_gpu_seconds": 10.,
                 "entries": [{"state": "complete", "charged_seconds": 10., "command": ["PRIVATE_COMMAND_MARKER"]}]})
    final = frozen["output"].with_name("final")
    result = build(frozen, final, ledger)
    assert result["compute_accounting"]["available"]
    for name in ("models.json", "reference-results.json", "per-image.csv", "paired-per-image.csv", "runtime.csv"):
        assert (final / name).read_bytes() == (frozen["output"] / name).read_bytes()
    assert (final / "results.json").read_bytes() != (frozen["output"] / "results.json").read_bytes()
    assert "PRIVATE_COMMAND_MARKER" not in (final / "results.json").read_text()


def test_active_or_enlarged_ledger_is_refused(frozen):
    ledger = frozen["results"].parent / "ledger.json"
    for cap, status in ((43200, "running"), (43201, "idle")):
        put(ledger, {"authorized_gpu_seconds": cap, "status": status, "consumed_gpu_seconds": 10.,
                     "entries": [{"state": "complete", "charged_seconds": 10.}]})
        with pytest.raises(ValueError): build(frozen, budget=ledger)
        assert not frozen["output"].exists()

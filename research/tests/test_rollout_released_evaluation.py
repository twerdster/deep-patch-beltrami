"""Synthetic CPU replay contracts; no image files, checkpoints or GPU are opened."""
from argparse import Namespace
import copy
from datetime import datetime, timezone
import importlib.util
import json
import os
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location("new_rollout_replay_tests", ROOT / "tools/evaluate_rollout_released.py")
subject = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(subject)
SHA = "1" * 64


def put(path, value):
    path.write_text(json.dumps(value, indent=2) + "\n")
    return subject.protocol.file_hash(path)


@pytest.fixture
def manifest():
    return {"schema": subject.SCHEMA, "purpose": "bsd68", "evaluation_scope": subject.SCOPE,
        "sigma": 25, "offset": 25, "selection_sha256": SHA, "results_sha256": SHA,
        "private_evaluator_sha256": SHA, "protocol_helper_sha256": subject.protocol.file_hash(subject.HELPER_PATH),
        "source_pins": subject.rollout_artifacts.source_pins(), "selection_created_utc": "2026-01-01T00:00:00+00:00",
        "lease_exclusions": ["old-training"], "expected_results": {"filename": "reference-results.json", "sha256": SHA},
        "assets": [{"model": name, "role": role, "filename": name + ".pt", "sha256": subject.REACTION_SHA256 if name == "released-reaction" else SHA,
            "metadata": {"implementation": "beltrami_operator" if name == "released-reaction" else "beltrami_rollout",
                "model_config": {"stages": stages}, "sigma": 25, "grayscale_offset": 25,
                "full_depth_updates": 20, "training_step": 52000 if name == "released-reaction" else 300}}
            for name, role, stages in zip(subject.MODEL_NAMES, ("control", "candidate", "candidate", "benchmark"), (5, 32, 64, 5))],
        "comparisons": copy.deepcopy(subject.COMPARISONS), "timing_warmup": 2, "timing_repeats": 3,
        "bootstrap_samples": 1000, "bootstrap_seed": 7}


def test_valid_manifest_pin_and_source_checks(tmp_path, manifest):
    path = tmp_path / "models.json"
    pin = put(path, manifest)
    assert subject.read_manifest(path, pin) == manifest
    with pytest.raises(ValueError, match="independent pin"):
        subject.read_manifest(path, "0" * 64)


@pytest.mark.parametrize("mutation", ["scope", "cohort", "roles", "pairs", "depth", "reaction", "full-depth", "source", "helper", "time", "bootstrap", "path"])
def test_manifest_rejects_wrong_protocol_cohort_or_source(manifest, mutation):
    if mutation == "scope": manifest["evaluation_scope"] = "untouched holdout"
    elif mutation == "cohort": manifest["assets"].pop()
    elif mutation == "roles": manifest["assets"][0]["role"] = "candidate"
    elif mutation == "pairs": manifest["comparisons"].pop()
    elif mutation == "depth": manifest["assets"][1]["metadata"]["model_config"]["stages"] = 16
    elif mutation == "reaction": manifest["assets"][3]["sha256"] = SHA
    elif mutation == "full-depth": manifest["assets"][1]["metadata"]["full_depth_updates"] = 19
    elif mutation == "source": manifest["source_pins"]["beltrami_rollout"] = SHA
    elif mutation == "helper": manifest["protocol_helper_sha256"] = SHA
    elif mutation == "time": manifest["selection_created_utc"] = "2999-01-01T00:00:00+00:00"
    elif mutation == "bootstrap": manifest["bootstrap_samples"] = True
    else: manifest["expected_results"]["filename"] = "../private.json"
    with pytest.raises(ValueError): subject.validate_cohort(manifest)


@pytest.fixture
def budget():
    return {"authorized_gpu_seconds": 43200, "status": "running", "entries": [{"state": "running", "run_id": "new-replay",
        "supervisor_pid": 100, "process_pid": 101, "started_utc": "2026-02-01T00:00:00+00:00", "allowance_seconds": 900}]}


def test_exact_new_exclusive_lease(manifest, budget):
    assert subject.assert_new_lease(budget, manifest, "new-replay", 101, 100) is budget["entries"][0]


@pytest.mark.parametrize("mutation", ["budget", "duplicate", "old", "parent", "pid", "unbounded", "before", "terminated"])
def test_lease_rejects_wrong_identity_or_budget(manifest, budget, mutation):
    lease = budget["entries"][0]
    if mutation == "budget": budget["authorized_gpu_seconds"] += 1
    elif mutation == "duplicate": budget["entries"].append(copy.deepcopy(lease))
    elif mutation == "old": manifest["lease_exclusions"].append("new-replay")
    elif mutation == "parent": lease["supervisor_pid"] = 999
    elif mutation == "pid": lease["process_pid"] = 999
    elif mutation == "unbounded": lease["allowance_seconds"] = 1801
    elif mutation == "before": lease["started_utc"] = "2025-01-01T00:00:00+00:00"
    else: lease["termination_reason"] = "budget"
    with pytest.raises(ValueError): subject.assert_new_lease(budget, manifest, "new-replay", 101, 100)


def test_pid_wait_is_bounded_and_wrong_published_pid_fails_immediately(tmp_path, manifest, budget, monkeypatch):
    args = Namespace(budget_file=tmp_path / "budget", manifest=tmp_path / "manifest", output=tmp_path / "output",
                     gpu_lock=tmp_path / "lock", lease_id="new-replay")
    lease = budget["entries"][0]
    lease.update(supervisor_pid=os.getppid(), process_pid=os.getpid(), command=[str(Path(p).resolve()) for p in
        (subject.__file__, args.manifest, args.output, args.budget_file, args.gpu_lock)])
    assert subject.wait_for_published_lease(args, manifest, read_budget=lambda p: budget) >= 0
    lease["process_pid"] += 1
    with pytest.raises(ValueError, match="child PID"):
        subject.wait_for_published_lease(args, manifest, read_budget=lambda p: budget, sleep=lambda n: pytest.fail("must not sleep"))
    del lease["process_pid"]
    clock = iter((0., 0., 9.))
    with pytest.raises(ValueError, match="within9seconds"):
        subject.wait_for_published_lease(args, manifest, read_budget=lambda p: budget, monotonic=lambda: next(clock), sleep=lambda n: None)


@pytest.fixture
def quality(tmp_path):
    reference = {"complete": True, "selection_sha256": SHA, "evaluation_scope": subject.SCOPE, "models": [], "comparisons": []}
    for name, role in zip(subject.MODEL_NAMES, ("control", "candidate", "candidate", "benchmark")):
        reference["models"].append({"name": name, "role": role, "parameters": 1, "mean_psnr": 20.,
            "images": [{"name": f"synthetic-{i}", "shape_native": [1, 1, 2, 3], "noisy_native_sha256": SHA,
                "label_native_sha256": SHA, "mse": 650.25, "psnr": 20., "clipped_pixels": 0} for i in range(68)]})
    for pair in subject.COMPARISONS:
        reference["comparisons"].append({**pair, "mean_delta_db": 0.})
    path = tmp_path / "reference.json"
    pin = put(path, reference)
    replay = copy.deepcopy(reference)
    replay["manifest"] = {"expected_results": {"sha256": pin}, "selection_sha256": SHA}
    replay_path = tmp_path / "replay.json"
    return path, pin, replay_path, replay


def test_exact_272_record_reproduction_receipt(quality):
    path, pin, replay_path, value = quality
    replay_pin = put(replay_path, value)
    receipt = subject.compare_replay(path, replay_path, pin, replay_pin)
    assert receipt["model_image_records"] == 272 and not receipt["prediction_pixel_arrays_compared"]


@pytest.mark.parametrize("mutation", ["incomplete", "reference-pin", "selection", "model", "shape", "input", "mse", "psnr", "clip", "pair"])
def test_score_or_provenance_mismatch_is_not_reproduction(quality, mutation):
    path, pin, replay_path, value = quality
    if mutation == "incomplete": value["complete"] = False
    elif mutation == "reference-pin": value["manifest"]["expected_results"]["sha256"] = "0" * 64
    elif mutation == "selection": value["manifest"]["selection_sha256"] = "0" * 64
    elif mutation == "model": value["models"].pop()
    elif mutation == "pair": value["comparisons"][0]["mean_delta_db"] = 1.
    else:
        image = value["models"][0]["images"][0]
        key, item = {"shape": ("shape_native", [1, 1, 3, 2]), "input": ("noisy_native_sha256", "0" * 64),
            "mse": ("mse", 1.), "psnr": ("psnr", 21.), "clip": ("clipped_pixels", 1)}[mutation]
        image[key] = item
    replay_pin = put(replay_path, value)
    with pytest.raises(ValueError): subject.compare_replay(path, replay_path, pin, replay_pin)

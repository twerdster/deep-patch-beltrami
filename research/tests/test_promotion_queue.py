"""Prepared queues reject mutations, exhausted budgets and stale terminal evidence."""
import copy
import importlib.util
import json
import shutil
from pathlib import Path
import sys
import time

import pytest
import torch

TOOLS = Path(__file__).resolve().parents[1] / "tools"


def load(name):
    spec = importlib.util.spec_from_file_location(name, TOOLS / (name + ".py"))
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


queue = load("run_prepared_promotions")
promote = load("promote_run")


def put(path, value):
    path.write_text(json.dumps(value) + "\n")


@pytest.fixture
def prepared(tmp_path, monkeypatch):
    monkeypatch.setattr(promote, "ROOT", tmp_path)
    monkeypatch.setattr(queue, "ROOT", tmp_path)
    (tmp_path / "tools").mkdir()
    shutil.copyfile(TOOLS / "lease_runner.py", tmp_path / "tools/lease_runner.py")
    source = tmp_path / "src/beltrami_operator"
    source.mkdir(parents=True)
    (source / "__init__.py").write_text("")
    (source / "train.py").write_text("# Synthetic numerical source.\n")
    (tmp_path / "configs").mkdir()
    (tmp_path / "data").mkdir()
    config = tmp_path / "configs/control.json"
    put(config, {"steps": 20, "model": {}})
    for name in ("train400", "validation32"):
        put(tmp_path / "data" / (name + ".json"), {"split": name})
    parent = tmp_path / "runs/pilot-001"
    parent.mkdir(parents=True)
    put(parent / "status.json", {"state": "stopped", "reason": "budget", "step": 10})
    put(parent / "metadata.json", {})
    ckpt = {"source_sha256": promote.numerical_source_hash(), "config_sha256": promote.digest(config),
            "config": queue.read(config), "manifest_sha256": promote.digest(tmp_path / "data/train400.json"),
            "validation_manifest_sha256": promote.digest(tmp_path / "data/validation32.json"),
            "model": {"w": torch.ones(1)}, "optimizer": {"moment": torch.zeros(1)},
            "step": 10, "gpu_seconds": 1., "best_psnr": 28.}
    torch.save(ckpt, parent / "last.pt")
    torch.save(ckpt, parent / "best.pt")
    ledger = {"authorized_gpu_seconds": 43200, "entries": [
        {"run_id": "pilot-001", "state": "complete", "returncode": 0, "charged_seconds": 2.}]}
    put(tmp_path / "BUDGET.json", ledger)
    decision = tmp_path / "DECISION.md"
    decision.write_text("Validation decision; no test access.\n")
    model_dir, gpu_lock = tmp_path / "models", tmp_path / "gpu.lock"
    output = tmp_path / "runs/promote-001"
    prepared = promote.prepare(parent, output, config, .04, decision, model_dir, gpu_lock)
    row = {"parent": parent.name, "output": output.name, "config": config.name, "hours": .04}
    plan = {"schema": 1, "state": "prepared_not_launched", "decision": decision.name, "runs": [row]}
    put(tmp_path / "plan.json", plan)
    return tmp_path, row, prepared, decision, model_dir, gpu_lock, ledger


def verify(items):
    root, row, _, decision, model_dir, gpu_lock, ledger = items
    return queue.verify_prepared(root, row, decision, model_dir, gpu_lock, ledger)


def test_prepared_schema_and_budget_pass_without_modification(prepared):
    root, row, original, *_ = prepared
    queue.validate_plan(queue.read(root / "plan.json"))
    assert queue.budget_idle(root, [row])["entries"][0]["run_id"] == "pilot-001"
    assert verify(prepared) == original


@pytest.mark.parametrize("kind", ["command", "decision", "source", "config", "manifest", "checkpoint", "status"])
def test_artifact_mutations_refuse_execution(prepared, kind):
    root, row, plan, decision, *_ = prepared
    output = root / "runs" / row["output"]
    if kind == "command":
        plan["command"] += ["--unexpected"]
        put(output / "promotion.json", plan)
    elif kind == "checkpoint":
        (output / "parent-last.pt").write_bytes(b"corruption")
    elif kind == "status":
        put(output / "status.json", {"state": "running"})
    else:
        path = {"decision": decision, "source": root / "src/beltrami_operator/train.py",
                "config": root / "configs/control.json", "manifest": root / "data/train400.json"}[kind]
        path.write_text(path.read_text() + "\n")
    with pytest.raises(ValueError):
        verify(prepared)


@pytest.mark.parametrize("mutation", ["running", "used", "overspent", "nan", "authorization"])
def test_budget_rejects_unresolved_reused_and_overcommitted_leases(prepared, mutation):
    root, row, *rest = prepared
    ledger = queue.read(root / "BUDGET.json")
    if mutation == "running":
        ledger["entries"][0]["state"] = "running"
    elif mutation == "used":
        ledger["entries"][0]["run_id"] = row["output"]
    elif mutation == "overspent":
        ledger["entries"][0]["charged_seconds"] = 43100
    elif mutation == "nan":
        ledger["entries"][0]["charged_seconds"] = float("nan")
    else:
        ledger["authorized_gpu_seconds"] = 43201
    put(root / "BUDGET.json", ledger)
    with pytest.raises(ValueError):
        queue.budget_idle(root, [row])


def terminal_fixture(prepared):
    root, row, plan, *rest = prepared
    output = root / "runs" / row["output"]
    started = time.time() - .1
    last = torch.load(output / "parent-last.pt", weights_only=False)
    last.update(step=11, gpu_seconds=3.)
    torch.save(last, output / "last.pt")
    put(output / "status.json", {"state": "stopped", "reason": "budget", "step": 11,
        "utc": queue.utc(), "checkpoint": str(output / "last.pt")})
    ledger = queue.read(root / "BUDGET.json")
    ledger["entries"].append({"run_id": row["output"], "state": "complete", "returncode": 0,
        "charged_seconds": 2., "allowance_seconds": row["hours"] * 3600})
    put(root / "BUDGET.json", ledger)
    return root, row, plan, started


def test_clean_budget_stop_is_valid_terminal_state(prepared):
    args = terminal_fixture(prepared)
    result = queue.verify_terminal(*args, 0)
    assert result["training_status"]["step"] == 11


@pytest.mark.parametrize("kind", ["returncode", "stale", "step", "nonfinite", "signal", "lease"])
def test_terminal_failures_do_not_advance_queue(prepared, kind):
    args = terminal_fixture(prepared)
    root, row, plan, started = args
    output = root / "runs" / row["output"]
    if kind == "stale":
        status = queue.read(output / "status.json")
        status["utc"] = "2020-01-01T00:00:00+00:00"
        put(output / "status.json", status)
    elif kind in ("step", "nonfinite"):
        last = torch.load(output / "last.pt", weights_only=False)
        if kind == "step":
            last["step"] = 10
        else:
            last["optimizer"]["moment"].fill_(float("inf"))
        torch.save(last, output / "last.pt")
    elif kind == "signal":
        status = queue.read(output / "status.json")
        status["reason"] = "signal"
        put(output / "status.json", status)
    elif kind == "lease":
        ledger = queue.read(root / "BUDGET.json")
        ledger["entries"][-1]["state"] = "failed"
        put(root / "BUDGET.json", ledger)
    with pytest.raises(ValueError):
        queue.verify_terminal(*args, 1 if kind == "returncode" else 0)


def test_status_exists_never_relaunches(prepared, monkeypatch):
    root, *_ = prepared
    status = root / "queue-status.json"
    put(status, {"state": "failed"})
    monkeypatch.setattr(sys, "argv", ["runner", "--plan", str(root / "plan.json"), "--status", str(status),
        "--model-dir", str(root / "models"), "--gpu-lock", str(root / "gpu.lock")])
    with pytest.raises(FileExistsError):
        queue.main()
    assert queue.read(status) == {"state": "failed"}

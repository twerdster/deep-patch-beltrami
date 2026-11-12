"""Run a frozen sequence of prepared continuations without rebuilding commands."""
import argparse
from datetime import datetime, timezone
import fcntl
import hashlib
import json
import math
import os
from pathlib import Path
import signal
import subprocess
import sys
import time

import torch

ROOT = Path(__file__).resolve().parents[1]
IMPLEMENTATIONS = {"beltrami_operator", "beltrami_precision"}
RESERVE_SECONDS = 1800
GLOBAL_GUARD_SECONDS = 60


def utc():
    return datetime.now(timezone.utc).isoformat()


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def source_hash(root, implementation):
    if implementation not in IMPLEMENTATIONS:
        raise ValueError("Unknown numerical implementation")
    paths = sorted((root / "src" / implementation).glob("*.py"))
    if not paths:
        raise ValueError("Missing numerical source")
    value = hashlib.sha256()
    for path in paths:
        value.update(path.name.encode())
        value.update(path.read_bytes())
    return value.hexdigest()


def finite(value):
    if torch.is_tensor(value):
        return bool(torch.isfinite(value).all())
    if isinstance(value, dict):
        return all(finite(v) for v in value.values())
    if isinstance(value, (list, tuple)):
        return all(finite(v) for v in value)
    return not isinstance(value, float) or math.isfinite(value)


def read(path):
    return json.loads(path.read_text())


def save(path, state):
    temporary = path.with_suffix(path.suffix + ".partial")
    temporary.write_text(json.dumps(state, indent=2, allow_nan=False) + "\n")
    temporary.replace(path)


def basename(value):
    if not isinstance(value, str) or not value or Path(value).name != value or value in (".", ".."):
        raise ValueError("Plan paths must be nonempty basenames")
    return value


def budget_idle(root, remaining):
    """Check the actual shared lock, ledger state, and the whole remaining queue."""
    with (root / "BUDGET.lock").open("a+") as lock:
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as error:
            raise ValueError("A compute supervisor owns the budget lock") from error
        ledger = read(root / "BUDGET.json")
        entries = ledger["entries"]
        if any(e["state"] == "running" for e in entries):
            raise ValueError("A live or unresolved compute lease exists")
        authorization = ledger["authorized_gpu_seconds"]
        charges = [e["charged_seconds"] for e in entries]
        if (not math.isfinite(authorization) or not 0 < authorization <= 43200 or
                any(not math.isfinite(v) or v < 0 for v in charges)):
            raise ValueError("Invalid aggregate budget accounting")
        names = [e["run_id"] for e in entries]
        if len(names) != len(set(names)):
            raise ValueError("Duplicate lease identifiers in ledger")
        if any(r["output"] in names for r in remaining):
            raise ValueError("A prepared output already has a lease; never relaunch")
        requested = sum(r["hours"] * 3600 for r in remaining)
        if sum(charges) + requested > authorization - RESERVE_SECONDS - GLOBAL_GUARD_SECONDS:
            raise ValueError("Queue would consume the final evaluation reserve")
        return ledger


def checkpoint(path, root, config, expected_source):
    # These are trusted locally generated research checkpoints, never downloads.
    value = torch.load(path, map_location="cpu", weights_only=False)
    expected = {"source_sha256": expected_source, "config_sha256": digest(config),
                "manifest_sha256": digest(root / "data/train400.json"),
                "validation_manifest_sha256": digest(root / "data/validation32.json")}
    if any(value.get(k) != v for k, v in expected.items()) or value.get("config") != read(config):
        raise ValueError("Checkpoint source/config/data hashes differ")
    if (type(value.get("step")) is not int or value["step"] < 0 or
            not isinstance(value.get("gpu_seconds"), (int, float)) or
            not math.isfinite(value["gpu_seconds"]) or value["gpu_seconds"] < 0 or
            not isinstance(value.get("best_psnr"), (int, float)) or
            not math.isfinite(value["best_psnr"]) or
            not finite(value["model"]) or not finite(value["optimizer"])):
        raise ValueError("Checkpoint contains invalid or nonfinite state")
    return value


def expected_command(root, row, prepared, model_dir, gpu_lock):
    output = root / "runs" / row["output"]
    return [sys.executable, str(root / "tools/lease_runner.py"),
            "--budget-file", str(root / "BUDGET.json"), "--run-id", row["output"],
            "--hours", str(row["hours"]), "--log", str(output / "console.log"), "--",
            sys.executable, "-m", prepared["implementation"] + ".train",
            "--config", str(root / "configs" / row["config"]),
            "--data-root", str(root / "data/original400"),
            "--manifest", str(root / "data/train400.json"),
            "--validation-root", str(root / "data/validation32"),
            "--validation-manifest", str(root / "data/validation32.json"),
            "--model-dir", str(model_dir), "--output", str(output),
            "--gpu-lock", str(gpu_lock), "--max-gpu-hours", str(prepared["cumulative_inner_hours"]),
            "--resume", str(output / "parent-last.pt")]


def verify_prepared(root, row, decision, model_dir, gpu_lock, ledger):
    output = root / "runs" / row["output"]
    parent = root / "runs" / row["parent"]
    config = root / "configs" / row["config"]
    for name in ("status.json", "metadata.json", "last.pt", "failure.pt", "failure.txt", "events.jsonl", "console.log"):
        if (output / name).exists():
            raise ValueError("Prepared output already contains execution evidence: " + name)
    prepared = read(output / "promotion.json")
    if (prepared["parent"] != str(parent) or prepared["new_lease_hours"] != row["hours"] or
            prepared["decision_sha256"] != digest(decision) or
            digest(output / "promotion-decision.txt") != digest(decision) or
            prepared["config_sha256"] != digest(config)):
        raise ValueError("Prepared promotion differs from the reviewed decision/plan")
    implementation = prepared["implementation"]
    expected_source = source_hash(root, implementation)
    if prepared["source_sha256"] != expected_source:
        raise ValueError("Prepared numerical source changed")
    metadata = read(parent / "metadata.json")
    if metadata.get("implementation", "beltrami_operator") != implementation:
        raise ValueError("Parent implementation differs")
    precision = read(config).get("model", {}).get("metric_precision", "float32")
    if prepared.get("metric_precision") != precision or (implementation == "beltrami_operator" and precision != "float32"):
        raise ValueError("Prepared precision differs")
    leases = [e for e in ledger["entries"] if e["run_id"] == row["parent"]]
    if len(leases) != 1 or leases[0]["state"] != "complete" or leases[0].get("returncode") != 0:
        raise ValueError("Parent lease did not finish successfully")
    parent_status = read(parent / "status.json")
    if not (parent_status.get("state") == "complete" or
            (parent_status.get("state") == "stopped" and parent_status.get("reason") == "budget")):
        raise ValueError("Parent training did not end cleanly")
    for name in ("status.json", "metadata.json"):
        if digest(parent / name) != digest(output / ("parent-" + name)):
            raise ValueError("Frozen parent metadata/status differs")
    for frozen, original, key in (("parent-last.pt", "last.pt", "parent_last_sha256"),
                                  ("best.pt", "best.pt", "parent_best_sha256")):
        if digest(output / frozen) != prepared[key] or digest(parent / original) != prepared[key]:
            raise ValueError("Frozen parent checkpoint hash differs")
    last = checkpoint(output / "parent-last.pt", root, config, expected_source)
    best = checkpoint(output / "best.pt", root, config, expected_source)
    if (last["step"] != parent_status["step"] or last["step"] != prepared["parent_step"] or
            last["step"] >= last["config"]["steps"] or best["step"] > last["step"] or
            best["gpu_seconds"] > last["gpu_seconds"] or best["best_psnr"] != last["best_psnr"]):
        raise ValueError("Parent checkpoint chronology is inconsistent")
    inner = last["gpu_seconds"] / 3600 + row["hours"] - .02
    if prepared["cumulative_inner_hours"] != inner or not 0 < inner <= 12:
        raise ValueError("Prepared cumulative allowance differs")
    if prepared["command"] != expected_command(root, row, prepared, model_dir, gpu_lock):
        raise ValueError("Prepared command differs from exact continuation schema")
    return prepared


def validate_plan(plan):
    if (plan.get("schema") != 1 or plan.get("state") != "prepared_not_launched" or
            not isinstance(plan.get("runs"), list) or not plan["runs"]):
        raise ValueError("Require a nonempty prepared schema1 promotion plan")
    basename(plan["decision"])
    outputs = []
    for row in plan["runs"]:
        if set(row) != {"parent", "output", "config", "hours"}:
            raise ValueError("Unexpected promotion plan fields")
        for key in ("parent", "output", "config"):
            basename(row[key])
        if (not isinstance(row["hours"], (int, float)) or isinstance(row["hours"], bool) or
                not math.isfinite(row["hours"]) or not .03 < row["hours"] <= 12):
            raise ValueError("Invalid promotion allowance")
        if row["output"] in outputs or row["parent"] == row["output"]:
            raise ValueError("Promotion outputs must be unique and new")
        outputs.append(row["output"])


def verify_terminal(root, row, prepared, launched_at, returncode):
    output = root / "runs" / row["output"]
    ledger = read(root / "BUDGET.json")
    leases = [e for e in ledger["entries"] if e["run_id"] == row["output"]]
    if (returncode != 0 or len(leases) != 1 or leases[0]["state"] != "complete" or
            leases[0].get("returncode") != 0 or any(e["state"] == "running" for e in ledger["entries"])):
        raise ValueError("Continuation lease failed or remains unresolved")
    lease = leases[0]
    if (lease.get("termination_reason") or lease.get("forced_kill") or
            lease.get("allowance_seconds") != row["hours"] * 3600 or
            not math.isfinite(lease["charged_seconds"]) or lease["charged_seconds"] < 0 or
            lease["charged_seconds"] > row["hours"] * 3600 + GLOBAL_GUARD_SECONDS):
        raise ValueError("Unexpected lease termination or accounting")
    status = read(output / "status.json")
    config = root / "configs" / row["config"]
    if not (status.get("state") == "complete" or
            (status.get("state") == "stopped" and status.get("reason") == "budget")):
        raise ValueError("Unexpected terminal training status")
    if (datetime.fromisoformat(status["utc"]).timestamp() < launched_at or
            (output / "last.pt").stat().st_mtime < launched_at - 1 or
            status.get("checkpoint") != str(output / "last.pt")):
        raise ValueError("Terminal checkpoint/status is stale")
    last = checkpoint(output / "last.pt", root, config, prepared["source_sha256"])
    best = checkpoint(output / "best.pt", root, config, prepared["source_sha256"])
    if (last["step"] != status.get("step") or last["step"] <= prepared["parent_step"] or
            best["step"] > last["step"] or best["best_psnr"] != last["best_psnr"] or
            (status["state"] == "complete" and last["step"] != last["config"]["steps"])):
        raise ValueError("Terminal checkpoint step/best chronology differs")
    return {"run_id": row["output"], "returncode": returncode, "lease": lease,
            "training_status": status, "last_sha256": digest(output / "last.pt"),
            "best_sha256": digest(output / "best.pt")}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--plan", type=Path, required=True)
    parser.add_argument("--status", type=Path, required=True)
    parser.add_argument("--model-dir", type=Path, required=True)
    parser.add_argument("--gpu-lock", type=Path, required=True)
    parser.add_argument("--validate-only", action="store_true")
    args = parser.parse_args()
    root = ROOT
    args.plan, args.status = args.plan.resolve(), args.status.resolve()
    args.model_dir, args.gpu_lock = args.model_dir.resolve(), args.gpu_lock.resolve()
    plan = read(args.plan)
    validate_plan(plan)
    if args.status.exists():
        raise FileExistsError("Queue status exists; preserve evidence and never relaunch")
    # The queue lock spans all children; each child separately owns BUDGET.lock.
    with args.status.with_suffix(args.status.suffix + ".lock").open("a+") as queue_lock:
        try:
            fcntl.flock(queue_lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as error:
            raise ValueError("Another promotion queue owns this status path") from error
        if args.status.exists():
            raise FileExistsError("Queue status already exists")
        ledger = budget_idle(root, plan["runs"])
        decision = root / plan["decision"]
        prepared = [verify_prepared(root, r, decision, args.model_dir, args.gpu_lock, ledger)
                    for r in plan["runs"]]
        pins = {str(p): digest(p) for p in [args.plan, decision, root / "tools/lease_runner.py"] +
                [root / "runs" / r["output"] / "promotion.json" for r in plan["runs"]]}
        if args.validate_only:
            print(json.dumps({"state": "validated_not_launched", "pins": pins,
                              "commands": [p["command"] for p in prepared]}, indent=2))
            return 0
        state = {"state": "running", "pid": os.getpid(), "started_utc": utc(),
                 "plan": str(args.plan), "pins": pins, "results": []}
        # O_EXCL closes the final status-file race even against a different lock user.
        with args.status.open("x") as output:
            output.write(json.dumps(state, indent=2) + "\n")
        child = None
        stopping = False
        def stop(signum, _frame):
            nonlocal stopping
            stopping = True
            if child is not None and child.poll() is None:
                try:
                    os.killpg(child.pid, signum)
                except ProcessLookupError:
                    pass
        signal.signal(signal.SIGTERM, stop)
        signal.signal(signal.SIGINT, stop)
        try:
            for index, (row, preparation) in enumerate(zip(plan["runs"], prepared)):
                if stopping:
                    raise InterruptedError("Queue signal requested; no next run will launch")
                if any(digest(Path(p)) != value for p, value in pins.items()):
                    raise ValueError("Frozen plan, decision, preparation or lease runner changed")
                ledger = budget_idle(root, plan["runs"][index:])
                verify_prepared(root, row, decision, args.model_dir, args.gpu_lock, ledger)
                launched_at = time.time()
                state.update(current_run=row["output"], current_implementation=preparation["implementation"],
                             current_started_utc=utc(), updated_utc=utc())
                save(args.status, state)
                child = subprocess.Popen(preparation["command"], cwd=root, start_new_session=True)
                state.update(lease_pid=child.pid, updated_utc=utc())
                save(args.status, state)
                # A signal arriving during Popen is forwarded after its PID is known.
                if stopping:
                    os.killpg(child.pid, signal.SIGTERM)
                while child.poll() is None:
                    state.update(updated_utc=utc())
                    save(args.status, state)
                    time.sleep(2)
                result = verify_terminal(root, row, preparation, launched_at, child.returncode)
                state["results"].append(result)
                save(args.status, state)
                child = None
                if stopping:
                    raise InterruptedError("Queue interrupted after child termination")
            state.update(state="complete", ended_utc=utc(), current_run=None, lease_pid=None)
            save(args.status, state)
            return 0
        except BaseException as error:
            if child is not None and child.poll() is None:
                os.killpg(child.pid, signal.SIGTERM)
                try:
                    child.wait(timeout=35)
                except subprocess.TimeoutExpired:
                    os.killpg(child.pid, signal.SIGKILL)
                    child.wait()
            state.update(state="interrupted" if stopping else "failed", error=str(error),
                         ended_utc=utc(), child_returncode=child.returncode if child else None)
            save(args.status, state)
            print(str(error), file=sys.stderr, flush=True)
            return 1


if __name__ == "__main__":
    sys.exit(main())

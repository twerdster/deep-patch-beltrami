"""Prepare a verified continuation and optionally execute its bounded lease."""
import argparse
from datetime import datetime, timezone
import hashlib
import json
import math
from pathlib import Path
import shutil
import subprocess
import sys

import torch

ROOT = Path(__file__).resolve().parents[1]
IMPLEMENTATIONS = frozenset(("beltrami_operator", "beltrami_precision"))


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def numerical_source_hash(implementation="beltrami_operator"):
    if not isinstance(implementation, str) or implementation not in IMPLEMENTATIONS:
        raise ValueError(f"Unknown operator implementation: {implementation}")
    paths = sorted((ROOT / "src" / implementation).glob("*.py"))
    if not paths:
        raise ValueError(f"Missing numerical source for implementation: {implementation}")
    result = hashlib.sha256()
    for path in paths:
        result.update(path.name.encode())
        result.update(path.read_bytes())
    return result.hexdigest()


def implementation_precision(implementation, config, metadata):
    """Keep precision experiments explicit without importing either trainer."""
    model = config.get("model", {})
    if not isinstance(model, dict):
        raise ValueError("Parent model config must be an object")
    if implementation == "beltrami_precision":
        precision = model.get("metric_precision")
        if precision not in ("float64", "float32"):
            raise ValueError("Precision implementation needs explicit float64 or diagnostic float32 metric_precision")
        recorded = metadata.get("numerics", {}).get("metric_precision", precision)
        if recorded != precision:
            raise ValueError("Parent precision metadata and model config differ")
        return precision
    if "metric_precision" in model:
        raise ValueError("Legacy implementation cannot declare metric_precision")
    return "float32"


def finite_state(value):
    """Optimizer moments matter as well as the visible model parameters."""
    if torch.is_tensor(value):
        return bool(torch.isfinite(value).all())
    if isinstance(value, dict):
        return all(finite_state(item) for item in value.values())
    if isinstance(value, (list, tuple)):
        return all(finite_state(item) for item in value)
    if isinstance(value, float):
        return math.isfinite(value)
    return True


def prepare(parent, output, config, hours, decision, model_dir, gpu_lock):
    """Never overwrite a parent or continue a checkpoint under different code."""
    if not 0.03 < hours <= 12:
        raise ValueError("Continuation allowance must exceed .03 and be at most12 hours")
    if output.exists():
        raise FileExistsError(f"New continuation output already exists: {output}")
    if not decision.is_file() or not decision.read_text().strip():
        raise ValueError("Record validation-based promotion rationale first")
    status = json.loads((parent / "status.json").read_text())
    if status["state"] not in ("complete", "stopped"):
        raise ValueError("Parent must be verified complete or cleanly stopped")
    if status["state"] == "stopped" and status.get("reason") != "budget":
        raise ValueError("A signal stop needs separate diagnosis before promotion")
    metadata = json.loads((parent / "metadata.json").read_text())
    implementation = metadata.get("implementation", "beltrami_operator")
    current_source = numerical_source_hash(implementation)
    ledger = json.loads((ROOT / "BUDGET.json").read_text())
    if any(e["state"] == "running" for e in ledger["entries"]):
        raise ValueError("A live or unresolved lease still owns the budget")
    parent_leases = [e for e in ledger["entries"] if e["run_id"] == parent.name]
    if len(parent_leases) != 1 or parent_leases[0]["state"] != "complete" or parent_leases[0].get("returncode") != 0:
        raise ValueError("Parent needs one successfully terminated compute lease")
    charges = [e["charged_seconds"] for e in ledger["entries"]]
    if any(not math.isfinite(charge) or charge < 0 for charge in charges):
        raise ValueError("Budget contains invalid charged time")
    authorized = ledger["authorized_gpu_seconds"]
    if not math.isfinite(authorized) or not 0 < authorized <= 12 * 3600:
        raise ValueError("Budget authorization must not exceed12 hours")
    spent = sum(charges)
    if spent + hours * 3600 > authorized - 60 - 1800:
        raise ValueError("Continuation would consume the final .5h evaluation reserve")
    run_id = output.name
    if any(e["run_id"] == run_id for e in ledger["entries"]):
        raise ValueError("A unique new lease identifier is required")
    last_path, best_path = parent / "last.pt", parent / "best.pt"
    parent_hashes = {name: digest(path) for name, path in (("last", last_path), ("best", best_path))}
    # Only locally generated, trusted checkpoints are accepted by this helper.
    last = torch.load(last_path, map_location="cpu", weights_only=False)
    best = torch.load(best_path, map_location="cpu", weights_only=False)
    config_hash = digest(config)
    expected_config = json.loads(config.read_text())
    metric_precision = implementation_precision(implementation, expected_config, metadata)
    for name, checkpoint in (("last", last), ("best", best)):
        if checkpoint["source_sha256"] != current_source:
            raise ValueError(f"{name} checkpoint numerical source differs")
        if checkpoint["config_sha256"] != config_hash:
            raise ValueError(f"{name} checkpoint config differs")
        if checkpoint["config"] != expected_config:
            raise ValueError(f"{name} checkpoint config contents differ")
        for key, path in (("manifest_sha256", ROOT / "data/train400.json"),
                          ("validation_manifest_sha256", ROOT / "data/validation32.json")):
            if checkpoint[key] != digest(path):
                raise ValueError(f"{name} checkpoint {key} differs")
        if not finite_state(checkpoint["model"]) or not finite_state(checkpoint["optimizer"]):
            raise ValueError(f"{name} checkpoint model or optimizer state is not finite")
        if not math.isfinite(checkpoint["gpu_seconds"]) or checkpoint["gpu_seconds"] < 0:
            raise ValueError(f"{name} checkpoint elapsed time is invalid")
        if not math.isfinite(checkpoint["best_psnr"]):
            raise ValueError(f"{name} checkpoint best score is not finite")
        if not isinstance(checkpoint["step"], int) or checkpoint["step"] < 0:
            raise ValueError(f"{name} checkpoint step is invalid")
    if last["step"] != status["step"]:
        raise ValueError("Parent last checkpoint and terminal status steps differ")
    if best["step"] > last["step"] or best["gpu_seconds"] > last["gpu_seconds"]:
        raise ValueError("Parent best checkpoint is later than its last checkpoint")
    if best["best_psnr"] != last["best_psnr"]:
        raise ValueError("Parent last and best checkpoints disagree on the best score")
    if last["step"] >= last["config"]["steps"]:
        raise ValueError("Parent already reached its immutable step ceiling")
    prior_hours = last["gpu_seconds"] / 3600
    inner_hours = prior_hours + hours - .02
    if inner_hours > 12:
        raise ValueError("Cumulative inner allowance would exceed12 hours")
    output.mkdir(parents=True)
    frozen_last = output / "parent-last.pt"
    shutil.copy2(last_path, frozen_last)
    shutil.copy2(best_path, output / "best.pt")
    shutil.copy2(decision, output / "promotion-decision.txt")
    for name in ("metadata.json", "status.json"):
        shutil.copy2(parent / name, output / ("parent-" + name))
    for name, path, original in (("last", frozen_last, last_path),
                                  ("best", output / "best.pt", best_path)):
        if digest(path) != parent_hashes[name] or digest(original) != parent_hashes[name]:
            raise IOError("Checkpoint changed while it was copied")
    command = [sys.executable, str(ROOT / "tools/lease_runner.py"),
               "--budget-file", str(ROOT / "BUDGET.json"), "--run-id", run_id,
               "--hours", str(hours), "--log", str(output / "console.log"), "--",
               sys.executable, "-m", implementation + ".train",
               "--config", str(config), "--data-root", str(ROOT / "data/original400"),
               "--manifest", str(ROOT / "data/train400.json"),
               "--validation-root", str(ROOT / "data/validation32"),
               "--validation-manifest", str(ROOT / "data/validation32.json"),
               "--model-dir", str(model_dir), "--output", str(output),
               "--gpu-lock", str(gpu_lock), "--max-gpu-hours", str(inner_hours),
               "--resume", str(frozen_last)]
    plan = {"prepared_utc": datetime.now(timezone.utc).isoformat(),
            "parent": str(parent), "parent_step": last["step"],
            "parent_last_sha256": digest(frozen_last),
            "parent_best_sha256": digest(output / "best.pt"),
            "implementation": implementation, "metric_precision": metric_precision,
            "source_sha256": current_source, "config_sha256": config_hash,
            "new_lease_hours": hours, "cumulative_inner_hours": inner_hours,
            "decision_sha256": digest(decision), "command": command}
    (output / "promotion.json").write_text(json.dumps(plan, indent=2) + "\n")
    return plan


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--parent", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--hours", type=float, required=True)
    parser.add_argument("--decision", type=Path, required=True)
    parser.add_argument("--model-dir", type=Path, required=True)
    parser.add_argument("--gpu-lock", type=Path, required=True)
    parser.add_argument("--launch", action="store_true")
    args = parser.parse_args()
    plan = prepare(args.parent.resolve(), args.output.resolve(), args.config.resolve(),
                   args.hours, args.decision.resolve(), args.model_dir.resolve(),
                   args.gpu_lock.resolve())
    print(json.dumps(plan, indent=2), flush=True)
    if args.launch:
        return subprocess.call(plan["command"], cwd=ROOT)
    return 0


if __name__ == "__main__":
    sys.exit(main())

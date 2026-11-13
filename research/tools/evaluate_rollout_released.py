#!/usr/bin/env python3
"""Replay safe 5/32/64-stage weights on the reused BSD68 benchmark.

Actual replay requires a new exclusive bounded lease, the existing shared GPU
lock and a 10 GiB CUDA cap. It is reproduction, never checkpoint selection.
No raw training checkpoint or private evaluator is needed.
"""
import argparse
from datetime import datetime, timezone
import importlib.util
import json
import math
import os
from pathlib import Path
import re
import sys
import time

import numpy as np
import torch
from beltrami_research import artifacts, rollout_artifacts

ROOT = Path(__file__).resolve().parents[1]
HELPER_PATH = ROOT / "tools/evaluate_finalists.py"
_spec = importlib.util.spec_from_file_location("rollout_replay_protocol", HELPER_PATH)
protocol = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(protocol)
_spec = importlib.util.spec_from_file_location("rollout_replay_existing", ROOT / "tools/evaluate_released.py")
_existing = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_existing)
MODEL_NAMES = ("rollout-5", "rollout-32", "rollout-64", "released-reaction")
COMPARISONS = [
    {"candidate": "rollout-32", "reference": "rollout-5"},
    {"candidate": "rollout-64", "reference": "rollout-5"},
    {"candidate": "rollout-64", "reference": "rollout-32"},
    *[{"candidate": name, "reference": "released-reaction"} for name in MODEL_NAMES[:3]],
]
REACTION_SHA256 = "9a085c0745136eaee86bfaf0b2e1eb31926eb3bb0526169134fa3a89cf2cfed5"
SCHEMA = "deep-beltrami-rollout-released-v1"
SCOPE = "Reused BSD68 benchmark, already exposed in the previous research round; not untouched held-out evidence."
DIGEST = re.compile(r"[0-9a-f]{64}\Z")


def read(path):
    def reject(value):
        raise ValueError("Non-finite JSON constant: " + value)
    return json.loads(Path(path).read_text(), parse_constant=reject)


def require(condition, message):
    if not condition:
        raise ValueError(message)


def digest(value):
    require(isinstance(value, str) and DIGEST.fullmatch(value), "Invalid SHA256")
    return value


def validate_cohort(manifest):
    require(manifest.get("schema") == SCHEMA and manifest.get("purpose") == "bsd68"
            and manifest.get("evaluation_scope") == SCOPE, "Unsupported rollout manifest/scope")
    require(manifest.get("sigma") == 25 and manifest.get("offset") == 25, "Archived sigma/offset differs")
    require([x.get("model") for x in manifest["assets"]] == list(MODEL_NAMES)
            and [x.get("role") for x in manifest["assets"]] == ["control", "candidate", "candidate", "benchmark"],
            "The fixed four-model cohort/order/roles differ")
    require(manifest["comparisons"] == COMPARISONS, "The six predeclared comparisons differ")
    filenames = [entry["filename"] for entry in manifest["assets"]]
    require(len(set(filenames)) == len(filenames), "Safe artifact filenames must be unique")
    for filename in filenames:
        _existing.artifact_path(Path.cwd(), filename)
    for i, entry in enumerate(manifest["assets"]):
        metadata = entry["metadata"]
        expected = "beltrami_rollout" if i < 3 else "beltrami_operator"
        require(metadata["implementation"] == expected, "Asset implementation differs")
        require(metadata["model_config"]["stages"] == (5, 32, 64, 5)[i], "Asset target depth differs")
        require(metadata["sigma"] == 25 and metadata["grayscale_offset"] == 25, "Asset intensity protocol differs")
        if i < 3:
            require(type(metadata["full_depth_updates"]) is int and metadata["full_depth_updates"] >= 20,
                    "Selected rollout lacks20full-depthupdates")
        else:
            require(entry["sha256"] == REACTION_SHA256 and metadata["training_step"] == 52000,
                    "Released reaction pin differs")
    for key, minimum in (("timing_warmup", 2), ("timing_repeats", 3),
                         ("bootstrap_samples", 1000), ("bootstrap_seed", 0)):
        require(type(manifest[key]) is int and manifest[key] >= minimum, "Invalid timing/bootstrap setting")
    for key in ("selection_sha256", "results_sha256", "private_evaluator_sha256", "protocol_helper_sha256"):
        digest(manifest[key])
    require(manifest["protocol_helper_sha256"] == protocol.file_hash(HELPER_PATH), "Protocol helper source changed")
    require(manifest["source_pins"] == rollout_artifacts.source_pins(), "Numerical namespace source changed")
    created = datetime.fromisoformat(manifest["selection_created_utc"])
    require(created.tzinfo is not None and created <= datetime.now(timezone.utc), "Invalid selection timestamp")
    require(isinstance(manifest["lease_exclusions"], list)
            and all(isinstance(x, str) and re.fullmatch(r"[A-Za-z0-9_.-]+", x) for x in manifest["lease_exclusions"]),
            "Invalid pre-existing lease identities")
    digest(manifest["expected_results"]["sha256"])
    _existing.artifact_path(Path.cwd(), manifest["expected_results"]["filename"])
    return manifest


def read_manifest(path, expected_sha):
    digest(expected_sha)
    require(protocol.file_hash(path) == expected_sha, "Manifest SHA256 differs from independent pin")
    return validate_cohort(read(path))


def verify_asset(entry, path, device="cpu"):
    digest(entry["sha256"])
    require(type(entry["bytes"]) is int and entry["bytes"] > 0, "Invalid artifact size")
    require(path.stat().st_size == entry["bytes"] and protocol.file_hash(path) == entry["sha256"],
            "Safe artifact file bytes differ")
    if entry["model"] == "released-reaction":
        require(entry["sha256"] == REACTION_SHA256, "Released reaction pin differs")
        model, value = artifacts.load(path, device)
        require(set(value) == _existing._METADATA_KEYS | {"state_dict"}
                and set(value["provenance"]) == _existing._PROVENANCE_KEYS,
                "Released reaction contains unsupported metadata")
    else:
        model, value = rollout_artifacts.load_safe(path, device=device, expected_sha256=entry["sha256"])
    require({key: value for key, value in value.items() if key != "state_dict"} == entry["metadata"],
            "Manifest and safe artifact metadata differ")
    require(sum(p.numel() for p in model.parameters()) == entry["metadata"]["parameters"], "Parameter count differs")
    require(protocol.file_hash(path) == entry["sha256"], "Artifact changed during loading")
    return model


def preflight(manifest, weights_dir):
    paths = {}
    for entry in manifest["assets"]:
        path = _existing.artifact_path(weights_dir, entry["filename"])
        model = verify_asset(entry, path)
        del model
        paths[entry["model"]] = path
    return paths


def assert_new_lease(budget, manifest, lease_id, pid, parent_pid, *, allow_pending_pid=False):
    require(budget.get("authorized_gpu_seconds") == 43200 and budget.get("status") == "running",
            "Replay requires the existing live12hledger")
    running = [x for x in budget["entries"] if x["state"] == "running"]
    require(len(running) == 1 and running[0]["run_id"] == lease_id
            and lease_id not in manifest["lease_exclusions"], "Replay requires exactly one NEW exclusive lease")
    lease = running[0]
    require(lease.get("supervisor_pid") == parent_pid, "Lease parent PID differs")
    if not (allow_pending_pid and "process_pid" not in lease):
        require(lease.get("process_pid") == pid, "Lease child PID differs")
    require(datetime.fromisoformat(lease["started_utc"]) > datetime.fromisoformat(manifest["selection_created_utc"]),
            "Lease must start after selection")
    allowance = lease["allowance_seconds"]
    require(isinstance(allowance, (int, float)) and math.isfinite(allowance) and 0 < allowance <= 1800
            and not lease.get("termination_reason"), "Replay lease must be bounded to at most.5h")
    return lease


def wait_for_published_lease(args, manifest, *, read_budget=read, monotonic=time.monotonic, sleep=time.sleep):
    began = monotonic()
    while True:
        budget = read_budget(args.budget_file)
        lease = assert_new_lease(budget, manifest, args.lease_id, os.getpid(), os.getppid(), allow_pending_pid=True)
        command = lease.get("command", [])
        require(all(str(Path(path).resolve()) in command for path in
                    (__file__, args.manifest, args.output, args.budget_file, args.gpu_lock)),
                "Lease command does not identify this exact replay/manifest/output/ledger/lock")
        waited = monotonic() - began
        if "process_pid" in lease:
            return waited
        require(waited < 9, "Matching lease failed to publish its PID within9seconds")
        sleep(min(.25, 9 - waited))


def evaluate(args):
    require(sys.platform == "linux" and args.device.startswith("cuda"), "Actual replay requires leased Linux CUDA")
    manifest = read_manifest(args.manifest, args.manifest_sha256)
    require(args.gpu_lock.is_file(), "Supply the existing shared GPU lock; never create a replacement")
    require(b"lease_runner.py" in Path(f"/proc/{os.getppid()}/cmdline").read_bytes(), "Parent is not lease_runner")
    waited = wait_for_published_lease(args, manifest)
    args.output.mkdir(parents=True, exist_ok=False)
    report_path, status_path = args.output / "results.json", args.output / "status.json"
    started = time.monotonic()
    report = {"schema": SCHEMA, "complete": False, "started_utc": protocol.utc(),
              "manifest_sha256": args.manifest_sha256, "manifest": manifest,
              "evaluation_scope": SCOPE, "lease_id": args.lease_id,
              "lease_pid_publication_wait_seconds": waited, "models": [], "comparisons": []}
    def status(state, **extra):
        protocol.atomic_json(status_path, {"state": state, "utc": protocol.utc(), "pid": os.getpid(),
            "seconds": time.monotonic() - started, "lease_id": args.lease_id, **extra})
    try:
        status("preflight")
        paths = preflight(manifest, args.weights_dir or args.manifest.resolve().parent)
        with protocol.gpu_lock(args.gpu_lock):
            device = protocol.configure_device(args.device, args.threads, args.max_cuda_gib)
            report.update(dataset="bsd68", evaluator_sha256=protocol.file_hash(__file__),
                protocol_helper_sha256=protocol.file_hash(HELPER_PATH),
                environment={"python": sys.version, "torch": torch.__version__, "numpy": np.__version__,
                    "cuda": torch.version.cuda, "device": str(device), "device_name": torch.cuda.get_device_name(device),
                    "max_cuda_gib": args.max_cuda_gib, "cpu_threads": args.threads, "tf32": False})
            # The first image access is after immutable safe-asset/lease/lock guards.
            cases, provenance = protocol.dataset_cases({"purpose": "bsd68", "sigma": 25}, args)
            require(len(cases) == 68, "Replay requires all68archivedinputs")
            report.update(image_count=68, dataset_provenance=provenance)
            for entry in manifest["assets"]:
                status("loading_model", model=entry["model"])
                torch.cuda.reset_peak_memory_stats(device)
                model = verify_asset(entry, paths[entry["model"]], device)
                result = {"name": entry["model"], "role": entry["role"],
                          "parameters": entry["metadata"]["parameters"], "complete": False, "images": []}
                report["models"].append(result)
                _existing.evaluate_model(model, result, cases, manifest, device, status, report_path, report)
                result.update(peak_cuda_allocated_bytes=torch.cuda.max_memory_allocated(device),
                              peak_cuda_reserved_bytes=torch.cuda.max_memory_reserved(device))
                del model
                torch.cuda.empty_cache()
            by_name = {x["name"]: x for x in report["models"]}
            report["comparisons"] = [{**pair, **protocol.paired_summary(by_name[pair["candidate"]]["images"],
                by_name[pair["reference"]]["images"], manifest["bootstrap_seed"], manifest["bootstrap_samples"])}
                for pair in COMPARISONS]
            require(protocol.file_hash(args.manifest) == args.manifest_sha256, "Manifest changed during replay")
            for entry in manifest["assets"]:
                require(protocol.file_hash(paths[entry["model"]]) == entry["sha256"], "Safe asset changed during replay")
            report.update(complete=True, ended_utc=protocol.utc(), seconds_total=time.monotonic() - started)
            protocol.atomic_json(report_path, report)
            status("complete", models=4, images=68)
    except BaseException as error:
        report.update(failure={"type": type(error).__name__, "message": str(error)}, ended_utc=protocol.utc())
        protocol.atomic_json(report_path, report)
        status("failed", error=report["failure"])
        raise
    return report


def compare_replay(reference_path, replay_path, expected_reference_sha, expected_replay_sha):
    """CPU-only exact score/hash verification; times and prediction arrays differ in scope."""
    require(protocol.file_hash(reference_path) == digest(expected_reference_sha)
            and protocol.file_hash(replay_path) == digest(expected_replay_sha), "Reference/replay file SHA differs")
    reference, replay = read(reference_path), read(replay_path)
    require(reference.get("complete") is True and replay.get("complete") is True
            and not replay.get("failure"), "Incomplete reference/replay")
    require(reference["evaluation_scope"] == SCOPE and replay["evaluation_scope"] == SCOPE, "Benchmark scope differs")
    require(replay["manifest"]["expected_results"]["sha256"] == expected_reference_sha
            and replay["manifest"]["selection_sha256"] == reference["selection_sha256"], "Replayed reference/selection pin differs")
    require([x["name"] for x in reference["models"]] == list(MODEL_NAMES)
            and [x["name"] for x in replay["models"]] == list(MODEL_NAMES), "Replayed model cohort differs")
    count = 0
    keys = ("name", "shape_native", "noisy_native_sha256", "label_native_sha256", "mse", "psnr", "clipped_pixels")
    for expected, actual in zip(reference["models"], replay["models"]):
        require(actual.get("complete", True) and len(expected["images"]) == len(actual["images"]) == 68,
                "Incomplete68image replay")
        require(actual["role"] == expected["role"] and actual["parameters"] == expected["parameters"]
                and actual["mean_psnr"] == expected["mean_psnr"], "Model identity/quality differs")
        for left, right in zip(expected["images"], actual["images"]):
            require({key: left[key] for key in keys} == {key: right[key] for key in keys}, "Per-image replay differs")
            count += 1
    stripped = lambda rows: [{k: v for k, v in row.items() if k not in ("candidate_role", "reference_role")} for row in rows]
    require(stripped(reference["comparisons"]) == stripped(replay["comparisons"]), "Paired statistics differ")
    return {"schema": "deep-beltrami-rollout-reproduction-check-v1", "complete": True,
            "reference_sha256": expected_reference_sha, "replay_sha256": expected_replay_sha,
            "model_count": 4, "image_count": 68, "model_image_records": count,
            "exact_input_label_shape_mse_psnr_clipping_and_paired_statistics": True,
            "prediction_pixel_arrays_compared": False, "runtime_identity_required": False,
            "evaluation_scope": SCOPE}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    run = commands.add_parser("evaluate")
    for name in ("manifest", "asset-dir", "clean-dir", "output", "gpu-lock", "budget-file"):
        run.add_argument("--" + name, type=Path, required=True)
    run.add_argument("--manifest-sha256", required=True)
    run.add_argument("--lease-id", required=True)
    run.add_argument("--weights-dir", type=Path)
    run.add_argument("--cache-dir", type=Path)
    run.add_argument("--device", default="cuda:0")
    run.add_argument("--threads", type=int, default=4)
    run.add_argument("--max-cuda-gib", type=float, default=10)
    check = commands.add_parser("compare")
    for name in ("reference", "replay", "output"):
        check.add_argument("--" + name, type=Path, required=True)
    for name in ("reference-sha256", "replay-sha256"):
        check.add_argument("--" + name, required=True)
    args = parser.parse_args()
    if args.command == "evaluate":
        require(args.threads >= 1 and math.isfinite(args.max_cuda_gib) and 0 < args.max_cuda_gib <= 10,
                "Require positive threads and CUDA cap in(0,10]GiB")
        evaluate(args)
    else:
        receipt = compare_replay(args.reference, args.replay, args.reference_sha256, args.replay_sha256)
        with args.output.open("x") as stream:
            json.dump(receipt, stream, indent=2, allow_nan=False)
            stream.write("\n")
    return 0


if __name__ == "__main__":
    sys.exit(main())

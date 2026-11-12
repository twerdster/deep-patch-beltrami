#!/usr/bin/env python3
"""Summarize recorded validation32 pilot evidence without loading model weights."""
import argparse
import csv
from datetime import datetime, timezone
import hashlib
import json
import math
from pathlib import Path, PurePosixPath
import random
import statistics
import tempfile


TRAIN_SHA = "f8a22e7177107990eff632eef8385a17e40398903bf2d6be0bc5a8cae448c8c0"
VALIDATION_SHA = "27736aa4658ffbd9f072c8990b9d370ae8734431b70818edf5a04fbe4d59780e"
INITIAL_PSNR = 28.216729835605314
PILOTS = ("control-patch100", "finetune-frozen", "finetune-positive",
          "deeper-positive", "wider-positive", "reaction-frozen")
IMPLEMENTATIONS = {"beltrami_operator": "float32", "beltrami_precision": "float64"}
ROOT = Path(__file__).resolve().parents[1]
RECORDED_PROJECT_ROOT = PurePosixPath(ROOT.as_posix())
CAUTION = (
    "VALIDATION32 MODEL-SELECTION RESULTS ONLY. Each best checkpoint is the maximum "
    "over that run's verified ancestry and observed primary-noise validation means; ongoing runs have "
    "only a best-so-far. Paired image-bootstrap intervals condition on these selected "
    "checkpoints, this one fixed noise draw, and this one training seed. They do not "
    "correct for checkpoint/trial selection or estimate training/noise-seed variation. "
    "They are not final BSD68 results. No promotion is performed by this tool."
)


def sha256(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def finite_tree(value):
    if isinstance(value, float):
        return math.isfinite(value)
    if isinstance(value, dict):
        return all(finite_tree(x) for x in value.values())
    if isinstance(value, list):
        return all(finite_tree(x) for x in value)
    return True


def nonnegative_number(value):
    return isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(value) and value >= 0


def snapshot_evidence_path(root, path):
    resolved = path.resolve()
    if not resolved.is_relative_to(root.resolve()):
        raise ValueError(f"evidence file resolves outside the local snapshot: {path.name}")
    return resolved


def read_snapshot_json(root, path, issues, default=None):
    try:
        return read_json(snapshot_evidence_path(root, path), issues, default)
    except (OSError, ValueError) as error:
        issues.append(str(error))
        return default


def read_json(path, issues, default=None):
    try:
        value = json.loads(path.read_text())
        if not isinstance(value, dict):
            raise ValueError("expected JSON object")
        if not finite_tree(value):
            raise ValueError("non-finite numeric value")
        return value
    except (OSError, ValueError) as error:
        issues.append(f"{path.name}: {error}")
        return default


def read_events(path, issues):
    if not path.exists():
        issues.append("events.jsonl unavailable")
        return []
    text = path.read_text()
    lines = text.splitlines()
    result = []
    for index, line in enumerate(lines):
        try:
            event = json.loads(line)
            if not isinstance(event, dict) or not finite_tree(event):
                raise ValueError("non-finite or non-object event")
            result.append(event)
        except ValueError as error:
            if index == len(lines) - 1 and not text.endswith("\n"):
                issues.append("events.jsonl has a partial final line; snapshot may be live")
            else:
                issues.append(f"events.jsonl line {index + 1}: {error}")
    return result


def read_validation(path, issues):
    value = read_json(path, issues)
    if value is None:
        return None
    try:
        step = int(path.stem.removeprefix("validation-"))
        images = value["images"]
        keys = [(item["image"], item["pixels_sha256"]) for item in images]
        if len(images) != 32 or len(set(keys)) != 32:
            raise ValueError("expected 32 unique image/hash pairs")
        if len({item["image"] for item in images}) != 32:
            raise ValueError("duplicate image names")
        mean = statistics.fmean(item["psnr"] for item in images)
        if abs(mean - value["mean_psnr"]) > 1e-9:
            raise ValueError("reported mean does not match per-image PSNR")
        if value.get("crop") != 0 or value.get("clipping") != "prediction only, float [0,255]":
            raise ValueError("unexpected crop/clipping protocol")
        return {"step": step, "mean_psnr": mean,
                "images": {key: item["psnr"] for key, item in zip(keys, images)},
                "path": str(path), "sha256": sha256(path)}
    except (KeyError, TypeError, ValueError) as error:
        issues.append(f"{path.name}: {error}")
        return None


def percentile(values, q):
    position = (len(values) - 1) * q
    index = int(position)
    fraction = position - index
    return values[index] * (1 - fraction) + values[min(index + 1, len(values) - 1)] * fraction


def comparison_seed(base, run_id, kind):
    token = f"{base}:{run_id}:{kind}".encode()
    return int.from_bytes(hashlib.sha256(token).digest()[:8], "big")


def paired(candidate, reference, repeats, seed):
    if candidate is None or reference is None:
        return {"available": False, "reason": "validation image scores unavailable"}
    if set(candidate["images"]) != set(reference["images"]):
        return {"available": False, "reason": "image names/decoded hashes differ"}
    keys = sorted(candidate["images"])
    differences = [candidate["images"][key] - reference["images"][key] for key in keys]
    rng = random.Random(seed)
    n = len(differences)
    samples = sorted(statistics.fmean(rng.choices(differences, k=n)) for _ in range(repeats))
    return {"available": True, "n": n, "mean_delta_db": statistics.fmean(differences),
            "ci95_low_db": percentile(samples, .025), "ci95_high_db": percentile(samples, .975),
            "bootstrap_repeats": repeats, "bootstrap_seed": seed,
            "candidate_step": candidate["step"], "reference_step": reference["step"],
            "per_image": [{"image": key[0], "pixels_sha256": key[1],
                           "candidate_psnr": candidate["images"][key],
                           "reference_psnr": reference["images"][key], "delta_db": delta}
                          for key, delta in zip(keys, differences)]}


def implementation_config(metadata):
    implementation = metadata.get("implementation", "beltrami_operator")
    model = metadata.get("config", {}).get("model", {})
    precision = model.get("metric_precision", IMPLEMENTATIONS.get(implementation, "unknown"))
    return implementation, precision


def local_source_hash(root, implementation="beltrami_operator"):
    # Declared namespaces are data, never arbitrary paths or imported modules.
    if implementation not in IMPLEMENTATIONS:
        return None
    paths = sorted((root / "src" / implementation).glob("*.py"))
    if not paths:
        return None
    digest = hashlib.sha256()
    for path in paths:
        digest.update(path.name.encode())
        digest.update(path.read_bytes())
    return digest.hexdigest()


def local_manifest_path(root, declared, recorded_root=RECORDED_PROJECT_ROOT):
    """Map only project-relative paths or one explicit recorded project root.

    Metadata paths identify evidence; they must never grant access outside the
    supplied snapshot, including through parent components or local symlinks.
    The recorded Linux prefix is interpreted lexically, not resolved on this host.
    """
    if not isinstance(declared, str) or not declared or "\x00" in declared:
        raise ValueError("manifest path is missing or invalid")
    recorded_root = PurePosixPath(recorded_root)
    if not recorded_root.is_absolute() or ".." in recorded_root.parts:
        raise ValueError("recorded project root must be absolute without parent traversal")
    recorded = PurePosixPath(declared)
    if ".." in recorded.parts:
        raise ValueError("manifest path contains parent traversal")
    if recorded.is_absolute():
        try:
            relative = recorded.relative_to(recorded_root)
        except ValueError as error:
            raise ValueError("absolute manifest path is outside the explicit recorded project root") from error
    else:
        relative = recorded
    snapshot = root.resolve()
    local = (snapshot / Path(*relative.parts)).resolve()
    if not local.is_relative_to(snapshot):
        raise ValueError("manifest path resolves outside the local snapshot")
    return local


def latest_common_pair(candidate, reference):
    """Select by step availability, without considering validation scores."""
    if reference is None:
        return None, None
    candidates = {value["step"]: value for value in candidate["validations"]}
    references = {value["step"]: value for value in reference["validations"]}
    common = set(candidates) & set(references)
    if not common:
        return None, None
    step = max(common)
    return candidates[step], references[step]


def _load_run_local(path, root, ledger, recorded_root=RECORDED_PROJECT_ROOT):
    issues = []
    metadata = read_snapshot_json(root, path / "metadata.json", issues, {})
    status = read_snapshot_json(root, path / "status.json", issues, {})
    try:
        events = read_events(snapshot_evidence_path(root, path / "events.jsonl"), issues)
    except (OSError, ValueError) as error:
        issues.append(str(error))
        events = []
    validations = []
    for candidate in sorted(path.glob("validation-*.json")):
        try:
            value = read_validation(snapshot_evidence_path(root, candidate), issues)
            if value is not None:
                validations.append(value)
        except (OSError, ValueError) as error:
            issues.append(str(error))
    config = metadata.get("config", {})
    model = config.get("model", {})
    implementation, metric_precision = implementation_config(metadata)
    initial = next((v for v in validations if v["step"] == 0), None)
    best = max(validations, key=lambda v: (v["mean_psnr"], -v["step"])) if validations else None
    validation_events = {int(e["step"]): e for e in events if e.get("event") == "validation"}
    for value in validations:
        event = validation_events.get(value["step"])
        value["gpu_seconds"] = event.get("gpu_seconds") if event else None
        value["event_utc"] = event.get("utc") if event else None
        value["origin_run_id"] = path.name
        if event is None:
            issues.append(f"validation step {value['step']} lacks event (snapshot may be live)")
        elif abs(event["mean_psnr"] - value["mean_psnr"]) > 1e-9:
            issues.append(f"validation step {value['step']} event/file means differ")
    steps = [int(status.get("step", 0))] + [int(e.get("step", 0)) for e in events]
    updates = max(steps)
    batch, patch = config.get("batch_size"), config.get("patch_size")
    for e in events:
        if e.get("event") == "progress" and batch and e.get("images_seen") != e["step"] * batch:
            issues.append("progress images_seen disagrees with step × batch")
            break
    leases = [e for e in ledger.get("entries", []) if e.get("run_id") == path.name]
    lease_seconds = sum(e["charged_seconds"] for e in leases) if leases and all(nonnegative_number(e.get("charged_seconds")) for e in leases) else None
    if leases and lease_seconds is None:
        issues.append("lease charge unavailable or invalid; charged time cannot be assumed zero")
    lease_allowance = sum(e["allowance_seconds"] for e in leases) if leases and all(nonnegative_number(e.get("allowance_seconds")) for e in leases) else None
    state = status.get("state", "unknown")
    if state in ("training", "initializing", "evaluating"):
        state_label = f"ongoing ({state}; snapshot)"
    elif state == "stopped":
        state_label = f"stopped ({status.get('reason', 'unspecified')}); ceiling not complete"
    elif state == "complete":
        state_label = "complete (step ceiling)"
    else:
        state_label = state
    progress = [e for e in events if e.get("event") == "progress"]
    failure = state == "failed" or any(e.get("event") == "failure" for e in events)
    failure_messages = [str(status.get("error", ""))] + [str(e.get("error", "")) for e in events if e.get("event") == "failure"]
    finite_bad = (any("non-finite" in issue for issue in issues) or
                  any("non-finite" in message.lower() or "nonfinite" in message.lower() for message in failure_messages))
    source = metadata.get("environment", {}).get("source_sha256")
    initialization = metadata.get("initialization", {})
    data_ok = (metadata.get("manifest_sha256") == TRAIN_SHA and
               metadata.get("validation_manifest_sha256") == VALIDATION_SHA and
               metadata.get("train_images") == 400 and metadata.get("validation_images") == 32)
    data_local = {}
    manifest_evidence = {}
    for label, key in (("training", "manifest"), ("validation", "validation_manifest")):
        declared = metadata.get("paths", {}).get(key)
        expected = metadata.get("manifest_sha256" if label == "training" else "validation_manifest_sha256")
        evidence = {"declared_path": declared, "recorded_project_root": str(recorded_root),
                    "local_path": None, "recorded_sha256": expected, "local_sha256": None}
        try:
            local = local_manifest_path(root, declared, recorded_root)
            evidence["local_path"] = str(local)
            if local.is_file():
                evidence["local_sha256"] = sha256(local)
                data_local[label] = "matches recorded SHA" if evidence["local_sha256"] == expected else "MISMATCH"
                if data_local[label] == "MISMATCH":
                    issues.append(f"{label} local manifest SHA differs from recorded SHA")
                    data_ok = False
            else:
                data_local[label] = "local manifest unavailable"
        except (OSError, ValueError) as error:
            evidence["error"] = str(error)
            data_local[label] = "recorded path unavailable" if declared is None else "recorded path rejected"
            if declared is not None:
                issues.append(f"{label} manifest path rejected: {error}")
                data_ok = False
        manifest_evidence[label] = evidence
    validation_evidence = manifest_evidence["validation"]
    validation_manifest = (Path(validation_evidence["local_path"])
                           if validation_evidence["local_path"] else None)
    image_identity_check = "local manifest unavailable"
    if (validation_manifest is not None and validation_manifest.is_file() and
            validation_evidence["local_sha256"] == metadata.get("validation_manifest_sha256")):
        records = read_json(validation_manifest, issues, {}).get("images", [])
        identities = {(record["path"], record["pixels_sha256"]) for record in records if record.get("split") == "validation"}
        bad = [value for value in validations if set(value["images"]) != identities]
        image_identity_check = "all validation image names/hashes match manifest" if not bad else "MISMATCH"
        if bad:
            issues.append("saved validation image identities differ from local audited manifest")
            data_ok = False
    parity_ok = initial is not None and abs(initial["mean_psnr"] - INITIAL_PSNR) <= 1e-4
    sigma_protocol = config.get("sigma") == 25 and config.get("offset") == 25 and config.get("validation_seed") == 742913
    if not sigma_protocol:
        issues.append("not the primary sigma25/+25/validation-seed742913 protocol")
    if implementation not in IMPLEMENTATIONS:
        issues.append(f"unsupported implementation namespace: {implementation!r}")
    if metric_precision not in ("float32", "float64"):
        issues.append(f"unsupported declared metric precision: {metric_precision!r}")
    if implementation == "beltrami_precision":
        declared_precisions = [metadata.get("numerics", {}).get("metric_precision"), initialization.get("metric_precision")]
        if any(value is not None and value != metric_precision for value in declared_precisions):
            issues.append("config/metadata/initialization metric precision declarations disagree")
    if implementation == "beltrami_operator" and metric_precision != "float32":
        issues.append("legacy implementation does not implement the declared metric precision")
    if implementation == "beltrami_precision":
        warmstart = ("cohort step0 comparison pending; historical byte/mean parity not required" if initial is not None
                     else "no step0 score; cohort warmstart not assessed")
    else:
        warmstart = ("declared preserving; initial mean parity within1e-4dB" if initialization.get("function_preserving") and parity_ok else (
            "no step0 score; resume parity not assessed" if initial is None else "warmstart declaration/initial parity FAIL"))
    checks = {
        "finite": "FAIL/non-finite evidence" if finite_bad else (
            "job failure; inspect evidence" if failure else (
                "recorded progress/validation finite; no replay" if progress and validations else "insufficient logs")),
        "data": "audited manifest hashes/counts match" if data_ok else "MISMATCH or metadata missing",
        "local_manifests": data_local,
        "manifest_path_evidence": manifest_evidence,
        "validation_image_identities": image_identity_check,
        "source": "local numerical source SHA matches declared namespace" if source and source == local_source_hash(root, implementation) else (
            "local source SHA differs/unavailable" if source else "source SHA missing"),
        "source_sha256": source,
        "source_namespace": implementation,
        "warmstart": warmstart,
        "initial_checkpoint_sha256": initialization.get("source_sha256"),
        "primary_sigma_noise_protocol": sigma_protocol,
        "failure_observed": failure,
    }
    width, stages = model.get("width"), model.get("stages")
    expected_parameters = stages * (98 * width * width + 149 * width + bool(model.get("reaction"))) if width and stages else None
    checks["parameter_count"] = "matches declared architecture" if metadata.get("parameters") == expected_parameters else "MISMATCH/missing"
    checks["expected_parameters"] = expected_parameters
    if not leases:
        issues.append("no matching aggregate-budget lease entry")
    return {"run_id": path.name, "path": str(path), "metadata": metadata, "status": status,
            "implementation": implementation, "metric_precision": metric_precision,
            "state": state_label, "config": config, "model": model, "checks": checks,
            "issues": issues, "updates": updates, "images_seen": updates * batch if batch else None,
            "pixels_seen": updates * batch * patch * patch if batch and patch else None,
            "lease_seconds_snapshot": lease_seconds,
            "lease_allowance_seconds": lease_allowance,
            "lease_state": ",".join(e.get("state", "unknown") for e in leases),
            "lease_snapshot_utc": max((e.get("last_checked_utc", e.get("ended_utc", e.get("started_utc", ""))) for e in leases), default=None),
            "inner_cumulative_seconds": status.get("gpu_seconds"),
            "validation_count": len(validations), "initial": initial, "best": best,
            "validations": validations}


def resource_trend(validations, parent_step):
    """Predeclared compute-allocation heuristic, never a significance test."""
    values = sorted({value["step"]: value for value in validations
                     if value["step"] > parent_step}.values(), key=lambda value: value["step"])
    count = len(values)
    quarter = max(1, count // 4)
    first = statistics.median(value["mean_psnr"] for value in values[:quarter]) if count else None
    last = statistics.median(value["mean_psnr"] for value in values[-quarter:]) if count else None
    delta = last - first if count else None
    return {"resource_decision_only": True, "parent_step_excluded": parent_step,
            "unique_new_validation_count": count, "quarter_count": quarter,
            "rounding": "max(1, n // 4)", "minimum_count_for_assessment": 4,
            "first_quarter_median_psnr": first, "last_quarter_median_psnr": last,
            "last_minus_first_db": delta, "threshold_db": .002,
            "no_further_extension": delta < .002 if count >= 4 else None,
            "evaluated_global_steps": [value["step"] for value in values]}


def _parent_link(run, path, root, plan, recorded_root):
    """Validate saved JSON linkage before inheriting any parent evidence."""
    parent = local_manifest_path(root, plan.get("parent"), recorded_root)
    runs_root = (root.resolve() / "runs").resolve()
    if parent.parent != runs_root or not parent.is_dir() or parent == path.resolve():
        raise ValueError("parent must be a different direct run directory in this snapshot")
    evidence = {"declared_parent": plan.get("parent"), "local_parent": str(parent),
                "json_checks": {}, "artifact_checks": {}}
    documents = {}
    for name in ("metadata.json", "status.json"):
        original = snapshot_evidence_path(root, parent / name)
        copied = snapshot_evidence_path(root, path / ("parent-" + name))
        if not original.is_file() or not copied.is_file():
            raise ValueError(f"missing original or retained parent {name}")
        original_sha, copied_sha = sha256(original), sha256(copied)
        evidence["json_checks"][name] = {"parent_sha256": original_sha,
                                         "retained_sha256": copied_sha}
        if original_sha != copied_sha:
            raise ValueError(f"retained parent {name} differs from immutable parent")
        documents[name] = json.loads(copied.read_text())
        if not isinstance(documents[name], dict) or not finite_tree(documents[name]):
            raise ValueError(f"invalid retained parent {name}")
    parent_meta, parent_status = documents["metadata.json"], documents["status.json"]
    step = plan.get("parent_step")
    if not isinstance(step, int) or isinstance(step, bool) or step < 0 or step != parent_status.get("step"):
        raise ValueError("promotion parent_step disagrees with retained terminal status")
    if (parent_status.get("state") not in ("complete", "stopped") or
            parent_status.get("state") == "stopped" and parent_status.get("reason") != "budget"):
        raise ValueError("parent is not complete or cleanly budget-stopped")
    metadata = run["metadata"]
    if not metadata:
        raise ValueError("continuation metadata not available yet")
    if metadata.get("config") != parent_meta.get("config"):
        raise ValueError("continuation and parent configurations differ")
    for key in ("config_sha256", "manifest_sha256", "validation_manifest_sha256",
                "parameters", "trainable_parameters", "train_images", "validation_images"):
        if metadata.get(key) != parent_meta.get(key):
            raise ValueError(f"continuation and parent {key} differ")
    for key in ("config_sha256",):
        token = plan.get(key)
        if (not isinstance(token, str) or len(token) != 64 or
                any(character not in "0123456789abcdef" for character in token) or token != metadata.get(key)):
            raise ValueError(f"promotion plan and metadata {key} differ")
    source = metadata.get("environment", {}).get("source_sha256")
    if (not isinstance(source, str) or len(source) != 64 or
            any(character not in "0123456789abcdef" for character in source) or
            source != plan.get("source_sha256") or source != parent_meta.get("environment", {}).get("source_sha256")):
        raise ValueError("promotion plan, parent and child numerical source hashes differ")
    implementation, precision = implementation_config(metadata)
    if ((implementation, precision) != implementation_config(parent_meta) or
            implementation != plan.get("implementation") or precision != plan.get("metric_precision")):
        raise ValueError("promotion namespace or metric precision differs")
    if (metadata.get("initialization", {}).get("source_sha256") !=
            parent_meta.get("initialization", {}).get("source_sha256")):
        raise ValueError("historical initialization checkpoint hashes differ")
    resume = metadata.get("paths", {}).get("resume")
    if local_manifest_path(root, resume, recorded_root) != (path / "parent-last.pt").resolve():
        raise ValueError("child resume path does not identify its retained parent-last checkpoint")
    for key, files in (("parent_last_sha256", (parent / "last.pt", path / "parent-last.pt")),
                       ("parent_best_sha256", (parent / "best.pt",))):
        expected = plan.get(key)
        if (not isinstance(expected, str) or len(expected) != 64 or
                any(character not in "0123456789abcdef" for character in expected)):
            raise ValueError(f"missing or invalid promotion {key}")
        for artifact in files:
            artifact = snapshot_evidence_path(root, artifact)
            recorded = {"recorded_sha256": expected, "available": artifact.is_file()}
            if artifact.is_file():
                recorded["local_sha256"] = sha256(artifact)
                if recorded["local_sha256"] != expected:
                    raise ValueError(f"retained parent artifact hash differs: {artifact.name}")
            evidence["artifact_checks"][str(artifact)] = recorded
    # The child's best.pt may have improved and is deliberately not compared
    # with parent_best_sha256. The original parent's best remains immutable.
    decision = snapshot_evidence_path(root, path / "promotion-decision.txt")
    if decision.is_file():
        if not plan.get("decision_sha256") or sha256(decision) != plan["decision_sha256"]:
            raise ValueError("retained promotion decision hash differs")
        evidence["decision_sha256"] = plan["decision_sha256"]
    else:
        evidence["decision_sha256"] = "retained decision unavailable"
    evidence["promotion_json_sha256"] = sha256(path / "promotion.json")
    return parent, step, evidence


def load_run(path, root, ledger, recorded_root=RECORDED_PROJECT_ROOT, _seen=None):
    """Load one reporting row and, for a verified continuation, its ancestry."""
    path = path.resolve()
    seen = frozenset() if _seen is None else _seen
    if path in seen or len(seen) >= 32:
        raise ValueError("cyclic or overlong promotion ancestry")
    run = _load_run_local(path, root, ledger, recorded_root)
    own_validations = list(run["validations"])
    own_updates, own_seconds = run["updates"], run["lease_seconds_snapshot"]
    own_allowance = run["lease_allowance_seconds"]
    run.update(lineage_run_ids=[path.name], lineage_complete=True, lineage_evidence=None,
               continuation_start_step=0, continuation_updates=own_updates,
               continuation_images_seen=run["images_seen"], continuation_pixels_seen=run["pixels_seen"],
               continuation_lease_seconds_snapshot=own_seconds,
               continuation_lease_allowance_seconds=own_allowance,
               continuation_validation_count=len(own_validations),
               evaluation_call_count=len(own_validations), continuation_trend=None)
    promotion_path = path / "promotion.json"
    resumed = bool(run["metadata"].get("paths", {}).get("resume"))
    if promotion_path.exists() and not run["metadata"]:
        run["state"] = "prepared (not started in snapshot)"
    if promotion_path.exists() or resumed:
        try:
            if not promotion_path.is_file():
                raise ValueError("resumed run has no promotion.json linkage")
            promotion_path = snapshot_evidence_path(root, promotion_path)
            plan = read_json(promotion_path, run["issues"])
            if plan is None:
                raise ValueError("promotion.json invalid or unavailable")
            parent_path, parent_step, evidence = _parent_link(run, path, root, plan, recorded_root)
            if parent_path in seen | {path}:
                raise ValueError("cyclic promotion ancestry")
            parent = load_run(parent_path, root, ledger, recorded_root, seen | {path})
            if not parent["lineage_complete"]:
                raise ValueError("parent ancestry is incomplete or inconsistent")
            if any(value["step"] > parent_step for value in parent["validations"]):
                raise ValueError("parent validation exceeds its frozen terminal step")
            if any(value["step"] < parent_step for value in own_validations) or own_updates < parent_step:
                raise ValueError("child evidence predates its declared resume step")
            inherited = {value["step"]: value for value in parent["validations"]}
            duplicate_steps = sorted(set(inherited) & {value["step"] for value in own_validations})
            for value in own_validations:
                prior = inherited.get(value["step"])
                if prior is not None and abs(prior["mean_psnr"] - value["mean_psnr"]) > 1e-4:
                    run["issues"].append(f"parent/child validation at step {value['step']} differs by more than 1e-4dB")
                inherited[value["step"]] = value
            run["validations"] = [inherited[step] for step in sorted(inherited)]
            run["initial"] = inherited.get(0)
            run["best"] = max(run["validations"], key=lambda value: (value["mean_psnr"], -value["step"])) if inherited else None
            run["validation_count"] = len(inherited)
            run["evaluation_call_count"] += parent["evaluation_call_count"]
            run["lineage_run_ids"] = parent["lineage_run_ids"] + [path.name]
            run["continuation_start_step"] = parent_step
            run["continuation_updates"] = own_updates - parent_step
            batch, patch = run["config"].get("batch_size"), run["config"].get("patch_size")
            run["continuation_images_seen"] = run["continuation_updates"] * batch if batch else None
            run["continuation_pixels_seen"] = run["continuation_updates"] * batch * patch * patch if batch and patch else None
            run["continuation_trend"] = resource_trend(own_validations, parent_step)
            run["continuation_trend"]["provisional"] = run["status"].get("state") not in ("complete", "stopped")
            evidence["deduplicated_global_steps_child_overrides"] = duplicate_steps
            evidence["parent_issues"] = parent["issues"]
            evidence["checkpoint_scope"] = "SHA checks only where binary artifacts are copied; tensors are never loaded"
            run["lineage_evidence"] = evidence
            if run["initial"] is not None and run["implementation"] == "beltrami_operator":
                run["checks"]["warmstart"] = (
                    "inherited parent step0 mean within1e-4dB; child resumed from frozen parent-last"
                    if abs(run["initial"]["mean_psnr"] - INITIAL_PSNR) <= 1e-4 else
                    "inherited step0 differs from historical reference; inspect initialization")
            ids = run["lineage_run_ids"]
            entries = [entry for entry in ledger.get("entries", []) if entry.get("run_id") in ids]
            if (len(set(ids)) != len(ids) or any(sum(entry.get("run_id") == run_id for entry in entries) != 1 for run_id in ids) or
                    not all(nonnegative_number(entry.get("charged_seconds")) for entry in entries)):
                run["lease_seconds_snapshot"] = None
                run["lease_allowance_seconds"] = None
                run["issues"].append("cumulative lease unavailable: each unique ancestor and child needs one ledger entry with a finite nonnegative charge")
            else:
                run["lease_seconds_snapshot"] = sum(entry["charged_seconds"] for entry in entries)
                run["lease_allowance_seconds"] = sum(entry["allowance_seconds"] for entry in entries) if all(nonnegative_number(entry.get("allowance_seconds")) for entry in entries) else None
            run["lease_state"] = ",".join(entry.get("state", "unknown") for entry in entries)
            run["lease_snapshot_utc"] = max((entry.get("last_checked_utc", entry.get("ended_utc", entry.get("started_utc", ""))) for entry in entries), default=None)
        except (OSError, ValueError, KeyError, TypeError) as error:
            run["lineage_complete"] = False
            run["lineage_evidence"] = {"error": str(error)}
            run["issues"].append(f"promotion ancestry rejected/unavailable: {error}")
            # Cumulative steps cannot be divided by the child's lease alone.
            run["lease_seconds_snapshot"] = None
            run["lease_allowance_seconds"] = None
            run["continuation_start_step"] = None
            run["continuation_updates"] = None
            run["continuation_images_seen"] = None
            run["continuation_pixels_seen"] = None
    run["best_origin_run_id"] = run["best"].get("origin_run_id") if run["best"] else None
    return run


def comparison_kind(run):
    model, patch = run["model"], run["config"].get("patch_size")
    if model.get("reaction"):
        return "reaction vs frozen control", "frozen", patch
    if model.get("stages", 5) > 5 or model.get("width", 80) > 80:
        return "capacity vs same-metric control", model.get("metric"), patch
    if run.get("implementation", "beltrami_operator") == "beltrami_precision":
        return f"continued {model.get('metric')} precision control (self)", model.get("metric"), patch
    if model.get("metric") == "positive":
        return "positive vs frozen metric", "frozen", patch
    if patch == 100:
        return "patch100 vs patch160", "frozen", 160
    return "continued frozen control (self)", model.get("metric"), patch


def matched_control(run, runs):
    kind, metric, patch = comparison_kind(run)
    if not run.get("lineage_complete", True):
        return kind, None
    if "(self)" in kind:
        return kind, run
    config = run["config"]
    candidates = []
    for other in runs:
        if not other.get("lineage_complete", True):
            continue
        model, other_config = other["model"], other["config"]
        if (run.get("implementation", "beltrami_operator") != other.get("implementation", "beltrami_operator") or
                run.get("metric_precision", "float32") != other.get("metric_precision", "float32")):
            continue
        if (model.get("stages") != 5 or model.get("width") != 80 or model.get("reaction") or
                model.get("metric") != metric or other_config.get("patch_size") != patch):
            continue
        if any(run["model"].get(key, default) != model.get(key, default)
               for key, default in (("zero_dc", False), ("step_scale", 1000.0))):
            continue
        keys = ("sigma", "offset", "seed", "validation_seed", "batch_size", "learning_rate",
                "weight_decay", "gradient_clip", "deterministic")
        if any(config.get(key) != other_config.get(key) for key in keys):
            continue
        if any(run["metadata"].get(key) != other["metadata"].get(key)
               for key in ("manifest_sha256", "validation_manifest_sha256")):
            continue
        if run["checks"]["source_sha256"] != other["checks"]["source_sha256"]:
            continue
        if run["checks"]["initial_checkpoint_sha256"] != other["checks"]["initial_checkpoint_sha256"]:
            continue
        if other["best"] is not None:
            candidates.append(other)
    # Multiple matched seeds/runs must be resolved explicitly, not by their scores.
    return kind, candidates[0] if len(candidates) == 1 else None


def record_initial_comparison(run, control, repeats, seed):
    """Keep intentional precision changes separate from cohort warmstart checks."""
    initial = run["initial"]
    protocol = run["checks"]["primary_sigma_noise_protocol"]
    run["legacy_initial_reference_mean_db"] = INITIAL_PSNR if protocol else None
    run["initial_mean_delta_vs_legacy_db"] = (initial["mean_psnr"] - INITIAL_PSNR
                                               if initial is not None and protocol else None)
    run["delta_cohort_initial"] = {"available": False, "reason": "legacy cohort uses historical initial check"}
    run["initial_control_max_abs_image_delta_db"] = None
    if run["implementation"] != "beltrami_precision":
        return
    pair = paired(initial, control["initial"] if control else None, repeats, seed)
    run["delta_cohort_initial"] = pair
    declared = run["metadata"].get("initialization", {}).get("function_preserving")
    if not declared:
        run["checks"]["warmstart"] = "mathematical preservation not declared; inspect initialization"
    elif initial is None:
        run["checks"]["warmstart"] = "no step0 score; cohort warmstart not assessed"
    elif control is run:
        run["checks"]["warmstart"] = "precision base step0 recorded; mathematical preservation declared; historical byte/mean parity not required"
    elif not pair["available"]:
        run["checks"]["warmstart"] = "same-implementation/precision step0 control unavailable; historical byte/mean parity not required"
    else:
        maximum = max(abs(item["delta_db"]) for item in pair["per_image"])
        run["initial_control_max_abs_image_delta_db"] = maximum
        run["checks"]["warmstart"] = (
            "declared preserving; step0 per-image PSNR within1e-4dB of same-implementation/precision control"
            if maximum <= 1e-4 else "step0 score mismatch with same-implementation/precision control; inspect warmstart")


def fmt(value, digits=5):
    return "—" if value is None else f"{value:.{digits}f}"


def delta_text(value):
    if not value.get("available"):
        return "unavailable"
    return (f"{value['mean_delta_db']:+.5f} "
            f"[{value['ci95_low_db']:+.5f}, {value['ci95_high_db']:+.5f}]")


def report(root, output, repeats=20000, seed=20261112, plot=True, pattern="pilot-*",
           recorded_root=RECORDED_PROJECT_ROOT):
    output.mkdir(parents=True, exist_ok=True)
    issues = []
    ledger = read_json(root / "BUDGET.json", issues, {})
    runs = [load_run(path, root, ledger, recorded_root) for path in sorted((root / "runs").glob(pattern)) if path.is_dir()]
    for run in runs:
        run["delta_initial"] = paired(run["best"], run["initial"], repeats,
                                      comparison_seed(seed, run["run_id"], "initial"))
        kind, control = matched_control(run, runs)
        run["comparison"] = kind
        run["control_run_id"] = control["run_id"] if control else None
        record_initial_comparison(run, control, repeats, comparison_seed(seed, run["run_id"], "cohort_initial"))
        run["delta_control"] = paired(run["best"], control["best"] if control else None,
                                      repeats, comparison_seed(seed, run["run_id"], "control"))
        common_candidate, common_reference = latest_common_pair(run, control)
        run["delta_common_step"] = paired(common_candidate, common_reference, repeats,
                                          comparison_seed(seed, run["run_id"], "common_step"))
        run["common_step_candidate_psnr"] = common_candidate["mean_psnr"] if common_candidate else None
        run["common_step_control_psnr"] = common_reference["mean_psnr"] if common_reference else None
    rows = []
    for run in runs:
        model, config, best = run["model"], run["config"], run["best"]
        rows.append({"run_id": run["run_id"], "state": run["state"],
                     "implementation": run["implementation"], "metric_precision": run["metric_precision"],
                     "architecture": f"{model.get('stages', '?')}×{model.get('width', '?')}",
                     "metric": model.get("metric"), "reaction": bool(model.get("reaction")),
                     "patch": config.get("patch_size"), "parameters": run["metadata"].get("parameters"),
                     "expected_parameters": run["checks"]["expected_parameters"],
                     "trainable_parameters": run["metadata"].get("trainable_parameters"),
                     "seed": config.get("seed"), "validation_seed": config.get("validation_seed"),
                     "updates": run["updates"], "images_seen": run["images_seen"], "pixels_seen": run["pixels_seen"],
                     "lineage_complete": run["lineage_complete"],
                     "lineage_run_ids": " -> ".join(run["lineage_run_ids"]),
                     "continuation_start_step": run["continuation_start_step"],
                     "continuation_updates": run["continuation_updates"],
                     "continuation_images_seen": run["continuation_images_seen"],
                     "continuation_pixels_seen": run["continuation_pixels_seen"],
                     "lease_hours_snapshot": run["lease_seconds_snapshot"] / 3600 if run["lease_seconds_snapshot"] is not None else None,
                     "continuation_lease_hours_snapshot": run["continuation_lease_seconds_snapshot"] / 3600 if run["continuation_lease_seconds_snapshot"] is not None else None,
                     "continuation_updates_per_lease_second_snapshot": run["continuation_updates"] / run["continuation_lease_seconds_snapshot"] if run["continuation_lease_seconds_snapshot"] and run["continuation_updates"] is not None else None,
                     "continuation_pixels_per_lease_second_snapshot": run["continuation_pixels_seen"] / run["continuation_lease_seconds_snapshot"] if run["continuation_lease_seconds_snapshot"] and run["continuation_pixels_seen"] is not None else None,
                     "lease_allowance_seconds": run["lease_allowance_seconds"],
                     "lease_state": run["lease_state"], "lease_snapshot_utc": run["lease_snapshot_utc"],
                     "updates_per_lease_second_snapshot": run["updates"] / run["lease_seconds_snapshot"] if run["lease_seconds_snapshot"] else None,
                     "pixels_per_lease_second_snapshot": run["pixels_seen"] / run["lease_seconds_snapshot"] if run["lease_seconds_snapshot"] and run["pixels_seen"] is not None else None,
                     "inner_cumulative_seconds": run["inner_cumulative_seconds"],
                     "initial_psnr": run["initial"]["mean_psnr"] if run["initial"] else None,
                     "legacy_initial_reference_mean_db": run["legacy_initial_reference_mean_db"],
                     "initial_mean_delta_vs_legacy_db": run["initial_mean_delta_vs_legacy_db"],
                     "initial_control_mean_delta_db": run["delta_cohort_initial"].get("mean_delta_db"),
                     "initial_control_max_abs_image_delta_db": run["initial_control_max_abs_image_delta_db"],
                     "best_step": best["step"] if best else None, "best_psnr": best["mean_psnr"] if best else None,
                     "best_origin_run_id": run["best_origin_run_id"],
                     "evaluation_call_count": run["evaluation_call_count"],
                     "continuation_validation_count": run["continuation_validation_count"],
                     "resource_new_validation_count": (run["continuation_trend"] or {}).get("unique_new_validation_count"),
                     "resource_quarter_count": (run["continuation_trend"] or {}).get("quarter_count"),
                     "resource_last_minus_first_db": (run["continuation_trend"] or {}).get("last_minus_first_db"),
                     "resource_no_further_extension": (run["continuation_trend"] or {}).get("no_further_extension"),
                     "resource_trend_provisional": (run["continuation_trend"] or {}).get("provisional"),
                     "validation_checkpoints": run["validation_count"], "comparison": run["comparison"],
                     "control_run_id": run["control_run_id"],
                     "common_evaluated_step": run["delta_common_step"].get("candidate_step"),
                     "common_step_candidate_psnr": run["common_step_candidate_psnr"],
                     "common_step_control_psnr": run["common_step_control_psnr"],
                     **{f"delta_{kind}_{field}": run[f"delta_{kind}"].get(field)
                        for kind in ("initial", "control", "common_step")
                        for field in ("mean_delta_db", "ci95_low_db", "ci95_high_db")},
                     "finite_check": run["checks"]["finite"], "data_check": run["checks"]["data"],
                     "source_check": run["checks"]["source"], "warmstart_check": run["checks"]["warmstart"],
                     "issues": " | ".join(run["issues"])})
    fields = list(rows[0]) if rows else ["run_id", "state"]
    with (output / "trials.csv").open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fields)
        writer.writeheader()
        writer.writerows(rows)
    with (output / "paired-per-image.csv").open("w", newline="") as stream:
        fields = ("run_id", "comparison", "reference_run_id", "candidate_step", "reference_step",
                  "image", "pixels_sha256", "candidate_psnr", "reference_psnr", "delta_db")
        writer = csv.DictWriter(stream, fields)
        writer.writeheader()
        for run in runs:
            for kind in ("initial", "control", "common_step", "cohort_initial"):
                pair = run[f"delta_{kind}"]
                for image in pair.get("per_image", []):
                    writer.writerow({"run_id": run["run_id"], "comparison": kind,
                                     "reference_run_id": run["run_id"] if kind == "initial" else run["control_run_id"],
                                     "candidate_step": pair["candidate_step"], "reference_step": pair["reference_step"], **image})
    # JSON image scores use lists because tuple map keys cannot be serialized.
    serializable = []
    for run in runs:
        item = {key: value for key, value in run.items() if key not in ("initial", "best", "validations")}
        for key in ("initial", "best"):
            item[key] = {k: v for k, v in run[key].items() if k != "images"} if run[key] else None
        item["validation_files"] = [{k: v for k, v in value.items() if k != "images"} for value in run["validations"]]
        serializable.append(item)
    pending = ["pilot-" + name + "-001" for name in PILOTS
               if not (root / "runs" / ("pilot-" + name + "-001")).exists()] if pattern == "pilot-*" else []
    if pattern == "pilot-*":
        pending = sorted(set(pending) | {path.stem + "-001" for path in (root / "configs").glob("pilot-*.json")
                                        if not (root / "runs" / (path.stem + "-001")).exists()})
    summary = {"created_utc": datetime.now(timezone.utc).isoformat(), "label": "validation32-only", "run_glob": pattern,
               "caution": CAUTION, "bootstrap_seed_base": seed, "bootstrap_repeats": repeats,
               "recorded_project_root_mapping": str(recorded_root),
               "budget_snapshot_sha256": sha256(root / "BUDGET.json") if (root / "BUDGET.json").is_file() else None,
               "budget_consumed_seconds_snapshot": ledger.get("consumed_gpu_seconds"),
               "budget_authorized_seconds": ledger.get("authorized_gpu_seconds"),
               "issues": issues, "pending_expected_pilots": pending, "runs": serializable}
    plot_state = "disabled"
    plot_path = output / "validation-curves.png"
    if plot_path.exists():
        plot_path.unlink()
    if plot:
        plot_state = draw_curves(runs, plot_path)
    summary["plot"] = plot_state
    (output / "summary.json").write_text(json.dumps(summary, indent=2, allow_nan=False) + "\n")
    lines = ["# Validation32 experiment snapshot", "", CAUTION, "",
             "Compute is charged aggregate lease wall time at the recorded snapshot, including setup/evaluation. "
             "Inner cumulative time excludes initial setup. Image/pixel counts are training patch exposure, not distinct photographs. "
             "For verified promotions, updates, exposure, best scores and lease time include the unique parent ancestry; "
             "continuation-only figures are separate below. Historical pretraining before step0 is excluded. "
             "Compare recorded time, allowance, updates and pixels within the declared implementation/precision cohort. "
             "Live snapshot health is not a current process check.", "",
             "| Trial | Status | Operator / metric / patch | Parameters | Updates / images / pixels | Lease h | Best step / PSNR | Δ initial [95% CI] | Δ matched control [95% CI] |",
             "|---|---|---|---:|---|---:|---|---|---|"]
    for run, row in zip(runs, rows):
        lines.append(f"| {run['run_id']} | {run['state']} | {row['implementation']}: {row['architecture']} / {row['metric']}"
                     f" / metric-{row['metric_precision']}"
                     f"{' + reaction' if row['reaction'] else ''} / {row['patch']} | {row['parameters']} | "
                     f"{row['updates']} / {row['images_seen']} / {row['pixels_seen']} | {fmt(row['lease_hours_snapshot'], 4)} | "
                     f"{row['best_step']} / {fmt(row['best_psnr'])} | {delta_text(run['delta_initial'])} | {delta_text(run['delta_control'])} |")
    for name in pending:
        lines.append(f"| {name} | not started in snapshot | — | — | — | — | — | — | — |")
    lines += ["", f"Budget snapshot: {fmt(ledger.get('consumed_gpu_seconds'), 1)} / "
              f"{fmt(ledger.get('authorized_gpu_seconds'), 1)} seconds. Includes smoke/failure/evaluation leases beyond this table.",
              f"Plot: {plot_state}.", "", "## Latest common evaluated-step assessment", "",
              "This endpoint is the largest step evaluated by both candidate and matched control, selected "
              "without looking at scores. Batch size is matched; different patch sizes still imply unequal pixel exposure. "
              "Different architectures have unequal compute at the same update count. Throughput uses snapshot "
              "lease charges; a live ledger and status can be sampled a few seconds apart.", "",
              "| Trial | Matched control | Common updates | Candidate / control PSNR | Paired Δ [95% CI] | Updates / charged second |",
              "|---|---|---:|---|---|---:|"]
    for run, row in zip(runs, rows):
        lines.append(f"| {run['run_id']} | {run['control_run_id']} | {row['common_evaluated_step']} | "
                     f"{fmt(run['common_step_candidate_psnr'])} / {fmt(run['common_step_control_psnr'])} | "
                     f"{delta_text(run['delta_common_step'])} | {fmt(row['updates_per_lease_second_snapshot'], 2)} |")
    promotions = [(run, row) for run, row in zip(runs, rows)
                  if (Path(run["path"]) / "promotion.json").exists() or run["metadata"].get("paths", {}).get("resume")]
    if promotions:
        lines += ["", "## Continuation-only exposure and resource trend", "",
                  "Global updates and cumulative leases above include each verified ancestor once. "
                  "The table below counts only updates after the frozen parent-last step and only this child's lease. "
                  "Saved validation curves inherit parent step0 and earlier best; a child validation at the same global step "
                  "replaces that point in the curve, while the number of actual evaluations is retained separately. "
                  "Checkpoint tensors are never loaded; JSON linkage and available binary SHA checks have explicit scope in summary.json. "
                  "The validation/event files are recorded evidence with file hashes, not a cryptographic binding between checkpoint tensors and predictions.", "",
                  "| Continuation | Lineage verified | Parent step | New updates / images / pixels | Child lease h | Unique validations / actual evaluations | Best origin | Trend n / quarter | Last−first median dB | No further extension |",
                  "|---|---|---:|---|---:|---|---|---|---:|---|"]
        for run, row in promotions:
            trend = run["continuation_trend"] or {}
            extension = str(trend.get("no_further_extension")) + (" (provisional)" if trend.get("provisional") else "")
            lines.append(f"| {run['run_id']} | {run['lineage_complete']} | {run['continuation_start_step']} | "
                         f"{run['continuation_updates']} / {run['continuation_images_seen']} / {run['continuation_pixels_seen']} | "
                         f"{fmt(row['continuation_lease_hours_snapshot'], 4)} | {run['validation_count']} / {run['evaluation_call_count']} | "
                         f"{run['best_origin_run_id']} | {trend.get('unique_new_validation_count')} / {trend.get('quarter_count')} | "
                         f"{fmt(trend.get('last_minus_first_db'), 6)} | {extension} |")
        lines += ["", "Resource trend uses only unique child validation global steps strictly greater than parent_step. "
                  "Sort by global step, let q=max(1,n//4), and subtract the median PSNR of the first q from the last q. "
                  "Below0.002dB means no further extension under the predeclared compute-allocation rule; "
                  "fewer than4 new evaluations is unavailable. This is not a significance test or a mandate to extend. "
                  "A live snapshot trend is provisional; the extension criterion is assessed at the planned continuation's end. "
                  "Cumulative inner elapsed time is already restored by the trainer and is never summed across ancestors."]
    lines += ["", "## Evidence checks", "",
              "Logged finite losses/gradients and finite saved validation scores are evidence only for observed records. "
              "This report does not execute a numerical replay, inspect checkpoint tensors, verify remote processes, "
              "or independently redo the source-identity audit. A mean-parity check alone cannot establish pixel-level model parity.", ""]
    for run in runs:
        lines += [f"### {run['run_id']}", "",
                  f"- Comparison: {run['comparison']}; reference `{run['control_run_id']}` at its independently selected best.",
                  f"- Finite: {run['checks']['finite']}.", f"- Data: {run['checks']['data']}; local manifests: {run['checks']['local_manifests']}.",
                  f"- Image identities: {run['checks']['validation_image_identities']}; parameter count: {run['checks']['parameter_count']}.",
                  f"- Source: {run['checks']['source']}; SHA256 `{run['checks']['source_sha256']}`.",
                  f"- Implementation: `{run['implementation']}`; metric arithmetic `{run['metric_precision']}`; controls require both to match.",
                  f"- Warmstart: {run['checks']['warmstart']}; original SHA256 `{run['checks']['initial_checkpoint_sha256']}`.",
                  f"- Initial shift versus recorded legacy mean: {fmt(run['initial_mean_delta_vs_legacy_db'], 8)}dB "
                  "(arithmetic/protocol diagnostic, not a training gain or byte-parity requirement).",
                  f"- Step0 versus cohort control: {delta_text(run['delta_cohort_initial'])}; "
                  f"maximum absolute paired image-score shift {fmt(run['initial_control_max_abs_image_delta_db'], 8)}dB. "
                  "Saved scores cannot establish output-byte parity.",
                  f"- Recorded status UTC: {run['status'].get('utc')}; lease snapshot UTC: {run['lease_snapshot_utc']}.",
                  f"- Issues: {'; '.join(run['issues']) or 'none found in recorded evidence'}.", ""]
    lines += ["Bootstrap resamples the 32 paired image differences with replacement; percentile95% intervals, "
              f"{repeats:,} replicates, deterministic per-comparison seeds derived from base{seed}. "
              "Control matching requires the same numerical source, initial checkpoint, data manifests, training/noise seeds, "
              "implementation namespace, metric precision, sigma/offset, batch and optimizer settings. "
              "Deeper/wider use the same-metric control; reaction uses frozen; "
              "positive metric uses frozen; patch100 uses patch160. Ambiguous controls are left unavailable. "
              "The precision namespace's base5×80 is its cohort self-control; intentional arithmetic changes "
              "are reported against the recorded legacy initial mean without demanding historical byte parity. "
              "Different evaluation counts and best-of-many checkpoint selection can favor faster runs. "
              "Only final frozen-model evaluation on the reserved benchmark can assess generalization for this research round.", ""]
    (output / "REPORT.md").write_text("\n".join(lines))
    return summary


def draw_curves(runs, path):
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        return "skipped: Matplotlib unavailable; CSV/JSON/Markdown written"
    with plt.rc_context({"font.size": 9, "axes.spines.top": False, "axes.spines.right": False}):
        figure, axes = plt.subplots(1, 2, figsize=(12, 4.8), constrained_layout=True)
        for run in runs:
            values = [v for v in run["validations"] if v.get("gpu_seconds") is not None]
            if not values:
                continue
            model = run["model"]
            label = (f"{run['run_id'].removeprefix('pilot-').removesuffix('-001')} "
                     f"[{model.get('stages')}×{model.get('width')}, {model.get('metric')}, "
                     f"{'precision' if run['implementation'] == 'beltrami_precision' else run['implementation']}, "
                     f"metric-{run['metric_precision']}]")
            if run["state"].startswith("ongoing"):
                label += " (ongoing)"
            elif run["checks"]["failure_observed"]:
                label += f" (FAILED at update{run['updates']})"
            x = [v["gpu_seconds"] / 3600 for v in values]
            y = [v["mean_psnr"] for v in values]
            line, = axes[0].plot(x, y, linewidth=1.2, label=label)
            axes[1].plot([v["step"] for v in values], y, linewidth=1.2, color=line.get_color())
            best = max(values, key=lambda v: (v["mean_psnr"], -v["step"]))
            axes[0].plot(best["gpu_seconds"] / 3600, best["mean_psnr"], "o", ms=4, color=line.get_color())
            axes[1].plot(best["step"], best["mean_psnr"], "o", ms=4, color=line.get_color())
        for index, axis in enumerate(axes):
            axis.axhline(INITIAL_PSNR, linestyle="--", color=".4", linewidth=.8,
                         label="Recorded legacy initial" if index == 0 else None)
            axis.set_ylabel("Validation32 mean PSNR (dB)")
            axis.grid(alpha=.2)
        axes[0].set_xlabel("Recorded inner elapsed hours (startup excluded)")
        axes[1].set_xlabel("Completed optimizer updates")
        figure.legend(*axes[0].get_legend_handles_labels(), loc="upper center",
                      bbox_to_anchor=(.5, -.06), ncol=2, fontsize=7)
        figure.suptitle("Validation32 model-selection curves — primary fixed noise draw")
        figure.text(.5, -.035, "Dots: selected best observed checkpoints. Curves and intervals are not final BSD68 results.",
                    ha="center", fontsize=8)
        figure.savefig(path, dpi=200, bbox_inches="tight")
        plt.close(figure)
    return str(path)


def self_test():
    from copy import deepcopy
    scores = {(f"image{i}", f"sha{i}"): float(i) for i in range(32)}
    initial = {"step": 0, "images": scores}
    equal = {"step": 1, "images": dict(scores)}
    shifted = {"step": 2, "images": {key: value + .125 for key, value in scores.items()}}
    assert paired(equal, initial, 100, 7)["ci95_low_db"] == 0
    result = paired(shifted, initial, 100, 7)
    assert result["mean_delta_db"] == result["ci95_low_db"] == result["ci95_high_db"] == .125
    unequal = {"step": 2, "images": dict(list(scores.items())[:-1])}
    assert not paired(unequal, initial, 100, 7)["available"]
    assert paired(shifted, initial, 100, 7) == paired(shifted, initial, 100, 7)
    assert not finite_tree({"nested": [float("nan")]})
    with tempfile.TemporaryDirectory() as temporary:
        root = Path(temporary)
        path = root / "runs/pilot-finetune-frozen-001"
        path.mkdir(parents=True)
        metadata = {"config": {"model": {"stages": 5, "width": 80, "metric": "frozen"},
                               "batch_size": 2, "patch_size": 160, "sigma": 25, "offset": 25,
                               "seed": 20261112, "validation_seed": 742913},
                    "manifest_sha256": TRAIN_SHA, "validation_manifest_sha256": VALIDATION_SHA,
                    "train_images": 400, "validation_images": 32,
                    "initialization": {"function_preserving": True, "source_sha256": "original"},
                    "environment": {"source_sha256": "source"}}
        (path / "metadata.json").write_text(json.dumps(metadata))
        (path / "status.json").write_text(json.dumps({"state": "training", "step": 10, "gpu_seconds": 1}))
        (path / "events.jsonl").write_text(json.dumps({"event": "progress", "step": 10,
                                                       "loss": .1, "gradient_norm": 1, "images_seen": 20}) + "\n{partial")
        for step, delta in ((0, 0), (5, .125), (9, .125)):
            images = [{"image": key[0], "pixels_sha256": key[1], "psnr": value + delta}
                      for key, value in scores.items()]
            value = {"mean_psnr": statistics.fmean(item["psnr"] for item in images), "images": images,
                     "crop": 0, "clipping": "prediction only, float [0,255]"}
            (path / f"validation-{step:07d}.json").write_text(json.dumps(value))
        ledger = {"entries": [{"run_id": path.name, "charged_seconds": 2, "state": "running"}]}
        (root / "BUDGET.json").write_text(json.dumps(ledger))
        summary = report(root, root / "reports", 100, 7, False)
        run = summary["runs"][0]
        assert run["best"]["step"] == 5, "equal maxima choose earliest checkpoint"
        assert run["updates"] == 10 and run["images_seen"] == 20 and run["pixels_seen"] == 512000
        assert run["lease_seconds_snapshot"] == 2 and run["state"].startswith("ongoing")
        assert run["delta_initial"]["mean_delta_db"] == .125
        assert any("partial final line" in issue for issue in run["issues"])
        assert run["control_run_id"] == path.name and run["delta_control"]["mean_delta_db"] == 0
        assert run["delta_common_step"]["candidate_step"] == 9
        assert run["delta_common_step"]["mean_delta_db"] == 0
        frozen = load_run(path, root, ledger)
        assert frozen["implementation"] == "beltrami_operator" and frozen["metric_precision"] == "float32"
        positive = deepcopy(frozen)
        positive["run_id"] = "positive"
        positive["model"]["metric"] = "positive"
        deeper = deepcopy(positive)
        deeper["run_id"] = "deeper"
        deeper["model"]["stages"] = 8
        kind, control = matched_control(deeper, [frozen, positive, deeper])
        assert control is positive and kind == "capacity vs same-metric control"
        positive["config"]["seed"] = -1
        assert matched_control(deeper, [frozen, positive, deeper])[1] is None
        positive["config"]["seed"] = deeper["config"]["seed"]
        assert matched_control(deeper, [positive, deepcopy(positive)])[1] is None
        precision = deepcopy(positive)
        precision["run_id"] = "pilot-precision-control-positive-001"
        precision["implementation"] = "beltrami_precision"
        precision["metadata"]["implementation"] = "beltrami_precision"
        precision["metric_precision"] = "float64"
        precision["model"]["metric_precision"] = "float64"
        precision["initial"]["mean_psnr"] = INITIAL_PSNR + .01
        precision["initial"]["images"] = {key: INITIAL_PSNR + .01 + (index - 15.5) * .1
                                               for index, key in enumerate(scores)}
        precision["initial"]["mean_psnr"] = statistics.fmean(precision["initial"]["images"].values())
        precision_depth = deepcopy(precision)
        precision_depth["run_id"] = "arbitrary-name-without-architecture-hints"
        precision_depth["model"]["stages"] = 8
        kind, control = matched_control(precision_depth, [frozen, positive, precision, precision_depth])
        assert control is precision and kind == "capacity vs same-metric control"
        assert matched_control(precision, [positive, precision])[1] is precision
        assert matched_control(precision_depth, [positive, frozen])[1] is None, "namespaces cannot mix, even with identical source SHA"
        precision32 = deepcopy(precision)
        precision32["metric_precision"] = "float32"
        precision32["model"]["metric_precision"] = "float32"
        assert matched_control(precision_depth, [precision32])[1] is None, "metric precisions cannot mix"
        record_initial_comparison(precision_depth, precision, 100, 7)
        assert precision_depth["delta_cohort_initial"]["mean_delta_db"] == 0
        assert precision_depth["initial_control_max_abs_image_delta_db"] == 0
        assert "same-implementation/precision control" in precision_depth["checks"]["warmstart"]
        assert math.isclose(precision_depth["initial_mean_delta_vs_legacy_db"], .01, abs_tol=1e-12)
        # Namespace source verification never imports either numerical package.
        for namespace in IMPLEMENTATIONS:
            directory = root / "src" / namespace
            directory.mkdir(parents=True)
            (directory / "__init__.py").write_text(f'"""Synthetic {namespace} source-hash fixture."""\n')
        assert local_source_hash(root, "beltrami_operator") != local_source_hash(root, "beltrami_precision")
        assert local_source_hash(root, "../unsupported") is None
        fixture = deepcopy(metadata)
        fixture["implementation"] = "beltrami_precision"
        fixture["config"]["model"]["metric_precision"] = "float64"
        fixture["environment"]["source_sha256"] = local_source_hash(root, "beltrami_precision")
        fixture_path = root / "runs/pilot-precision-fixture-001"
        fixture_path.mkdir()
        (fixture_path / "metadata.json").write_text(json.dumps(fixture))
        (fixture_path / "status.json").write_text(json.dumps({"state": "failed", "step": 17,
                                                               "error": "Non-finite training prediction"}))
        (fixture_path / "events.jsonl").write_text("")
        fixture_run = load_run(fixture_path, root, {})
        assert fixture_run["checks"]["source"].startswith("local numerical source SHA matches")
        assert fixture_run["checks"]["finite"] == "FAIL/non-finite evidence"
        assert fixture_run["implementation"] == "beltrami_precision" and fixture_run["metric_precision"] == "float64"
        fixture["config"]["model"]["metric"] = "positive"
        fixture["numerics"] = {"metric_precision": "float64"}
        fixture["initialization"]["metric_precision"] = "float64"
        for stages, suffix in ((5, "control-positive"), (8, "deeper-positive")):
            pilot = root / "runs" / ("pilot-precision-" + suffix + "-001")
            pilot.mkdir()
            pilot_metadata = deepcopy(fixture)
            pilot_metadata["config"]["model"]["stages"] = stages
            pilot_metadata["parameters"] = stages * (98 * 80 * 80 + 149 * 80)
            (pilot / "metadata.json").write_text(json.dumps(pilot_metadata))
            (pilot / "status.json").write_text(json.dumps({"state": "stopped", "reason": "budget", "step": 10}))
            event_lines = []
            for step, delta in ((0, 0), (5, .125)):
                images = [{"image": key[0], "pixels_sha256": key[1], "psnr": value + delta}
                          for key, value in precision["initial"]["images"].items()]
                value = {"mean_psnr": statistics.fmean(item["psnr"] for item in images), "images": images,
                         "crop": 0, "clipping": "prediction only, float [0,255]"}
                (pilot / f"validation-{step:07d}.json").write_text(json.dumps(value))
                event_lines.append(json.dumps({"event": "validation", "step": step,
                                               "gpu_seconds": step + 1, "mean_psnr": value["mean_psnr"]}))
            (pilot / "events.jsonl").write_text("\n".join(event_lines) + "\n")
        integrated = report(root, root / "precision-report", 100, 7, False)
        integrated_depth = next(item for item in integrated["runs"] if item["run_id"] == "pilot-precision-deeper-positive-001")
        assert integrated_depth["control_run_id"] == "pilot-precision-control-positive-001"
        assert integrated_depth["delta_cohort_initial"]["mean_delta_db"] == 0
        assert integrated_depth["checks"]["parameter_count"] == "matches declared architecture"
        assert math.isclose(integrated_depth["initial_mean_delta_vs_legacy_db"], .01, abs_tol=1e-12)
        assert "beltrami_precision" in (root / "precision-report/trials.csv").read_text()
        different_steps = {"validations": [{"step": 0}, {"step": 5}, {"step": 8}]}
        other_steps = {"validations": [{"step": 0}, {"step": 5}, {"step": 10}]}
        a, b = latest_common_pair(different_steps, other_steps)
        assert a["step"] == b["step"] == 5
        assert (root / "reports/trials.csv").exists() and (root / "reports/paired-per-image.csv").exists()
        invalid = root / "bad.json"
        invalid.write_text('{"mean_psnr":NaN}')
        issues = []
        assert read_json(invalid, issues) is None and "non-finite" in issues[0]
    print("CPU self-tests passed: pairing/hash exclusion, bootstrap reproducibility, finite rejection, "
          "best tie selection, partial JSONL, ongoing status, exposure/lease accounting, "
          "namespace/precision cohorts, initial arithmetic shifts, output integration")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument("--output", type=Path, help="Default: ROOT/reports/validation32-snapshot")
    parser.add_argument("--run-glob", default="pilot-*", help="Run directory pattern; default pilot-*")
    parser.add_argument("--recorded-root", type=PurePosixPath, default=RECORDED_PROJECT_ROOT,
                        help="Explicit recorded project-root prefix to map into --root; defaults to this checkout; no basename guessing")
    parser.add_argument("--bootstrap", type=int, default=20000)
    parser.add_argument("--seed", type=int, default=20261112)
    parser.add_argument("--no-plot", action="store_true")
    parser.add_argument("--self-test", action="store_true")
    args = parser.parse_args()
    if args.self_test:
        self_test()
        return
    if args.bootstrap < 100:
        parser.error("--bootstrap must be at least100")
    if not args.recorded_root.is_absolute() or ".." in args.recorded_root.parts:
        parser.error("--recorded-root must be absolute without parent traversal")
    root = args.root.resolve()
    output = args.output or root / "reports/validation32-snapshot"
    summary = report(root, output, args.bootstrap, args.seed, not args.no_plot, args.run_glob,
                     args.recorded_root)
    print(f"Validation32-only snapshot: {len(summary['runs'])} run directories; "
          f"{len(summary['pending_expected_pilots'])} expected directories missing")
    print(f"Report: {output / 'REPORT.md'}")
    print(f"Plot: {summary['plot']}")
    for run in summary["runs"]:
        print(f"{run['run_id']}: {run['state']}; best={fmt(run['best']['mean_psnr'] if run['best'] else None)}dB; "
              f"Δ initial={delta_text(run['delta_initial'])}; Δ control={delta_text(run['delta_control'])}")
    print(CAUTION)


if __name__ == "__main__":
    main()

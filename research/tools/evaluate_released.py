#!/usr/bin/env python3
"""Evaluate public tensor-only finalists using the unchanged archived BSD68 inputs."""
import argparse
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
from beltrami_research import artifacts


_HELPER_PATH = Path(__file__).with_name("evaluate_finalists.py")
_SPEC = importlib.util.spec_from_file_location("beltrami_released_protocol", _HELPER_PATH)
protocol = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(protocol)
_DIGEST = re.compile(r"[0-9a-f]{64}\Z")
_METADATA_KEYS = {"format", "implementation", "source_sha256", "model_config", "sigma",
                  "grayscale_offset", "tensor_sha256", "parameters", "training_step", "provenance"}
_PROVENANCE_KEYS = {"training_checkpoint_sha256", "training_config_sha256", "training_manifest_sha256",
                    "validation_manifest_sha256", "original_checkpoint_sha256", "license", "citation"}


def require_digest(value, name):
    if not isinstance(value, str) or not _DIGEST.fullmatch(value):
        raise ValueError(f"Invalid SHA256: {name}")
    return value


def positive_integer(value, name, minimum=1):
    if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
        raise ValueError(f"Invalid integer: {name}")


def artifact_path(root, filename):
    """Manifest artifact names are portable relative paths contained in weights-dir."""
    if not isinstance(filename, str) or not filename or "\\" in filename:
        raise ValueError("Artifact filename must be a portable relative path")
    path = Path(filename)
    if path.is_absolute() or any(part in (".", "..") for part in path.parts):
        raise ValueError("Artifact filename must be a contained relative path")
    root = Path(root).resolve()
    result = (root / path).resolve()
    if not result.is_relative_to(root):
        raise ValueError("Artifact path escapes weights directory")
    return result


def validate_comparisons(manifest):
    baseline = manifest["baseline"]
    assets = manifest["assets"]
    names = [baseline["name"]] + [entry["model"] for entry in assets]
    if any(not isinstance(name, str) or not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.-]*", name)
           for name in names) or len(set(names)) != len(names):
        raise ValueError("Model names must be unique portable identifiers")
    if baseline.get("kind") != "historical" or baseline.get("role") != "baseline":
        raise ValueError("Exactly one original historical baseline is required")
    if (not assets or any(entry.get("role") not in ("candidate", "control") for entry in assets)
            or not any(entry["role"] == "candidate" for entry in assets)
            or not any(entry["role"] == "control" for entry in assets)):
        raise ValueError("Released assets require both candidate and continued-control roles")
    pairs = manifest["comparisons"]
    if not isinstance(pairs, list) or not pairs:
        raise ValueError("Explicit paired comparisons are required")
    seen = set()
    by_name = {entry["model"]: entry for entry in assets}
    for pair in pairs:
        if set(pair) != {"candidate", "reference"}:
            raise ValueError("Each comparison requires candidate and reference")
        key = pair["candidate"], pair["reference"]
        if (key[0] not in by_name or key[1] not in names or key[0] == key[1] or key in seen):
            raise ValueError("Invalid or duplicated paired comparison")
        seen.add(key)
    for entry in assets:
        name = entry["model"]
        if (name, baseline["name"]) not in seen:
            raise ValueError(f"Missing historical comparison: {name}")
        if entry["role"] != "candidate":
            continue
        controls = [by_name[reference] for candidate, reference in seen
                    if candidate == name and reference in by_name
                    and by_name[reference]["role"] == "control"]
        if not controls or not any(matched_control(entry, control) for control in controls):
            raise ValueError(f"Missing matched continued control: {name}")
        if any(not matched_control(entry, control) for control in controls):
            raise ValueError(f"Mismatched arithmetic/operator control: {name}")


def matched_control(candidate, control):
    if candidate["implementation"] != control["implementation"]:
        return False
    cfg, ref = candidate["model_config"], control["model_config"]
    if any(cfg.get(key) != ref.get(key) for key in set(cfg) | set(ref)
           if key not in ("reaction", "width", "stages")):
        return False
    if cfg.get("reaction"):
        return (not ref.get("reaction") and cfg["stages"] == ref["stages"]
                and cfg["width"] == ref["width"] and cfg.get("metric") == "frozen")
    if ref.get("reaction") or cfg.get("metric") != "positive":
        return False
    if cfg["stages"] > ref["stages"]:
        return (candidate["implementation"] == "beltrami_precision"
                and cfg["width"] == ref["width"] and ref["stages"] == 5)
    return cfg["stages"] == ref["stages"] and cfg["width"] >= ref["width"]


def read_manifest(path, expected_sha):
    require_digest(expected_sha, "manifest")
    if protocol.file_hash(path) != expected_sha:
        raise ValueError("Released manifest SHA256 differs from the explicit pin")
    def invalid_constant(value):
        raise ValueError(f"Non-finite JSON value: {value}")
    manifest = json.loads(Path(path).read_text(), parse_constant=invalid_constant)
    if manifest.get("schema") != 1 or manifest.get("purpose") != "bsd68":
        raise ValueError("Unsupported released-evaluation manifest schema/purpose")
    sigma = manifest["sigma"]
    if manifest["offset"] != protocol.archived_offset(sigma):
        raise ValueError("Manifest intensity protocol differs from the archived offset")
    for key in ("selection_sha256", "results_sha256", "evaluator_sha256"):
        require_digest(manifest[key], key)
    if manifest["evaluator_sha256"] != protocol.file_hash(_HELPER_PATH):
        raise ValueError("Frozen evaluation protocol helper changed")
    for key in ("timing_warmup", "timing_repeats", "bootstrap_samples"):
        positive_integer(manifest[key], key)
    positive_integer(manifest["bootstrap_seed"], "bootstrap_seed", 0)
    if manifest["timing_warmup"] < 2 or manifest["timing_repeats"] < 3:
        raise ValueError("Released timing requires at least two warmups and three repetitions")
    validate_comparisons(manifest)
    return manifest


def verify_asset(entry, path, sigma, offset, original_sha, device="cpu"):
    """Check all export metadata and bytes before opening any evaluation labels."""
    require_digest(entry["sha256"], "artifact")
    positive_integer(entry["bytes"], "artifact bytes")
    if path.stat().st_size != entry["bytes"] or protocol.file_hash(path) != entry["sha256"]:
        raise ValueError(f"Released artifact bytes differ: {entry['model']}")
    model, value = artifacts.load(path, device=device)
    if set(value) != _METADATA_KEYS | {"state_dict"}:
        raise ValueError("Artifact must contain only tensor state and supported export metadata")
    if value["format"] != "beltrami-operator-tensors-v1":
        raise ValueError("Unsupported released tensor format")
    for key in _METADATA_KEYS - {"format"}:
        if entry[key] != value[key]:
            raise ValueError(f"Manifest/export {key} differs: {entry['model']}")
    provenance = value["provenance"]
    if set(provenance) != _PROVENANCE_KEYS:
        raise ValueError("Unsupported export provenance fields")
    for key in _PROVENANCE_KEYS - {"license", "citation"}:
        require_digest(provenance[key], key)
    if (value["sigma"] != sigma or value["grayscale_offset"] != offset
            or provenance["original_checkpoint_sha256"] != original_sha):
        raise ValueError("Export sigma/offset/original initialization differs")
    if provenance["license"] != "MIT" or not isinstance(provenance["citation"], str):
        raise ValueError("Export license/citation is missing")
    positive_integer(value["training_step"], "training_step", 0)
    actual_parameters = sum(parameter.numel() for parameter in model.parameters())
    if value["parameters"] != actual_parameters:
        raise ValueError("Export parameter count differs from the strict loaded operator")
    if protocol.file_hash(path) != entry["sha256"]:
        raise ValueError("Released artifact changed during safe loading")
    return model


def preflight(manifest, weights_dir, model_dir):
    originals = protocol.original_files(model_dir, manifest["sigma"])
    expected = [{"name": Path(item["path"]).name, "sha256": item["sha256"]} for item in originals]
    baseline = manifest["baseline"]
    if (baseline["files"] != expected or baseline["parameters"] != 3195600
            or baseline["training_step"] != 40000):
        raise ValueError("Historical baseline differs from pinned original assets")
    paths = {}
    train, validation = set(), set()
    for entry in manifest["assets"]:
        path = artifact_path(weights_dir, entry["filename"])
        paths[entry["model"]] = path
        model = verify_asset(entry, path, manifest["sigma"], manifest["offset"], originals[0]["sha256"])
        del model
        train.add(entry["provenance"]["training_manifest_sha256"])
        validation.add(entry["provenance"]["validation_manifest_sha256"])
    if len(train) != 1 or len(validation) != 1:
        raise ValueError("Finalists must share the recorded training and selection manifests")
    return originals, paths


def evaluate(args):
    # Exclusive output preserves previous complete, partial and failed evidence.
    args.output.mkdir(parents=True, exist_ok=False)
    report_path, status_path = args.output / "results.json", args.output / "status.json"
    report = {"schema": 1, "complete": False, "started_utc": protocol.utc(),
              "manifest_sha256": args.manifest_sha256, "models": [], "comparisons": []}
    started = time.monotonic()
    def status(state, **extra):
        protocol.atomic_json(status_path, {"state": state, "utc": protocol.utc(),
            "pid": os.getpid(), "seconds": time.monotonic() - started, **extra})
    status("preflight")
    try:
        manifest = read_manifest(args.manifest, args.manifest_sha256)
        weights_dir = args.weights_dir or args.manifest.resolve().parent
        originals, paths = preflight(manifest, weights_dir, args.model_dir)
        report.update(manifest=manifest, evaluator_sha256=protocol.file_hash(__file__),
                      protocol_evaluator_sha256=protocol.file_hash(_HELPER_PATH), dataset="bsd68")
        with protocol.gpu_lock(args.gpu_lock):
            device = protocol.configure_device(args.device, args.threads, args.max_cuda_gib)
            report["environment"] = {"python": sys.version, "torch": torch.__version__,
                "numpy": np.__version__, "cuda": torch.version.cuda, "device": str(device),
                "device_name": torch.cuda.get_device_name(device) if device.type == "cuda" else "cpu",
                "deterministic_cudnn": True, "tf32": False,
                "max_cuda_gib": args.max_cuda_gib, "cpu_threads": args.threads}
            # The unchanged loader verifies all clean/noisy hashes before returning cases.
            cases, provenance = protocol.dataset_cases({"purpose": "bsd68", "sigma": manifest["sigma"]}, args)
            report.update(dataset_provenance=provenance, image_count=len(cases),
                          timing_scope="synchronized forward only; transfers and scoring excluded")
            entries = [manifest["baseline"]] + manifest["assets"]
            for entry in entries:
                historical = entry is manifest["baseline"]
                name = entry["name"] if historical else entry["model"]
                status("loading_model", model=name)
                if device.type == "cuda":
                    torch.cuda.reset_peak_memory_stats(device)
                if historical:
                    model = protocol.HistoricalOperator({"files": originals}, device)
                else:
                    model = verify_asset(entry, paths[name], manifest["sigma"], manifest["offset"],
                                         originals[0]["sha256"], device)
                result = {"name": name, "role": entry["role"], "parameters": entry["parameters"],
                          "images": [], "complete": False}
                report["models"].append(result)
                evaluate_model(model, result, cases, manifest, device, status, report_path, report)
                if device.type == "cuda":
                    result.update(peak_cuda_allocated_bytes=torch.cuda.max_memory_allocated(device),
                                  peak_cuda_reserved_bytes=torch.cuda.max_memory_reserved(device))
                del model
                if device.type == "cuda":
                    torch.cuda.empty_cache()
                protocol.atomic_json(report_path, report)
            by_name = {item["name"]: item for item in report["models"]}
            for pair in manifest["comparisons"]:
                report["comparisons"].append({**pair, **protocol.paired_summary(
                    by_name[pair["candidate"]]["images"], by_name[pair["reference"]]["images"],
                    manifest["bootstrap_seed"], manifest["bootstrap_samples"])})
            if protocol.file_hash(args.manifest) != args.manifest_sha256:
                raise ValueError("Manifest changed during evaluation")
            protocol.verify_files(originals)
            for entry in manifest["assets"]:
                if protocol.file_hash(paths[entry["model"]]) != entry["sha256"]:
                    raise ValueError("Artifact changed during evaluation")
            report.update(complete=True, ended_utc=protocol.utc(), seconds_total=time.monotonic() - started)
            protocol.atomic_json(report_path, report)
            status("complete", models=len(report["models"]), images=len(cases))
    except BaseException as error:
        report.update(failure={"type": type(error).__name__, "message": str(error)}, ended_utc=protocol.utc())
        protocol.atomic_json(report_path, report)
        status("failed", error=report["failure"])
        raise
    return report


def evaluate_model(model, result, cases, manifest, device, status, report_path, report):
    offset = manifest["offset"]
    with torch.inference_mode():
        tensor = torch.from_numpy(cases[0]["noisy"]).to(device) + offset
        for _ in range(manifest["timing_warmup"]):
            prediction = model(tensor)
            if not torch.isfinite(prediction).all():
                raise FloatingPointError(f"Non-finite warmup prediction: {result['name']}")
        protocol.sync(device)
        del tensor, prediction
        for index, case in enumerate(cases):
            status("evaluating", model=result["name"], image=case["name"], image_index=index, images=len(cases))
            tensor = torch.from_numpy(case["noisy"]).to(device) + offset
            times = []
            for _ in range(manifest["timing_repeats"]):
                protocol.sync(device)
                began = time.perf_counter()
                prediction = model(tensor)
                protocol.sync(device)
                times.append(time.perf_counter() - began)
                if not torch.isfinite(prediction).all():
                    raise FloatingPointError(f"Non-finite prediction: {result['name']}/{case['name']}")
            score = protocol.score_prediction(case["clean"], prediction.cpu().numpy(), offset)
            if score["psnr"] is None or not math.isfinite(score["psnr"]):
                raise FloatingPointError("Perfect or non-finite PSNR needs explicit inspection")
            item = {"name": case["name"], "shape_native": list(case["noisy"].shape),
                    "noisy_native_sha256": case["noisy_native_sha256"],
                    "label_native_sha256": case["label_native_sha256"], "seconds_forward": times,
                    "median_seconds_forward": float(np.median(times)), **score}
            if "archived_psnr" in case:
                item["archived_historical_psnr"] = case["archived_psnr"]
            result["images"].append(item)
            protocol.atomic_json(report_path, report)
            print(json.dumps({"model": result["name"], "image": case["name"],
                              "psnr": score["psnr"], "seconds_forward": times}), flush=True)
            del tensor, prediction
    result.update(complete=True, mean_psnr=float(np.mean([item["psnr"] for item in result["images"]])),
        median_seconds_forward=float(np.median([item["median_seconds_forward"] for item in result["images"]])),
        sum_seconds_forward=float(sum(sum(item["seconds_forward"]) for item in result["images"])))


def parser():
    result = argparse.ArgumentParser(description=__doc__)
    result.add_argument("--manifest", type=Path, required=True, help="Sanitized public models.json")
    result.add_argument("--manifest-sha256", required=True, help="Independent published SHA256 pin")
    result.add_argument("--weights-dir", type=Path, help="Existing local release tensors; defaults to manifest parent")
    result.add_argument("--model-dir", type=Path, required=True, help="Pinned original Caffe models directory")
    result.add_argument("--asset-dir", type=Path, required=True, help="Existing original noise release asset directory")
    result.add_argument("--clean-dir", type=Path, required=True, help="Existing pinned public BSD68 images")
    result.add_argument("--cache-dir", type=Path)
    result.add_argument("--output", type=Path, required=True, help="New evidence directory; never overwritten")
    result.add_argument("--gpu-lock", type=Path, required=True, help="Shared lock, including for CPU runs")
    result.add_argument("--device", default="cpu")
    result.add_argument("--threads", type=int, default=4)
    result.add_argument("--max-cuda-gib", type=float, default=10)
    return result


def main():
    args = parser().parse_args()
    if args.threads < 1 or not math.isfinite(args.max_cuda_gib) or not 0 < args.max_cuda_gib <= 10:
        raise ValueError("Use positive CPU threads and a CUDA cap in (0,10] GiB")
    evaluate(args)


if __name__ == "__main__":
    main()

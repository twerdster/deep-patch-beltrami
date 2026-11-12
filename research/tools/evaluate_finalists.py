#!/usr/bin/env python3
"""Freeze model selection, then evaluate shared inputs without test-time fitting."""
import argparse
from contextlib import contextmanager
from datetime import datetime, timezone
import fcntl
import hashlib
import importlib
import json
import math
import os
from pathlib import Path
import socket
import sys
import time

import numpy as np
import torch


def utc():
    return datetime.now(timezone.utc).isoformat()


def file_hash(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def array_hash(array):
    return hashlib.sha256(np.ascontiguousarray(array).tobytes()).hexdigest()


def atomic_json(path, value):
    path = Path(path)
    temporary = path.with_name(path.name + ".partial")
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")
    temporary.replace(path)


IMPLEMENTATIONS = ("beltrami_operator", "beltrami_precision")


def implementation_module(implementation):
    if implementation not in IMPLEMENTATIONS:
        raise ValueError(f"Unknown operator implementation: {implementation}")
    return importlib.import_module(implementation)


def model_classes(implementation):
    implementation_module(implementation)
    module = importlib.import_module(implementation + ".model")
    return module.BeltramiOperator, module.ModelConfig


def operator_source_hash(implementation="beltrami_operator"):
    package = implementation_module(implementation)
    digest = hashlib.sha256()
    for path in sorted(Path(package.__file__).parent.glob("*.py")):
        digest.update(path.name.encode())
        digest.update(path.read_bytes())
    return digest.hexdigest()


def archived_offset(sigma):
    factors = {5: 3, 10: 2, 15: 1, 20: 1, 25: 1, 30: 1, 40: 1, 50: .5}
    if sigma not in factors:
        raise ValueError(f"No original checkpoint/intensity protocol for sigma {sigma}")
    return sigma * factors[sigma]


def resolved(base, value):
    path = Path(value)
    return (base / path).resolve() if not path.is_absolute() else path.resolve()


def original_files(model_dir, sigma):
    from deep_patch_beltrami import assets
    directory = Path(model_dir)
    if (directory / "PBSolve_multi").is_dir():
        directory = directory / "PBSolve_multi"
    names = [f"PB_{sigma}_80_5__iter_40000.caffemodel"]
    names.extend(f"stage_{stage:03d}_deploy.prototxt" for stage in range(1, 6))
    pinned = {item["name"]: item["sha256"] for item in assets.metadata("models.json")["files"]}
    records = []
    for name in names:
        path = (directory / name).resolve()
        actual = file_hash(path)
        if pinned.get(name) != actual:
            raise ValueError(f"Original model file does not match public release: {name}")
        records.append({"path": str(path), "sha256": actual})
    return records


def verify_files(records):
    for record in records:
        if file_hash(record["path"]) != record["sha256"]:
            raise ValueError(f"Frozen artifact changed: {record['path']}")


def load_checkpoint(path):
    # Checkpoints are trusted, locally created training artifacts, never uploads.
    checkpoint = torch.load(path, map_location="cpu", weights_only=False)
    required = {"model", "config", "config_sha256", "source_sha256", "step", "gpu_seconds",
                "manifest_sha256", "validation_manifest_sha256", "initialization"}
    if required - checkpoint.keys():
        raise ValueError(f"Incomplete training checkpoint: {sorted(required - checkpoint.keys())}")
    if not checkpoint["model"] or any(
            not torch.isfinite(value).all() for value in checkpoint["model"].values()):
        raise FloatingPointError("Checkpoint contains empty or non-finite model tensors")
    return checkpoint


def validate_checkpoint_identity(checkpoint, config, config_hash, sigma, offset,
                                 source_hash, original_sha, train_sha, validation_sha):
    if checkpoint["config_sha256"] != config_hash or checkpoint["config"] != config:
        raise ValueError("Checkpoint/config mismatch")
    if config["sigma"] != sigma or config["offset"] != offset:
        raise ValueError("All finalists must share sigma and archived offset")
    if checkpoint["source_sha256"] != source_hash:
        raise ValueError("Training source differs; investigate before freezing")
    if (checkpoint["initialization"]["source_sha256"] != original_sha
            or checkpoint["initialization"]["source_sigma"] != sigma):
        raise ValueError("Checkpoint original initialization differs")
    if (checkpoint["manifest_sha256"] != train_sha
            or checkpoint["validation_manifest_sha256"] != validation_sha):
        raise ValueError("Checkpoint training/validation manifest identity differs")


def write_frozen_selection(path, value):
    """Exclusive creation and a separate digest make accidental edits detectable."""
    path = Path(path)
    receipt = path.with_name(path.name + ".sha256")
    if path.exists() or receipt.exists():
        raise FileExistsError("Selection already exists; preserve the original frozen decision")
    path.parent.mkdir(parents=True, exist_ok=True)
    data = (json.dumps(value, indent=2, allow_nan=False) + "\n").encode()
    with path.open("xb") as stream:
        stream.write(data)
        stream.flush()
        os.fsync(stream.fileno())
    with receipt.open("x") as stream:
        stream.write(hashlib.sha256(data).hexdigest() + "\n")


def read_frozen_selection(path):
    path = Path(path)
    digest = file_hash(path)
    if path.with_name(path.name + ".sha256").read_text().strip() != digest:
        raise ValueError("Selection digest changed after freezing")
    selection = json.loads(path.read_text())
    if selection.get("schema") != 1 or selection.get("purpose") not in ("bsd68", "validation32"):
        raise ValueError("Unsupported selection schema/purpose")
    created = datetime.fromisoformat(selection["created_utc"])
    if created.tzinfo is None or created > datetime.now(timezone.utc):
        raise ValueError("Selection timestamp must precede evaluation")
    if selection["offset"] != archived_offset(selection["sigma"]):
        raise ValueError("Intensity protocol differs from the archived fixed protocol")
    return selection, digest


def freeze(args):
    spec = json.loads(args.spec.read_text())
    base = args.spec.resolve().parent
    if not spec.get("reason", "").strip():
        raise ValueError("Record the validation-only selection/stopping reason")
    sigma = spec["sigma"]
    offset = archived_offset(sigma)
    implementation_hashes = {}
    originals = original_files(args.model_dir, sigma)
    train_sha = file_hash(args.training_manifest)
    validation_sha = file_hash(args.validation_manifest)
    models, seen = [], set()
    for item in spec["models"]:
        name, role = item["name"], item["role"]
        if not name or name in seen or role not in ("baseline", "control", "candidate"):
            raise ValueError("Model names must be unique and roles baseline/control/candidate")
        seen.add(name)
        entry = {"name": name, "role": role, "kind": item.get("kind", "checkpoint")}
        if entry["kind"] == "historical":
            if role != "baseline":
                raise ValueError("The historical checkpoint must have baseline role")
            entry.update(files=originals, parameters=3195600, training_step=40000)
        elif entry["kind"] == "checkpoint":
            if role == "baseline":
                raise ValueError("Use the original released graph as the historical baseline")
            implementation = item.get("implementation", "beltrami_operator")
            BeltramiOperator, ModelConfig = model_classes(implementation)
            if implementation not in implementation_hashes:
                implementation_hashes[implementation] = operator_source_hash(implementation)
            source_hash = implementation_hashes[implementation]
            checkpoint_path = resolved(base, item["checkpoint"])
            config_path = resolved(base, item["config"])
            checkpoint_sha = file_hash(checkpoint_path)
            checkpoint = load_checkpoint(checkpoint_path)
            config = json.loads(config_path.read_text())
            validate_checkpoint_identity(checkpoint, config, file_hash(config_path), sigma, offset,
                                         source_hash, originals[0]["sha256"], train_sha, validation_sha)
            model = BeltramiOperator(ModelConfig(**config["model"]))
            model.load_state_dict(checkpoint["model"], strict=True)
            if file_hash(checkpoint_path) != checkpoint_sha:
                raise ValueError(f"Checkpoint changed while freezing: {name}; stop the writer first")
            entry.update(implementation=implementation,
                         checkpoint=str(checkpoint_path), checkpoint_sha256=checkpoint_sha,
                         config=str(config_path), config_sha256=file_hash(config_path),
                         model_config=config["model"], parameters=sum(p.numel() for p in model.parameters()),
                         training_step=checkpoint["step"], training_gpu_seconds=checkpoint["gpu_seconds"],
                         training_manifest_sha256=checkpoint["manifest_sha256"],
                         validation_manifest_sha256=checkpoint["validation_manifest_sha256"],
                         training_source_sha256=checkpoint["source_sha256"],
                         initialization=checkpoint["initialization"])
            del model, checkpoint
        else:
            raise ValueError(f"Unsupported model kind: {entry['kind']}")
        models.append(entry)
    roles = [item["role"] for item in models]
    if roles.count("baseline") != 1 or "control" not in roles or "candidate" not in roles:
        raise ValueError("Freeze one historical baseline, a continued control and a candidate")
    controls = [item for item in models if item["role"] == "control"]
    baseline = next(item["name"] for item in models if item["role"] == "baseline")
    comparisons = spec.get("comparisons") or [
        {"candidate": item["name"], "reference": baseline}
        for item in models if item["role"] != "baseline"] + [
        {"candidate": item["name"], "reference": control["name"]}
        for item in models if item["role"] == "candidate" for control in controls]
    for comparison in comparisons:
        if (comparison["candidate"] not in seen or comparison["reference"] not in seen
                or comparison["candidate"] == comparison["reference"]):
            raise ValueError("Invalid predeclared paired comparison")
    selection = {"schema": 1, "created_utc": utc(), "purpose": args.purpose,
                 "reason": spec["reason"], "selection_data": "validation32 only; no BSD68 model selection",
                 "sigma": sigma, "offset": offset, "models": models, "comparisons": comparisons,
                 "training_manifest": str(args.training_manifest.resolve()),
                 "training_manifest_sha256": train_sha,
                 "validation_manifest": str(args.validation_manifest.resolve()),
                 "validation_manifest_sha256": validation_sha,
                 "implementation_source_sha256": implementation_hashes,
                 "evaluator_sha256": file_hash(__file__),
                 "timing_warmup": args.warmup, "timing_repeats": args.timing_repeats,
                 "bootstrap_seed": args.bootstrap_seed, "bootstrap_samples": args.bootstrap_samples,
                 "protocol": {"input_axes": "N,C,W,H", "input_scale": "0-255 float32, unclipped",
                              "output": "subtract fixed offset; reject nonfinite; clip float [0,255]",
                              "mse_dtype": "float64", "peak": 255, "crop": 0,
                              "augmentation": "none", "aggregation": "mean per-image PSNR in dB"}}
    if args.purpose == "validation32":
        if args.validation_seed is None:
            raise ValueError("Validation purpose requires a fixed manifest and noise seed")
        selection.update(validation_seed=args.validation_seed)
    write_frozen_selection(args.selection, selection)
    print(json.dumps({"selection": str(args.selection), "sha256": file_hash(args.selection),
                      "models": [item["name"] for item in models], "purpose": args.purpose}))


@contextmanager
def gpu_lock(path):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a+") as stream:
        try:
            fcntl.flock(stream, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as error:
            raise RuntimeError("Shared GPU lock is held; refusing a concurrent evaluation") from error
        yield


def configure_device(name, threads, max_cuda_gib):
    torch.set_num_threads(threads)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    device = torch.device(name)
    if device.type == "cuda":
        if device.index is None:
            device = torch.device("cuda", torch.cuda.current_device())
        total = torch.cuda.get_device_properties(device).total_memory
        torch.cuda.set_per_process_memory_fraction(min(.9, max_cuda_gib * 1024 ** 3 / total), device)
    elif device.type != "cpu":
        raise ValueError("Use CPU or CUDA so numerical/runtime behavior is documented")
    return device


def sync(device):
    if device.type == "cuda":
        torch.cuda.synchronize(device)


class HistoricalOperator:
    """Unchanged released Caffe graphs, with clipping left to the common evaluator."""
    def __init__(self, entry, device):
        from deep_patch_beltrami.caffe_model import load_weights
        from deep_patch_beltrami.inference import CaffeGraph
        verify_files(entry["files"])
        weights = load_weights(entry["files"][0]["path"])
        self.graphs = [CaffeGraph(record["path"], weights, device) for record in entry["files"][1:]]

    def __call__(self, tensor):
        for graph in self.graphs:
            tensor = graph(tensor)
        return tensor


def make_model(entry, device):
    if entry["kind"] == "historical":
        return HistoricalOperator(entry, device)
    BeltramiOperator, ModelConfig = model_classes(entry.get("implementation", "beltrami_operator"))
    verify_files([{"path": entry["checkpoint"], "sha256": entry["checkpoint_sha256"]},
                  {"path": entry["config"], "sha256": entry["config_sha256"]}])
    checkpoint = load_checkpoint(entry["checkpoint"])
    model = BeltramiOperator(ModelConfig(**entry["model_config"]))
    model.load_state_dict(checkpoint["model"], strict=True)
    return model.to(device).eval()


def verify_implementation_sources(selection):
    checkpoint_entries = [entry for entry in selection["models"] if entry["kind"] == "checkpoint"]
    used = {entry.get("implementation", "beltrami_operator") for entry in checkpoint_entries}
    if "implementation_source_sha256" in selection:
        hashes = selection["implementation_source_sha256"]
        if set(hashes) != used:
            raise ValueError("Frozen implementation source map differs from actual finalist implementations")
    else:
        # Old manifests contain only the original namespace and one exact source hash.
        if used != {"beltrami_operator"}:
            raise ValueError("Legacy selection cannot declare a different implementation")
        hashes = {"beltrami_operator": selection["operator_source_sha256"]}
    for implementation, expected in hashes.items():
        if operator_source_hash(implementation) != expected:
            raise ValueError(f"Operator source changed after freezing: {implementation}")
    for entry in checkpoint_entries:
        expected = hashes[entry.get("implementation", "beltrami_operator")]
        if entry["training_source_sha256"] != expected:
            raise ValueError(f"Checkpoint source differs from its frozen implementation: {entry['name']}")


def verify_artifacts(selection):
    """Reject mutated live checkpoints before opening any evaluation labels."""
    verify_files([{"path": selection[f"{split}_manifest"],
                   "sha256": selection[f"{split}_manifest_sha256"]}
                  for split in ("training", "validation")])
    for entry in selection["models"]:
        if entry["kind"] == "historical":
            verify_files(entry["files"])
        else:
            verify_files([{"path": entry["checkpoint"], "sha256": entry["checkpoint_sha256"]},
                          {"path": entry["config"], "sha256": entry["config_sha256"]}])


def score_prediction(clean, raw_prediction, offset):
    raw_prediction = np.asarray(raw_prediction)
    if raw_prediction.shape != clean.shape:
        raise ValueError(f"Prediction shape differs: {raw_prediction.shape} versus {clean.shape}")
    if not np.isfinite(raw_prediction).all():
        raise FloatingPointError("Non-finite prediction before output clipping")
    prediction = np.clip(raw_prediction - offset, 0, 255)
    error = prediction.astype(np.float64) - np.asarray(clean, dtype=np.float64)
    mse = float(np.mean(error * error, dtype=np.float64))
    # Exact predictions are represented explicitly, never hidden by an MSE floor.
    psnr = None if mse == 0 else float(10 * math.log10(255 ** 2 / mse))
    return {"mse": mse, "psnr": psnr, "perfect_prediction": mse == 0,
            "clipped_pixels": int(np.count_nonzero((raw_prediction - offset < 0) |
                                                     (raw_prediction - offset > 255)))}


def paired_summary(candidate, reference, seed, samples):
    if len(candidate) != len(reference) or len(candidate) < 2:
        raise ValueError("Paired statistics require matching nonempty image sets")
    if [item["name"] for item in candidate] != [item["name"] for item in reference]:
        raise ValueError("Paired image order/identity differs")
    if any(item["noisy_native_sha256"] != other["noisy_native_sha256"]
           for item, other in zip(candidate, reference)):
        raise ValueError("Paired noisy inputs differ")
    delta = np.array([item["psnr"] - other["psnr"] for item, other in zip(candidate, reference)],
                     dtype=np.float64)
    if not np.isfinite(delta).all():
        raise FloatingPointError("Cannot bootstrap non-finite PSNR changes")
    rng = np.random.default_rng(seed)
    means = []
    for start in range(0, samples, 1000):
        draws = rng.integers(len(delta), size=(min(1000, samples - start), len(delta)))
        means.extend(delta[draws].mean(axis=1))
    lower, upper = np.quantile(means, [.025, .975])
    return {"images": len(delta), "mean_delta_db": float(delta.mean()),
            "median_delta_db": float(np.median(delta)), "minimum_delta_db": float(delta.min()),
            "maximum_delta_db": float(delta.max()), "improved_images": int((delta > 0).sum()),
            "tied_images": int((delta == 0).sum()), "worse_images": int((delta < 0).sum()),
            "bootstrap_mean_95_percentile_ci_db": [float(lower), float(upper)],
            "bootstrap_seed": seed, "bootstrap_samples": samples,
            "uncertainty_scope": "paired images in this fixed noise realization; excludes training/selection uncertainty",
            "per_image_delta_db": [{"name": item["name"], "delta_db": float(change)}
                                   for item, change in zip(candidate, delta)]}


def dataset_cases(selection, args):
    sigma = selection["sigma"]
    if selection["purpose"] == "bsd68":
        from deep_patch_beltrami import assets
        from deep_patch_beltrami.reproduce import PublicBSD68
        # Explicit local roots prevent unaccounted downloads or a changed dataset.
        if args.clean_dir is None or args.asset_dir is None:
            raise ValueError("BSD68 requires explicit existing --clean-dir and --asset-dir")
        suite = PublicBSD68(cache_dir=args.cache_dir, asset_dir=args.asset_dir, clean_dir=args.clean_dir)
        try:
            if len(suite.names) != 68:
                raise ValueError("Final evaluation requires all 68 pinned images")
            cases = []
            for name in suite.names:
                clean, noisy, archived, digest = suite.case(sigma, name)
                cases.append({"name": name, "clean": clean, "noisy": noisy,
                              "noisy_native_sha256": digest, "label_native_sha256": array_hash(clean),
                              "archived_psnr": archived})
            provenance = {"loader": "deep_patch_beltrami.reproduce.PublicBSD68",
                          "public_bsd68_manifest_sha256": hashlib.sha256(
                              json.dumps(assets.metadata("bsd68.json"), sort_keys=True).encode()).hexdigest(),
                          "noise_asset_sha256": assets.metadata("release-assets.json")["assets"]["bsd68_noise"]["sha256"]}
            return cases, provenance
        finally:
            suite.close()
    implementation = next(entry.get("implementation", "beltrami_operator")
                          for entry in selection["models"] if entry["kind"] == "checkpoint")
    data_module = importlib.import_module(implementation + ".data")
    manifest = selection["validation_manifest"]
    if file_hash(manifest) != selection["validation_manifest_sha256"]:
        raise ValueError("Frozen validation manifest changed")
    if args.validation_root is None:
        raise ValueError("Validation evaluation requires --validation-root")
    images, records, _ = data_module.load_images(args.validation_root, manifest, "validation")
    if len(images) != 32:
        raise ValueError("Expected all 32 source-disjoint validation images")
    cases = []
    for noisy, clean, record in data_module.validation_pairs(images, records, sigma, selection["validation_seed"], 0):
        noisy = noisy.numpy()
        cases.append({"name": record["path"], "clean": clean, "noisy": noisy,
                      "noisy_native_sha256": array_hash(noisy), "label_native_sha256": array_hash(clean)})
    return cases, {"loader": implementation + ".data.validation_pairs",
                   "validation_manifest_sha256": selection["validation_manifest_sha256"],
                   "validation_seed": selection["validation_seed"]}


def evaluate(args):
    selection, selection_hash = read_frozen_selection(args.selection)
    verify_implementation_sources(selection)
    if selection["evaluator_sha256"] != file_hash(__file__):
        raise ValueError("Evaluator source changed after freezing")
    verify_artifacts(selection)
    # Create the evidence directory exclusively; never overwrite partial/failure results.
    args.output.mkdir(parents=True, exist_ok=False)
    report_path, status_path = args.output / "results.json", args.output / "status.json"
    report = {"schema": 1, "started_utc": utc(), "complete": False,
              "selection_sha256": selection_hash, "selection": selection,
              "models": [], "comparisons": []}
    started = time.monotonic()
    def status(state, **extra):
        atomic_json(status_path, {"state": state, "utc": utc(), "pid": os.getpid(),
                                 "seconds": time.monotonic() - started, **extra})
    status("initializing", selection_sha256=selection_hash)
    try:
        with gpu_lock(args.gpu_lock):
            device = configure_device(args.device, args.threads, args.max_cuda_gib)
            report["environment"] = {"host": socket.gethostname(), "python": sys.version,
                                     "torch": torch.__version__, "numpy": np.__version__,
                                     "cuda": torch.version.cuda, "device": str(device),
                                     "device_name": torch.cuda.get_device_name(device) if device.type == "cuda" else "cpu",
                                     "deterministic_cudnn": True, "tf32": False,
                                     "max_cuda_gib": args.max_cuda_gib, "cpu_threads": args.threads}
            cases, provenance = dataset_cases(selection, args)
            report.update(dataset=selection["purpose"], dataset_provenance=provenance,
                          image_count=len(cases), timing_scope="synchronized forward only; transfers and scoring excluded")
            offset = selection["offset"]
            for entry in selection["models"]:
                model_name = entry["name"]
                status("loading_model", model=model_name)
                if device.type == "cuda":
                    torch.cuda.reset_peak_memory_stats(device)
                model = make_model(entry, device)
                result = {"name": model_name, "role": entry["role"], "parameters": entry["parameters"],
                          "images": [], "complete": False}
                report["models"].append(result)
                with torch.inference_mode():
                    warmup = torch.from_numpy(cases[0]["noisy"]).to(device) + offset
                    for _ in range(selection["timing_warmup"]):
                        value = model(warmup)
                        if not torch.isfinite(value).all():
                            raise FloatingPointError(f"Non-finite warmup prediction: {model_name}")
                    sync(device)
                    del warmup, value
                    for index, case in enumerate(cases):
                        status("evaluating", model=model_name, image=case["name"],
                               image_index=index, images=len(cases))
                        tensor = torch.from_numpy(case["noisy"]).to(device) + offset
                        times = []
                        for _ in range(selection["timing_repeats"]):
                            sync(device)
                            began = time.perf_counter()
                            prediction = model(tensor)
                            sync(device)
                            times.append(time.perf_counter() - began)
                            if not torch.isfinite(prediction).all():
                                raise FloatingPointError(f"Non-finite raw prediction: {model_name}/{case['name']}")
                        score = score_prediction(case["clean"], prediction.cpu().numpy(), offset)
                        if score["psnr"] is None:
                            raise FloatingPointError("Perfect prediction needs explicit infinite-PSNR reporting; inspect inputs")
                        item = {"name": case["name"], "shape_native": list(case["noisy"].shape),
                                "noisy_native_sha256": case["noisy_native_sha256"],
                                "label_native_sha256": case["label_native_sha256"],
                                "seconds_forward": times, "median_seconds_forward": float(np.median(times)), **score}
                        if "archived_psnr" in case:
                            item["archived_historical_psnr"] = case["archived_psnr"]
                        result["images"].append(item)
                        atomic_json(report_path, report)
                        print(json.dumps({"model": model_name, "image": case["name"],
                                          "psnr": score["psnr"], "seconds_forward": times}), flush=True)
                        del tensor, prediction
                result.update(complete=True, mean_psnr=float(np.mean([item["psnr"] for item in result["images"]])),
                              median_seconds_forward=float(np.median([item["median_seconds_forward"] for item in result["images"]])),
                              sum_seconds_forward=float(sum(sum(item["seconds_forward"]) for item in result["images"])))
                if device.type == "cuda":
                    result.update(peak_cuda_allocated_bytes=torch.cuda.max_memory_allocated(device),
                                  peak_cuda_reserved_bytes=torch.cuda.max_memory_reserved(device))
                atomic_json(report_path, report)
                del model
                if device.type == "cuda":
                    torch.cuda.empty_cache()
            by_name = {item["name"]: item for item in report["models"]}
            for pair in selection["comparisons"]:
                report["comparisons"].append({**pair, **paired_summary(
                    by_name[pair["candidate"]]["images"], by_name[pair["reference"]]["images"],
                    selection["bootstrap_seed"], selection["bootstrap_samples"])})
            report.update(complete=True, ended_utc=utc(), seconds_total=time.monotonic() - started)
            atomic_json(report_path, report)
            status("complete", models=len(report["models"]), images=len(cases))
    except BaseException as error:
        report.update(failure={"type": type(error).__name__, "message": str(error)}, ended_utc=utc())
        atomic_json(report_path, report)
        status("failed", error=report["failure"])
        raise


def self_test():
    """Small CPU fixtures test protocol mechanics, never stand in for measured results."""
    import tempfile
    import unittest

    class ProtocolTests(unittest.TestCase):
        def test_native_axes_offset_and_float64_mse(self):
            clean = np.arange(6, dtype=np.float32).reshape(1, 1, 2, 3)
            raw = clean + 25 + np.float32(.125)
            score = score_prediction(clean, raw, 25)
            self.assertEqual(score["mse"], .125 ** 2)
            self.assertAlmostEqual(score["psnr"], 10 * math.log10(255 ** 2 / .125 ** 2), places=12)
            with self.assertRaises(ValueError):
                score_prediction(clean, raw.transpose(0, 1, 3, 2), 25)

        def test_output_only_clipping(self):
            clean = np.array([0, 255], dtype=np.float32).reshape(1, 1, 1, 2)
            raw = np.array([-20, 300], dtype=np.float32).reshape(clean.shape) + 25
            result = score_prediction(clean, raw, 25)
            self.assertTrue(result["perfect_prediction"])
            self.assertEqual(result["clipped_pixels"], 2)

        def test_nonfinite_rejected_before_clip(self):
            clean = np.zeros((1, 1, 1, 1), dtype=np.float32)
            for value in (np.inf, -np.inf, np.nan):
                with self.assertRaises(FloatingPointError):
                    score_prediction(clean, np.full_like(clean, value), 25)

        def test_paired_bootstrap_and_identity(self):
            reference = [{"name": str(i), "psnr": float(i + 20), "noisy_native_sha256": str(i)} for i in range(4)]
            candidate = [{**item, "psnr": item["psnr"] + .25} for item in reference]
            result = paired_summary(candidate, reference, 71, 1200)
            self.assertEqual(result["mean_delta_db"], .25)
            self.assertEqual(result["bootstrap_mean_95_percentile_ci_db"], [.25, .25])
            self.assertEqual(result["improved_images"], 4)
            self.assertEqual(result, paired_summary(candidate, reference, 71, 1200))
            candidate[0]["noisy_native_sha256"] = "changed"
            with self.assertRaises(ValueError):
                paired_summary(candidate, reference, 71, 1200)

        def test_selection_exclusive_and_tamper_proof(self):
            with tempfile.TemporaryDirectory() as directory:
                path = Path(directory) / "selection.json"
                record = {"schema": 1, "purpose": "bsd68", "created_utc": utc(), "sigma": 25, "offset": 25}
                write_frozen_selection(path, record)
                self.assertEqual(read_frozen_selection(path)[0], record)
                with self.assertRaises(FileExistsError):
                    write_frozen_selection(path, record)
                path.write_text(path.read_text() + " ")
                with self.assertRaises(ValueError):
                    read_frozen_selection(path)

        def test_shared_lock_refuses_duplicate(self):
            with tempfile.TemporaryDirectory() as directory:
                path = Path(directory) / "gpu.lock"
                with gpu_lock(path):
                    with self.assertRaises(RuntimeError):
                        with gpu_lock(path):
                            pass
                with gpu_lock(path):
                    pass

        def test_checkpoint_provenance_rejects_each_mismatch(self):
            import copy
            config = {"sigma": 25, "offset": 25, "model": {"width": 80}}
            checkpoint = {"config": config, "config_sha256": "config", "source_sha256": "source",
                          "manifest_sha256": "train", "validation_manifest_sha256": "validation",
                          "initialization": {"source_sha256": "original", "source_sigma": 25}}
            def check(value):
                validate_checkpoint_identity(value, config, "config", 25, 25,
                                             "source", "original", "train", "validation")
            check(checkpoint)
            for key in ("config_sha256", "source_sha256", "manifest_sha256", "validation_manifest_sha256"):
                mutated = copy.deepcopy(checkpoint)
                mutated[key] = "changed"
                with self.subTest(key=key), self.assertRaises(ValueError):
                    check(mutated)
            for key in ("source_sha256", "source_sigma"):
                mutated = copy.deepcopy(checkpoint)
                mutated["initialization"][key] = "changed"
                with self.subTest(key=key), self.assertRaises(ValueError):
                    check(mutated)
            mutated = copy.deepcopy(checkpoint)
            mutated["config"]["model"]["width"] = 112
            with self.assertRaises(ValueError):
                check(mutated)

        def test_implementation_class_selection(self):
            from beltrami_operator.model import BeltramiOperator as Original, ModelConfig as OriginalConfig
            from beltrami_precision.model import BeltramiOperator as Precision, ModelConfig as PrecisionConfig
            self.assertEqual(model_classes("beltrami_operator"), (Original, OriginalConfig))
            self.assertEqual(model_classes("beltrami_precision"), (Precision, PrecisionConfig))
            self.assertNotEqual(Original, Precision)
            self.assertEqual(PrecisionConfig(metric_precision="float64").metric_precision, "float64")

        def test_implementation_source_and_entry_tamper(self):
            import copy
            hashes = {name: operator_source_hash(name) for name in IMPLEMENTATIONS}
            entries = [{"name": name, "kind": "checkpoint", "implementation": name,
                        "training_source_sha256": digest} for name, digest in hashes.items()]
            selection = {"models": entries, "implementation_source_sha256": hashes}
            verify_implementation_sources(selection)
            for name in IMPLEMENTATIONS:
                mutated = copy.deepcopy(selection)
                mutated["implementation_source_sha256"][name] = "changed"
                with self.subTest(implementation=name), self.assertRaises(ValueError):
                    verify_implementation_sources(mutated)
            mutated = copy.deepcopy(selection)
            mutated["models"][1]["implementation"] = "beltrami_operator"
            with self.assertRaises(ValueError):
                verify_implementation_sources(mutated)
            mutated = copy.deepcopy(selection)
            mutated["models"][1]["training_source_sha256"] = hashes["beltrami_operator"]
            with self.assertRaises(ValueError):
                verify_implementation_sources(mutated)

        def test_legacy_selection_and_unknown_implementation(self):
            digest = operator_source_hash()
            legacy = {"operator_source_sha256": digest, "models": [
                {"name": "legacy", "kind": "checkpoint", "training_source_sha256": digest}]}
            verify_implementation_sources(legacy)
            legacy["models"][0]["implementation"] = "beltrami_precision"
            with self.assertRaises(ValueError):
                verify_implementation_sources(legacy)
            for name in ("os", "beltrami_operator.model", "unknown"):
                with self.subTest(name=name), self.assertRaises(ValueError):
                    model_classes(name)
                with self.subTest(name=name), self.assertRaises(ValueError):
                    operator_source_hash(name)
            with self.assertRaises(ValueError):
                verify_implementation_sources({"implementation_source_sha256": {"os": "not-a-source-hash"},
                    "models": [{"name": "unknown", "kind": "checkpoint", "implementation": "os",
                                "training_source_sha256": "not-a-source-hash"}]})

    result = unittest.TextTestRunner(verbosity=2).run(unittest.defaultTestLoader.loadTestsFromTestCase(ProtocolTests))
    return 0 if result.wasSuccessful() else 1


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    frozen = commands.add_parser("freeze", help="Record the validation-only decision without evaluating labels")
    frozen.add_argument("--spec", required=True, type=Path)
    frozen.add_argument("--selection", required=True, type=Path)
    frozen.add_argument("--model-dir", required=True, type=Path)
    frozen.add_argument("--purpose", choices=("bsd68", "validation32"), required=True)
    frozen.add_argument("--training-manifest", required=True, type=Path)
    frozen.add_argument("--validation-manifest", required=True, type=Path)
    frozen.add_argument("--validation-seed", type=int)
    frozen.add_argument("--warmup", type=int, default=2)
    frozen.add_argument("--timing-repeats", type=int, default=3)
    frozen.add_argument("--bootstrap-seed", type=int, default=20261112)
    frozen.add_argument("--bootstrap-samples", type=int, default=20000)
    run = commands.add_parser("evaluate", help="Evaluate the frozen entries once in a new evidence directory")
    run.add_argument("--selection", required=True, type=Path)
    run.add_argument("--output", required=True, type=Path)
    run.add_argument("--gpu-lock", required=True, type=Path)
    run.add_argument("--device", default="cuda:0")
    run.add_argument("--threads", type=int, default=4)
    run.add_argument("--max-cuda-gib", type=float, default=10)
    run.add_argument("--cache-dir", type=Path)
    run.add_argument("--clean-dir", type=Path)
    run.add_argument("--asset-dir", type=Path)
    run.add_argument("--validation-root", type=Path)
    commands.add_parser("self-test", help="Run CPU-only protocol tests without loading research datasets")
    args = parser.parse_args()
    if args.command == "self-test":
        return self_test()
    if args.command == "freeze":
        if args.warmup < 1 or args.timing_repeats < 1 or args.bootstrap_samples < 1000:
            parser.error("Require warmup>=1, timing-repeats>=1, bootstrap-samples>=1000")
        freeze(args)
    else:
        if not 0 < args.max_cuda_gib <= 10 or args.threads < 1:
            parser.error("Require 0<max-cuda-gib<=10 and threads>=1")
        evaluate(args)
    return 0


if __name__ == "__main__":
    sys.exit(main())

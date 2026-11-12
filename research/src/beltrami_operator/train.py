"""Single-process budgeted experiments with resumable state and finite checks."""
import argparse
from dataclasses import asdict
from datetime import datetime, timezone
import fcntl
import hashlib
import json
import math
import os
from pathlib import Path
import random
import shutil
import signal
import socket
import subprocess
import sys
import time
import traceback
import numpy as np
import torch
from .data import PatchSampler, load_images, validation_pairs
from .model import BeltramiOperator, ModelConfig


def utc():
    return datetime.now(timezone.utc).isoformat()


def atomic_json(path, value):
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")
    os.replace(temporary, path)


def file_hash(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def source_hash():
    digest = hashlib.sha256()
    for path in sorted(Path(__file__).parent.glob("*.py")):
        digest.update(path.name.encode())
        digest.update(path.read_bytes())
    return digest.hexdigest()


def environment():
    return {"host": socket.gethostname(), "python": sys.version, "torch": torch.__version__,
            "numpy": np.__version__, "cuda": torch.version.cuda,
            "cudnn": torch.backends.cudnn.version(),
            "device": torch.cuda.get_device_name(0) if torch.cuda.is_available() else "cpu",
            "source_sha256": source_hash()}


@torch.no_grad()
def evaluate(model, images, records, config, device):
    model.eval()
    scores = []
    for noisy, clean, record in validation_pairs(images, records, config["sigma"],
                                                config["validation_seed"], config["offset"]):
        prediction = model(noisy.to(device)) - config["offset"]
        if not torch.isfinite(prediction).all():
            raise FloatingPointError(f"Non-finite validation prediction: {record['path']}")
        prediction = prediction.clamp(0, 255)
        prediction = prediction.cpu().numpy().astype(np.float64)
        mse = float(np.mean((prediction - clean.astype(np.float64)) ** 2))
        scores.append({"image": record["path"], "pixels_sha256": record["pixels_sha256"],
                       "psnr": 10 * math.log10(255 ** 2 / max(mse, 1e-300))})
    model.train()
    return {"mean_psnr": float(np.mean([item["psnr"] for item in scores])),
            "images": scores, "clipping": "prediction only, float [0,255]", "crop": 0}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--validation-root", type=Path)
    parser.add_argument("--validation-manifest", type=Path)
    parser.add_argument("--model-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--gpu-lock", type=Path, required=True)
    parser.add_argument("--max-gpu-hours", type=float, required=True)
    parser.add_argument("--deadline-utc", help="Absolute ISO8601 deadline; earliest budget wins")
    parser.add_argument("--resume", type=Path)
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()
    if args.max_gpu_hours <= 0 or args.max_gpu_hours > 12:
        parser.error("Per-run budget must be positive and at most12GPU-hours")
    config = json.loads(args.config.read_text())
    if bool(args.validation_root) != bool(args.validation_manifest):
        parser.error("Provide both external validation root and manifest")
    required = {"sigma", "offset", "seed", "validation_seed", "steps", "batch_size", "patch_size",
                "learning_rate", "evaluate_every", "checkpoint_every", "log_every", "model"}
    if required - config.keys():
        parser.error(f"Missing configuration fields: {sorted(required - config.keys())}")
    config_hash = file_hash(args.config)
    args.gpu_lock.parent.mkdir(parents=True, exist_ok=True)
    lock = args.gpu_lock.open("a+")
    try:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError:
        raise SystemExit("Shared GPU lock is held; refusing to queue a duplicate process")
    args.output.mkdir(parents=True, exist_ok=True)
    if (args.output / "status.json").exists() and not args.resume:
        raise SystemExit("Output already contains status; resume explicitly or choose a new run")
    stop = {"requested": False, "signal": None}
    def request_stop(signum, _frame):
        stop.update(requested=True, signal=signum)
    signal.signal(signal.SIGTERM, request_stop)
    signal.signal(signal.SIGINT, request_stop)
    random.seed(config["seed"])
    np.random.seed(config["seed"])
    torch.manual_seed(config["seed"])
    torch.set_num_threads(config.get("cpu_threads", 4))
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = config.get("deterministic", True)
    device = torch.device(args.device)
    if device.type == "cuda" and device.index is None:
        device = torch.device("cuda", torch.cuda.current_device())
    if device.type == "cuda":
        torch.cuda.set_per_process_memory_fraction(
            min(0.9, config.get("max_cuda_gib", 10) * 1024 ** 3 /
                torch.cuda.get_device_properties(device).total_memory), device)
    train_images, train_records, train_manifest = load_images(args.data_root, args.manifest, "train")
    validation_root = args.validation_root or args.data_root
    validation_manifest_path = args.validation_manifest or args.manifest
    val_images, val_records, val_manifest = load_images(validation_root, validation_manifest_path, "validation")
    overlap = {r["pixels_sha256"] for r in train_records} & {r["pixels_sha256"] for r in val_records}
    if overlap:
        raise ValueError("Training and validation decoded image hashes overlap")
    sampler = PatchSampler(train_images, config["patch_size"], config["seed"])
    model = BeltramiOperator(ModelConfig(**config["model"]))
    initialization = model.initialize_original(args.model_dir, config["sigma"])
    model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=config["learning_rate"],
                                 weight_decay=config.get("weight_decay", 0.0))
    step, prior_seconds, best_psnr = 0, 0.0, -math.inf
    if args.resume:
        checkpoint = torch.load(args.resume, map_location=device, weights_only=False)
        if checkpoint["config_sha256"] != config_hash or checkpoint["manifest_sha256"] != file_hash(args.manifest):
            raise ValueError("Resume configuration or data manifest differs")
        if checkpoint["validation_manifest_sha256"] != file_hash(validation_manifest_path):
            raise ValueError("Resume validation manifest differs")
        model.load_state_dict(checkpoint["model"])
        optimizer.load_state_dict(checkpoint["optimizer"])
        step, prior_seconds, best_psnr = checkpoint["step"], checkpoint["gpu_seconds"], checkpoint["best_psnr"]
        sampler.rng.bit_generator.state = checkpoint["sampler_rng"]
        torch.set_rng_state(checkpoint["torch_rng"].cpu())
        if device.type == "cuda":
            torch.cuda.set_rng_state_all([state.cpu() for state in checkpoint["cuda_rng"]])
    started = time.monotonic()
    hard_deadline = (datetime.fromisoformat(args.deadline_utc.replace("Z", "+00:00")).timestamp()
                     if args.deadline_utc else math.inf)
    metadata = {"created_utc": utc(), "pid": os.getpid(), "environment": environment(),
                "config": config, "config_sha256": config_hash, "initialization": initialization,
                "parameters": sum(p.numel() for p in model.parameters()),
                "trainable_parameters": sum(p.numel() for p in model.parameters() if p.requires_grad),
                "manifest_sha256": file_hash(args.manifest),
                "validation_manifest_sha256": file_hash(validation_manifest_path),
                "train_images": len(train_images), "validation_images": len(val_images),
                "validation_prior_checkpoint_exposure": val_manifest.get("prior_checkpoint_exposure", "unknown"),
                "authorized_gpu_hours_this_run": args.max_gpu_hours,
                "accounting": "conservative process wall time including initialization/evaluation; no idle deduction",
                "deadline_utc": args.deadline_utc,
                "paths": {k: str(v) for k, v in vars(args).items() if isinstance(v, Path)}}
    atomic_json(args.output / "metadata.json", metadata)
    log = (args.output / "events.jsonl").open("a", buffering=1)
    last_evaluation = None
    def elapsed():
        return prior_seconds + time.monotonic() - started
    def event(kind, **values):
        item = {"utc": utc(), "event": kind, "step": step, "gpu_seconds": elapsed(), **values}
        log.write(json.dumps(item, allow_nan=False) + "\n")
        print(json.dumps(item, allow_nan=False), flush=True)
        return item
    def status(state, **extra):
        values = {"state": state, "utc": utc(), "pid": os.getpid(), "step": step,
                  "target_steps": config["steps"], "gpu_seconds": elapsed(),
                  "budget_gpu_seconds": args.max_gpu_hours * 3600,
                  "last_evaluation": last_evaluation,
                  "checkpoint": str(args.output / "last.pt"),
                  "disk_free_bytes": shutil.disk_usage(args.output).free, **extra}
        if device.type == "cuda":
            free, total = torch.cuda.mem_get_info(device)
            values.update(cuda_free_bytes=free, cuda_total_bytes=total,
                          cuda_allocated_bytes=torch.cuda.memory_allocated(device),
                          cuda_reserved_bytes=torch.cuda.memory_reserved(device),
                          cuda_peak_allocated_bytes=torch.cuda.max_memory_allocated(device))
        atomic_json(args.output / "status.json", values)
    def save(name="last.pt"):
        value = {"model": model.state_dict(), "optimizer": optimizer.state_dict(),
                 "step": step, "gpu_seconds": elapsed(), "best_psnr": best_psnr,
                 "config": config, "config_sha256": config_hash,
                 "manifest_sha256": file_hash(args.manifest),
                 "validation_manifest_sha256": file_hash(validation_manifest_path),
                 "sampler_rng": sampler.rng.bit_generator.state, "torch_rng": torch.get_rng_state(),
                 "cuda_rng": torch.cuda.get_rng_state_all() if device.type == "cuda" else [],
                 "initialization": initialization, "source_sha256": source_hash()}
        temporary = args.output / (name + ".tmp")
        torch.save(value, temporary)
        os.replace(temporary, args.output / name)
    def validation():
        nonlocal last_evaluation, best_psnr
        result = evaluate(model, val_images, val_records, config, device)
        last_evaluation = {"step": step, "mean_psnr": result["mean_psnr"], "utc": utc()}
        atomic_json(args.output / f"validation-{step:07d}.json", result)
        improved = result["mean_psnr"] > best_psnr
        best_psnr = max(best_psnr, result["mean_psnr"])
        if improved:
            save("best.pt")
        save()
        event("validation", mean_psnr=result["mean_psnr"], best_psnr=best_psnr, improved=improved)
    try:
        status("initializing")
        validation()
        status("training")
        recent_losses = []
        while step < config["steps"]:
            if stop["requested"] or elapsed() >= args.max_gpu_hours * 3600 or time.time() >= hard_deadline:
                reason = "signal" if stop["requested"] else "budget"
                event("stop", reason=reason, signal=stop["signal"])
                save()
                status("stopped", reason=reason)
                return
            noisy, clean = sampler.batch(config["batch_size"], config["sigma"], config["offset"])
            optimizer.zero_grad(set_to_none=True)
            prediction = model(noisy.to(device))
            if not torch.isfinite(prediction).all():
                raise FloatingPointError("Non-finite training prediction")
            loss = (prediction - clean.to(device)).square().mean() / (255.0 ** 2)
            if not torch.isfinite(loss):
                raise FloatingPointError("Non-finite training loss")
            loss.backward()
            for name, parameter in model.named_parameters():
                if parameter.grad is not None and not torch.isfinite(parameter.grad).all():
                    raise FloatingPointError(f"Non-finite gradient: {name}")
            gradient_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), config.get("gradient_clip", 1.0))
            optimizer.step()
            model.project_constraints()
            for name, parameter in model.named_parameters():
                if not torch.isfinite(parameter).all():
                    raise FloatingPointError(f"Non-finite parameter: {name}")
            step += 1
            recent_losses.append(float(loss.detach()))
            if step % config["log_every"] == 0:
                event("progress", loss=float(np.mean(recent_losses)), gradient_norm=float(gradient_norm),
                      images_seen=step * config["batch_size"])
                recent_losses = []
                status("training")
                if shutil.disk_usage(args.output).free < config.get("minimum_disk_gib", 5) * 1024 ** 3:
                    raise OSError("Disk space below configured training floor")
            if step % config["evaluate_every"] == 0:
                status("evaluating")
                validation()
                status("training")
            elif step % config["checkpoint_every"] == 0:
                save()
        status("evaluating")
        validation()
        event("complete", best_psnr=best_psnr)
        status("complete")
    except Exception as error:
        details = traceback.format_exc()
        (args.output / "failure.txt").write_text(details)
        save("failure.pt")
        event("failure", error=str(error))
        status("failed", error=str(error))
        raise
    finally:
        log.close()


if __name__ == "__main__":
    main()

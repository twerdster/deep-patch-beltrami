"""Budgeted progressive-depth training of metric-driven Beltrami operators."""
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
import time
import traceback

import numpy as np
import torch

from beltrami_precision.data import PatchSampler, load_images, validation_pairs
from .model import BeltramiOperator, ModelConfig


def utc():
    return datetime.now(timezone.utc).isoformat()


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def source_pins():
    root = Path(__file__).resolve().parents[1]
    pins = {}
    for namespace in ("beltrami_operator", "beltrami_precision", "beltrami_rollout"):
        h = hashlib.sha256()
        for path in sorted((root / namespace).glob("*.py")):
            h.update(path.name.encode())
            h.update(path.read_bytes())
        pins[namespace] = h.hexdigest()
    return pins


def atomic_json(path, value):
    temp = path.with_suffix(path.suffix + ".partial")
    temp.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")
    temp.replace(path)


def schedule(config, progress):
    """One budget-aware warmup/cosine cycle; chronology is saved in every log."""
    progress = min(1.0, max(0.0, progress))
    warmup = config.get("warmup_fraction", 0.05)
    floor = config.get("lr_floor_fraction", 0.1)
    factor = (max(0.05, progress / warmup) if progress < warmup else
              floor + (1 - floor) * 0.5 *
              (1 + math.cos(math.pi * (progress - warmup) / (1 - warmup))))
    target = config["model"]["stages"]
    active = target
    for transition in config["curriculum"]:
        if progress < transition["until_fraction"]:
            active = transition["stages"]
            break
    auxiliary = config.get("auxiliary_weight", 0.05) * max(
        0.0, 1 - progress / config.get("auxiliary_until_fraction", 0.3))
    return factor, active, auxiliary


def optimizer_groups(model, config):
    groups = {}
    for name, parameter in model.named_parameters():
        if not parameter.requires_grad:
            continue
        index = int(name.split(".")[1])
        refinement = index >= 5
        decay = (config["weight_decay"] if name.endswith("weight") and
                 "metric_" not in name else 0.0)
        peak = config["learning_rate"] * (
            config.get("refinement_lr_multiplier", 5.0) if refinement else 1.0)
        groups.setdefault((peak, decay), []).append(parameter)
    return [{"params": values, "lr": peak, "peak_lr": peak, "weight_decay": decay}
            for (peak, decay), values in groups.items()]


def finite_state(value):
    if torch.is_tensor(value):
        return bool(torch.isfinite(value).all())
    if isinstance(value, dict):
        return all(finite_state(x) for x in value.values())
    if isinstance(value, (tuple, list)):
        return all(finite_state(x) for x in value)
    return not isinstance(value, float) or math.isfinite(value)


def validate_config(config):
    if config["model"]["stages"] not in (5, 32, 64):
        raise ValueError("This cohort has exactly 5,32,64-stage arms")
    if config["patch_size"] != 180 or config["microbatch"] != 1 or config["accumulation"] != 2:
        raise ValueError("Cohort requires 180px/microbatch1/accumulation2")
    if not 0 < config["warmup_fraction"] < 1:
        raise ValueError("Invalid warmup")
    previous = 0
    for row in config["curriculum"]:
        if not previous < row["until_fraction"] <= 1 or not 5 <= row["stages"] <= config["model"]["stages"]:
            raise ValueError("Invalid depth curriculum")
        previous = row["until_fraction"]
    if config["curriculum"][-1] != {"until_fraction": 1.0, "stages": config["model"]["stages"]}:
        raise ValueError("Curriculum must end at target depth")


def main():
    p = argparse.ArgumentParser(description=__doc__)
    for name in ("config", "data-root", "manifest", "validation-root", "validation-manifest",
                 "warmstart", "output", "gpu-lock"):
        p.add_argument("--" + name, type=Path, required=True)
    p.add_argument("--max-gpu-hours", type=float, required=True)
    p.add_argument("--device", default="cuda:0")
    p.add_argument("--resume", type=Path)
    p.add_argument("--smoke", action="store_true", help="Two full-depth updates and one full-image validation")
    args = p.parse_args()
    if not 0 < args.max_gpu_hours <= 12:
        p.error("Invalid budget")
    config = json.loads(args.config.read_text())
    validate_config(config)
    expected_pins = config["source_pins"]
    if source_pins() != expected_pins:
        raise ValueError("Numerical or training source has changed from frozen configuration")
    config_sha = digest(args.config)
    manifest_sha = digest(args.manifest)
    validation_sha = digest(args.validation_manifest)
    args.output.mkdir(parents=True, exist_ok=True)
    if (args.output / "status.json").exists() and not args.resume:
        raise ValueError("Existing output; an explicit new lease and resume are required")
    args.gpu_lock.parent.mkdir(parents=True, exist_ok=True)
    lock = args.gpu_lock.open("a+")
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    started = time.monotonic()
    stop = {"requested": False, "signal": None}
    def request_stop(signum, frame):
        stop.update(requested=True, signal=signum)
    signal.signal(signal.SIGTERM, request_stop)
    signal.signal(signal.SIGINT, request_stop)
    torch.set_num_threads(config.get("cpu_threads", 4))
    random.seed(config["seed"])
    np.random.seed(config["seed"])
    torch.manual_seed(config["seed"])
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    device = torch.device(args.device)
    if device.type == "cuda":
        torch.cuda.set_per_process_memory_fraction(
            config["max_cuda_gib"] * 1024 ** 3 /
            torch.cuda.get_device_properties(device).total_memory, device)
    train_images, train_records, _ = load_images(args.data_root, args.manifest, "train")
    val_images, val_records, _ = load_images(args.validation_root, args.validation_manifest, "validation")
    if {x["pixels_sha256"] for x in train_records} & {x["pixels_sha256"] for x in val_records}:
        raise ValueError("Training-validation overlap")
    sampler = PatchSampler(train_images, config["patch_size"], config["seed"])
    model = BeltramiOperator(ModelConfig(**config["model"]))
    initialization = model.initialize_from_safe(args.warmstart, config["warmstart_sha256"])
    model.to(device)
    optimizer = torch.optim.AdamW(optimizer_groups(model, config), betas=(0.9, 0.999), eps=1e-8)
    step, prior_seconds, best_psnr = 0, 0.0, None
    last_evaluation = None
    best_step = None
    active_previous = None
    full_depth_updates = 0
    phase_updates = {}
    initialization_psnr = None
    if args.resume:
        if args.resume.resolve().parent != args.output.resolve():
            raise ValueError("Resume requires the original output; preserve its verified BEST")
        saved = torch.load(args.resume, map_location=device, weights_only=False)
        for key, value in {"config_sha256": config_sha, "manifest_sha256": manifest_sha,
                           "validation_manifest_sha256": validation_sha, "source_pins": expected_pins}.items():
            if saved.get(key) != value:
                raise ValueError("Resume guard differs: " + key)
        if not finite_state(saved["model"]) or not finite_state(saved["optimizer"]):
            raise ValueError("Cannot resume nonfinite state")
        model.load_state_dict(saved["model"])
        optimizer.load_state_dict(saved["optimizer"])
        step, prior_seconds = saved["step"], saved["gpu_seconds"]
        best_psnr, best_step = saved["best_psnr"], saved["best_step"]
        last_evaluation = saved["last_evaluation"]
        full_depth_updates = saved["full_depth_updates"]
        phase_updates = saved["phase_updates"]
        initialization_psnr = saved["initialization_psnr"]
        if best_psnr is not None:
            prior_best = torch.load(args.output / "best.pt", map_location="cpu", weights_only=False)
            if (prior_best["config_sha256"] != config_sha or prior_best["source_pins"] != expected_pins or
                    prior_best["step"] != best_step or not finite_state(prior_best["model"])):
                raise ValueError("Original BEST differs from resumed lineage")
        sampler.rng.bit_generator.state = saved["sampler_rng"]
        torch.set_rng_state(saved["torch_rng"].cpu())
        if device.type == "cuda":
            torch.cuda.set_rng_state_all([x.cpu() for x in saved["cuda_rng"]])
    def elapsed():
        return prior_seconds + time.monotonic() - started
    limit = args.max_gpu_hours * 3600
    log = (args.output / "events.jsonl").open("a", buffering=1)
    def event(kind, **values):
        item = {"utc": utc(), "event": kind, "step": step, "gpu_seconds": elapsed(), **values}
        line = json.dumps(item, allow_nan=False)
        log.write(line + "\n")
        print(line, flush=True)
    def status(state, **values):
        item = {"state": state, "utc": utc(), "pid": os.getpid(), "step": step,
                "target_steps": config["steps"], "gpu_seconds": elapsed(),
                "budget_gpu_seconds": limit, "target_stages": config["model"]["stages"],
                "active_stages": active_previous, "last_evaluation": last_evaluation,
                "full_depth_updates": full_depth_updates, "phase_updates": phase_updates,
                "initialization_psnr": initialization_psnr,
                "best_psnr": best_psnr, "best_step": best_step,
                "checkpoint": str(args.output / "last.pt"),
                "disk_free_bytes": shutil.disk_usage(args.output).free, **values}
        if device.type == "cuda":
            item.update(cuda_allocated_bytes=torch.cuda.memory_allocated(device),
                        cuda_reserved_bytes=torch.cuda.memory_reserved(device),
                        cuda_peak_allocated_bytes=torch.cuda.max_memory_allocated(device))
        atomic_json(args.output / "status.json", item)
    def save(name="last.pt"):
        saved = {"model": model.state_dict(), "optimizer": optimizer.state_dict(), "step": step,
                 "gpu_seconds": elapsed(), "best_psnr": best_psnr, "best_step": best_step,
                 "last_evaluation": last_evaluation, "config": config, "config_sha256": config_sha,
                 "full_depth_updates": full_depth_updates, "phase_updates": phase_updates,
                 "initialization_psnr": initialization_psnr,
                 "manifest_sha256": manifest_sha, "validation_manifest_sha256": validation_sha,
                 "source_pins": expected_pins, "initialization": initialization,
                 "sampler_rng": sampler.rng.bit_generator.state, "torch_rng": torch.get_rng_state(),
                 "cuda_rng": torch.cuda.get_rng_state_all() if device.type == "cuda" else []}
        path = args.output / name
        temp = path.with_suffix(path.suffix + ".partial")
        torch.save(saved, temp)
        temp.replace(path)
    @torch.no_grad()
    def validation():
        nonlocal best_psnr, best_step, last_evaluation, initialization_psnr
        status("evaluating", evaluation_target_stages=config["model"]["stages"])
        model.eval()
        scores = []
        pairs = validation_pairs(val_images, val_records, config["sigma"],
                                 config["validation_seed"], config["offset"])
        for index, (noisy, clean, record) in enumerate(pairs):
            prediction = model(noisy.to(device)) - config["offset"]
            if not torch.isfinite(prediction).all():
                raise FloatingPointError("Nonfinite full-depth validation: " + record["path"])
            prediction = prediction.clamp(0, 255).cpu().numpy().astype(np.float64)
            mse = float(np.mean((prediction - clean.astype(np.float64)) ** 2))
            scores.append({"image": record["path"], "pixels_sha256": record["pixels_sha256"],
                           "mse": mse, "psnr": 10 * math.log10(255 ** 2 / max(mse, 1e-300))})
            if args.smoke:
                break
            if (index + 1) % 4 == 0:
                status("evaluating", evaluation_images_done=index + 1, evaluation_images_total=len(val_images))
        result = {"mean_psnr": float(np.mean([x["psnr"] for x in scores])), "images": scores,
                  "step": step, "utc": utc(), "stages": config["model"]["stages"],
                  "full_depth_updates": full_depth_updates,
                  "selection_eligible": args.smoke or full_depth_updates >= config["minimum_full_depth_updates"],
                  "smoke_one_image": args.smoke, "clipping": "output-only float[0,255]", "crop": 0}
        atomic_json(args.output / f"validation-{step:07d}.json", result)
        last_evaluation = {k: result[k] for k in ("mean_psnr", "step", "utc", "stages")}
        if step == 0:
            initialization_psnr = result["mean_psnr"]
        improved = result["selection_eligible"] and (best_psnr is None or result["mean_psnr"] > best_psnr)
        if improved:
            best_psnr, best_step = result["mean_psnr"], step
            save("best.pt")
        save()
        event("validation", **last_evaluation, improved=improved, best_psnr=best_psnr, best_step=best_step,
              selection_eligible=result["selection_eligible"], full_depth_updates=full_depth_updates)
        model.train()
    metadata = {"implementation": "beltrami_rollout", "created_utc": utc(), "pid": os.getpid(),
                "config": config, "config_sha256": config_sha, "source_pins": expected_pins,
                "manifest_sha256": manifest_sha, "validation_manifest_sha256": validation_sha,
                "initialization": initialization, "parameters": sum(p.numel() for p in model.parameters()),
                "torch": torch.__version__, "numpy": np.__version__, "cuda": torch.version.cuda,
                "device": str(device), "budget_seconds": limit,
                "sampling": "400 recovered180pxcrops,freshnoise,D4,effectivebatch2",
                "scope": "validation-only exploration; BSD68 already exposed in previous round",
                "accounting": "outer exclusive lease includes all startup, failures and evaluation"}
    atomic_json(args.output / "metadata.json", metadata)
    try:
        validation()
        status("training")
        recent = []
        while step < (2 if args.smoke else config["steps"]):
            if stop["requested"] or elapsed() >= limit:
                break
            factor, active, auxiliary = schedule(config, elapsed() / limit)
            if args.smoke:
                active, factor, auxiliary = config["model"]["stages"], 1.0, 0.05
            if active != active_previous:
                active_previous = active
                event("curriculum", active_stages=active, target_stages=config["model"]["stages"])
            for group in optimizer.param_groups:
                group["lr"] = group["peak_lr"] * factor
            optimizer.zero_grad(set_to_none=True)
            loss_sum, final_sum = 0.0, 0.0
            for _ in range(config["accumulation"]):
                noisy, clean = sampler.batch(config["microbatch"], config["sigma"], config["offset"])
                clean = clean.to(device)
                prediction, intermediate = model(noisy.to(device), active_stages=active,
                                                 return_intermediates=True)
                if not torch.isfinite(prediction).all():
                    raise FloatingPointError("Nonfinite training prediction")
                final_loss = (prediction - clean).square().mean() / 255 ** 2
                aux_values = [v for k, v in intermediate.items() if k < active]
                loss = final_loss
                if aux_values and auxiliary > 0:
                    loss = loss + auxiliary * torch.stack([
                        (v - clean).square().mean() / 255 ** 2 for v in aux_values]).mean()
                if not torch.isfinite(loss):
                    raise FloatingPointError("Nonfinite training loss")
                (loss / config["accumulation"]).backward()
                loss_sum += float(loss.detach()) / config["accumulation"]
                final_sum += float(final_loss.detach()) / config["accumulation"]
            for name, parameter in model.named_parameters():
                if parameter.grad is not None and not torch.isfinite(parameter.grad).all():
                    raise FloatingPointError("Nonfinite gradient: " + name)
            norm = torch.nn.utils.clip_grad_norm_(model.parameters(), config["gradient_clip"])
            if not torch.isfinite(norm):
                raise FloatingPointError("Nonfinite aggregate gradient norm")
            optimizer.step()
            model.project_constraints()
            if not finite_state(model.state_dict()):
                raise FloatingPointError("Nonfinite model parameters")
            step += 1
            phase_updates[str(active)] = phase_updates.get(str(active), 0) + 1
            if active == config["model"]["stages"]:
                full_depth_updates += 1
            recent.append(loss_sum)
            if step % config["log_every"] == 0 or args.smoke:
                event("progress", loss=float(np.mean(recent)), final_loss=final_sum,
                      gradient_norm=float(norm), active_stages=active, lr_factor=factor,
                      auxiliary_weight=auxiliary, images_seen=step * config["accumulation"],
                      learning_rates=[g["lr"] for g in optimizer.param_groups])
                recent = []
                status("training", finite_gradients=True, finite_parameters=True)
                if shutil.disk_usage(args.output).free < config["minimum_disk_gib"] * 1024 ** 3:
                    raise OSError("Disk floor reached")
            if step % config["evaluate_every"] == 0 and limit - elapsed() > 180:
                validation()
                status("training")
            elif step % config["checkpoint_every"] == 0:
                save()
        # The outer lease leaves at least120s after this inner training limit.
        if not stop["requested"]:
            validation()
        if args.smoke and step != 2:
            raise ValueError("GPU smoke did not finish both full-depth updates")
        if not finite_state(optimizer.state_dict()):
            raise FloatingPointError("Nonfinite terminal optimizer")
        save()
        event("complete", reason="signal" if stop["requested"] else
              ("steps" if step >= config["steps"] or args.smoke else "budget"),
              finite_model=finite_state(model.state_dict()), finite_optimizer=finite_state(optimizer.state_dict()))
        status("complete", reason="signal" if stop["requested"] else "budget_or_steps",
               finite_model=True, finite_optimizer=finite_state(optimizer.state_dict()))
    except Exception as error:
        (args.output / "failure.txt").write_text(traceback.format_exc())
        save("failure.pt")
        event("failure", error=str(error))
        status("failed", error=str(error))
        raise
    finally:
        log.close()


if __name__ == "__main__":
    main()

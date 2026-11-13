"""Export a pinned, trusted local BEST as safe long-rollout tensors on CPU.

Raw checkpoints use pickle and must come from this project's trusted local
training process. Public outputs load with weights_only=True and contain no
optimizer, random-generator state, image pixels or machine paths. This tool
never requests CUDA or scores BSD68. Stage diagnostics are synthetic checks,
not evidence that every nominal stage improves denoising quality.
"""
import argparse
from dataclasses import asdict
from datetime import datetime, timezone
import json
import math
from pathlib import Path
import tempfile

import torch

from beltrami_rollout.model import BeltramiOperator, ModelConfig
from beltrami_research.rollout_artifacts import (
    CITATION, FORMAT, NAMESPACES, WARMSTART_SHA256, load_safe, numerical_metadata,
    sha, source_pins, tensor_hash,
)


def finite(value):
    if torch.is_tensor(value):
        return bool(torch.isfinite(value).all())
    if isinstance(value, dict):
        return all(finite(item) for item in value.values())
    if isinstance(value, (tuple, list)):
        return all(finite(item) for item in value)
    return not isinstance(value, float) or math.isfinite(value)


def trusted_best(checkpoint_path, config_path, manifest_path, validation_manifest_path,
                 expected_checkpoint_sha256, expected_step):
    """Verify selection/provenance before unpickling and exporting trusted BEST."""
    checkpoint_path, config_path = Path(checkpoint_path), Path(config_path)
    manifest_path, validation_manifest_path = Path(manifest_path), Path(validation_manifest_path)
    if checkpoint_path.name != "best.pt":
        raise ValueError("Export a selected best.pt, not a terminal or initialization checkpoint")
    if sha(checkpoint_path) != expected_checkpoint_sha256:
        raise ValueError("Trusted checkpoint SHA256 differs from explicit pin")
    config = json.loads(config_path.read_text())
    pins = source_pins()
    if config["source_pins"] != pins:
        raise ValueError("Installed source differs from frozen training configuration")
    # This is the ONLY unsafe deserialization, restricted to the explicitly
    # hash-pinned trusted local checkpoint supplied by the caller.
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    for key, expected in (("config_sha256", sha(config_path)),
                          ("manifest_sha256", sha(manifest_path)),
                          ("validation_manifest_sha256", sha(validation_manifest_path)),
                          ("source_pins", pins), ("config", config)):
        if checkpoint.get(key) != expected:
            raise ValueError("Trusted checkpoint guard differs: " + key)
    stages = config["model"]["stages"]
    if stages not in (5, 32, 64) or config["sigma"] != 25 or config["offset"] != 25:
        raise ValueError("Unsupported cohort depth, sigma or grayscale offset")
    if (not isinstance(expected_step, int) or expected_step <= 0
            or checkpoint.get("step") != expected_step or checkpoint.get("best_step") != expected_step):
        raise ValueError("BEST step differs from explicit selected step")
    minimum = max(20, config["minimum_full_depth_updates"])
    full_updates = checkpoint.get("full_depth_updates")
    if not isinstance(full_updates, int) or not minimum <= full_updates <= expected_step:
        raise ValueError("BEST has insufficient full-target-depth training updates")
    phases = checkpoint.get("phase_updates", {})
    if (phases.get(str(stages)) != full_updates
            or any(not isinstance(count, int) or count < 0 for count in phases.values())
            or sum(phases.values()) != expected_step):
        raise ValueError("BEST phase chronology differs from full-depth update count")
    if not finite(checkpoint["model"]) or not finite(checkpoint["optimizer"]):
        raise FloatingPointError("Trusted BEST model or optimizer is nonfinite")
    if (checkpoint.get("initialization", {}).get("source_sha256") != WARMSTART_SHA256
            or config.get("warmstart_sha256") != WARMSTART_SHA256):
        raise ValueError("BEST warmstart differs from the selected reaction model")
    validation_path = checkpoint_path.parent / f"validation-{expected_step:07d}.json"
    validation = json.loads(validation_path.read_text())
    last = checkpoint.get("last_evaluation", {})
    if (validation.get("step") != expected_step or validation.get("stages") != stages
            or not validation.get("selection_eligible") or validation.get("smoke_one_image")
            or validation.get("full_depth_updates") != full_updates
            or last.get("step") != expected_step or last.get("stages") != stages
            or last.get("mean_psnr") != validation.get("mean_psnr")
            or checkpoint.get("best_psnr") != validation.get("mean_psnr")):
        raise ValueError("BEST does not match its eligible full-depth validation")
    records = json.loads(validation_manifest_path.read_text())["images"]
    records = [record for record in records if record["split"] == "validation"]
    expected_records = {(record["path"], record["pixels_sha256"]) for record in records}
    scores = validation["images"]
    actual_records = {(record["image"], record["pixels_sha256"]) for record in scores}
    if (len(records) != 32 or len(scores) != 32 or len(expected_records) != 32
            or len(actual_records) != 32 or actual_records != expected_records or not finite(scores)):
        raise ValueError("BEST validation must cover the exact32 independent images finitely")
    mean = sum(record["psnr"] for record in scores) / 32
    if not math.isclose(mean, validation["mean_psnr"], rel_tol=0, abs_tol=1e-12):
        raise ValueError("BEST validation mean differs from per-image scores")
    return checkpoint, config, validation_path


@torch.no_grad()
def stage_activity(model, native):
    """Parameter support and actual synthetic fixture updates, not useful-depth proof."""
    current, noisy = native, native
    rows = []
    for index, stage in enumerate(model.stages):
        scale = 1.0 if index < 5 else model.config.refinement_scale
        updated = stage(current, noisy, scale)
        if not torch.isfinite(updated).all():
            raise FloatingPointError(f"Nonfinite synthetic diagnostic at stage{index + 1}")
        projection = stage.projection.weight.detach().double().reshape(-1)
        update = (updated - current).double()
        rows.append({"stage": index + 1, "fixed_update_scale": scale,
                     "projection_l2_norm": float(torch.linalg.vector_norm(projection)),
                     "nonzero_projection_channels": int(torch.count_nonzero(projection)),
                     "width": model.config.width, "reaction": float(stage.reaction),
                     "fixture_update_rms_pixels": float(update.square().mean().sqrt()),
                     "fixture_update_max_abs_pixels": float(update.abs().max()),
                     "exact_identity_on_this_fixture": bool(torch.equal(updated, current))})
        current = updated
    return rows


def export(checkpoint_path, config_path, manifest_path, validation_manifest_path,
           expected_checkpoint_sha256, expected_step, output, report_path):
    """Write a new exclusive asset/report only after CPU identity checks pass."""
    if torch.cuda.is_initialized():
        raise RuntimeError("Use a CPU-only process; CUDA is already initialized")
    output, report_path = Path(output), Path(report_path)
    if output.exists() or report_path.exists():
        raise FileExistsError("Never overwrite safe assets or verification evidence")
    checkpoint, config, validation_path = trusted_best(
        checkpoint_path, config_path, manifest_path, validation_manifest_path,
        expected_checkpoint_sha256, expected_step)
    model = BeltramiOperator(ModelConfig(**config["model"])).eval()
    model.load_state_dict(checkpoint["model"], strict=True)
    state = {name: tensor.detach().cpu().contiguous().clone()
             for name, tensor in model.state_dict().items()}
    value = {"format": FORMAT, "implementation": "beltrami_rollout", "source_pins": source_pins(),
             "model_config": asdict(model.config), "numerics": numerical_metadata(model.config),
             "sigma": 25, "grayscale_offset": 25, "state_dict": state,
             "tensor_sha256": tensor_hash(state),
             "parameters": sum(parameter.numel() for parameter in model.parameters()),
             "training_step": expected_step, "full_depth_updates": checkpoint["full_depth_updates"],
             "phase_updates": checkpoint["phase_updates"],
             "provenance": {"training_checkpoint_sha256": expected_checkpoint_sha256,
                            "training_config_sha256": sha(config_path),
                            "training_manifest_sha256": sha(manifest_path),
                            "validation_manifest_sha256": sha(validation_manifest_path),
                            "selection_validation_sha256": sha(validation_path),
                            "warmstart_sha256": WARMSTART_SHA256, "license": "MIT", "citation": CITATION}}
    output.parent.mkdir(parents=True, exist_ok=True)
    report_path.parent.mkdir(parents=True, exist_ok=True)
    temporary = None
    try:
        with tempfile.NamedTemporaryFile(prefix=".rollout-export-", suffix=".pt", dir=output.parent,
                                         delete=False) as stream:
            temporary = Path(stream.name)
            torch.save(value, stream)
        safe_model, safe = load_safe(temporary)
        if set(state) != set(safe["state_dict"]) or tensor_hash(checkpoint["model"]) != safe["tensor_sha256"]:
            raise ValueError("Safe export tensors differ from trusted BEST")
        for name, tensor in state.items():
            other = safe["state_dict"][name]
            if (tensor.dtype != other.dtype or tensor.shape != other.shape
                    or tensor.numpy().tobytes() != other.numpy().tobytes()):
                raise ValueError("Safe export tensor bytes differ: " + name)
        generator = torch.Generator(device="cpu").manual_seed(20261113)
        checks = []
        activity = None
        with torch.no_grad():
            for shape in ((1, 1, 9, 13), (1, 1, 1, 11), (1, 1, 13, 1)):
                fixture = torch.randn(shape, generator=generator) * 25 + 130
                reference, actual = model(fixture), safe_model(fixture)
                if not torch.isfinite(reference).all() or not torch.isfinite(actual).all():
                    raise FloatingPointError("Nonfinite CPU raw/safe geometry fixture")
                if not torch.equal(reference, actual):
                    raise ValueError("CPU raw/safe predictions are not exact")
                checks.append({"native_shape": list(shape), "finite": True,
                               "prediction_bytes_exact": reference.numpy().tobytes() == actual.numpy().tobytes(),
                               "max_absolute_difference": float((reference - actual).abs().max())})
                if activity is None:
                    activity = stage_activity(model, fixture)
        if torch.cuda.is_initialized():
            raise RuntimeError("CPU export unexpectedly initialized CUDA")
        metadata = {key: item for key, item in value.items() if key != "state_dict"}
        serialized_metadata = json.dumps(metadata, allow_nan=False)
        if any(token in serialized_metadata for token in ("/Users/", "/home/", "/Volumes/", "optimizer", "rng")):
            raise ValueError("Unsafe private metadata in public asset")
        report = {"schema": "deep-beltrami-rollout-export-check-v1",
                  "verified_utc": datetime.now(timezone.utc).isoformat(),
                  "export_sha256": sha(temporary), "export_bytes": temporary.stat().st_size,
                  "tensor_count": len(state), "all_tensor_bytes_preserved": True,
                  "raw_model_and_optimizer_finite": True, "safe_weights_only_reader": True,
                  "cpu_geometry_checks": checks, "stage_activity": activity,
                  "stage_activity_scope": "Synthetic parameter/update diagnostics; not proof of useful denoising steps",
                  "torch": str(torch.__version__), "cpu_threads": torch.get_num_threads(),
                  "cuda_initialized": False, "bsd68_accessed": False, "accuracy_scored": False,
                  "asset_metadata": metadata}
        # Exclusive creation avoids overwriting results even if another exporter
        # races after the initial existence check. Stream copy supports exFAT.
        with temporary.open("rb") as source, output.open("xb") as destination:
            while block := source.read(1024 * 1024):
                destination.write(block)
        with report_path.open("x") as stream:
            json.dump(report, stream, indent=2, allow_nan=False)
            stream.write("\n")
        return report
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("checkpoint", "config", "manifest", "validation-manifest", "output", "report"):
        parser.add_argument("--" + name, type=Path, required=True)
    parser.add_argument("--expected-checkpoint-sha256", required=True)
    parser.add_argument("--expected-step", type=int, required=True)
    args = parser.parse_args()
    torch.set_num_threads(1)
    report = export(args.checkpoint, args.config, args.manifest, args.validation_manifest,
                    args.expected_checkpoint_sha256, args.expected_step, args.output, args.report)
    print(json.dumps(report, indent=2, allow_nan=False))


if __name__ == "__main__":
    main()

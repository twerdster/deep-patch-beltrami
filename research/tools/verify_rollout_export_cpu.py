#!/usr/bin/env python3
"""Verify a selected safe export against trusted raw state on a fixed crop.

Run only after the joint terminal selection is frozen. The raw checkpoint is
unpickled only after its explicit file SHA256 matches. This is a CPU export and
image-adapter identity check, without quality scoring, CUDA or BSD68 access.
"""
import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import platform
import re

import numpy as np
from PIL import Image
import torch

from beltrami_research import rollout_artifacts as artifacts
from beltrami_research.rollout_cli import denoise
from beltrami_rollout.model import BeltramiOperator, ModelConfig
from beltrami_rollout.train import finite_state


CROP = {"row_start": 7, "row_stop": 24, "column_start": 11, "column_stop": 30}
_DIGEST = re.compile(r"[0-9a-f]{64}\Z")


def require(condition, message):
    if not condition:
        raise ValueError(message)


def read(path):
    def reject(value):
        raise ValueError("Non-finite JSON constant: " + value)
    return json.loads(Path(path).read_text(), parse_constant=reject)


def pin(path, expected, description):
    require(isinstance(expected, str) and _DIGEST.fullmatch(expected)
            and artifacts.file_sha(path) == expected, description + " SHA256 differs")


def crop_pixels(root, manifest_path, expected_manifest_sha):
    """Open only the first validation image, after file/manifest containment checks."""
    pin(manifest_path, expected_manifest_sha, "Validation manifest")
    records = [record for record in read(manifest_path)["images"] if record["split"] == "validation"]
    require(len(records) == 32 and len({x["path"] for x in records}) == 32
            and len({x["pixels_sha256"] for x in records}) == 32,
            "Expected the exact32-image validation manifest")
    record = records[0]
    relative = Path(record["path"])
    require(not relative.is_absolute() and ".." not in relative.parts and "\\" not in str(relative),
            "Validation image must be a contained relative path")
    root = Path(root).resolve()
    path = (root / relative).resolve()
    require(path.is_relative_to(root), "Validation image escapes its supplied root")
    pin(path, record["file_sha256"], "Validation image file")
    with Image.open(path) as image:
        require(image.mode == "L", "Validation image must retain original grayscale pixels")
        pixels = np.asarray(image, dtype=np.uint8).copy()
    require(list(pixels.shape) == record["shape_hw"], "Validation image dimensions differ")
    require(hashlib.sha256(pixels.tobytes()).hexdigest() == record["pixels_sha256"],
            "Validation decoded pixel SHA256 differs")
    require(pixels.shape[0] >= CROP["row_stop"] and pixels.shape[1] >= CROP["column_stop"],
            "Validation image is smaller than the fixed crop")
    crop = pixels[CROP["row_start"]:CROP["row_stop"], CROP["column_start"]:CROP["column_stop"]].copy()
    return crop, record


def trusted_raw(path, expected_sha, safe):
    pin(path, expected_sha, "Trusted checkpoint")
    require(Path(path).name == "best.pt", "Verify the selected best.pt checkpoint")
    provenance = safe["provenance"]
    require(expected_sha == provenance["training_checkpoint_sha256"], "Safe/raw checkpoint pin differs")
    # This is trusted-local deserialization, never a public model-loading path.
    raw = torch.load(path, map_location="cpu", weights_only=False)
    require(raw.get("source_pins") == safe["source_pins"] == artifacts.SOURCE_PINS,
            "Raw checkpoint numerical source differs")
    config = raw["config"]
    require(config.get("source_pins") == safe["source_pins"] and config["model"] == safe["model_config"]
            and config.get("sigma") == 25 and config.get("offset") == 25
            and config.get("warmstart_sha256") == artifacts.WARMSTART_SHA256,
            "Raw checkpoint configuration differs")
    for raw_key, public_key in (("config_sha256", "training_config_sha256"),
                                ("manifest_sha256", "training_manifest_sha256"),
                                ("validation_manifest_sha256", "validation_manifest_sha256")):
        require(raw.get(raw_key) == provenance[public_key], "Raw provenance differs: " + raw_key)
    step = safe["training_step"]
    require(raw.get("step") == raw.get("best_step") == step
            and raw.get("full_depth_updates") == safe["full_depth_updates"]
            and raw.get("phase_updates") == safe["phase_updates"]
            and safe["full_depth_updates"] >= max(20, config["minimum_full_depth_updates"]),
            "Raw selected step/full-target chronology differs")
    require(raw.get("initialization", {}).get("source_sha256") == artifacts.WARMSTART_SHA256,
            "Raw reaction warmstart differs")
    require(finite_state(raw["model"]) and finite_state(raw["optimizer"]),
            "Raw model or optimizer is nonfinite")
    validation_path = Path(path).parent / f"validation-{step:07d}.json"
    pin(validation_path, provenance["selection_validation_sha256"], "Selected validation")
    validation = read(validation_path)
    require(validation.get("step") == step and validation.get("stages") == safe["model_config"]["stages"]
            and validation.get("full_depth_updates") == safe["full_depth_updates"]
            and validation.get("selection_eligible") is True and not validation.get("smoke_one_image")
            and validation.get("mean_psnr") == raw.get("best_psnr")
            and raw.get("last_evaluation", {}).get("step") == step
            and raw["last_evaluation"].get("mean_psnr") == validation.get("mean_psnr"),
            "Raw BEST differs from selected eligible full-depth validation")
    return raw


@torch.no_grad()
def adapter_check(raw_model, safe_model, pixels):
    grayscale = pixels.ndim == 2
    channels = pixels[None] if grayscale else pixels.transpose(2, 0, 1)
    native = np.ascontiguousarray(channels.transpose(0, 2, 1)[:, None], dtype=np.float32)
    value = torch.from_numpy(native) + 25
    reference, actual = raw_model(value), safe_model(value)
    require(torch.isfinite(reference).all() and torch.isfinite(actual).all(),
            "Nonfinite raw/safe prediction before clipping")
    require(reference.dtype == actual.dtype == torch.float32 and reference.shape == actual.shape == value.shape,
            "Raw/safe prediction shape or dtype differs")
    reference_bytes, actual_bytes = reference.numpy().tobytes(), actual.numpy().tobytes()
    require(reference_bytes == actual_bytes, "Raw/safe CPU prediction bytes differ")
    expected = np.clip(reference.numpy() - 25, 0, 255)[:, 0].transpose(0, 2, 1)
    expected = expected[0] if grayscale else expected.transpose(1, 2, 0)
    adapted = denoise(safe_model, pixels, offset=25, device="cpu")
    require(adapted.shape == pixels.shape and adapted.dtype == np.float32
            and adapted.tobytes() == expected.tobytes(), "Public image adapter axes/offset/clipping differs")
    return {"pixel_shape": list(pixels.shape), "native_shape": list(value.shape),
            "per_channel_offset": 25, "input_native_float32_sha256": hashlib.sha256(value.numpy().tobytes()).hexdigest(),
            "raw_prediction_sha256": hashlib.sha256(reference_bytes).hexdigest(),
            "safe_prediction_sha256": hashlib.sha256(actual_bytes).hexdigest(),
            "prediction_bytes_exact": True, "adapter_bytes_exact": True, "finite_before_clip": True,
            "clipped_pixels": int(np.count_nonzero((reference.numpy() - 25 < 0) | (reference.numpy() - 25 > 255)))}


def verify(checkpoint, expected_checkpoint_sha256, safe_path, expected_safe_sha256,
           validation_root, validation_manifest, report_path):
    report_path = Path(report_path)
    require(not torch.cuda.is_initialized(), "Use a CPU-only process; CUDA is already initialized")
    if report_path.exists():
        raise FileExistsError("Never overwrite export verification evidence")
    torch.set_num_threads(1)
    pin(checkpoint, expected_checkpoint_sha256, "Trusted checkpoint")
    safe_model, safe = artifacts.load_safe(safe_path, expected_sha256=expected_safe_sha256)
    raw = trusted_raw(checkpoint, expected_checkpoint_sha256, safe)
    state = raw["model"]
    require(set(state) == set(safe["state_dict"]), "Raw/safe tensor names differ")
    for name, tensor in state.items():
        exported = safe["state_dict"][name]
        require(tensor.dtype == exported.dtype and tensor.shape == exported.shape
                and tensor.detach().cpu().contiguous().numpy().tobytes()
                == exported.detach().cpu().contiguous().numpy().tobytes(), "Raw/safe tensor bytes differ: " + name)
    require(artifacts.tensor_hash(state) == safe["tensor_sha256"], "Raw/safe tensor digest differs")
    raw_model = BeltramiOperator(ModelConfig(**safe["model_config"])).eval()
    raw_model.load_state_dict(state, strict=True)
    crop, record = crop_pixels(validation_root, validation_manifest, safe["provenance"]["validation_manifest_sha256"])
    pixels = crop.astype(np.float32)
    rgb = np.stack([pixels, pixels[::-1], pixels[:, ::-1]], axis=-1)
    fixtures = (pixels, pixels[:1], pixels[:, :1], rgb)
    checks = [adapter_check(raw_model, safe_model, image) for image in fixtures]
    pin(checkpoint, expected_checkpoint_sha256, "Trusted checkpoint")
    pin(safe_path, expected_safe_sha256, "Safe artifact")
    pin(validation_manifest, safe["provenance"]["validation_manifest_sha256"], "Validation manifest")
    require(not torch.cuda.is_initialized(), "CPU verification unexpectedly initialized CUDA")
    receipt = {"schema": "deep-beltrami-rollout-cpu-export-verification-v1", "complete": True,
        "verified_utc": datetime.now(timezone.utc).isoformat(), "checkpoint_sha256": expected_checkpoint_sha256,
        "safe_sha256": expected_safe_sha256, "tensor_sha256": safe["tensor_sha256"], "tensor_count": len(state),
        "all_tensor_bytes_preserved": True, "raw_model_and_optimizer_finite": True,
        "source_pins": safe["source_pins"], "model_config": safe["model_config"],
        "training_step": safe["training_step"], "full_depth_updates": safe["full_depth_updates"],
        "phase_updates": safe["phase_updates"], "validation_manifest_sha256": safe["provenance"]["validation_manifest_sha256"],
        "selected_validation_sha256": safe["provenance"]["selection_validation_sha256"],
        "fixture": {"validation_record_index": 0, "image_file_sha256": record["file_sha256"],
                    "decoded_pixels_sha256": record["pixels_sha256"], "image_shape_hw": record["shape_hw"],
                    "crop": CROP, "crop_uint8_sha256": hashlib.sha256(crop.tobytes()).hexdigest(),
                    "rgb_scope": "three deterministic transforms of the same grayscale crop"},
        "checks": checks, "python": platform.python_version(), "torch": str(torch.__version__),
        "numpy": np.__version__, "device": "cpu", "cpu_threads": 1,
        "verification_tool_sha256": artifacts.file_sha(__file__),
        "cuda_initialized": False, "bsd68_accessed": False, "accuracy_scored": False,
        "scope": "Export and image-adapter identity on a fixed hash-verified validation crop; no quality or useful-depth claim"}
    report_path.parent.mkdir(parents=True, exist_ok=True)
    with report_path.open("x") as stream:
        json.dump(receipt, stream, indent=2, allow_nan=False)
        stream.write("\n")
    return receipt


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("checkpoint", "safe", "validation-root", "validation-manifest", "report"):
        parser.add_argument("--" + name, type=Path, required=True)
    for name in ("expected-checkpoint-sha256", "expected-safe-sha256"):
        parser.add_argument("--" + name, required=True)
    args = parser.parse_args()
    result = verify(args.checkpoint, args.expected_checkpoint_sha256, args.safe, args.expected_safe_sha256,
                    args.validation_root, args.validation_manifest, args.report)
    print(json.dumps(result, indent=2, allow_nan=False))


if __name__ == "__main__":
    main()

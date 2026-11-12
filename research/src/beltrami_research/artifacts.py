"""Tensor-only operator artifacts with source and original-checkpoint provenance."""
from dataclasses import asdict
import hashlib
import importlib
import json
from pathlib import Path

import torch


IMPLEMENTATIONS = ("beltrami_operator", "beltrami_precision")


def source_hash(implementation):
    if implementation not in IMPLEMENTATIONS:
        raise ValueError(f"Unsupported implementation: {implementation}")
    module = importlib.import_module(implementation)
    result = hashlib.sha256()
    for path in sorted(Path(module.__file__).parent.glob("*.py")):
        result.update(path.name.encode())
        result.update(path.read_bytes())
    return result.hexdigest()


def tensor_hash(state):
    result = hashlib.sha256()
    for name, tensor in sorted(state.items()):
        result.update(name.encode())
        result.update(str(tensor.dtype).encode())
        result.update(json.dumps(list(tensor.shape)).encode())
        result.update(tensor.detach().cpu().contiguous().numpy().tobytes())
    return result.hexdigest()


def classes(implementation):
    source_hash(implementation)
    module = importlib.import_module(implementation + ".model")
    return module.BeltramiOperator, module.ModelConfig


def load(path, device="cpu"):
    # Public inference assets contain plain metadata and tensors, never optimizer/RNG objects.
    value = torch.load(path, map_location="cpu", weights_only=True)
    if value.get("format") != "beltrami-operator-tensors-v1":
        raise ValueError("Unsupported operator artifact format")
    implementation = value["implementation"]
    if value["source_sha256"] != source_hash(implementation):
        raise ValueError("Artifact numerical source differs from the installed implementation")
    state = value["state_dict"]
    if not state or any(not torch.is_tensor(item) or not torch.isfinite(item).all()
                        for item in state.values()):
        raise FloatingPointError("Artifact contains empty or non-finite tensors")
    if tensor_hash(state) != value["tensor_sha256"]:
        raise ValueError("Learned tensor digest mismatch")
    Operator, Config = classes(implementation)
    model = Operator(Config(**value["model_config"]))
    model.load_state_dict(state, strict=True)
    return model.eval().to(device), value


def trusted_checkpoint(checkpoint_path, config_path, implementation):
    """Check provenance before converting or resuming trusted local training state."""
    expected_source = source_hash(implementation)
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    config_bytes = Path(config_path).read_bytes()
    config = json.loads(config_bytes)
    if checkpoint["config_sha256"] != hashlib.sha256(config_bytes).hexdigest():
        raise ValueError("Training configuration hash differs")
    if checkpoint["config"] != config or checkpoint["source_sha256"] != expected_source:
        raise ValueError("Training configuration/source identity differs")
    from deep_patch_beltrami.assets import metadata
    name = checkpoint["initialization"]["source_name"]
    original = {item["name"]: item["sha256"] for item in metadata("models.json")["files"]}
    if original.get(name) != checkpoint["initialization"]["source_sha256"]:
        raise ValueError("Original initialization does not match the published historical checkpoint")
    if checkpoint["initialization"]["source_sigma"] != config["sigma"]:
        raise ValueError("Original initialization sigma differs")
    if name != f"PB_{config['sigma']}_80_5__iter_40000.caffemodel":
        raise ValueError("Original initialization file differs from the declared sigma")
    return checkpoint, config


def finite(value):
    if torch.is_tensor(value):
        return bool(torch.isfinite(value).all())
    if isinstance(value, dict):
        return all(finite(item) for item in value.values())
    if isinstance(value, (list, tuple)):
        return all(finite(item) for item in value)
    if isinstance(value, float):
        import math
        return math.isfinite(value)
    return True


def validate_resume(checkpoint_path, config_path, implementation, manifest,
                    validation_manifest, model_dir):
    checkpoint, config = trusted_checkpoint(checkpoint_path, config_path, implementation)
    for key, path in (("manifest_sha256", manifest),
                      ("validation_manifest_sha256", validation_manifest)):
        if checkpoint[key] != hashlib.sha256(Path(path).read_bytes()).hexdigest():
            raise ValueError(f"Resume {key} differs")
    if not finite(checkpoint["model"]) or not finite(checkpoint["optimizer"]):
        raise FloatingPointError("Resume model or optimizer state is not finite")
    if (not isinstance(checkpoint["step"], int) or checkpoint["step"] < 0
            or not isinstance(checkpoint["gpu_seconds"], (int, float))
            or not finite(checkpoint["gpu_seconds"]) or checkpoint["gpu_seconds"] < 0
            or not finite(checkpoint["best_psnr"])):
        raise ValueError("Resume step, elapsed time or best validation score is invalid")
    directory = Path(model_dir)
    if (directory / "PBSolve_multi").is_dir():
        directory = directory / "PBSolve_multi"
    path = directory / checkpoint["initialization"]["source_name"]
    if hashlib.sha256(path.read_bytes()).hexdigest() != checkpoint["initialization"]["source_sha256"]:
        raise ValueError("Resume original model bytes differ")
    Operator, Config = classes(implementation)
    model = Operator(Config(**config["model"]))
    model.load_state_dict(checkpoint["model"], strict=True)
    return {"implementation": implementation, "source_sha256": checkpoint["source_sha256"],
            "step": checkpoint["step"], "gpu_seconds": checkpoint["gpu_seconds"]}


def export(checkpoint_path, config_path, implementation, output):
    """Export only a trusted locally generated training checkpoint."""
    checkpoint, config = trusted_checkpoint(checkpoint_path, config_path, implementation)
    Operator, Config = classes(implementation)
    model = Operator(Config(**config["model"]))
    model.load_state_dict(checkpoint["model"], strict=True)
    state = {name: tensor.detach().cpu().contiguous().clone()
             for name, tensor in model.state_dict().items()}
    if any(not torch.isfinite(tensor).all() for tensor in state.values()):
        raise FloatingPointError("Cannot publish non-finite learned tensors")
    value = {"format": "beltrami-operator-tensors-v1", "implementation": implementation,
             "source_sha256": checkpoint["source_sha256"], "model_config": asdict(model.config),
             "sigma": config["sigma"], "grayscale_offset": config["offset"],
             "state_dict": state, "tensor_sha256": tensor_hash(state),
             "parameters": sum(parameter.numel() for parameter in model.parameters()),
             "training_step": checkpoint["step"],
             "provenance": {
                 "training_checkpoint_sha256": hashlib.sha256(Path(checkpoint_path).read_bytes()).hexdigest(),
                 "training_config_sha256": checkpoint["config_sha256"],
                 "training_manifest_sha256": checkpoint["manifest_sha256"],
                 "validation_manifest_sha256": checkpoint["validation_manifest_sha256"],
                 "original_checkpoint_sha256": checkpoint["initialization"]["source_sha256"],
                 "license": "MIT", "citation": "Wetzler, Topics in Computational Geometric Vision, 2017, Chapter 4 Section 4.5"}}
    with Path(output).open("xb") as stream:
        torch.save(value, stream)
    return {key: item for key, item in value.items() if key != "state_dict"}

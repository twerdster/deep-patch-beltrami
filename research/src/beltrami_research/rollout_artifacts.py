"""Strict tensor-only inference assets for the frozen long-rollout operator.

This reader never deserializes raw training state. File hashes establish an
artifact's identity when an external expected hash is supplied; the embedded
tensor digest detects corruption but is not an authenticity signature.
"""
from dataclasses import asdict, fields
import hashlib
import importlib
import json
from pathlib import Path
import re

import torch

from beltrami_rollout.model import BeltramiOperator, ModelConfig


FORMAT = "deep-beltrami-rollout-tensors-v1"
SOURCE_PINS = {
    "beltrami_operator": "6c2e5996b03d1521e56f098ec839efc7eb6ba366e3f0f0647f05461177c2c7c1",
    "beltrami_precision": "300c797fde457bb053c37c449382ec1ff59ced938fdd35b35e4b27e760caeae2",
    "beltrami_rollout": "dae895a39713ee4d3d6d64c35f8c3a9320c8ae98e8e8315c3c71585c7c2dbdad",
}
NAMESPACES = tuple(SOURCE_PINS)
WARMSTART_SHA256 = "9a085c0745136eaee86bfaf0b2e1eb31926eb3bb0526169134fa3a89cf2cfed5"
CITATION = "Aaron Wetzler, Topics in Computational Geometric Vision, PhD thesis,2017,Chapter4."
_KEYS = {"format", "implementation", "source_pins", "model_config", "numerics",
         "sigma", "grayscale_offset", "state_dict", "tensor_sha256", "parameters",
         "training_step", "full_depth_updates", "phase_updates", "provenance"}
_PROVENANCE_HASHES = {"training_checkpoint_sha256", "training_config_sha256",
                      "training_manifest_sha256", "validation_manifest_sha256",
                      "selection_validation_sha256", "warmstart_sha256"}


def file_sha(path):
    """Hash the full artifact, independently of the embedded tensor digest."""
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        while block := stream.read(1024 * 1024):
            digest.update(block)
    return digest.hexdigest()


sha = file_sha


def source_pins():
    """Hash filenames and bytes exactly as the frozen training guards do."""
    pins = {}
    for namespace in NAMESPACES:
        directory = Path(importlib.import_module(namespace).__file__).parent
        digest = hashlib.sha256()
        for path in sorted(directory.glob("*.py")):
            digest.update(path.name.encode())
            digest.update(path.read_bytes())
        pins[namespace] = digest.hexdigest()
    return pins


def tensor_hash(state):
    """Canonical sorted names, dtype, JSON shape and contiguous payload bytes."""
    digest = hashlib.sha256()
    for name, tensor in sorted(state.items()):
        digest.update(name.encode())
        digest.update(str(tensor.dtype).encode())
        digest.update(json.dumps(list(tensor.shape)).encode())
        digest.update(tensor.detach().cpu().contiguous().numpy().tobytes())
    return digest.hexdigest()


def numerical_metadata(config):
    return {"parameters": "float32", "dense_convolutions": "float32",
            "metric_convolutions_determinant_flux": "float64", "metric_parameterization": "softplus",
            "native_axes": "N,C,W,H", "intensity_scale": "0–255", "prefix_stages": 5,
            "nominal_stages": config.stages, "refinement_scale": config.refinement_scale,
            "output_clipping_in_model": False, "determinant_clipping": False,
            "global_stability_guarantee": False}


def _positive_integer(value):
    return type(value) is int and value > 0


def _sha256(value):
    return isinstance(value, str) and re.fullmatch(r"[0-9a-f]{64}", value) is not None


def _configuration(value):
    expected_keys = {field.name for field in fields(ModelConfig)}
    if not isinstance(value, dict) or set(value) != expected_keys:
        raise ValueError("Safe model configuration has unexpected or missing fields")
    if (type(value["stages"]) is not int or value["stages"] not in (5, 32, 64)
            or type(value["width"]) is not int or value["width"] != 80
            or type(value["checkpoint_blocks"]) is not int or value["checkpoint_blocks"] != 4
            or value["zero_dc"] is not False or value["reaction"] is not True
            or type(value["step_scale"]) is not float or value["step_scale"] != 1000.0
            or type(value["refinement_scale"]) is not float
            or value["refinement_scale"] != 1.0 / max(1, value["stages"] - 5)):
        raise ValueError("Safe model configuration differs from the frozen cohort")
    config = ModelConfig(**value)
    if asdict(config) != value:
        raise ValueError("Safe model configuration is not canonical")
    return config


def _metadata(value, config):
    if (type(value["sigma"]) is not int or value["sigma"] != 25
            or type(value["grayscale_offset"]) is not int or value["grayscale_offset"] != 25):
        raise ValueError("Safe model has invalid sigma or intensity offset")
    step, full = value["training_step"], value["full_depth_updates"]
    if not _positive_integer(step) or not _positive_integer(full) or not 20 <= full <= step:
        raise ValueError("Safe model has insufficient or invalid full-depth training")
    phases = value["phase_updates"]
    allowed_phases = {5: {"5"}, 32: {"16", "32"}, 64: {"16", "32", "64"}}[config.stages]
    if (not isinstance(phases, dict) or not phases or not set(phases) <= allowed_phases
            or any(type(count) is not int or count < 0 for count in phases.values())
            or sum(phases.values()) != step or phases.get(str(config.stages)) != full):
        raise ValueError("Safe model phase chronology differs from its training counts")
    provenance = value["provenance"]
    if (not isinstance(provenance, dict)
            or set(provenance) != _PROVENANCE_HASHES | {"license", "citation"}
            or any(not _sha256(provenance[key]) for key in _PROVENANCE_HASHES)
            or provenance["warmstart_sha256"] != WARMSTART_SHA256
            or provenance["license"] != "MIT" or provenance["citation"] != CITATION):
        raise ValueError("Safe model has invalid or private provenance metadata")
    if value["numerics"] != numerical_metadata(config):
        raise ValueError("Safe model numerical metadata differs from configuration")


def load_safe(path, device="cpu", expected_sha256=None):
    """Load only safe tensors, enforce source/protocol guards, then move device.

    Pass a separately obtained full-file SHA256 (for example from a published
    manifest) to pin the actual released asset before deserialization.
    """
    if expected_sha256 is not None:
        if not _sha256(expected_sha256) or file_sha(path) != expected_sha256:
            raise ValueError("Safe artifact SHA256 differs from the external pin")
    pins = source_pins()
    if pins != SOURCE_PINS:
        raise ValueError("Installed source differs from the frozen numerical namespaces")
    value = torch.load(path, map_location="cpu", weights_only=True)
    if (not isinstance(value, dict) or set(value) != _KEYS
            or value["format"] != FORMAT or value["implementation"] != "beltrami_rollout"):
        raise ValueError("Unsupported safe long-rollout format")
    if value["source_pins"] != pins:
        raise ValueError("Installed source differs from safe model source pins")
    config = _configuration(value["model_config"])
    _metadata(value, config)
    state = value["state_dict"]
    if not isinstance(state, dict) or not state or any(not isinstance(name, str) for name in state):
        raise ValueError("Safe model has invalid tensor names")
    if any(type(tensor) is not torch.Tensor or tensor.dtype != torch.float32
           or tensor.layout != torch.strided or not torch.isfinite(tensor).all()
           for tensor in state.values()):
        raise FloatingPointError("Safe model tensors must be finite dense float32")
    if not _sha256(value["tensor_sha256"]) or tensor_hash(state) != value["tensor_sha256"]:
        raise ValueError("Safe model exact learned tensor digest differs")
    model = BeltramiOperator(config)
    expected = model.state_dict()
    if (set(state) != set(expected)
            or any(state[name].shape != expected[name].shape for name in expected)):
        raise ValueError("Safe model tensor names or shapes differ from configuration")
    if (not _positive_integer(value["parameters"])
            or sum(parameter.numel() for parameter in model.parameters()) != value["parameters"]):
        raise ValueError("Safe model parameter count differs")
    model.load_state_dict(state, strict=True)
    return model.eval().to(device), value


load = load_safe

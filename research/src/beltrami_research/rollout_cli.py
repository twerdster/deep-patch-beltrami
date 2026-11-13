"""Denoise grayscale/RGB images with a tensor-only long-rollout model.

Each channel is an independent grayscale observation, using the fixed +25
intensity offset. Inputs remain unclipped float32 on the 0–255 scale; only
finite predictions have the offset removed and are finally clipped.
"""
import argparse
import json
from pathlib import Path

import numpy as np
from PIL import Image, ImageOps
import torch

from .rollout_artifacts import load_safe


def image_pixels(path):
    """Read an upright 8-bit grayscale or RGB image as 0–255 float32 pixels."""
    with Image.open(path) as original:
        image = ImageOps.exif_transpose(original)
        grayscale = image.mode in ("1", "L", "I", "I;16", "F")
        return np.asarray(image.convert("L" if grayscale else "RGB"), dtype=np.float32)


@torch.no_grad()
def denoise(model, pixels, offset=25, device="cpu"):
    """Adapt H,W or H,W,3 pixels to independent native N,1,W,H channels.

    The returned array is float32. Negative and greater-than-255 noisy input
    values are allowed; no input clipping or integer quantization occurs here.
    """
    pixels = np.asarray(pixels)
    if (pixels.ndim not in (2, 3) or pixels.shape[0] < 1 or pixels.shape[1] < 1
            or (pixels.ndim == 3 and pixels.shape[2] != 3)
            or pixels.dtype.kind not in "uif"):
        raise ValueError("Use a nonempty grayscale H,W or RGB H,W,3 numeric array")
    if type(offset) not in (int, float) or offset != 25:
        raise ValueError("Long-rollout inference requires the fixed +25 channel offset")
    if not np.isfinite(pixels).all():
        raise FloatingPointError("Non-finite input pixels")
    pixels = pixels.astype(np.float32)
    if not np.isfinite(pixels).all():
        raise FloatingPointError("Input pixels overflow float32")
    grayscale = pixels.ndim == 2
    channels = pixels[None] if grayscale else pixels.transpose(2, 0, 1)
    native = np.ascontiguousarray(channels.transpose(0, 2, 1)[:, None])
    value = torch.from_numpy(native).to(device) + offset
    if not torch.isfinite(value).all():
        raise FloatingPointError("Non-finite offset input pixels")
    prediction = model(value)
    if (not isinstance(prediction, torch.Tensor) or prediction.shape != value.shape
            or prediction.dtype != torch.float32):
        raise ValueError("Operator prediction must preserve native shape and float32 dtype")
    if not torch.isfinite(prediction).all():
        raise FloatingPointError("Non-finite operator prediction before clipping")
    output = (prediction - offset).clamp(0, 255).cpu().numpy()[:, 0].transpose(0, 2, 1)
    return output[0] if grayscale else output.transpose(1, 2, 0)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("input", type=Path)
    parser.add_argument("output", type=Path, help="New PNG path; existing files are never overwritten")
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--expected-sha256", help="Optional independently obtained full model SHA256")
    parser.add_argument("--device", default="cpu", choices=("cpu", "cuda", "cuda:0"))
    parser.add_argument("--threads", type=int, default=1, help="CPU worker threads (default:1)")
    args = parser.parse_args(argv)
    if args.threads < 1:
        parser.error("--threads must be positive")
    if args.output.suffix.lower() != ".png":
        parser.error("The output must use .png for lossless 8-bit pixels")
    if args.output.exists():
        raise FileExistsError("Never overwrite an existing image")
    torch.set_num_threads(args.threads)
    model, metadata = load_safe(args.checkpoint, args.device, args.expected_sha256)
    pixels = image_pixels(args.input)
    result = denoise(model, pixels, metadata["grayscale_offset"], args.device)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("xb") as stream:
        Image.fromarray(np.rint(result).astype(np.uint8)).save(stream, format="PNG")
    print(json.dumps({"sigma": metadata["sigma"], "model": metadata["model_config"],
                      "parameters": metadata["parameters"], "shape": list(result.shape),
                      "per_channel_offset": metadata["grayscale_offset"],
                      "output": "rounded and clipped 8-bit PNG"}))


if __name__ == "__main__":
    main()

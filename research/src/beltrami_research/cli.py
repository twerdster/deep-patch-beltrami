"""Denoise an image with an explicitly supplied tensor-only experimental model."""
import argparse
import json
from pathlib import Path

import numpy as np
from PIL import Image
import torch

from .artifacts import load


def image_pixels(path):
    with Image.open(path) as image:
        gray = image.mode in ("1", "L", "I", "I;16", "F")
        return np.asarray(image.convert("L" if gray else "RGB"), dtype=np.float32)


@torch.no_grad()
def denoise(model, pixels, offset, device):
    grayscale = pixels.ndim == 2
    channels = pixels[None] if grayscale else pixels.transpose(2, 0, 1)
    native = np.ascontiguousarray(channels.transpose(0, 2, 1)[:, None], dtype=np.float32)
    applied_offset = offset if grayscale else 0
    value = torch.from_numpy(native + applied_offset).to(device)
    prediction = model(value)
    if not torch.isfinite(prediction).all():
        raise FloatingPointError("Non-finite operator prediction before clipping")
    output = (prediction - applied_offset).clamp(0, 255).cpu().numpy()[:, 0].transpose(0, 2, 1)
    return output[0] if grayscale else output.transpose(1, 2, 0)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("input", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--device", default="cpu", choices=("cpu", "cuda", "cuda:0"))
    args = parser.parse_args()
    model, metadata = load(args.checkpoint, args.device)
    pixels = image_pixels(args.input)
    result = denoise(model, pixels, metadata["grayscale_offset"], args.device)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(np.rint(result).astype(np.uint8)).save(args.output)
    print(json.dumps({"sigma": metadata["sigma"], "model": metadata["model_config"],
                      "parameters": metadata["parameters"], "shape": list(result.shape)}))


if __name__ == "__main__":
    main()

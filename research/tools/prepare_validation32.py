"""Fetch pinned BSDS validation sources and verify every generated grayscale PNG."""
import argparse
import hashlib
import io
import json
from pathlib import Path

import numpy as np
from PIL import Image

from deep_patch_beltrami.assets import download


ROOT = Path(__file__).resolve().parents[1]


def encode(source, expected):
    import cv2
    with Image.open(source) as image:
        rgb = np.asarray(image.convert("RGB"))
    gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)
    if list(gray.shape) != expected["shape_hw"]:
        raise ValueError("Decoded source dimensions differ from the pinned validation image")
    if hashlib.sha256(gray.tobytes()).hexdigest() != expected["pixels_sha256"]:
        raise ValueError("Decoded grayscale pixels differ; check decoder/conversion versions")
    output = io.BytesIO()
    Image.fromarray(gray).save(output, format="PNG")
    data = output.getvalue()
    if hashlib.sha256(data).hexdigest() != expected["file_sha256"]:
        raise ValueError("PNG encoding differs despite matching pixels; use Pillow 13 or supply the exact PNGs")
    return data


def prepare(output, cache):
    sources = json.loads((ROOT / "manifests/validation32-sources.json").read_text())
    targets = json.loads((ROOT / "data/validation32.json").read_text())
    by_id = {Path(record["path"]).stem: record for record in targets["images"]}
    if len(by_id) != 32 or len(sources["images"]) != 32:
        raise ValueError("Expected the complete 32-image independent validation split")
    output.mkdir(parents=True, exist_ok=True)
    cache.mkdir(parents=True, exist_ok=True)
    for record in sources["images"]:
        expected = by_id[Path(record["path"]).stem]
        target = output / expected["path"]
        if target.exists():
            if hashlib.sha256(target.read_bytes()).hexdigest() != expected["file_sha256"]:
                raise FileExistsError(f"Preserving a conflicting existing image: {target.name}")
            continue
        source = download(record["url"], cache / Path(record["path"]).name, record["sha256"])
        data = encode(source, expected)
        with target.open("xb") as stream:
            stream.write(data)
    print(json.dumps({"images": 32, "all_file_and_pixel_hashes_verified": True}))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=ROOT / "data/validation32")
    parser.add_argument("--cache", type=Path, default=ROOT / "downloads/validation32")
    args = parser.parse_args()
    prepare(args.output, args.cache)


if __name__ == "__main__":
    main()

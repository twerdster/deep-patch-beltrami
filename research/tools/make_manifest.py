"""Record one explicitly supplied split without altering or copying photographs."""
import argparse
import hashlib
import json
from pathlib import Path

from beltrami_operator.data import image_array


def build(root, split, exposure):
    records = []
    for path in sorted(root.rglob("*.png")):
        relative = path.relative_to(root)
        if any(part.startswith(".") for part in relative.parts):
            continue
        image = image_array(path)
        records.append({"path": relative.as_posix(), "shape_hw": list(image.shape),
                        "file_sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
                        "pixels_sha256": hashlib.sha256(image.tobytes()).hexdigest(),
                        "split": split})
    if not records:
        raise ValueError("No visible grayscale PNG images found")
    return {"schema": 1, "images": records, "prior_checkpoint_exposure": exposure,
            "intensity_scale": "0-255 uint8 clean; float32 unclipped noisy",
            "tensor_axes": "N,C,W,H", "split_policy": "Explicit separately supplied split"}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--split", choices=("train", "validation"), required=True)
    parser.add_argument("--prior-checkpoint-exposure", choices=("yes", "no", "unknown"),
                        default="unknown")
    args = parser.parse_args()
    value = build(args.root, args.split, args.prior_checkpoint_exposure)
    with args.output.open("x") as stream:
        stream.write(json.dumps(value, indent=2) + "\n")
    print(json.dumps({"images": len(value["images"]), "split": args.split}))


if __name__ == "__main__":
    main()

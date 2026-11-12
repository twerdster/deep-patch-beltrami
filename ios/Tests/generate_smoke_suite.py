#!/usr/bin/env python3
"""Create native smoke-test references for all eight models and RGB handling.

Only procedural arrays and the released model tensors are used. These eleven
cases check model selection, grayscale offsets, natural image axes and RGB
channel handling; they are not the full 544-case BSD68 benchmark.
"""

import argparse
import hashlib
import json
from pathlib import Path
import platform
import time


def save_array(path, array):
    raw = array.astype("<f4", copy=False).tobytes(order="C")
    path.write_bytes(raw)
    return {"file": path.name, "bytes": len(raw),
            "sha256": hashlib.sha256(raw).hexdigest()}


def write_manifest(output, manifest):
    temporary = output / "manifest.json.partial"
    temporary.write_text(json.dumps(manifest, indent=2) + "\n")
    temporary.replace(output / "manifest.json")


def generate(args):
    import numpy as np
    import torch
    from deep_patch_beltrami import assets
    from deep_patch_beltrami.inference import DeepBeltrami, configure
    from deep_patch_beltrami.reproduce import OFFSET_FACTORS, SIGMAS

    model_dir = assets.models(cache_dir=args.cache_dir, asset_dir=args.asset_dir,
                              model_dir=args.model_dir)
    configure("cpu", args.threads)
    cases = [(f"sigma{sigma}-gray-h193-w257", sigma, 193, 257, 1)
             for sigma in SIGMAS]
    cases += [(f"sigma25-rgb-h{height}-w{width}", 25, height, width, 3)
              for height, width in ((2, 3), (79, 111), (193, 257))]
    args.output.mkdir(parents=True, exist_ok=True)
    manifest = {"schema": 1, "suite": "procedural-deep-all-models-rgb-v1",
                "source": "procedural; no photographs",
                "array_layout": "natural row-major H,W,channels; grayscale H,W",
                "dtype": "little-endian float32", "intensity_peak": 255,
                "native_model_layout": "N,C,W,H", "input_clipping": None,
                "prediction_clipping": [0, 255],
                "reference_backend": "PyTorch CPU float32",
                "offset_protocol": "archived grayscale; zero RGB",
                "threads": args.threads, "python": platform.python_version(),
                "numpy": np.__version__, "torch": torch.__version__,
                "expected_case_count": len(cases), "complete": False, "cases": []}
    write_manifest(args.output, manifest)
    models = {}
    checkpoint_hashes = {}
    for case_id, sigma, height, width, channels in cases:
        if sigma not in models:
            models[sigma] = DeepBeltrami(model_dir, sigma, "cpu")
            checkpoint = model_dir / "PBSolve_multi" / f"PB_{sigma}_80_5__iter_40000.caffemodel"
            checkpoint_hashes[sigma] = assets.sha256(checkpoint)
        row, col = np.mgrid[:height, :width]
        base = (45 + 90*row/max(1, height-1) + 50*col/max(1, width-1)
                + 18*np.sin(row/7)*np.cos(col/11)).astype(np.float32)
        if channels == 1:
            clean = base
        else:
            # Deliberately distinct channels expose ordering or shared-channel errors.
            clean = np.stack((base, 210 - .65*base + 9*np.sin(col/5),
                              25 + .8*base + 11*np.cos(row/9)), axis=-1).astype(np.float32)
        seed = 2017 + sigma*100000 + height*100 + width + channels
        rng = np.random.default_rng(seed)
        noisy = np.ascontiguousarray(clean + rng.normal(0, sigma, clean.shape).astype(np.float32))
        # Include values outside the display range so input clipping is detectable.
        noisy.flat[0] = -7.25
        noisy.flat[-1] = 267.5
        if channels == 1:
            native = np.ascontiguousarray(noisy.T[None, None])
            offset = float(sigma*OFFSET_FACTORS[sigma])
        else:
            native = np.ascontiguousarray(noisy.transpose(2, 1, 0)[None])
            offset = 0.0
        started = time.perf_counter()
        predicted = models[sigma](native, offset=offset, debug=True)
        elapsed = time.perf_counter() - started
        expected = (predicted[0, 0].T if channels == 1
                    else predicted[0].transpose(2, 1, 0))
        expected = np.ascontiguousarray(expected, dtype=np.float32)
        if expected.shape != noisy.shape or not np.isfinite(expected).all():
            raise FloatingPointError(f"Invalid reference prediction: {case_id}")
        directory = args.output / case_id
        directory.mkdir(exist_ok=True)
        record = {"id": case_id, "sigma": sigma, "width": width, "height": height,
                  "channels": channels, "offset": offset, "seed": seed,
                  "construction": "Asymmetric procedural ramp/sinusoid plus float32 AWGN; first/last scalar forced outside display range",
                  "input": save_array(directory / "input.f32", noisy),
                  "expected": save_array(directory / "expected.f32", expected),
                  "native_input_shape": list(native.shape),
                  "native_input_sha256": hashlib.sha256(native.tobytes(order="C")).hexdigest(),
                  "natural_input_min": float(noisy.min()), "natural_input_max": float(noisy.max()),
                  "expected_min": float(expected.min()), "expected_max": float(expected.max()),
                  "checkpoint_file": f"PB_{sigma}_80_5__iter_40000.caffemodel",
                  "checkpoint_sha256": checkpoint_hashes[sigma],
                  "all_finite": True, "reference_seconds": elapsed}
        (directory / "case.json").write_text(json.dumps(record, indent=2) + "\n")
        manifest["cases"].append(record)
        write_manifest(args.output, manifest)
        print(json.dumps({"id": case_id, "offset": offset,
                          "seconds": elapsed, "finite": True}), flush=True)
    manifest["complete"] = len(manifest["cases"]) == manifest["expected_case_count"]
    write_manifest(args.output, manifest)
    return manifest


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=Path("results/ios-deep-smoke-fixtures"))
    parser.add_argument("--model-dir", type=Path, help="Existing verified extracted models")
    parser.add_argument("--cache-dir", type=Path, help="Model download/extraction cache")
    parser.add_argument("--asset-dir", type=Path, help="Directory containing the released model archive")
    parser.add_argument("--threads", type=int, default=4)
    args = parser.parse_args(argv)
    if args.threads < 1:
        parser.error("--threads must be positive")
    try:
        generate(args)
    except ImportError as error:
        parser.exit(1, f"Install the package first: python -m pip install -e .\n{error}\n")
    except (OSError, ValueError, RuntimeError, FloatingPointError) as error:
        parser.exit(1, f"{error}\n")


if __name__ == "__main__":
    main()

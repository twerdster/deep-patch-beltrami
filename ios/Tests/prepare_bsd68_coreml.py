#!/usr/bin/env python3
"""Prepare pinned BSD68 float32 inputs for the native macOS Core ML runner.

Use an existing Python environment with this project installed. Full preparation
writes about 380 MB of arrays; an uncached original-noise asset needs another
324 MB. Keep --work-dir on suitable storage. No fitting or training occurs.
"""
from pathlib import Path
import argparse
import hashlib
import json
import time

import numpy as np

from deep_patch_beltrami import assets
from deep_patch_beltrami.reproduce import PublicBSD68, SIGMAS, OFFSET_FACTORS, PUBLISHED


def sigma_list(value):
    try:
        selected = tuple(int(item) for item in value.split(","))
    except ValueError as error:
        raise argparse.ArgumentTypeError("Use comma-separated integer sigmas") from error
    if not selected or len(set(selected)) != len(selected) or any(s not in SIGMAS for s in selected):
        raise argparse.ArgumentTypeError("Select unique sigmas from 5,10,15,20,25,30,40,50")
    return selected


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--work-dir", type=Path, required=True)
    parser.add_argument("--cache-dir", type=Path, help="Defaults to WORK_DIR/cache")
    parser.add_argument("--asset-dir", type=Path, help="Existing pinned release assets, including original-noise HDF5")
    parser.add_argument("--clean-dir", type=Path, help="Existing pinned BSD68 PNGs")
    parser.add_argument("--limit", type=int, help="Images per sigma for a smoke run; default is all 68")
    parser.add_argument("--sigmas", type=sigma_list, default=SIGMAS, help="Comma-separated sigmas; default is all eight")
    args = parser.parse_args()
    if args.limit is not None and not 1 <= args.limit <= 68:
        parser.error("--limit must be between 1 and 68")
    work = args.work_dir.expanduser().resolve()
    destination = work / "fixtures"
    destination.mkdir(parents=True, exist_ok=True)
    cache = args.cache_dir if args.cache_dir is not None else work / "cache"
    suite = PublicBSD68(cache_dir=cache, asset_dir=args.asset_dir, clean_dir=args.clean_dir)
    repo = Path(__file__).resolve().parents[2]
    reference_path = repo / "docs/results/bsd68.json"
    reference = json.loads(reference_path.read_text())
    references = {(c["sigma"], c["name"]): c for c in reference["cases"]}
    names = suite.names[:args.limit] if args.limit is not None else suite.names
    manifest = {
        "schema": 1, "dataset": "bsd68", "complete": False,
        "array_layout": "natural row-major H,W little-endian float32",
        "original_input_layout": "N,C,W,H", "offset_protocol": "archived",
        "sigmas": list(args.sigmas), "case_count": len(names)*len(args.sigmas),
        "published": {str(k): v for k, v in PUBLISHED["bsd68"].items()},
        "reference_backend": "validated PyTorch report",
        "reference_report_sha256": assets.sha256(reference_path),
        "noise_asset_sha256": assets.metadata("release-assets.json")["assets"]["bsd68_noise"]["sha256"],
        "labels": {}, "cases": [], "source_hashes": {},
    }
    for relative in (
        "src/deep_patch_beltrami/reproduce.py", "src/deep_patch_beltrami/assets.py",
        "ios/DeepBeltramiCoreML/DeepBeltramiCoreML.swift",
        "ios/Tests/prepare_bsd68_coreml.py", "ios/Tests/RunBSD68CoreML.swift",
        "ios/Tests/run_bsd68_coreml.sh",
    ):
        path = repo / relative
        if path.is_file():
            manifest["source_hashes"][relative] = assets.sha256(path)
    started = time.perf_counter()
    try:
        for sigma in args.sigmas:
            for name in names:
                label, noisy, archived, digest = suite.case(sigma, name)
                if not np.isfinite(noisy).all():
                    raise FloatingPointError(f"Non-finite original input: {name}, sigma{sigma}")
                width, height = noisy.shape[2:]
                natural = np.ascontiguousarray(noisy[0, 0].T, dtype="<f4")
                relative = Path("inputs") / str(sigma) / (name+".f32")
                path = destination / relative
                path.parent.mkdir(parents=True, exist_ok=True)
                natural.tofile(path)
                if name not in manifest["labels"]:
                    label_relative = Path("labels") / (name+".f32")
                    label_path = destination / label_relative
                    label_path.parent.mkdir(parents=True, exist_ok=True)
                    np.ascontiguousarray(label[0, 0].T, dtype="<f4").tofile(label_path)
                    manifest["labels"][name] = {
                        "path": str(label_relative), "natural_sha256": assets.sha256(label_path),
                        "native_sha256": assets.array_sha256(label), "width": width, "height": height,
                    }
                ref = references[sigma, name]
                if digest != ref["noisy_native_sha256"]:
                    raise ValueError("Validated reference and original input hashes differ")
                manifest["cases"].append({
                    "name": name, "sigma": sigma, "width": width, "height": height,
                    "offset": sigma*OFFSET_FACTORS[sigma], "input": str(relative),
                    "input_natural_sha256": assets.sha256(path), "noisy_native_sha256": digest,
                    "label": manifest["labels"][name]["path"],
                    "label_natural_sha256": manifest["labels"][name]["natural_sha256"],
                    "reference_psnr": ref["psnr"], "archived_psnr": archived,
                })
            print(json.dumps({"sigma": sigma, "prepared_cases": len(manifest["cases"])}), flush=True)
    finally:
        suite.close()
    manifest.update(
        complete=True, all_original_input_hashes_verified=True,
        full_published_suite=set(args.sigmas) == set(SIGMAS) and len(names) == 68,
        preparation_seconds=time.perf_counter()-started,
        input_bytes=sum((destination/c["input"]).stat().st_size for c in manifest["cases"]),
        label_bytes=sum((destination/c["path"]).stat().st_size for c in manifest["labels"].values()),
    )
    encoded = (json.dumps(manifest, indent=2)+"\n").encode()
    (destination / "manifest.json").write_bytes(encoded)
    print(json.dumps({"prepared_cases": len(manifest["cases"]), "complete": True,
                      "manifest_sha256": hashlib.sha256(encoded).hexdigest()}), flush=True)


if __name__ == "__main__":
    main()

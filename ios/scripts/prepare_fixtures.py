"""Generate procedural float-array fixtures for native Deep Patch-Beltrami checks.

Run from the repository root after installing the Python package. Only the
released model tensors are downloaded; no benchmark photographs are needed.
"""
from pathlib import Path
import argparse
import hashlib
import json
import platform
import runpy
import shutil
import tempfile
import time


SIZES = ((1, 1), (1, 17), (19, 1), (2, 3), (7, 9), (31, 47),
         (127, 129), (129, 127), (128, 128), (256, 256), (321, 481))
CASE_IDS = tuple(f"h{height}-w{width}" for height, width in SIZES)


def array_digest(array):
    return hashlib.sha256(array.tobytes(order="C")).hexdigest()


def save_array(path, array):
    raw = array.astype("<f4", copy=False).tobytes(order="C")
    path.write_bytes(raw)
    return dict(file=path.name, bytes=len(raw), sha256=hashlib.sha256(raw).hexdigest())


def write_manifest(output, manifest):
    temporary = output / "manifest.json.partial"
    temporary.write_text(json.dumps(manifest, indent=2) + "\n")
    temporary.replace(output / "manifest.json")


def merge_smoke(output, manifest, smoke, smoke_directory):
    """Join completed all-model/RGB references with the geometry suite."""
    if not smoke.get("complete") or len(smoke["cases"]) != 11:
        raise ValueError("The all-model/RGB smoke references are incomplete")
    existing_ids = {case["id"] for case in manifest["cases"]}
    smoke_ids = [case["id"] for case in smoke["cases"]]
    if len(set(smoke_ids)) != len(smoke_ids) or existing_ids.intersection(smoke_ids):
        raise ValueError("Duplicate fixture IDs while merging smoke references")
    geometry_full = existing_ids == set(CASE_IDS)
    geometry_requested = manifest.get("requested_cases", [case["id"] for case in manifest["cases"]])
    manifest["complete"] = False
    write_manifest(output, manifest)
    for case in smoke["cases"]:
        shutil.copytree(smoke_directory / case["id"], output / case["id"], dirs_exist_ok=True)
    manifest["geometry_reference"] = {key: manifest.pop(key) for key in
        ("sigma", "offset", "checkpoint_file", "checkpoint_sha256") if key in manifest}
    manifest["suite"] = "procedural-deep-native-geometry-all-models-rgb-v1"
    manifest["offset_protocol"] = "archived grayscale; zero RGB"
    manifest["array_layout"] = "natural row-major H,W,channels; grayscale H,W"
    manifest["smoke_reference"] = {key: value for key, value in smoke.items() if key != "cases"}
    manifest["cases"].extend(smoke["cases"])
    manifest["requested_cases"] = geometry_requested + smoke_ids
    manifest["expected_case_count"] = len(manifest["cases"])
    manifest["include_smoke"] = True
    manifest["full_suite"] = geometry_full and len(manifest["cases"]) == len(SIZES) + 11
    manifest["complete"] = True
    write_manifest(output, manifest)
    return manifest


def generate(args):
    import numpy as np
    import torch
    from deep_patch_beltrami import assets
    from deep_patch_beltrami.inference import DeepBeltrami, configure, psnr

    selected = set(args.case or CASE_IDS)
    sizes = [(height, width) for height, width in SIZES
             if f"h{height}-w{width}" in selected]
    model_dir = assets.models(cache_dir=args.cache_dir, asset_dir=args.asset_dir,
                              model_dir=args.model_dir)
    configure("cpu", args.threads)
    model = DeepBeltrami(model_dir, 25, "cpu")
    checkpoint = model_dir / "PBSolve_multi" / "PB_25_80_5__iter_40000.caffemodel"
    output = args.output
    output.mkdir(parents=True, exist_ok=True)
    manifest = dict(
        schema=1, suite="procedural-deep-native-parity-v1", sigma=25,
        offset=25.0, offset_protocol="archived", source="procedural; no photographs",
        array_layout="natural row-major H,W", dtype="little-endian float32",
        native_model_layout="N,C,W,H", prediction_clipping=[0, 255],
        reference_backend="PyTorch CPU float32", threads=args.threads,
        python=platform.python_version(), numpy=np.__version__, torch=torch.__version__,
        checkpoint_file=checkpoint.name, checkpoint_sha256=assets.sha256(checkpoint),
        requested_cases=[f"h{height}-w{width}" for height, width in sizes],
        include_smoke=args.include_smoke,
        expected_case_count=len(sizes) + (11 if args.include_smoke else 0),
        complete=False, full_suite=False, cases=[])
    write_manifest(output, manifest)

    for height, width in sizes:
        case_id = f"h{height}-w{width}"
        directory = output / case_id
        directory.mkdir(exist_ok=True)
        row, col = np.mgrid[:height, :width]
        clean = (70 + 80*row/max(1, height-1) + 35*col/max(1, width-1)
                 + 20*np.sin(row/7)*np.cos(col/11)).astype(np.float32)
        seed = 2017 + 1000*height + width
        noise = np.random.default_rng(seed).normal(0, 25, clean.shape).astype(np.float32)
        noisy = np.ascontiguousarray(clean + noise)
        noisy[0, 0] = -7.25
        if noisy.size > 1:
            noisy[-1, -1] = 267.5
        native = np.ascontiguousarray(noisy.T[None, None])
        started = time.perf_counter()
        output_native = model(native, offset=25)
        elapsed = time.perf_counter() - started
        expected = np.ascontiguousarray(output_native[0, 0].T)
        if not np.isfinite(expected).all():
            raise FloatingPointError(f"Non-finite reference prediction: {case_id}")

        record = dict(
            id=case_id, height=height, width=width, channels=1, sigma=25,
            seed=seed,
            construction="Asymmetric smooth ramp plus sinusoid and AWGN; corner values forced to -7.25 and 267.5 when distinct",
            input=save_array(directory / "input.f32", noisy),
            expected=save_array(directory / "expected.f32", expected),
            clean=save_array(directory / "clean.f32", clean),
            native_input_shape=list(native.shape), native_input_sha256=array_digest(native),
            natural_input_min=float(noisy.min()), natural_input_max=float(noisy.max()),
            expected_min=float(expected.min()), expected_max=float(expected.max()),
            all_finite=True, reference_seconds=elapsed,
            noisy_psnr_db=psnr(clean, noisy), expected_psnr_db=psnr(clean, expected))
        (directory / "case.json").write_text(json.dumps(record, indent=2) + "\n")
        manifest["cases"].append(record)
        write_manifest(output, manifest)
        print(json.dumps(dict(id=case_id, finite=True,
                              expected_sha256=record["expected"]["sha256"])), flush=True)

    manifest["complete"] = not args.include_smoke
    manifest["full_suite"] = len(manifest["cases"]) == len(SIZES)
    write_manifest(output, manifest)
    if args.include_smoke:
        helper = Path(__file__).resolve().parents[1] / "Tests/generate_smoke_suite.py"
        smoke_generate = runpy.run_path(str(helper))["generate"]
        with tempfile.TemporaryDirectory(prefix="deep-smoke-", dir=output.parent) as temporary:
            smoke_args = argparse.Namespace(output=Path(temporary), cache_dir=args.cache_dir,
                asset_dir=args.asset_dir, model_dir=args.model_dir, threads=args.threads)
            smoke = smoke_generate(smoke_args)
            return merge_smoke(output, manifest, smoke, Path(temporary))
    return manifest


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=Path("ios/DeepBeltramiApp/Fixtures"),
                        help="Fixture output directory (default: ios/DeepBeltramiApp/Fixtures)")
    parser.add_argument("--cache-dir", type=Path, help="Model download/extraction cache")
    parser.add_argument("--asset-dir", type=Path, help="Directory containing verified release files")
    parser.add_argument("--model-dir", type=Path, help="Existing verified extracted model directory")
    parser.add_argument("--threads", type=int, default=4, help="CPU reference threads (default: 4)")
    parser.add_argument("--case", action="append", choices=CASE_IDS,
                        help="Generate only this case; repeat to select several (default: all 11)")
    parser.add_argument("--include-smoke", action="store_true",
                        help="Append eight noise-specific grayscale and three RGB references (22 cases by default)")
    args = parser.parse_args(argv)
    if args.threads < 1:
        parser.error("--threads must be positive")
    try:
        generate(args)
    except ImportError as error:
        parser.exit(1, f"Install the package first: python -m pip install -e .\n{error}\n")
    except (OSError, ValueError, RuntimeError) as error:
        parser.exit(1, f"{error}\n")


if __name__ == "__main__":
    main()

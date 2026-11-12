"""Package validated, portable Core ML models into per-sigma release assets.

Only the selected .mlpackage trees, a numerical manifest and the MIT license
enter each archive. Compiled device caches and Torch tracing artifacts do not.
"""
import argparse
import gzip
import hashlib
import io
import json
from pathlib import Path
import tarfile


SIGMAS = [5, 10, 15, 20, 25, 30, 40, 50]
OFFSETS = {5: 15, 10: 20, 15: 15, 20: 20, 25: 25, 30: 30, 40: 40, 50: 25}


def sha256(path):
    result = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            result.update(block)
    return result.hexdigest()


def package_files(package):
    return sorted(p for p in package.rglob("*") if p.is_file() and not p.name.startswith("._"))


def package_record(directory, sigma, precision):
    export = json.loads((directory / "manifest.json").read_text())
    matches = [m for m in export["models"] if m["sigma"] == sigma and m["precision"] == precision]
    if len(matches) != 1:
        raise ValueError(f"Expected one validated {sigma}/{precision} model in {directory}")
    model = matches[0]
    if not model["validation"] or not all(v["finite"] for v in model["validation"]):
        raise ValueError("Every packaged model requires finite conversion checks")
    package = directory / model["package"]
    tree = hashlib.sha256()
    files = []
    for path in package_files(package):
        relative = str(path.relative_to(package))
        data = path.read_bytes()
        if any(marker in data for marker in (b"/Users/", b"/home/", b"Dropbox")):
            raise ValueError(f"Private path marker in {package.name}/{relative}")
        tree.update(relative.encode() + b"\0")
        tree.update(data)
        files.append(dict(path=relative, size=len(data), sha256=hashlib.sha256(data).hexdigest()))
    if tree.hexdigest() != model["package_sha256"]:
        raise ValueError(f"Package changed since validation: {package.name}")
    record = dict(package=package.name, precision=precision,
                  boundary_mode=export.get("boundary_mode", "native_rectangle"),
                  flexible=export["flexible"], maximum_size=export["maximum_size"],
                  package_sha256=tree.hexdigest(), files=files,
                  validation_reference="Original PyTorch graph on CPU, float32, raw output before external offset/clipping",
                  torch=export["torch"], coremltools=export["coremltools"],
                  validation=model["validation"])
    # Reject NaN/Infinity in validation errors, not just the finite flag.
    json.dumps(record, allow_nan=False)
    return package, record


def add_bytes(archive, name, data):
    info = tarfile.TarInfo(name)
    info.size = len(data)
    info.mode = 0o644
    info.mtime = 0
    archive.addfile(info, io.BytesIO(data))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--quality-dir", type=Path, required=True)
    parser.add_argument("--fast256-dir", type=Path, required=True)
    parser.add_argument("--fast512-dir", type=Path, required=True)
    parser.add_argument("--original-model-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--sigmas", type=int, nargs="+", choices=SIGMAS, default=SIGMAS)
    parser.add_argument("--tag", default="v1.1.0")
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    license_text = (Path(__file__).resolve().parents[1] / "LICENSE").read_bytes()
    source_manifest = json.loads((args.original_model_dir / "manifest.json").read_text())
    released = dict(schema=1, tag=args.tag, assets={})
    for sigma in args.sigmas:
        name = f"deep-beltrami-coreml-sigma{sigma}-{args.tag}.tar.gz"
        target = args.output_dir / name
        if target.exists():
            raise FileExistsError(f"Refusing to replace a versioned asset: {target}")
        selections = [package_record(args.quality_dir, sigma, "fp32"),
                      package_record(args.fast256_dir, sigma, "mixed16"),
                      package_record(args.fast512_dir, sigma, "mixed16")]
        checkpoint = args.original_model_dir / "PBSolve_multi" / f"PB_{sigma}_80_5__iter_40000.caffemodel"
        source = next(item for item in source_manifest["files"] if item["name"] == checkpoint.name)
        checkpoint_hash = sha256(checkpoint)
        if checkpoint_hash != source["sha256"]:
            raise ValueError(f"Source inference checkpoint changed: {checkpoint.name}")
        manifest = dict(schema=1, sigma=sigma, stages=5, feature_channels=80,
                        checkpoint=checkpoint.name, inference_checkpoint_sha256=checkpoint_hash,
                        original_checkpoint_sha256=source["original_file_sha256"],
                        learned_tensors_sha256=source["learned_tensors_sha256"],
                        saved_training_steps=40000, thesis_training_steps=100000,
                        input="Unclipped float32 intensities in native N,C,W,H order",
                        grayscale_offset=OFFSETS[sigma], rgb_offset=0,
                        postprocessing="Subtract the external offset and clip predictions to [0,255].",
                        halo=45, receptive_field=91,
                        tree_hash="SHA256 of sorted UTF-8 relative path + NUL + file bytes; excludes ._* files",
                        models=[record for _, record in selections])
        encoded = (json.dumps(manifest, indent=2, sort_keys=True, allow_nan=False) + "\n").encode()
        temporary = target.with_name(target.name + ".part")
        with temporary.open("wb") as raw:
            with gzip.GzipFile(filename="", mode="wb", fileobj=raw, mtime=0) as compressed:
                with tarfile.open(fileobj=compressed, mode="w|") as archive:
                    add_bytes(archive, "manifest.json", encoded)
                    add_bytes(archive, "LICENSE.txt", license_text)
                    for package, _ in selections:
                        for path in package_files(package):
                            add_bytes(archive, f"models/{package.name}/{path.relative_to(package)}", path.read_bytes())
        temporary.replace(target)
        released["assets"][name] = dict(sigma=sigma, size=target.stat().st_size, sha256=sha256(target),
            url=f"https://github.com/twerdster/deep-patch-beltrami/releases/download/{args.tag}/{name}")
        print(json.dumps(dict(file=name, **released["assets"][name])), flush=True)
    (args.output_dir / "coreml-release-assets.json").write_text(json.dumps(released, indent=2) + "\n")


if __name__ == "__main__":
    main()

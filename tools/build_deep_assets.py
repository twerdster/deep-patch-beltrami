"""Build portable inference assets from author-supplied checkpoints and inputs.

Clean benchmark images are excluded. Checkpoints retain learned tensors only.
This maintenance command requires an installed checkout.
"""
from pathlib import Path
import argparse
import hashlib
import json
import tarfile
import gzip
import io
import h5py
import numpy as np
from PIL import Image
from deep_patch_beltrami.caffe_model import fields, load_weights

SIGMAS = (5, 10, 15, 20, 25, 30, 40, 50)
SCUNET_COMMIT = "52e440a80a655b01e0b41e9dd9bfe599bc11625e"
MIT_LICENSE = """MIT License

Copyright (c) Aaron Wetzler

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
"""


def sha(path):
    result = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            result.update(block)
    return result.hexdigest()


def array_sha(array):
    return hashlib.sha256(np.ascontiguousarray(array).tobytes()).hexdigest()


def encode_varint(value):
    out = bytearray()
    while value >= 128:
        out.append((value & 127) | 128)
        value >>= 7
    out.append(value)
    return bytes(out)


def encode_field(number, wire, value):
    header = encode_varint(number * 8 + wire)
    if wire == 0:
        return header + encode_varint(value)
    if wire == 2:
        return header + encode_varint(len(value)) + value
    return header + value


def strip_checkpoint(source, output):
    """Keep only weighted layer names and each blob's shape/data fields."""
    messages = []
    for number, wire, value in fields(source.read_bytes()):
        if number != 100:
            continue
        name, blobs = None, []
        for field, layer_wire, layer_value in fields(value):
            if field == 1:
                name = layer_value
            elif field == 7:
                blob = b"".join(encode_field(f, w, v) for f, w, v in fields(layer_value)
                                if f in (1, 2, 3, 4, 5, 7))
                blobs.append(encode_field(7, 2, blob))
        if blobs:
            if name is None:
                raise ValueError("Weighted layer without a name")
            layer = encode_field(1, 2, name) + b"".join(blobs)
            messages.append(encode_field(100, 2, layer))
    output.write_bytes(b"".join(messages))
    old, new = load_weights(source), load_weights(output)
    assert old.keys() == new.keys()
    tensor_hash = hashlib.sha256()
    for name in old:
        assert len(old[name]) == len(new[name])
        for a, b in zip(old[name], new[name]):
            assert np.array_equal(a, b), name
            tensor_hash.update(name.encode())
            tensor_hash.update(np.asarray(a.shape, dtype="<i8").tobytes())
            tensor_hash.update(a.astype("<f4").tobytes())
    return tensor_hash.hexdigest()


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--models-source", type=Path, required=True)
    p.add_argument("--bsd68-suite", type=Path, required=True)
    p.add_argument("--bsd68-clean", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--metadata-output", type=Path, required=True)
    args = p.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    args.metadata_output.mkdir(parents=True, exist_ok=True)
    staging = args.output / "model-staging" / "models" / "PBSolve_multi"
    staging.mkdir(parents=True, exist_ok=True)
    model_records, contents = [], []
    for sigma in SIGMAS:
        name = f"PB_{sigma}_80_5__iter_40000.caffemodel"
        src, dst = args.models_source / name, staging / name
        tensor_hash = strip_checkpoint(src, dst)
        record = dict(name=name, sigma=sigma, original_file_sha256=sha(src), sha256=sha(dst),
                      learned_tensors_sha256=tensor_hash, bytes=dst.stat().st_size,
                      sanitation="Only weighted layer names and blob shape/data fields retained; learned tensors unchanged")
        model_records.append(record)
        contents.append(dst)
    for stage in range(1, 6):
        name = f"stage_{stage:03d}_deploy.prototxt"
        dst = staging / name
        dst.write_bytes((args.models_source / name).read_bytes())
        model_records.append(dict(name=name, stage=stage, sha256=sha(dst), bytes=dst.stat().st_size))
        contents.append(dst)
    manifest_file = staging.parent / "manifest.json"
    manifest_file.write_text(json.dumps(dict(schema=1, checkpoint_step=40000, stages=5,
                                            features=80, files=model_records), indent=2)+"\n")
    contents.append(manifest_file)
    license_file = staging.parent / "LICENSE.txt"
    license_file.write_text("Deep Patch-Beltrami pretrained parameters and deployment graphs\n\n"+MIT_LICENSE)
    contents.append(license_file)
    model_asset = args.output / "deep-beltrami-models-v1.tar.gz"
    # Fixed timestamps make an unchanged export byte reproducible.
    with model_asset.open("wb") as raw, gzip.GzipFile(filename="", fileobj=raw, mode="wb", mtime=0) as gz:
        with tarfile.open(fileobj=gz, mode="w") as archive:
            for item in sorted(contents):
                info = tarfile.TarInfo(str(item.relative_to(staging.parent.parent)))
                payload = item.read_bytes()
                info.size, info.mtime, info.mode = len(payload), 0, 0o644
                archive.addfile(info, io.BytesIO(payload))
    print(f"Exported tensors unchanged: {model_asset.name}", flush=True)

    noise_asset = args.output / "bsd68-original-noise-v1.h5"
    metadata = dict(schema=1, image_source_repository="cszn/SCUNet", image_source_commit=SCUNET_COMMIT,
                    image_source_notice="BSD68 images retain their original dataset terms and are downloaded separately.",
                    native_axis_order="N,C,Matlab-column,Matlab-row", sigmas=list(SIGMAS), images=[], cases=[])
    with h5py.File(args.bsd68_suite, "r") as suite, h5py.File(noise_asset, "w") as out:
        out.attrs["schema"] = 1
        out.attrs["description"] = "Original author-generated Gaussian residuals; no clean images or noisy images included"
        noise = out.create_group("noise")
        for name in sorted(suite["label"]):
            clean_path = args.bsd68_clean / (name+".png")
            natural = np.asarray(Image.open(clean_path).convert("L"), dtype=np.float32)
            label = natural.T[None, None]
            assert np.array_equal(label, suite["label"][name][:]), name
            metadata["images"].append(dict(name=name, filename=clean_path.name, png_sha256=sha(clean_path),
                                            label_native_sha256=array_sha(label), shape=list(label.shape),
                                            url=f"https://raw.githubusercontent.com/cszn/SCUNet/{SCUNET_COMMIT}/testsets/bsd68/{clean_path.name}"))
        for sigma in SIGMAS:
            group = noise.create_group(str(sigma))
            for name in sorted(suite["label"]):
                y = suite["label"][name][:]
                z = suite[f"data/{sigma}/{name}"][:]
                residual = z.astype(np.float64)-y.astype(np.float64)
                reconstructed = (y.astype(np.float64)+residual).astype(np.float32)
                assert np.array_equal(reconstructed, z), (sigma, name)
                ds = group.create_dataset(name, data=residual, compression="gzip", compression_opts=1, shuffle=True)
                ds.attrs["noisy_native_sha256"] = array_sha(z)
                methods = json.loads(suite[f"data/{sigma}/{name}"].attrs["methods"])
                metadata["cases"].append(dict(name=name, sigma=sigma, noisy_native_sha256=array_sha(z),
                                              noise_sha256=array_sha(residual),
                                              archived_deep_psnr=methods["Beltrami-Deep"]["psnr"]))
            print(f"Noise-only sigma{sigma}: all 68 inputs reconstructed bit-for-bit", flush=True)
    (args.metadata_output / "bsd68.json").write_text(json.dumps(metadata, indent=2)+"\n")
    assets = dict(schema=1, repository="twerdster/deep-patch-beltrami", tag="v1.0.0", assets={})
    for key, item in [("models", model_asset), ("bsd68_noise", noise_asset)]:
        assets["assets"][key] = dict(filename=item.name, bytes=item.stat().st_size, sha256=sha(item),
                                    url=f"https://github.com/twerdster/deep-patch-beltrami/releases/download/v1.0.0/{item.name}")
    (args.metadata_output / "release-assets.json").write_text(json.dumps(assets, indent=2)+"\n")
    (args.metadata_output / "models.json").write_text(json.dumps(dict(files=model_records), indent=2)+"\n")
    print(json.dumps(assets, indent=2), flush=True)


if __name__ == "__main__":
    main()

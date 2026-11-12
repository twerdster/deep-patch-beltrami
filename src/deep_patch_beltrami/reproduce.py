"""Evaluate the released model with original inputs and published targets."""
from pathlib import Path
import json
import time
import h5py
import numpy as np
import torch
from . import assets
from .inference import DeepBeltrami, configure, psnr

SIGMAS = (5, 10, 15, 20, 25, 30, 40, 50)
OFFSET_FACTORS = dict(zip(SIGMAS, (3, 2, 1, 1, 1, 1, 1, .5)))
PUBLISHED = {
    "bsd68": dict(zip(SIGMAS, (37.70,33.52,31.37,29.92,28.87,28.04,26.81,25.95))),
    "portraits100": dict(zip(SIGMAS, (39.84,36.30,34.38,33.03,31.98,31.15,29.81,28.83))),
}


class PublicBSD68:
    def __init__(self, cache_dir=None, asset_dir=None, clean_dir=None):
        self.labels = assets.clean_bsd68(cache_dir, clean_dir)
        self.file = h5py.File(assets.asset("bsd68_noise", cache_dir, asset_dir), "r")
        self.names = sorted(self.labels)
        self.cases = {(r["sigma"],r["name"]):r for r in assets.metadata("bsd68.json")["cases"]}

    def case(self, sigma, name):
        label = self.labels[name]
        noise = self.file[f"noise/{sigma}/{name}"][:]
        noisy = (label.astype(np.float64)+noise).astype(np.float32)
        record = self.cases[sigma,name]
        digest = assets.array_sha256(noisy)
        if digest != record["noisy_native_sha256"]:
            raise ValueError(f"Original noisy input hash mismatch: {name}, sigma{sigma}")
        return label, noisy, record["archived_deep_psnr"], digest

    def close(self):
        self.file.close()


class LocalSuite:
    """Compatible HDF5: /label/name and /data/sigma/name in native N,C,W,H."""
    def __init__(self, path):
        self.file = h5py.File(path, "r")
        self.names = sorted(self.file["label"])

    def case(self, sigma, name):
        label = self.file["label"][name][:]
        dataset = self.file[f"data/{sigma}/{name}"]
        noisy = dataset[:]
        digest = assets.array_sha256(noisy)
        expected = dataset.attrs.get("sha256")
        if expected is not None and str(expected) != digest:
            raise ValueError(f"Local input hash mismatch: {name}, sigma{sigma}")
        methods = json.loads(dataset.attrs.get("methods", "{}"))
        archived = methods.get("Beltrami-Deep", {}).get("psnr")
        return label, noisy, archived, digest

    def close(self):
        self.file.close()


def evaluate(suite, model_dir, dataset="bsd68", device="cpu", sigmas=SIGMAS,
             threads=4, offset_protocol="archived", limit=None, output=None):
    configure(device, threads)
    cases, means = [], []
    expected_images = {"bsd68":68, "portraits100":100}.get(dataset)
    names = suite.names[:limit] if limit is not None else suite.names
    full = limit is None and tuple(sigmas) == SIGMAS and len(names) == expected_images
    report = dict(dataset=dataset, checkpoint_step=40000, features=80, stages=5,
                  offset_protocol=offset_protocol, device=device, input_axis_order="N,C,W,H",
                  cases=cases, means=means, complete=False)
    if not names:
        raise ValueError("The suite contains no images")
    for sigma in sigmas:
        if sigma not in SIGMAS:
            raise ValueError(f"No released checkpoint for sigma{sigma}")
        model = DeepBeltrami(model_dir, sigma, device)
        offset = sigma*OFFSET_FACTORS[sigma] if offset_protocol == "archived" else 0
        start = time.perf_counter()
        for name in names:
            label, noisy, archived, digest = suite.case(sigma,name)
            prediction = model(noisy, offset)
            actual = psnr(label,prediction)
            case = dict(name=name, sigma=sigma, psnr=actual, offset=offset,
                        noisy_native_sha256=digest, finite=bool(np.isfinite(prediction).all()))
            if archived is not None:
                case.update(archived_psnr=archived, delta_db=actual-archived)
            cases.append(case)
        subset = [c for c in cases if c["sigma"] == sigma]
        mean = dict(sigma=sigma, images=len(subset), mean_psnr=float(np.mean([c["psnr"] for c in subset])),
                    seconds=time.perf_counter()-start, offset=offset)
        if full and dataset in PUBLISHED:
            target = PUBLISHED[dataset][sigma]
            mean.update(published_psnr=target, matches_published_rounding=f"{mean['mean_psnr']:.2f}"==f"{target:.2f}")
        if all("archived_psnr" in c for c in subset):
            mean.update(archived_mean_psnr=float(np.mean([c["archived_psnr"] for c in subset])),
                        max_abs_psnr_delta_db=float(max(abs(c["delta_db"]) for c in subset)))
        if device.startswith("cuda"):
            mean["peak_cuda_allocated_bytes"] = torch.cuda.max_memory_allocated()
            mean["peak_cuda_reserved_bytes"] = torch.cuda.max_memory_reserved()
        means.append(mean)
        print(json.dumps(mean), flush=True)
        if output is not None:
            Path(output).parent.mkdir(parents=True, exist_ok=True)
            Path(output).write_text(json.dumps(report,indent=2)+"\n")
        del model
    report["complete"] = True
    report["full_published_suite"] = full
    report["all_finite"] = all(c["finite"] for c in cases)
    report["matches_all_published_means"] = full and all(m.get("matches_published_rounding",False) for m in means)
    if output is not None:
        Path(output).write_text(json.dumps(report,indent=2)+"\n")
    if full and offset_protocol == "archived" and not report["matches_all_published_means"]:
        raise RuntimeError("The run completed but at least one published mean did not reproduce; inspect the JSON")
    return report

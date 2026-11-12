"""Denoise a procedural image with an original Deep Patch-Beltrami model."""
from pathlib import Path
import argparse
import json

import numpy as np
from PIL import Image, ImageDraw

from deep_patch_beltrami import assets
from deep_patch_beltrami.inference import DeepBeltrami, configure, psnr
from deep_patch_beltrami.reproduce import OFFSET_FACTORS, SIGMAS


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=Path("results/demo"))
    parser.add_argument("--sigma", type=int, choices=SIGMAS, default=25)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--cache-dir", type=Path)
    parser.add_argument("--asset-dir", type=Path)
    args = parser.parse_args()
    import torch
    device = ("cuda" if torch.cuda.is_available() else "cpu") if args.device == "auto" else args.device
    configure(device)
    row, col = np.mgrid[:192, :224]
    clean = 105 + 35*np.sin(row/19) + 25*np.cos(col/23)
    clean += 45*((row-96)**2 + (col-112)**2 < 40**2)
    clean = clean.astype(np.float32)
    noisy = clean + np.random.default_rng(2017).normal(0, args.sigma, clean.shape).astype(np.float32)
    model = DeepBeltrami(assets.models(args.cache_dir, args.asset_dir), args.sigma, device)
    denoised = model(noisy.T[None, None], args.sigma*OFFSET_FACTORS[args.sigma])[0, 0].T
    metrics = dict(sigma=args.sigma, seed=2017, device=device,
                   noisy_psnr_db=psnr(clean, noisy), denoised_psnr_db=psnr(clean, denoised),
                   finite=bool(np.isfinite(denoised).all()),
                   image="procedural sinusoidal surface and circular intensity step",
                   input_clipping="none", prediction_clipping="0..255",
                   offset_protocol="archived")
    args.output.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(args.output/"arrays.npz", clean=clean, noisy=noisy, denoised=denoised)
    previews = []
    titles = [f"Noisy: {metrics['noisy_psnr_db']:.2f} dB",
              f"Deep Patch-Beltrami: {metrics['denoised_psnr_db']:.2f} dB", "Clean reference"]
    for name, array, title in zip(("noisy", "denoised", "clean"), (noisy, denoised, clean), titles):
        preview = Image.fromarray(np.clip(array, 0, 255).round().astype(np.uint8))
        preview.save(args.output/f"{name}.png")
        panel = Image.new("RGB", (preview.width, preview.height+28), "white")
        panel.paste(preview, (0, 28))
        ImageDraw.Draw(panel).text((6, 7), title, fill="black")
        previews.append(panel)
    comparison = Image.new("RGB", (sum(p.width for p in previews)+16, previews[0].height), "white")
    x = 0
    for panel in previews:
        comparison.paste(panel, (x, 0))
        x += panel.width+8
    comparison.save(args.output/"comparison.png")
    (args.output/"metrics.json").write_text(json.dumps(metrics, indent=2)+"\n")
    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()

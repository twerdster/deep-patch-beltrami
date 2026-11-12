"""Hash-verified clean images, deterministic sampling, and fresh Gaussian noise."""
from pathlib import Path
import hashlib
import json
import numpy as np
import torch
from PIL import Image


def image_array(path):
    with Image.open(path) as image:
        if image.mode != "L":
            raise ValueError(f"Expected original grayscale image, got {image.mode}: {path.name}")
        return np.asarray(image, dtype=np.uint8).copy()


def load_images(root, manifest_path, split):
    manifest = json.loads(Path(manifest_path).read_text())
    images, records = [], []
    for record in manifest["images"]:
        if record["split"] != split:
            continue
        path = Path(root) / record["path"]
        if hashlib.sha256(path.read_bytes()).hexdigest() != record["file_sha256"]:
            raise ValueError(f"Image file hash mismatch: {record['path']}")
        image = image_array(path)
        if hashlib.sha256(image.tobytes()).hexdigest() != record["pixels_sha256"]:
            raise ValueError(f"Decoded image hash mismatch: {record['path']}")
        images.append(image)
        records.append(record)
    if not images:
        raise ValueError(f"No {split} images in manifest")
    return images, records, manifest


class PatchSampler:
    def __init__(self, images, patch_size, seed):
        self.images = images
        self.patch_size = patch_size
        self.rng = np.random.default_rng(seed)
        if min(min(image.shape) for image in images) < patch_size:
            raise ValueError("Training images smaller than requested patch size")

    def batch(self, batch_size, sigma, offset):
        patches = []
        size = self.patch_size
        for _ in range(batch_size):
            image = self.images[int(self.rng.integers(len(self.images)))]
            row = int(self.rng.integers(image.shape[0] - size + 1))
            col = int(self.rng.integers(image.shape[1] - size + 1))
            patch = image[row:row + size, col:col + size]
            patch = np.rot90(patch, int(self.rng.integers(4)))
            if self.rng.integers(2):
                patch = patch[:, ::-1]
            # The original Caffe interface uses width before height.
            patches.append(np.ascontiguousarray(patch.T, dtype=np.float32))
        clean = np.stack(patches)[:, None]
        noise = self.rng.standard_normal(clean.shape, dtype=np.float32) * sigma
        return torch.from_numpy(clean + noise + offset), torch.from_numpy(clean + offset)


def validation_pairs(images, records, sigma, seed, offset):
    for image, record in zip(images, records):
        item_seed = int.from_bytes(hashlib.sha256(
            f"{seed}:{record['pixels_sha256']}".encode()).digest()[:8], "little")
        rng = np.random.default_rng(item_seed)
        clean = image.T.copy().astype(np.float32)[None, None]
        noisy = clean + rng.standard_normal(clean.shape, dtype=np.float32) * sigma
        yield torch.from_numpy(noisy + offset), clean, record

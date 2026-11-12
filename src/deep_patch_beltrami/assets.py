"""Verified downloads and preparation of the public reproduction inputs."""
from concurrent.futures import ThreadPoolExecutor
from importlib.resources import files
from pathlib import Path
import hashlib
import json
import os
import shutil
import tarfile
import tempfile
import urllib.error
import urllib.request
import uuid
import numpy as np
from PIL import Image


def metadata(name):
    return json.loads(files("deep_patch_beltrami").joinpath("data", name).read_text())


def sha256(path):
    result = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024*1024), b""):
            result.update(block)
    return result.hexdigest()


def array_sha256(array):
    return hashlib.sha256(np.ascontiguousarray(array).tobytes()).hexdigest()


def default_cache():
    root = Path(os.environ.get("XDG_CACHE_HOME", Path.home()/".cache"))
    return root / "deep-patch-beltrami" / "v1"


def download(url, destination, expected_sha256):
    destination = Path(destination)
    if destination.is_file() and sha256(destination) == expected_sha256:
        return destination
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_name(destination.name+".partial-"+uuid.uuid4().hex)
    try:
        request = urllib.request.Request(url, headers={"User-Agent": "deep-patch-beltrami/1.0"})
        with urllib.request.urlopen(request, timeout=60) as response, temporary.open("wb") as out:
            shutil.copyfileobj(response, out, length=1024*1024)
        actual = sha256(temporary)
        if actual != expected_sha256:
            raise ValueError(f"SHA256 mismatch for {destination.name}: {actual}")
        temporary.replace(destination)
    except (urllib.error.URLError, OSError) as error:
        raise RuntimeError(f"Could not download {url}: {error}") from error
    finally:
        temporary.unlink(missing_ok=True)
    return destination


def asset(name, cache_dir=None, asset_dir=None):
    record = metadata("release-assets.json")["assets"][name]
    if asset_dir is not None:
        path = Path(asset_dir)/record["filename"]
        if not path.is_file():
            raise FileNotFoundError(path)
        if sha256(path) != record["sha256"]:
            raise ValueError(f"SHA256 mismatch for local release asset: {path.name}")
        return path
    cache = Path(cache_dir) if cache_dir is not None else default_cache()
    return download(record["url"], cache/"downloads"/record["filename"], record["sha256"])


def valid_models(directory):
    directory = Path(directory)
    for record in metadata("models.json")["files"]:
        path = directory/"PBSolve_multi"/record["name"]
        if not path.is_file() or sha256(path) != record["sha256"]:
            return False
    return True


def models(cache_dir=None, asset_dir=None, model_dir=None):
    if model_dir is not None:
        path = Path(model_dir)
        if not valid_models(path):
            raise ValueError("Model directory does not match the released inference checkpoints")
        return path
    cache = Path(cache_dir) if cache_dir is not None else default_cache()
    destination = cache/"models"
    if valid_models(destination):
        return destination
    archive_path = asset("models", cache, asset_dir)
    cache.mkdir(parents=True, exist_ok=True)
    allowed = {"models/PBSolve_multi/"+record["name"] for record in metadata("models.json")["files"]}
    allowed.update(("models/manifest.json", "models/LICENSE.txt"))
    with tempfile.TemporaryDirectory(prefix="model-extract-", dir=cache) as temporary:
        temporary = Path(temporary)
        with tarfile.open(archive_path, "r:gz") as archive:
            for member in archive.getmembers():
                if member.name not in allowed or not member.isfile():
                    raise ValueError(f"Unexpected release archive member: {member.name}")
                target = temporary/member.name
                target.parent.mkdir(parents=True, exist_ok=True)
                with archive.extractfile(member) as source, target.open("wb") as out:
                    shutil.copyfileobj(source, out)
        if not valid_models(temporary/"models"):
            raise ValueError("Extracted checkpoint hashes do not match the release manifest")
        if destination.exists():
            shutil.rmtree(destination)
        (temporary/"models").replace(destination)
    return destination


def clean_bsd68(cache_dir=None, clean_dir=None):
    cache = Path(cache_dir) if cache_dir is not None else default_cache()
    directory = Path(clean_dir) if clean_dir is not None else cache/"bsd68-clean"
    records = metadata("bsd68.json")["images"]

    def prepare(record):
        path = directory/record["filename"]
        if clean_dir is None:
            download(record["url"], path, record["png_sha256"])
        if not path.is_file() or sha256(path) != record["png_sha256"]:
            raise ValueError(f"BSD68 PNG does not match the pinned public image: {record['filename']}")
        array = np.asarray(Image.open(path).convert("L"), dtype=np.float32).T[None, None]
        if array_sha256(array) != record["label_native_sha256"]:
            raise ValueError(f"BSD68 decoded pixels do not match: {record['filename']}")
        return record["name"], array

    with ThreadPoolExecutor(max_workers=8) as pool:
        return dict(pool.map(prepare, records))

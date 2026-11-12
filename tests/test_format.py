"""Lightweight checks that do not download models or require PyTorch."""
import hashlib
import struct
import numpy as np
import pytest
from deep_patch_beltrami.assets import download, metadata
from deep_patch_beltrami.caffe_model import load_weights


def varint(value):
    result = bytearray()
    while value > 127:
        result.append((value & 127) | 128)
        value >>= 7
    result.append(value)
    return bytes(result)


def length_field(number, value):
    return varint(number*8+2)+varint(len(value))+value


def test_checkpoint_tensor_order_and_unknown_fields(tmp_path):
    # A 2-by-2 Caffe tensor with a training-only field beside the numeric blob.
    shape = length_field(1,b"\x02\x02")
    values = struct.pack("<4f",1.25,-2.5,3.75,4.5)
    blob = length_field(7,shape)+length_field(5,values)
    layer = length_field(1,b"filter-001")+length_field(7,blob)
    training_metadata = length_field(1,b"unused training metadata")
    path = tmp_path/"example.caffemodel"
    path.write_bytes(training_metadata+length_field(100,layer))
    weights = load_weights(path)
    assert list(weights)==["filter-001"]
    np.testing.assert_array_equal(weights["filter-001"][0],
                                  np.array([[1.25,-2.5],[3.75,4.5]],dtype=np.float32))


def test_download_hash_failure_is_atomic(tmp_path):
    source = tmp_path/"source.bin"
    source.write_bytes(b"released contents")
    destination = tmp_path/"cache"/"asset.bin"
    with pytest.raises(ValueError,match="SHA256 mismatch"):
        download(source.as_uri(),destination,"0"*64)
    assert not destination.exists()
    assert list(destination.parent.glob("*.partial-*"))==[]
    expected = hashlib.sha256(source.read_bytes()).hexdigest()
    assert download(source.as_uri(),destination,expected)==destination
    assert destination.read_bytes()==source.read_bytes()


def test_public_fixture_manifest_is_complete_and_pinned():
    reference = metadata("bsd68.json")
    assert len(reference["images"])==68
    assert len(reference["cases"])==544
    assert len({(r["name"],r["sigma"]) for r in reference["cases"]})==544
    commit = reference["image_source_commit"]
    assert len(commit)==40
    for record in reference["images"]:
        assert f"/{commit}/" in record["url"]
        assert len(record["png_sha256"])==64
        assert len(record["label_native_sha256"])==64
    release = metadata("release-assets.json")
    assert release["tag"]=="v1.0.0"
    assert set(release["assets"])=={"models","bsd68_noise"}
    for record in release["assets"].values():
        assert "/releases/download/v1.0.0/" in record["url"]
        assert len(record["sha256"])==64

"""Read archived Caffe protobuf weights without Caffe or generated protobuf code.

Only known NetParameter/LayerParameter/BlobProto fields are interpreted. All
other protobuf fields are skipped by their wire type, preserving the source.
"""
from pathlib import Path
import re
import numpy as np


def varint(buf, pos=0):
    value, shift = 0, 0
    while True:
        b = buf[pos]
        pos += 1
        value |= (b & 127) << shift
        if b < 128:
            return value, pos
        shift += 7
        if shift > 70:
            raise ValueError("Invalid protobuf varint")


def fields(buf):
    pos = 0
    while pos < len(buf):
        tag, pos = varint(buf, pos)
        number, wire = tag >> 3, tag & 7
        if wire == 0:
            value, pos = varint(buf, pos)
        elif wire == 2:
            length, pos = varint(buf, pos)
            value = buf[pos:pos + length]
            pos += length
        elif wire in (1, 5):
            length = 8 if wire == 1 else 4
            value = buf[pos:pos + length]
            pos += length
        else:
            raise ValueError(f"Unsupported protobuf wire type {wire}")
        if pos > len(buf):
            raise ValueError("Truncated protobuf")
        yield number, wire, value


def blob_array(buf):
    shape, old_shape, data = [], [0] * 4, []
    for number, wire, value in fields(buf):
        if number == 7:
            for sf, sw, sv in fields(value):
                if sf == 1 and sw == 2:
                    pos = 0
                    while pos < len(sv):
                        dim, pos = varint(sv, pos)
                        shape.append(dim)
                elif sf == 1 and sw == 0:
                    shape.append(sv)
        elif number in (1, 2, 3, 4):
            old_shape[number - 1] = value
        elif number == 5:
            data.append(np.frombuffer(value, dtype="<f4"))
    array = np.concatenate(data) if len(data) > 1 else data[0].copy()
    return array.reshape(shape or old_shape)


def load_weights(path):
    result = {}
    for number, wire, value in fields(Path(path).read_bytes()):
        if number != 100:
            continue
        name, blobs = None, []
        for lf, lw, lv in fields(value):
            if lf == 1:
                name = lv.decode()
            elif lf == 7:
                blobs.append(blob_array(lv))
        if blobs:
            if not name or name in result:
                raise ValueError(f"Invalid or duplicate weight layer: {name}")
            result[name] = blobs
    if not result:
        raise ValueError("No modern Caffe weight layers found")
    return result


def load_prototxt(path):
    """Parse the small protobuf-text subset used by the archived deploy graphs."""
    source = re.sub(r"#[^\n]*", "", Path(path).read_text())
    tokens = re.findall(r'"(?:\\.|[^"\\])*"|[{}:]|[^\s{}:]+', source)
    pos = 0

    def message():
        nonlocal pos
        result = {}
        while pos < len(tokens) and tokens[pos] != "}":
            name = tokens[pos]
            pos += 1
            if tokens[pos] == "{":
                pos += 1
                value = message()
                assert tokens[pos] == "}"
                pos += 1
            else:
                assert tokens[pos] == ":", (name, tokens[pos])
                pos += 1
                token = tokens[pos]
                pos += 1
                if token.startswith('"'):
                    value = token[1:-1]
                else:
                    try:
                        value = float(token) if any(c in token for c in ".eE") else int(token)
                    except ValueError:
                        value = token
            result.setdefault(name, []).append(value)
        return result

    return message()


def first(message, key, default=None):
    return message.get(key, [default])[0]

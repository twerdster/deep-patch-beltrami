"""Faithful forward-only execution of the archived Deep Patch-Beltrami graphs.

No fitting or training is performed. Tensors keep native HDF5/Caffe axis order,
which is the transpose of natural MATLAB images and exactly the archived input.
"""
from pathlib import Path
import collections
import math
import numpy as np
import torch
import torch.nn.functional as F
from .caffe_model import first, load_prototxt, load_weights


class CaffeGraph:
    def __init__(self, prototxt, weights, device="cpu"):
        self.layers = load_prototxt(prototxt)["layer"]
        self.weights = {
            name: [torch.from_numpy(np.ascontiguousarray(a)).to(device) for a in arrays]
            for name, arrays in weights.items()
        }
        self.users = collections.Counter(
            bottom for layer in self.layers for bottom in layer.get("bottom", [])
        )

    @torch.inference_mode()
    def __call__(self, data, debug=False):
        blobs = {"data": data}
        users = self.users.copy()
        for layer in self.layers:
            kind, name = first(layer, "type"), first(layer, "name")
            bottoms = layer.get("bottom", [])
            x = [blobs[b] for b in bottoms]
            if kind == "Padding":
                p = first(layer, "padding_param")
                pad = tuple(int(first(p, k, 0)) for k in ("pad_l", "pad_r", "pad_t", "pad_b"))
                method = first(p, "pad_method", "CONSTANT")
                if method == "REPEAT":
                    y = F.pad(x[0], pad, mode="replicate")
                elif method == "CONSTANT":
                    y = F.pad(x[0], pad, value=float(first(p, "pad_value", 0)))
                else:
                    raise ValueError(method)
            elif kind == "Convolution":
                p = first(layer, "convolution_param")
                arrays = self.weights[name]
                padding = (int(first(p, "pad_h", first(p, "pad", 0))),
                           int(first(p, "pad_w", first(p, "pad", 0))))
                stride = (int(first(p, "stride_h", first(p, "stride", 1))),
                          int(first(p, "stride_w", first(p, "stride", 1))))
                y = F.conv2d(x[0], arrays[0], arrays[1] if len(arrays) > 1 else None,
                             padding=padding, stride=stride, groups=int(first(p, "group", 1)))
            elif kind == "Power":
                p = first(layer, "power_param")
                scale, shift = float(first(p, "scale", 1)), float(first(p, "shift", 0))
                value = x[0] if scale == 1 and shift == 0 else x[0] * scale + shift
                y = value.pow(float(first(p, "power", 1)))
            elif kind == "Eltwise":
                p = first(layer, "eltwise_param")
                operation = first(p, "operation", "SUM")
                if operation == "PROD":
                    y = x[0] * x[1]
                    for value in x[2:]:
                        y = y * value
                elif operation == "SUM":
                    coefficients = p.get("coeff", [1] * len(x))
                    y = x[0] * coefficients[0]
                    for coefficient, value in zip(coefficients[1:], x[1:]):
                        y = y + coefficient * value
                else:
                    raise ValueError(operation)
            else:
                raise ValueError(f"Unsupported Caffe layer {kind}: {name}")
            if debug and not torch.isfinite(y).all():
                raise FloatingPointError(f"Non-finite output from layer {name}")
            for top in layer["top"]:
                blobs[top] = y
            for bottom in bottoms:
                users[bottom] -= 1
                if users[bottom] == 0:
                    del blobs[bottom]
        return y


class DeepBeltrami:
    def __init__(self, model_dir, sigma, device="cpu"):
        directory = Path(model_dir) / "PBSolve_multi"
        path = directory / f"PB_{sigma}_80_5__iter_40000.caffemodel"
        weights = load_weights(path)
        self.graphs = [CaffeGraph(directory / f"stage_{k:03d}_deploy.prototxt", weights, device)
                       for k in range(1, 6)]
        self.device = device
        self.path = str(path)

    def __call__(self, data, offset=0, debug=False):
        # Caffe models are grayscale; historical RGB evaluates each channel alone.
        channels = []
        for channel in range(data.shape[1]):
            tensor = torch.from_numpy(np.ascontiguousarray(data[:, channel:channel+1])).to(self.device)
            tensor = tensor + offset
            for graph in self.graphs:
                tensor = graph(tensor, debug=debug)
            channels.append((tensor - offset).cpu().numpy())
        result = np.clip(np.concatenate(channels, axis=1), 0, 255)
        if not np.isfinite(result).all():
            raise FloatingPointError("Non-finite denoised image")
        return result


def psnr(a, b):
    return float(10 * math.log10(255**2 / np.mean((a.astype(float) - b.astype(float))**2)))


def configure(device="cpu", threads=4):
    torch.set_num_threads(threads)
    if torch.get_num_interop_threads() != 1:
        torch.set_num_interop_threads(1)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cudnn.benchmark = False
    if device.startswith("cuda"):
        # Limit this process's CUDA allocation to 4 GiB.
        index = int(device.split(":")[1]) if ":" in device else 0
        total = torch.cuda.get_device_properties(index).total_memory
        torch.cuda.set_per_process_memory_fraction(min(.9, 4 * 1024**3 / total), index)

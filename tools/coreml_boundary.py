"""Static-tile evaluation with the original finite-rectangle boundaries.

Inputs are native N,C,W,H, with one channel.  ``noisy`` already contains any
external grayscale offset.  This module neither subtracts offsets nor clips.
Only the rectangle mask and clamped gather indices introduce new operations.
No metric determinant is altered, clamped, or regularized.
"""
import torch
import torch.nn.functional as F

from deep_patch_beltrami.caffe_model import first
from deep_patch_beltrami.inference import DeepBeltrami


GRADIENT_PREFIXES = ("Ix-", "Iy-")
FLUX_PREFIXES = (
    "gm05_mul_Iyg11_sub_Ixg12-",
    "gm05_mul_Ixg22_sub_Iyg12-",
)


def masked_graph_forward(graph, data, valid_mask, audit=None):
    """Execute one original stage, masking exactly two gradients/two fluxes."""
    blobs = {"data": data}
    users = graph.users.copy()
    for layer in graph.layers:
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
            arrays = graph.weights[name]
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

        gradient = kind == "Convolution" and name.startswith(GRADIENT_PREFIXES)
        flux = kind == "Eltwise" and name.startswith(FLUX_PREFIXES)
        if audit is not None and (gradient or flux or name.startswith(("detG-", "gm05-"))):
            # Audit before masking: multiplication by zero cannot hide a NaN.
            audit.append({"layer": name, "finite_before_mask": bool(torch.isfinite(y).all()),
                          "minimum": float(y.min()), "maximum": float(y.max()),
                          "mask_applied": gradient or flux})
        if gradient or flux:
            y = y * valid_mask
        for top in layer["top"]:
            blobs[top] = y
        for bottom in bottoms:
            users[bottom] -= 1
            if users[bottom] == 0:
                del blobs[bottom]
    return y


class MaskedBoundaryNetwork(torch.nn.Module):
    """Traceable five-stage network with exact finite-rectangle boundaries.

    forward(noisy, valid_mask, idx_x, idx_y) -> raw fixed-size output.
    ``valid_mask`` is 1 on [0:W,0:H], otherwise 0.  ``idx_x`` and ``idx_y``
    each have fixed length T and contain min(arange(T), W-1/H-1).
    Dtypes: noisy/mask float32 (float64 supported for validation); indices int32
    or int64.  Set record_audit=False (the default) when tracing/exporting.
    """
    def __init__(self, models, sigma=25, dtype=torch.float32, record_audit=False):
        super().__init__()
        self.source = DeepBeltrami(models, sigma, "cpu")
        self.record_audit = record_audit
        self.last_audit = []
        for graph in self.source.graphs:
            graph.weights = {name: [a.to(dtype=dtype) for a in arrays]
                             for name, arrays in graph.weights.items()}
            gradients = [l for l in graph.layers
                         if first(l, "type") == "Convolution" and first(l, "name").startswith(GRADIENT_PREFIXES)]
            fluxes = [l for l in graph.layers
                      if first(l, "type") == "Eltwise" and first(l, "name").startswith(FLUX_PREFIXES)]
            if len(gradients) != 2 or len(fluxes) != 2:
                raise ValueError("Expected precisely two gradient and two flux masks per stage")

    def forward(self, noisy, valid_mask, idx_x, idx_y):
        output = noisy
        audit = [] if self.record_audit else None
        for graph in self.source.graphs:
            # REPEAT extension must use this stage's current estimate.
            output = output.index_select(2, idx_x).index_select(3, idx_y)
            output = masked_graph_forward(graph, output, valid_mask, audit)
        if self.record_audit:
            self.last_audit = audit
        return output

    def interception_points(self):
        return [{"name": first(layer, "name"), "type": first(layer, "type"),
                 "tops": layer["top"]}
                for graph in self.source.graphs for layer in graph.layers
                if ((first(layer, "type") == "Convolution" and
                     first(layer, "name").startswith(GRADIENT_PREFIXES)) or
                    (first(layer, "type") == "Eltwise" and
                     first(layer, "name").startswith(FLUX_PREFIXES)))]


def pack_rectangle(data, tile_size=256, index_dtype=torch.int32):
    """Pack native [1,1,W,H] data in the top-left of a static square tile."""
    if data.ndim != 4 or tuple(data.shape[:2]) != (1, 1):
        raise ValueError("Expected native [1,1,W,H] tensor")
    width, height = data.shape[-2:]
    if not (0 < width <= tile_size and 0 < height <= tile_size):
        raise ValueError("Rectangle dimensions must be in 1..tile_size")
    noisy = data.new_zeros((1, 1, tile_size, tile_size))
    noisy[:, :, :width, :height] = data
    mask = data.new_zeros((1, 1, tile_size, tile_size))
    mask[:, :, :width, :height] = 1
    idx_x = torch.arange(tile_size, device=data.device, dtype=index_dtype).clamp(max=width-1)
    idx_y = torch.arange(tile_size, device=data.device, dtype=index_dtype).clamp(max=height-1)
    return noisy, mask, idx_x, idx_y

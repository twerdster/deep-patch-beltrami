"""Beltrami operator with explicitly selected high-precision metric arithmetic.

Learned parameters and dense analysis/divergence/projection convolutions remain
float32. The float64 variant evaluates metric convolutions, determinant and flux
in double precision to avoid cancellation in nearly collinear derivative banks.
It does not add an epsilon, clamp the determinant or change the flow equation.
"""
from dataclasses import asdict, dataclass
from pathlib import Path
import hashlib
import torch
from torch import nn
from torch.nn import functional as F


@dataclass(frozen=True)
class ModelConfig:
    width: int = 80
    stages: int = 5
    metric: str = "frozen"
    zero_dc: bool = False
    step_scale: float = 1000.0
    reaction: bool = False
    metric_precision: str = "float64"


class BeltramiStage(nn.Module):
    def __init__(self, config):
        super().__init__()
        c = config.width
        self.config = config
        self.dx = nn.Conv2d(1, c, 7, bias=False)
        self.dy = nn.Conv2d(1, c, 7, bias=False)
        self.divx = nn.Conv2d(c, c, 7, padding=3, bias=False)
        self.divy = nn.Conv2d(c, c, 7, padding=3, bias=False)
        self.projection = nn.Conv2d(c, 1, 1, bias=False)
        self.metric_weight = nn.Parameter(torch.ones(c, 1, 7, 7))
        self.metric_bias = nn.Parameter(torch.ones(c))
        if config.reaction:
            self.reaction = nn.Parameter(torch.zeros(()))
        else:
            self.register_parameter("reaction", None)
        if config.metric not in ("frozen", "positive", "unconstrained"):
            raise ValueError("metric must be frozen, positive, or unconstrained")
        if config.metric_precision not in ("float32", "float64"):
            raise ValueError("metric_precision must be float32 or float64")
        if config.metric == "frozen":
            self.metric_weight.requires_grad_(False)
            self.metric_bias.requires_grad_(False)

    def metric_parameters(self):
        if self.config.metric == "positive":
            return F.softplus(self.metric_weight), F.softplus(self.metric_bias)
        return self.metric_weight, self.metric_bias

    def derivative(self, value, layer):
        weight = layer.weight
        if self.config.zero_dc:
            weight = weight - weight.mean(dim=(-1, -2), keepdim=True)
        return F.conv2d(F.pad(value, (3, 3, 3, 3), mode="replicate"), weight)

    def forward(self, value, noisy=None):
        dx, dy = self.derivative(value, self.dx), self.derivative(value, self.dy)
        weight, bias = self.metric_parameters()
        if self.config.metric_precision == "float64":
            dx, dy, weight, bias = dx.double(), dy.double(), weight.double(), bias.double()
        g11 = F.conv2d(dx.square(), weight, bias, padding=3, groups=self.config.width)
        g12 = F.conv2d(dx * dy, weight, padding=3, groups=self.config.width)
        g22 = F.conv2d(dy.square(), weight, bias, padding=3, groups=self.config.width)
        determinant = g11 * g22 - g12.square()
        inverse_root = determinant.pow(-0.5)
        flux_x = (dx * g22 - dy * g12) * inverse_root
        flux_y = (dy * g11 - dx * g12) * inverse_root
        if self.config.metric_precision == "float64":
            flux_x, flux_y = flux_x.to(value.dtype), flux_y.to(value.dtype)
            inverse_root = inverse_root.to(value.dtype)
        divergence = self.divx(flux_x) + self.divy(flux_y)
        output = value + self.config.step_scale * self.projection(divergence * inverse_root)
        if self.config.reaction:
            output = output + self.reaction * (noisy - value)
        return output


class BeltramiOperator(nn.Module):
    """Native tensor axes are N,C,W,H; intensities use the 0–255 scale."""
    def __init__(self, config=ModelConfig()):
        super().__init__()
        if config.width < 80 or config.stages < 5:
            raise ValueError("Original-checkpoint initialization needs width >=80 and stages >=5")
        self.config = config
        self.stages = nn.ModuleList(BeltramiStage(config) for _ in range(config.stages))
        self.initialization = None

    def forward(self, value):
        noisy = value
        for stage in self.stages:
            value = stage(value, noisy)
        return value

    @torch.no_grad()
    def project_constraints(self):
        if self.config.reaction:
            for stage in self.stages:
                stage.reaction.clamp_(0, 1)

    @torch.no_grad()
    def initialize_original(self, model_directory, sigma=25):
        """Embed the author checkpoint preserving the mathematical initialization.

        Wider branches start as exact zero contributions. Extra stages copy
        the last original stage except for a zero output projection, so their
        first forward pass is the identity and their projection can learn.
        """
        from deep_patch_beltrami.caffe_model import load_weights
        directory = Path(model_directory)
        if (directory / "PBSolve_multi").is_dir():
            directory = directory / "PBSolve_multi"
        path = directory / f"PB_{sigma}_80_5__iter_40000.caffemodel"
        weights = load_weights(path)
        for index, stage in enumerate(self.stages):
            suffix = f"-{min(index + 1, 5):03d}"
            for metric in ("g12", "g22"):
                if not torch.equal(torch.from_numpy(weights[metric + suffix][0]),
                                   torch.from_numpy(weights["g11" + suffix][0])):
                    raise ValueError("Original graph metric weights are not shared")
            if not torch.equal(torch.from_numpy(weights["g11" + suffix][1]),
                               torch.from_numpy(weights["g22" + suffix][1])):
                raise ValueError("Original graph diagonal metric biases are not shared")
            for name, source in (("dx", "Ix"), ("dy", "Iy"),
                                 ("divx", "DpxNew"), ("divy", "DpyNew"),
                                 ("projection", "dt_mul_gradNew")):
                tensor = getattr(stage, name).weight
                tensor.zero_()
                original = torch.from_numpy(weights[source + suffix][0])
                slices = tuple(slice(0, n) for n in original.shape)
                tensor[slices].copy_(original)
            for tensor, source in ((stage.metric_weight, weights["g11" + suffix][0]),
                                   (stage.metric_bias, weights["g11" + suffix][1])):
                desired = torch.ones_like(tensor)
                desired[tuple(slice(0, n) for n in source.shape)] = torch.from_numpy(source)
                if self.config.metric == "positive":
                    if desired.min() <= 0:
                        raise ValueError("Cannot positively parameterize nonpositive original metric")
                    tensor.copy_(desired + torch.log(-torch.expm1(-desired)))
                else:
                    tensor.copy_(desired)
            if index >= 5:
                stage.projection.weight.zero_()
            if self.config.width > 80:
                # New feature derivatives learn immediately because their
                # small nonzero projection carries a gradient. The derivative
                # and all cross-channel divergence blocks are initially zero;
                # this preserves the original function exactly.
                stage.projection.weight[:, 80:].normal_(0, 1e-5)
                stage.divx.weight[80:, 80:].normal_(0, 1e-3)
                stage.divy.weight[80:, 80:].normal_(0, 1e-3)
        self.initialization = {
            "source_name": path.name, "source_sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
            "source_sigma": sigma, "config": asdict(self.config),
            "function_preserving": not self.config.zero_dc,
            "function_preserving_scope": "mathematical; changed arithmetic can alter rounding",
            "metric_precision": self.config.metric_precision,
            "arithmetic_change": (
                "float64 metric convolutions/determinant/flux; float32 learned dense operators"
                if self.config.metric_precision == "float64" else "legacy float32 arithmetic"),
            "metric_roundtrip": self.config.metric == "positive",
        }
        return self.initialization

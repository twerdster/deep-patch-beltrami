"""Long, untied Beltrami rollouts with identity-initialized refinements.

The first five stages preserve the selected reaction operator mathematically.
Metric/flux arithmetic is float64; all learned convolutions and parameters are
float32. This arithmetic change and positive-parameterization roundtrip can
change rounding relative to the published float32 reaction model. Positive
metrics do not imply global stability of the free learned flow update.
"""
from dataclasses import asdict, dataclass
import hashlib
from pathlib import Path

import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.checkpoint import checkpoint


def source_hash():
    """Hash namespace filenames and bytes, excluding caches and other packages."""
    digest = hashlib.sha256()
    for path in sorted(Path(__file__).parent.glob("*.py")):
        digest.update(path.name.encode())
        digest.update(path.read_bytes())
    return digest.hexdigest()


@dataclass(frozen=True)
class ModelConfig:
    width: int = 80
    stages: int = 32
    metric: str = "positive"
    zero_dc: bool = False
    step_scale: float = 1000.0
    reaction: bool = True
    metric_precision: str = "float64"
    checkpoint_blocks: int = 4
    refinement_scale: float | None = None

    def __post_init__(self):
        if self.width != 80 or self.stages < 5:
            raise ValueError("Long-rollout initialization requires width80 and stages>=5")
        if self.metric != "positive" or self.metric_precision != "float64":
            raise ValueError("Long rollouts require positive metrics and float64 metric arithmetic")
        if self.zero_dc or not self.reaction or self.step_scale != 1000.0:
            raise ValueError("Keep the selected reaction operator's derivatives and step_scale")
        if self.checkpoint_blocks < 0:
            raise ValueError("checkpoint_blocks must be nonnegative;0 disables checkpointing")
        if self.refinement_scale is None:
            object.__setattr__(self, "refinement_scale", 1.0 / max(1, self.stages - 5))
        if not 0 < self.refinement_scale <= 1:
            raise ValueError("refinement_scale must be positive and at most1")


class BeltramiStage(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        c = config.width
        self.dx = nn.Conv2d(1, c, 7, bias=False)
        self.dy = nn.Conv2d(1, c, 7, bias=False)
        self.divx = nn.Conv2d(c, c, 7, padding=3, bias=False)
        self.divy = nn.Conv2d(c, c, 7, padding=3, bias=False)
        self.projection = nn.Conv2d(c, 1, 1, bias=False)
        self.metric_weight = nn.Parameter(torch.ones(c, 1, 7, 7))
        self.metric_bias = nn.Parameter(torch.ones(c))
        self.reaction = nn.Parameter(torch.zeros(()))

    def metric_parameters(self):
        return F.softplus(self.metric_weight), F.softplus(self.metric_bias)

    def derivative(self, value, layer):
        return layer(F.pad(value, (3, 3, 3, 3), mode="replicate"))

    def forward(self, value, noisy, scale=1.0):
        dx = self.derivative(value, self.dx).double()
        dy = self.derivative(value, self.dy).double()
        weight, bias = self.metric_parameters()
        weight, bias = weight.double(), bias.double()
        g11 = F.conv2d(dx.square(), weight, bias, padding=3, groups=self.config.width)
        g12 = F.conv2d(dx * dy, weight, padding=3, groups=self.config.width)
        g22 = F.conv2d(dy.square(), weight, bias, padding=3, groups=self.config.width)
        inverse_root = (g11 * g22 - g12.square()).pow(-0.5)
        flux_x = ((dx * g22 - dy * g12) * inverse_root).to(value.dtype)
        flux_y = ((dy * g11 - dx * g12) * inverse_root).to(value.dtype)
        divergence = self.divx(flux_x) + self.divy(flux_y)
        update = self.config.step_scale * self.projection(divergence * inverse_root.to(value.dtype))
        output = value + scale * update
        return output + scale * self.reaction * (noisy - value)


class BeltramiOperator(nn.Module):
    """Native N,C,W,H tensors,0–255 intensities; no centering or output clipping."""

    def __init__(self, config=ModelConfig()):
        super().__init__()
        self.config = config
        self.stages = nn.ModuleList(BeltramiStage(config) for _ in range(config.stages))
        self.initialization = None

    def forward(self, value, active_stages=None, return_intermediates=False):
        active = self.config.stages if active_stages is None else active_stages
        if not isinstance(active, int) or not 5 <= active <= self.config.stages:
            raise ValueError("active_stages must be an integer between5 and configured stages")
        noisy = value
        intermediates = {}
        milestones = {5, 16, 32, 64, active}
        start = 0
        while start < active:
            end = min(active, start + max(1, self.config.checkpoint_blocks))
            # Cut blocks at milestones so auxiliary losses preserve exact stages.
            end = min([end] + [m for m in milestones if start < m < end])

            def run_block(current, observation, first=start, last=end):
                for index in range(first, last):
                    scale = 1.0 if index < 5 else self.config.refinement_scale
                    current = self.stages[index](current, observation, scale)
                return current

            if self.training and torch.is_grad_enabled() and self.config.checkpoint_blocks:
                value = checkpoint(run_block, value, noisy, use_reentrant=False,
                                   preserve_rng_state=False)
            else:
                value = run_block(value, noisy)
            if return_intermediates and end in milestones:
                intermediates[end] = value
            start = end
        return (value, intermediates) if return_intermediates else value

    @torch.no_grad()
    def project_constraints(self):
        for stage in self.stages:
            stage.reaction.clamp_(0, 1)

    @torch.no_grad()
    def initialize_from_safe(self, path, expected_sha256):
        """Load a pinned, tensor-only five-stage reaction model without raw state.

        Extra stages copy stage5's internal operator but zero projection AND
        reaction, making every added stage the identity with live first-step
        gradients into projection/reaction. All stage parameters remain untied.
        """
        path = Path(path)
        actual_sha256 = hashlib.sha256(path.read_bytes()).hexdigest()
        if not isinstance(expected_sha256, str) or actual_sha256 != expected_sha256:
            raise ValueError("Safe warmstart SHA256 does not match expected identity")
        payload = torch.load(path, map_location="cpu", weights_only=True)
        expected_config = {"width": 80, "stages": 5, "metric": "frozen", "zero_dc": False,
                           "step_scale": 1000.0, "reaction": True}
        if (payload.get("format") != "beltrami-operator-tensors-v1"
                or payload.get("implementation") != "beltrami_operator"
                or payload.get("model_config") != expected_config
                or payload.get("sigma") != 25 or payload.get("grayscale_offset") != 25):
            raise ValueError("Warmstart must be the selected five-stage sigma25 reaction operator")
        state = payload["state_dict"]
        expected_keys = {f"stages.{i}.{name}" for i in range(5)
                         for name in self.stages[0].state_dict()}
        if set(state) != expected_keys:
            raise ValueError("Safe warmstart has missing or unexpected tensors")
        if any(not isinstance(tensor, torch.Tensor) or tensor.dtype != torch.float32
               or not torch.isfinite(tensor).all() for tensor in state.values()):
            raise ValueError("Safe warmstart tensors must be finite float32")
        for index, stage in enumerate(self.stages):
            prefix = f"stages.{min(index, 4)}."
            source = {name: state[prefix + name].clone() for name in stage.state_dict()}
            for name in ("metric_weight", "metric_bias"):
                desired = source[name]
                if not (desired > 0).all():
                    raise ValueError("Safe warmstart metric must be strictly positive")
                source[name] = desired + torch.log(-torch.expm1(-desired))
            stage.load_state_dict(source, strict=True)
            if index >= 5:
                stage.projection.weight.zero_()
                stage.reaction.zero_()
        self.initialization = {
            "source_name": path.name, "source_sha256": actual_sha256,
            "source_training_step": payload.get("training_step"),
            "source_tensor_sha256": payload.get("tensor_sha256"),
            "config": asdict(self.config), "source_namespace_sha256": source_hash(),
            "prefix_stages": 5, "untied": True,
            "function_preserving_scope": "added stages exactly identity; prefix mathematical only",
            "arithmetic_change": "float64 metric/flux and positive-metric roundtrip change rounding",
            "extra_initialization": "copied final operator; projection and reaction zero",
            "global_stability_guarantee": False,
        }
        return self.initialization

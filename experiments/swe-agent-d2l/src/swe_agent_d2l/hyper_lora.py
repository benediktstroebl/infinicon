"""Minimal generated-LoRA model for SWE trajectory internalization.

This is intentionally small: the base Qwen model is frozen, the context encoder
is the frozen base transformer, and the hypernetwork emits per-example LoRA
factors for selected linear modules.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import torch
import torch.nn as nn


class GeneratedLoRALinear(nn.Module):
    def __init__(self, base: nn.Linear, *, alpha: float):
        super().__init__()
        self.base = base
        self.alpha = float(alpha)
        self.A: torch.Tensor | None = None
        self.B: torch.Tensor | None = None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.base(x)
        if self.A is None or self.B is None:
            return out
        rank = self.A.shape[-2]
        delta = torch.bmm(torch.bmm(x, self.A.transpose(-1, -2)), self.B.transpose(-1, -2))
        return out + (self.alpha / rank) * delta


LoraRegistry = dict[str, GeneratedLoRALinear]


def inject_generated_lora(
    model: nn.Module,
    *,
    target_modules: tuple[str, ...],
    alpha: float,
) -> LoraRegistry:
    registry: LoraRegistry = {}
    for parent_path, parent in model.named_modules():
        for child_name, child in list(parent.named_children()):
            if child_name in target_modules and isinstance(child, nn.Linear):
                wrapped = GeneratedLoRALinear(child, alpha=alpha)
                setattr(parent, child_name, wrapped)
                path = f"{parent_path}.{child_name}" if parent_path else child_name
                registry[path] = wrapped
    return registry


def set_lora(registry: LoraRegistry, tensors: dict[str, tuple[torch.Tensor, torch.Tensor]]) -> None:
    for path, module in registry.items():
        module.A, module.B = tensors[path]


def clear_lora(registry: LoraRegistry) -> None:
    for module in registry.values():
        module.A = None
        module.B = None


@dataclass
class HyperLoRAConfig:
    hidden_size: int
    rank: int
    alpha: float
    trunk_dim: int
    trunk_layers: int


class TrajectoryHyperNetwork(nn.Module):
    """Context -> generated LoRA factors for every registered module."""

    def __init__(self, registry: LoraRegistry, config: HyperLoRAConfig):
        super().__init__()
        self.registry_paths = list(registry)
        self.rank = config.rank
        self.alpha = config.alpha

        layers: list[nn.Module] = []
        in_dim = config.hidden_size
        for _ in range(config.trunk_layers):
            layers += [nn.Linear(in_dim, config.trunk_dim), nn.GELU()]
            in_dim = config.trunk_dim
        self.trunk = nn.Sequential(*layers)
        self.module_embed = nn.Embedding(len(self.registry_paths), config.trunk_dim)
        self.delta_head = nn.Sequential(
            nn.Linear(config.trunk_dim, config.trunk_dim),
            nn.GELU(),
            nn.Linear(config.trunk_dim, config.rank * config.rank),
        )

        self.A_params = nn.ParameterDict()
        self.B_params = nn.ParameterDict()
        for path, module in registry.items():
            key = _path_key(path)
            d_in = module.base.in_features
            d_out = module.base.out_features
            a = nn.Parameter(torch.empty(config.rank, d_in))
            nn.init.kaiming_uniform_(a, a=5**0.5)
            self.A_params[key] = a
            self.B_params[key] = nn.Parameter(torch.zeros(d_out, config.rank))

    @torch.no_grad()
    def encode_context(
        self,
        base_model: nn.Module,
        ctx_ids: torch.Tensor,
        ctx_attention_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        out = base_model.model(
            input_ids=ctx_ids,
            attention_mask=ctx_attention_mask,
            use_cache=False,
        )
        hidden = out.last_hidden_state
        if ctx_attention_mask is None:
            return hidden.mean(dim=1)
        mask = ctx_attention_mask.unsqueeze(-1).to(hidden.dtype)
        denom = mask.sum(dim=1).clamp_min(1)
        return (hidden * mask).sum(dim=1) / denom

    def forward(
        self,
        base_model: nn.Module,
        ctx_ids: torch.Tensor,
        ctx_attention_mask: torch.Tensor | None = None,
    ) -> dict[str, tuple[torch.Tensor, torch.Tensor]]:
        ctx = self.encode_context(base_model, ctx_ids, ctx_attention_mask)
        trunk = self.trunk(ctx)
        combined = trunk.unsqueeze(1) * self.module_embed.weight.unsqueeze(0)
        deltas = self.delta_head(combined).view(
            trunk.shape[0],
            len(self.registry_paths),
            self.rank,
            self.rank,
        )

        out: dict[str, tuple[torch.Tensor, torch.Tensor]] = {}
        batch = trunk.shape[0]
        for idx, path in enumerate(self.registry_paths):
            key = _path_key(path)
            A = self.A_params[key].unsqueeze(0).expand(batch, -1, -1)
            B_base = self.B_params[key]
            B = torch.einsum("or,brk->bok", B_base, deltas[:, idx])
            out[path] = (A, B)
        return out


def _path_key(path: str) -> str:
    return path.replace(".", "__")


def hyper_lora_state_dict(
    hypernet: TrajectoryHyperNetwork,
    *,
    extra: dict[str, Any] | None = None,
) -> dict[str, Any]:
    return {
        "hypernet": hypernet.state_dict(),
        "registry_paths": hypernet.registry_paths,
        "extra": extra or {},
    }

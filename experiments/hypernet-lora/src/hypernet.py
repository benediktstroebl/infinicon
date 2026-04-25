"""Hypernet: maps a context to per-module LoRA factors.

Encoder: frozen base, mean-pool last hidden states.
Trunk:   small MLP, d_model -> trunk_dim.
Heads:   shared MLP that emits a per-module r x r delta from (trunk_output * module_embed).
Per-module learned A_m [r, d_in] and B_m [d_out, r] (B_m zero-init so LoRA = 0 at start).
Effective LoRA: ΔW = B_m @ Δ_m(c) @ A_m, returned as (A_m, B_m @ Δ_m(c)) to LoRALinear.
"""

from __future__ import annotations

import torch
import torch.nn as nn

from .lora import LoraRegistry


def _path_key(path: str) -> str:
    """ParameterDict can't have '.' in keys."""
    return path.replace(".", "__")


class Hypernet(nn.Module):
    def __init__(
        self,
        registry: LoraRegistry,
        d_model: int,
        rank: int = 16,
        alpha: float = 16.0,
        trunk_dim: int = 512,
        trunk_layers: int = 2,
    ):
        super().__init__()
        self.rank = rank
        self.alpha = alpha
        self.module_paths: list[str] = list(registry.keys())
        n_modules = len(self.module_paths)

        # Trunk: d_model -> trunk_dim, with `trunk_layers` Linear+GELU stages.
        layers: list[nn.Module] = []
        in_dim = d_model
        for _ in range(trunk_layers):
            layers += [nn.Linear(in_dim, trunk_dim), nn.GELU()]
            in_dim = trunk_dim
        self.trunk = nn.Sequential(*layers)

        # Module embeddings: one per LoRALinear, modulate the shared head's input.
        self.module_embed = nn.Embedding(n_modules, trunk_dim)

        # Shared head: emits the per-module r x r delta.
        self.head = nn.Sequential(
            nn.Linear(trunk_dim, trunk_dim),
            nn.GELU(),
            nn.Linear(trunk_dim, rank * rank),
        )

        # Per-module learned A_m, B_m. B_m zero-init ⇒ initial LoRA delta is exactly 0.
        self.A_params = nn.ParameterDict()
        self.B_params = nn.ParameterDict()
        for path, mod in registry.items():
            d_in = mod.base.in_features
            d_out = mod.base.out_features
            key = _path_key(path)
            A = nn.Parameter(torch.empty(rank, d_in))
            nn.init.kaiming_uniform_(A, a=5**0.5)
            self.A_params[key] = A
            self.B_params[key] = nn.Parameter(torch.zeros(d_out, rank))

    @torch.no_grad()
    def encode(self, ctx_ids: torch.Tensor, base: nn.Module) -> torch.Tensor:
        """Forward the frozen base on context, mean-pool last hidden states. Returns [B, d_model]."""
        out = base.model(ctx_ids, use_cache=False)
        return out.last_hidden_state.mean(dim=1)

    def forward(
        self, ctx_ids: torch.Tensor, base: nn.Module,
    ) -> tuple[dict[str, torch.Tensor], dict[str, torch.Tensor]]:
        c = self.encode(ctx_ids, base)                          # [B, d_model]
        c_trunk = self.trunk(c)                                 # [B, trunk_dim]

        # Vectorize the head over all modules: combine c_trunk with each module embedding.
        mod_embs = self.module_embed.weight                     # [n_modules, trunk_dim]
        combined = c_trunk.unsqueeze(1) * mod_embs.unsqueeze(0)  # [B, n_modules, trunk_dim]
        deltas = self.head(combined).view(
            c_trunk.shape[0], -1, self.rank, self.rank,
        )                                                       # [B, n_modules, r, r]

        batch = c_trunk.shape[0]
        A_dict: dict[str, torch.Tensor] = {}
        B_dict: dict[str, torch.Tensor] = {}
        for i, path in enumerate(self.module_paths):
            key = _path_key(path)
            A_m = self.A_params[key]                            # [r, d_in]
            B_m = self.B_params[key]                            # [d_out, r]
            delta = deltas[:, i]                                # [B, r, r]

            A_dict[path] = A_m.unsqueeze(0).expand(batch, -1, -1)         # [B, r, d_in]
            B_dict[path] = torch.einsum("or, brk -> bok", B_m, delta)     # [B, d_out, r]

        return A_dict, B_dict

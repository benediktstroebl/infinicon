"""LoRALinear: a frozen base Linear plus per-step hypernet-emitted (A, B) tensors.

A and B are stored as plain Python attributes (not nn.Parameter, not buffers) so they
flow through autograd as regular tensors emitted by the hypernet, and so FSDP — which
only manages parameters and buffers — leaves them alone.
"""

from __future__ import annotations

import torch
import torch.nn as nn

# A registry maps a stable path string to the live LoRALinear instance. We hand
# references around instead of relying on `named_modules` paths, which change
# once FSDP wraps the model.
LoraRegistry = dict[str, "LoRALinear"]


class LoRALinear(nn.Module):
    """Wraps a frozen nn.Linear. Forward = base(x) + (alpha/r) * (x A^T) B^T when (A, B) are set."""

    def __init__(self, base: nn.Linear, alpha: float = 16.0):
        super().__init__()
        self.base = base
        self.alpha = float(alpha)
        self.A: torch.Tensor | None = None  # shape: [B, r, d_in]
        self.B: torch.Tensor | None = None  # shape: [B, d_out, r]

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.base(x)
        if self.A is None:
            return out
        r = self.A.shape[-2]
        scale = self.alpha / r
        # x: [..., d_in]; A: [B, r, d_in]; B: [B, d_out, r]
        # We assume x is [B, S, d_in] (batched), matching A and B's leading batch dim.
        xa = torch.bmm(x, self.A.transpose(-1, -2))      # [B, S, r]
        xab = torch.bmm(xa, self.B.transpose(-1, -2))    # [B, S, d_out]
        return out + scale * xab


def inject_lora(model: nn.Module, target_names: tuple[str, ...]) -> LoraRegistry:
    """Replace every nn.Linear whose immediate name is in target_names with a LoRALinear.

    Must be called before `accelerator.prepare(model)` so that FSDP wraps after injection.
    Returns a registry of {dotted_path: LoRALinear}; keep this for set_lora / clear_lora.
    """
    registry: LoraRegistry = {}
    for parent_path, parent in model.named_modules():
        for child_name, child in list(parent.named_children()):
            if child_name in target_names and isinstance(child, nn.Linear):
                wrapped = LoRALinear(child)
                setattr(parent, child_name, wrapped)
                path = f"{parent_path}.{child_name}" if parent_path else child_name
                registry[path] = wrapped
    return registry


def lora_shapes(registry: LoraRegistry) -> dict[str, tuple[int, int]]:
    """Per-LoRALinear (in_features, out_features), used by the hypernet to size its heads."""
    return {p: (m.base.in_features, m.base.out_features) for p, m in registry.items()}


def set_lora(
    registry: LoraRegistry,
    A_dict: dict[str, torch.Tensor],
    B_dict: dict[str, torch.Tensor],
) -> None:
    for path, mod in registry.items():
        mod.A = A_dict[path]
        mod.B = B_dict[path]


def clear_lora(registry: LoraRegistry) -> None:
    for mod in registry.values():
        mod.A = None
        mod.B = None


def mean_BA_norm(registry: LoraRegistry) -> torch.Tensor:
    """Diagnostic: mean Frobenius norm of (B @ A) across modules. Zero ⇒ LoRA collapsed."""
    norms = []
    for mod in registry.values():
        if mod.A is None:
            continue
        ba = torch.bmm(mod.B, mod.A)  # [B, d_out, d_in]
        norms.append(ba.flatten(1).norm(dim=-1).mean())
    return torch.stack(norms).mean() if norms else torch.tensor(0.0)

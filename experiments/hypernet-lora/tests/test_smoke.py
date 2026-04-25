"""Smoke tests that exercise the LoRA injection + hypernet logic on a tiny synthetic
Qwen3-shaped model (so we don't need to download the real one). Run with:

    .venv/bin/python -m tests.test_smoke

These tests cover:
1. LoRALinear: zero-LoRA invariance and non-zero delta application.
2. inject_lora / set_lora / clear_lora: registry plumbing on a Qwen3-like structure.
3. Hypernet zero-init invariance (B_m = 0 ⇒ student logits == base-no-context logits).
4. Hypernet output shape correctness.
5. KL gradient flow: hypernet params get gradients, base params don't.
"""

from __future__ import annotations

import sys
import traceback
from types import SimpleNamespace

import torch
import torch.nn as nn
import torch.nn.functional as F

from src.hypernet import Hypernet
from src.lora import LoRALinear, clear_lora, inject_lora, lora_shapes, mean_BA_norm, set_lora


# --------------------------- mock Qwen3-shaped model ---------------------------

class _Attn(nn.Module):
    def __init__(self, d):
        super().__init__()
        self.q_proj = nn.Linear(d, d, bias=False)
        self.k_proj = nn.Linear(d, d, bias=False)
        self.v_proj = nn.Linear(d, d, bias=False)
        self.o_proj = nn.Linear(d, d, bias=False)

    def forward(self, x):
        q, k, v = self.q_proj(x), self.k_proj(x), self.v_proj(x)
        scale = q.size(-1) ** -0.5
        attn = F.softmax((q @ k.transpose(-1, -2)) * scale, dim=-1)
        return self.o_proj(attn @ v)


class _MLP(nn.Module):
    def __init__(self, d, h):
        super().__init__()
        self.gate_proj = nn.Linear(d, h, bias=False)
        self.up_proj = nn.Linear(d, h, bias=False)
        self.down_proj = nn.Linear(h, d, bias=False)

    def forward(self, x):
        return self.down_proj(F.silu(self.gate_proj(x)) * self.up_proj(x))


class _Layer(nn.Module):
    def __init__(self, d, h):
        super().__init__()
        self.self_attn = _Attn(d)
        self.mlp = _MLP(d, h)

    def forward(self, x):
        return x + self.mlp(x + self.self_attn(x))


class _Inner(nn.Module):
    def __init__(self, vocab, d, h, n_layers):
        super().__init__()
        self.embed_tokens = nn.Embedding(vocab, d)
        self.layers = nn.ModuleList([_Layer(d, h) for _ in range(n_layers)])

    def forward(self, input_ids, **kw):
        x = self.embed_tokens(input_ids)
        for layer in self.layers:
            x = layer(x)
        return SimpleNamespace(last_hidden_state=x)


class MockBase(nn.Module):
    """Mimics enough of Qwen3ForCausalLM that our code paths exercise correctly."""

    def __init__(self, vocab=64, d=32, h=64, n_layers=3):
        super().__init__()
        self.config = SimpleNamespace(hidden_size=d, vocab_size=vocab, num_hidden_layers=n_layers)
        self.model = _Inner(vocab, d, h, n_layers)
        self.lm_head = nn.Linear(d, vocab, bias=False)

    def forward(self, input_ids, **kw):
        h = self.model(input_ids, **kw).last_hidden_state
        return SimpleNamespace(logits=self.lm_head(h))


TARGETS = ("q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj")


def _device():
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


# ----------------------------------- tests ------------------------------------

def test_lora_linear_zero_invariance():
    torch.manual_seed(0)
    base = nn.Linear(8, 8, bias=False)
    wrapped = LoRALinear(base, alpha=16.0)
    x = torch.randn(2, 4, 8)
    a = base(x)
    b = wrapped(x)
    assert torch.equal(a, b), "Without (A, B) set, LoRALinear must equal base."


def test_lora_linear_applies_delta():
    torch.manual_seed(1)
    d_in, d_out, r, B = 8, 6, 2, 3
    base = nn.Linear(d_in, d_out, bias=False)
    wrapped = LoRALinear(base, alpha=4.0)
    x = torch.randn(B, 5, d_in)
    A = torch.randn(B, r, d_in)
    Bm = torch.randn(B, d_out, r)
    wrapped.A, wrapped.B = A, Bm
    out = wrapped(x)
    expected = base(x) + (4.0 / r) * torch.bmm(torch.bmm(x, A.transpose(-1, -2)), Bm.transpose(-1, -2))
    assert torch.allclose(out, expected, atol=1e-5), "Delta application mismatch."


def test_inject_lora_replaces_targets():
    torch.manual_seed(2)
    m = MockBase(vocab=16, d=8, h=16, n_layers=2)
    registry = inject_lora(m, TARGETS)
    expected = 2 * len(TARGETS)
    assert len(registry) == expected, f"Expected {expected} LoRALinear modules, got {len(registry)}"
    for path, mod in registry.items():
        assert isinstance(mod, LoRALinear), f"{path} not a LoRALinear"
    # Forward still works.
    ids = torch.randint(0, 16, (2, 6))
    _ = m(ids)


def test_set_clear_lora_round_trip():
    torch.manual_seed(3)
    m = MockBase(vocab=16, d=8, h=16, n_layers=2)
    registry = inject_lora(m, TARGETS)
    ids = torch.randint(0, 16, (2, 6))

    base_logits = m(ids).logits.clone()

    shapes = lora_shapes(registry)
    A_dict, B_dict = {}, {}
    for path, (d_in, d_out) in shapes.items():
        A_dict[path] = torch.randn(2, 4, d_in) * 0.1
        B_dict[path] = torch.randn(2, d_out, 4) * 0.1
    set_lora(registry, A_dict, B_dict)

    perturbed = m(ids).logits
    assert not torch.allclose(perturbed, base_logits, atol=1e-4), "Setting LoRA must change logits."

    clear_lora(registry)
    restored = m(ids).logits
    assert torch.allclose(restored, base_logits, atol=1e-6), "After clear, logits must match base."


def test_hypernet_zero_init_invariance():
    """With B_m = 0 in the hypernet, the student must equal the base-no-context logits."""
    torch.manual_seed(4)
    dev = _device()
    m = MockBase(vocab=24, d=16, h=32, n_layers=2).to(dev)
    registry = inject_lora(m, TARGETS)
    hn = Hypernet(
        registry=registry, d_model=16, rank=4, alpha=4.0, trunk_dim=16, trunk_layers=2,
    ).to(dev)

    ctx = torch.randint(0, 24, (2, 7), device=dev)
    cont = torch.randint(0, 24, (2, 5), device=dev)

    base_logits = m(cont).logits.clone()

    A, B = hn(ctx, m)
    set_lora(registry, A, B)
    student_logits = m(cont).logits
    clear_lora(registry)

    assert torch.allclose(student_logits, base_logits, atol=1e-5), (
        "B_m is zero-init, so the student must exactly match base-no-context."
    )


def test_hypernet_output_shapes():
    torch.manual_seed(5)
    m = MockBase(vocab=24, d=16, h=32, n_layers=2)
    registry = inject_lora(m, TARGETS)
    hn = Hypernet(registry=registry, d_model=16, rank=4, alpha=4.0, trunk_dim=8, trunk_layers=1)
    ctx = torch.randint(0, 24, (3, 7))
    A, B = hn(ctx, m)
    assert set(A) == set(registry), "A_dict keys must match registry"
    assert set(B) == set(registry), "B_dict keys must match registry"
    for path, (d_in, d_out) in lora_shapes(registry).items():
        assert A[path].shape == (3, 4, d_in), (path, A[path].shape)
        assert B[path].shape == (3, d_out, 4), (path, B[path].shape)


def test_kl_gradient_flow():
    """Critical test: KL backward must produce gradients on hypernet params and NONE on base."""
    torch.manual_seed(6)
    dev = _device()
    m = MockBase(vocab=24, d=16, h=32, n_layers=2).to(dev)
    m.requires_grad_(False)
    registry = inject_lora(m, TARGETS)
    hn = Hypernet(
        registry=registry, d_model=16, rank=4, alpha=4.0, trunk_dim=16, trunk_layers=2,
    ).to(dev)

    # Force the hypernet's B_params off the zero init so gradients exist for both A and B params.
    with torch.no_grad():
        for p in hn.B_params.values():
            p.add_(torch.randn_like(p) * 0.01)

    ctx = torch.randint(0, 24, (2, 7), device=dev)
    cont = torch.randint(0, 24, (2, 5), device=dev)

    full = torch.cat([ctx, cont], dim=1)
    with torch.no_grad():
        teacher = m(full).logits[:, ctx.size(1):-1]

    A, B = hn(ctx, m)
    set_lora(registry, A, B)
    student = m(cont).logits[:, :-1]
    clear_lora(registry)

    loss = F.kl_div(
        F.log_softmax(student.float(), -1),
        F.log_softmax(teacher.float(), -1),
        reduction="batchmean", log_target=True,
    )
    loss.backward()

    have_grad = sum(int(p.grad is not None and p.grad.abs().sum() > 0)
                    for p in hn.parameters() if p.requires_grad)
    total = sum(1 for p in hn.parameters() if p.requires_grad)
    assert have_grad >= total // 2, f"Only {have_grad}/{total} hypernet params got gradients."

    base_grads = [p.grad for p in m.parameters()]
    assert all(g is None or g.abs().sum() == 0 for g in base_grads), (
        "Base params must not accumulate gradients."
    )


def test_mean_BA_norm_diagnostic():
    torch.manual_seed(7)
    m = MockBase(vocab=16, d=8, h=16, n_layers=1)
    registry = inject_lora(m, TARGETS)
    # No (A, B) set → norm 0.
    n0 = mean_BA_norm(registry).item()
    assert n0 == 0.0
    shapes = lora_shapes(registry)
    A = {p: torch.randn(1, 2, d_in) for p, (d_in, _) in shapes.items()}
    B = {p: torch.randn(1, d_out, 2) for p, (_, d_out) in shapes.items()}
    set_lora(registry, A, B)
    n1 = mean_BA_norm(registry).item()
    assert n1 > 0


# ----------------------------------- runner ------------------------------------

def main():
    tests = [
        test_lora_linear_zero_invariance,
        test_lora_linear_applies_delta,
        test_inject_lora_replaces_targets,
        test_set_clear_lora_round_trip,
        test_hypernet_zero_init_invariance,
        test_hypernet_output_shapes,
        test_kl_gradient_flow,
        test_mean_BA_norm_diagnostic,
    ]
    failed = 0
    for t in tests:
        try:
            t()
            print(f"  PASS  {t.__name__}")
        except Exception as e:
            failed += 1
            print(f"  FAIL  {t.__name__}: {e}")
            traceback.print_exc()
    print(f"\n{len(tests) - failed}/{len(tests)} passed")
    sys.exit(failed)


if __name__ == "__main__":
    main()

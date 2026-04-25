"""Integration test against the real Qwen/Qwen3-0.6B-Base model.

Validates:
  - inject_lora finds the right modules in the actual Qwen3 graph.
  - Hypernet builds and runs end-to-end on real shapes.
  - With B_m = 0, student forward is logit-identical to base-no-context (zero invariance).
  - One full KL training step runs and produces non-zero gradients on hypernet params.

Downloads ~1.2 GB the first time. Uses tiny ctx/cont lengths so it finishes in <1 min on MPS.
"""

from __future__ import annotations

import sys
import time
import traceback

import torch
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer

from src.hypernet import Hypernet
from src.lora import clear_lora, inject_lora, mean_BA_norm, set_lora


MODEL = "Qwen/Qwen3-0.6B-Base"
TARGETS = ("q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj")
CTX_LEN = 64
CONT_LEN = 16


def _device_and_dtype():
    if torch.backends.mps.is_available():
        # MPS support for bf16 is uneven across ops in some torch builds; fp32 is safer.
        return torch.device("mps"), torch.float32
    if torch.cuda.is_available():
        return torch.device("cuda"), torch.bfloat16
    return torch.device("cpu"), torch.float32


def _load(device, dtype):
    print(f"  loading {MODEL} ({dtype}) on {device}...")
    t0 = time.time()
    tokenizer = AutoTokenizer.from_pretrained(MODEL)
    base = AutoModelForCausalLM.from_pretrained(MODEL, torch_dtype=dtype).to(device).eval()
    base.requires_grad_(False)
    print(f"  loaded in {time.time() - t0:.1f}s, hidden_size={base.config.hidden_size}, "
          f"layers={base.config.num_hidden_layers}")
    return tokenizer, base


def test_inject_finds_real_modules():
    device, dtype = _device_and_dtype()
    _, base = _load(device, dtype)
    registry = inject_lora(base, TARGETS)
    expected = base.config.num_hidden_layers * len(TARGETS)
    assert len(registry) == expected, f"Expected {expected} LoRALinears, got {len(registry)}"
    sample = next(iter(registry))
    print(f"  ok: {len(registry)} LoRALinears injected; sample path: {sample}")


def test_real_zero_invariance():
    device, dtype = _device_and_dtype()
    tokenizer, base = _load(device, dtype)
    registry = inject_lora(base, TARGETS)
    hn = Hypernet(
        registry=registry, d_model=base.config.hidden_size, rank=4, alpha=4.0,
        trunk_dim=64, trunk_layers=1,
    ).to(device).to(dtype)

    ctx_ids = torch.randint(0, base.config.vocab_size, (1, CTX_LEN), device=device)
    cont_ids = torch.randint(0, base.config.vocab_size, (1, CONT_LEN), device=device)

    with torch.no_grad():
        base_logits = base(cont_ids).logits.clone()
        A, B = hn(ctx_ids, base)
        set_lora(registry, A, B)
        student_logits = base(cont_ids).logits
        clear_lora(registry)

    diff = (student_logits - base_logits).abs().max().item()
    assert diff < 1e-3, f"Zero-init student must match base; max abs diff = {diff:.3e}"
    print(f"  ok: max abs diff = {diff:.2e} (B_m = 0 ⇒ student == base)")


def test_real_kl_step_runs():
    device, dtype = _device_and_dtype()
    tokenizer, base = _load(device, dtype)
    registry = inject_lora(base, TARGETS)
    hn = Hypernet(
        registry=registry, d_model=base.config.hidden_size, rank=4, alpha=4.0,
        trunk_dim=64, trunk_layers=1,
    ).to(device).to(dtype)

    # Bump B_params off zero so the very first step produces non-zero LoRA + grads on B.
    with torch.no_grad():
        for p in hn.B_params.values():
            p.add_(torch.randn_like(p) * 0.01)

    ctx_ids = torch.randint(0, base.config.vocab_size, (1, CTX_LEN), device=device)
    cont_ids = torch.randint(0, base.config.vocab_size, (1, CONT_LEN), device=device)

    full = torch.cat([ctx_ids, cont_ids], dim=1)
    with torch.no_grad():
        teacher = base(full).logits[:, ctx_ids.size(1):-1]

    A, B = hn(ctx_ids, base)
    set_lora(registry, A, B)
    student = base(cont_ids).logits[:, :-1]
    clear_lora(registry)

    loss = F.kl_div(
        F.log_softmax(student.float(), -1),
        F.log_softmax(teacher.float(), -1),
        reduction="batchmean", log_target=True,
    )
    loss.backward()

    with_grad = sum(int(p.grad is not None and p.grad.abs().sum() > 0)
                    for p in hn.parameters() if p.requires_grad)
    total = sum(1 for p in hn.parameters() if p.requires_grad)
    base_no_grad = all(p.grad is None or p.grad.abs().sum() == 0 for p in base.parameters())

    n_norm = mean_BA_norm(registry).item() if A else 0.0
    print(f"  ok: kl={loss.item():.4f}, hypernet_params_with_grad={with_grad}/{total}, "
          f"base_no_grad={base_no_grad}")
    assert torch.isfinite(loss), "KL loss must be finite"
    assert with_grad >= total // 2, "At least half of hypernet params should have grad"
    assert base_no_grad, "Base params must not accumulate grads"


def main():
    tests = [test_inject_finds_real_modules, test_real_zero_invariance, test_real_kl_step_runs]
    failed = 0
    for t in tests:
        print(f"\n>>> {t.__name__}")
        try:
            t()
            print(f"  PASS  {t.__name__}")
        except Exception as e:
            failed += 1
            print(f"  FAIL  {t.__name__}: {e}")
            traceback.print_exc()
    print(f"\n{len(tests) - failed}/{len(tests)} integration tests passed")
    sys.exit(failed)


if __name__ == "__main__":
    main()

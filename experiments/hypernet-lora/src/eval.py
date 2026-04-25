"""Eval suite: held-out KL/perplexity, NIAH retrieval, negative-control.

Each function returns a dict of scalars suitable for wandb.log. Standalone main()
loads a hypernet checkpoint and runs the full suite.
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Iterable

import torch
import torch.nn.functional as F
import yaml
from transformers import AutoModelForCausalLM, AutoTokenizer

from .data import fineweb_pairs, niah_pairs
from .hypernet import Hypernet
from .lora import LoraRegistry, clear_lora, inject_lora, set_lora


def _ppl(logits: torch.Tensor, targets: torch.Tensor) -> float:
    logp = F.log_softmax(logits.float(), dim=-1)
    nll = -logp.gather(-1, targets.unsqueeze(-1)).squeeze(-1)
    return nll.mean().exp().item()


@torch.no_grad()
def eval_kl(
    base, hypernet, registry: LoraRegistry, batches: Iterable[dict], device,
) -> dict:
    """Held-out forward KL plus student / teacher / base-no-context perplexity."""
    kl_sum = ppl_s = ppl_t = ppl_n = 0.0
    n = 0
    for batch in batches:
        ctx = batch["ctx"].to(device)
        cont = batch["cont"].to(device)
        full = torch.cat([ctx, cont], dim=1)
        teacher = base(full).logits[:, ctx.size(1):-1]

        A, B = hypernet(ctx, base)
        set_lora(registry, A, B)
        student = base(cont).logits[:, :-1]
        clear_lora(registry)

        base_no = base(cont).logits[:, :-1]
        targets = cont[:, 1:]

        kl_sum += F.kl_div(
            F.log_softmax(student.float(), -1),
            F.log_softmax(teacher.float(), -1),
            reduction="batchmean", log_target=True,
        ).item()
        ppl_s += _ppl(student, targets)
        ppl_t += _ppl(teacher, targets)
        ppl_n += _ppl(base_no, targets)
        n += 1

    return {
        "kl_held_out": kl_sum / n,
        "student_ppl": ppl_s / n,
        "teacher_ppl": ppl_t / n,
        "base_no_ctx_ppl": ppl_n / n,
        "ppl_ratio_student_teacher": (ppl_s / n) / (ppl_t / n),
    }


@torch.no_grad()
def eval_niah(base, hypernet, registry, probes, device, tokenizer) -> dict:
    """Greedy-decode the answer under the hypernet-LoRA, check string containment."""
    by_depth: dict[float, list[bool]] = {}
    for p in probes:
        ctx = torch.tensor([p["ctx"]], dtype=torch.long, device=device)
        cont = torch.tensor([p["cont"]], dtype=torch.long, device=device)

        A, B = hypernet(ctx, base)
        set_lora(registry, A, B)
        ans_ids = tokenizer(p["answer"], add_special_tokens=False).input_ids
        gen = base.generate(
            cont, max_new_tokens=len(ans_ids) + 4, do_sample=False,
            pad_token_id=tokenizer.eos_token_id,
        )
        clear_lora(registry)

        new_text = tokenizer.decode(gen[0, cont.size(1):].tolist())
        by_depth.setdefault(p["depth"], []).append(p["answer"] in new_text)

    out = {f"niah_acc_d{int(d * 100):02d}": sum(v) / len(v) for d, v in by_depth.items()}
    out["niah_acc_overall"] = sum(sum(v) for v in by_depth.values()) / sum(
        len(v) for v in by_depth.values()
    )
    return out


@torch.no_grad()
def eval_negative_control(
    base, hypernet, registry: LoraRegistry, batches: Iterable[dict], device,
) -> dict:
    """Mismatched (ctx, cont) pairs. A non-cheating student should match base-no-ctx."""
    ppl_s = ppl_n = 0.0
    n = 0
    for batch in batches:
        ctx = batch["ctx"].to(device)
        cont_shuffled = torch.roll(batch["cont"], shifts=1, dims=0).to(device)

        A, B = hypernet(ctx, base)
        set_lora(registry, A, B)
        student = base(cont_shuffled).logits[:, :-1]
        clear_lora(registry)
        base_no = base(cont_shuffled).logits[:, :-1]

        targets = cont_shuffled[:, 1:]
        ppl_s += _ppl(student, targets)
        ppl_n += _ppl(base_no, targets)
        n += 1

    return {
        "neg_control_student_ppl": ppl_s / n,
        "neg_control_base_no_ctx_ppl": ppl_n / n,
        "neg_control_gap": (ppl_s - ppl_n) / n,           # ≈ 0 ⇒ student ignores wrong ctx
    }


def _build(cfg: dict, device):
    tokenizer = AutoTokenizer.from_pretrained(cfg["model"]["name"])
    base = AutoModelForCausalLM.from_pretrained(
        cfg["model"]["name"], torch_dtype=torch.bfloat16,
    ).to(device).eval()
    base.requires_grad_(False)
    registry = inject_lora(base, tuple(cfg["lora"]["targets"]))
    hypernet = Hypernet(
        registry=registry,
        d_model=base.config.hidden_size,
        rank=cfg["lora"]["rank"],
        alpha=cfg["lora"]["alpha"],
        trunk_dim=cfg["hypernet"]["trunk_dim"],
        trunk_layers=cfg["hypernet"]["trunk_layers"],
    ).to(device)
    return tokenizer, base, hypernet, registry


def _batched_pairs(it, bs):
    buf: list[dict] = []
    for ex in it:
        buf.append(ex)
        if len(buf) == bs:
            yield {
                "ctx": torch.tensor([b["ctx"] for b in buf], dtype=torch.long),
                "cont": torch.tensor([b["cont"] for b in buf], dtype=torch.long),
            }
            buf = []


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--ckpt", required=True)
    parser.add_argument("--n_eval_batches", type=int, default=64)
    parser.add_argument("--n_niah", type=int, default=300)
    args = parser.parse_args()

    cfg = yaml.safe_load(Path(args.config).read_text())
    device = torch.device("cuda")
    tokenizer, base, hypernet, registry = _build(cfg, device)
    hypernet.load_state_dict(torch.load(args.ckpt, map_location=device))
    hypernet.eval()

    pairs = fineweb_pairs(
        tokenizer,
        snapshot=cfg["data"]["snapshot"],
        context_len=cfg["data"]["context_len"],
        continuation_len=cfg["data"]["continuation_len"],
        min_doc_tokens=cfg["data"]["min_doc_tokens"],
        held_out=True,
    )
    bs = cfg["train"]["batch_size"]
    batches = _batched_pairs(pairs, bs)

    kl_batches = [next(batches) for _ in range(args.n_eval_batches)]
    neg_batches = [next(batches) for _ in range(args.n_eval_batches)]

    print("=== held-out KL ===");          print(eval_kl(base, hypernet, registry, kl_batches, device))
    print("=== negative control ===");     print(eval_negative_control(base, hypernet, registry, neg_batches, device))

    probes = niah_pairs(
        tokenizer, n=args.n_niah,
        filler_snapshot=cfg["data"]["niah_filler_snapshot"],
        context_len=cfg["data"]["context_len"],
    )
    print("=== NIAH ==="); print(eval_niah(base, hypernet, registry, probes, device, tokenizer))


if __name__ == "__main__":
    main()

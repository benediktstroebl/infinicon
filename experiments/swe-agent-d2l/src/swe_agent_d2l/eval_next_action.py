"""Evaluate tail-only, adapter, oracle, and mismatched-context baselines."""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from functools import partial
from pathlib import Path
from typing import Any

import torch
from datasets import load_from_disk
from torch.utils.data import DataLoader
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer

from .batching import collate_eval
from .config import load_yaml
from .hyper_lora import (
    HyperLoRAConfig,
    TrajectoryHyperNetwork,
    clear_lora,
    inject_generated_lora,
    set_lora,
)
from .losses import target_nll


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    parser.add_argument("--data", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--split", default="test")
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--limit", type=int, default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    cfg = load_yaml(args.config)
    tokenizer = AutoTokenizer.from_pretrained(cfg["model"]["name"], trust_remote_code=True)
    pad_token_id = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else tokenizer.eos_token_id

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    base = AutoModelForCausalLM.from_pretrained(
        cfg["model"]["name"],
        torch_dtype=_torch_dtype(cfg["model"].get("torch_dtype", "bfloat16")),
        attn_implementation=cfg["model"].get("attn_implementation", "sdpa"),
        trust_remote_code=True,
    ).to(device).eval()
    base.requires_grad_(False)

    registry = inject_generated_lora(
        base,
        target_modules=tuple(cfg["lora"]["target_modules"]),
        alpha=float(cfg["lora"]["alpha"]),
    )
    hypernet = TrajectoryHyperNetwork(
        registry,
        HyperLoRAConfig(
            hidden_size=base.config.hidden_size,
            rank=int(cfg["lora"]["rank"]),
            alpha=float(cfg["lora"]["alpha"]),
            trunk_dim=int(cfg["hypernet"]["trunk_dim"]),
            trunk_layers=int(cfg["hypernet"]["trunk_layers"]),
        ),
    ).to(base.device)
    state = torch.load(args.checkpoint, map_location=base.device)
    hypernet.load_state_dict(state["hypernet"])
    hypernet.eval()

    ds = load_from_disk(args.data)[args.split]
    if args.limit is not None:
        ds = ds.select(range(min(args.limit, len(ds))))
    loader = DataLoader(
        ds,
        batch_size=args.batch_size,
        shuffle=False,
        collate_fn=partial(collate_eval, pad_token_id=pad_token_id),
    )

    rows = []
    grouped: dict[str, list[dict[str, float]]] = defaultdict(list)
    for batch in tqdm(loader, desc=f"eval:{args.split}"):
        metrics = eval_batch(base, hypernet, registry, batch)
        rows.extend(metrics)
        for row in metrics:
            for key in (
                "target_kind",
                "tool_name",
                "context_length_bucket",
                "prompt_intent",
                "session_position",
            ):
                grouped[f"{key}:{row[key]}"].append(row)

    summary = {"overall": summarize(rows)}
    for key, vals in grouped.items():
        summary[key] = summarize(vals)
    print(json.dumps(summary, indent=2, sort_keys=True))


@torch.inference_mode()
def eval_batch(
    base,
    hypernet: TrajectoryHyperNetwork,
    registry,
    batch: dict[str, Any],
) -> list[dict[str, float]]:
    device = base.device
    input_ids = batch["input_ids"].to(device)
    attention_mask = batch["attention_mask"].to(device)
    labels = batch["labels"].to(device)

    tail_logits = base(
        input_ids=input_ids,
        attention_mask=attention_mask,
        use_cache=False,
    ).logits
    tail_nll = target_nll(tail_logits, labels)

    lora = hypernet(
        base,
        batch["ctx_ids"].to(device),
        batch["ctx_attention_mask"].to(device),
    )
    set_lora(registry, lora)
    adapter_logits = base(
        input_ids=input_ids,
        attention_mask=attention_mask,
        use_cache=False,
    ).logits
    clear_lora(registry)
    adapter_nll = target_nll(adapter_logits, labels)

    # Negative control: roll contexts inside the batch.
    rolled_ctx = torch.roll(batch["ctx_ids"], shifts=1, dims=0).to(device)
    rolled_mask = torch.roll(batch["ctx_attention_mask"], shifts=1, dims=0).to(device)
    bad_lora = hypernet(base, rolled_ctx, rolled_mask)
    set_lora(registry, bad_lora)
    bad_logits = base(
        input_ids=input_ids,
        attention_mask=attention_mask,
        use_cache=False,
    ).logits
    clear_lora(registry)
    bad_nll = target_nll(bad_logits, labels)

    if "teacher_input_ids" in batch:
        teacher_ids = batch["teacher_input_ids"].to(device)
        teacher_mask = batch["teacher_attention_mask"].to(device)
        teacher_logits = base(
            input_ids=teacher_ids,
            attention_mask=teacher_mask,
            use_cache=False,
        ).logits
        teacher_labels = torch.full_like(teacher_ids, -100)
        for i, (start, end) in enumerate(batch["teacher_response_start_end"].tolist()):
            teacher_labels[i, start:end] = teacher_ids[i, start:end]
        oracle_nll = target_nll(teacher_logits, teacher_labels)
    else:
        oracle_nll = torch.full_like(tail_nll, float("nan"))

    out = []
    for i in range(input_ids.shape[0]):
        denom = tail_nll[i] - oracle_nll[i]
        gap = (tail_nll[i] - adapter_nll[i]) / denom if torch.isfinite(denom) and abs(float(denom)) > 1e-8 else torch.tensor(float("nan"))
        bad_gap = (tail_nll[i] - bad_nll[i]) / denom if torch.isfinite(denom) and abs(float(denom)) > 1e-8 else torch.tensor(float("nan"))
        out.append(
            {
                "tail_nll": float(tail_nll[i].cpu()),
                "adapter_nll": float(adapter_nll[i].cpu()),
                "oracle_nll": float(oracle_nll[i].cpu()),
                "mismatch_nll": float(bad_nll[i].cpu()),
                "gap_closure": float(gap.cpu()),
                "mismatch_gap_closure": float(bad_gap.cpu()),
                "target_kind": batch.get("target_kind", ["unknown"])[i],
                "tool_name": _tool_name(batch.get("target_tools", [[]])[i]),
                "context_length_bucket": batch.get("context_length_bucket", ["unknown"])[i],
                "prompt_intent": batch.get("prompt_intent", ["unknown"])[i],
                "session_position": batch.get("session_position", ["unknown"])[i],
            }
        )
    return out


def summarize(rows: list[dict[str, float]]) -> dict[str, float]:
    numeric = [
        "tail_nll",
        "adapter_nll",
        "oracle_nll",
        "mismatch_nll",
        "gap_closure",
        "mismatch_gap_closure",
    ]
    out = {"count": len(rows)}
    for key in numeric:
        vals = [r[key] for r in rows if r[key] == r[key]]
        out[key] = sum(vals) / len(vals) if vals else float("nan")
    return out


def _tool_name(value: Any) -> str:
    if isinstance(value, str):
        return value or "none"
    if value:
        return str(value[0])
    return "none"


def _torch_dtype(name: str) -> torch.dtype:
    if name == "float16":
        return torch.float16
    if name == "float32":
        return torch.float32
    return torch.bfloat16


if __name__ == "__main__":
    main()

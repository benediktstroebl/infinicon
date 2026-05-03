"""Precompute full-context teacher top-k logprobs for target tokens."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import torch
from datasets import Dataset, DatasetDict, load_from_disk
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer

from .env import load_dotenv_from_parents


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--model", default="Qwen/Qwen3-4B-Instruct-2507")
    parser.add_argument("--split", default="train")
    parser.add_argument("--top-k", type=int, default=64)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--attn-implementation", default="sdpa")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    load_dotenv_from_parents()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
    pad_token_id = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else tokenizer.eos_token_id
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = AutoModelForCausalLM.from_pretrained(
        args.model,
        torch_dtype=torch.bfloat16,
        attn_implementation=args.attn_implementation,
        trust_remote_code=True,
    ).to(device).eval()

    data = load_from_disk(args.data)
    split_ds = data[args.split]
    if args.limit is not None:
        split_ds = split_ds.select(range(min(args.limit, len(split_ds))))

    rows: list[dict[str, Any]] = []
    for sample in tqdm(split_ds, desc=f"teacher:{args.split}"):
        enriched = dict(sample)
        vals, inds = compute_teacher_topk(model, sample, top_k=args.top_k, pad_token_id=pad_token_id)
        enriched["logprobs_vals"] = vals
        enriched["logprobs_indices"] = inds
        rows.append(enriched)

    out_dict = DatasetDict()
    for split_name in data.keys():
        if split_name == args.split:
            out_dict[split_name] = Dataset.from_list(rows)
        else:
            out_dict[split_name] = data[split_name]
    out_dict.save_to_disk(out)

    metadata = {
        "source": args.data,
        "model": args.model,
        "split": args.split,
        "top_k": args.top_k,
        "rows": len(rows),
    }
    (out / f"teacher_{args.split}_metadata.json").write_text(
        json.dumps(metadata, indent=2, sort_keys=True)
    )
    print(json.dumps(metadata, indent=2, sort_keys=True))


@torch.inference_mode()
def compute_teacher_topk(
    model: AutoModelForCausalLM,
    sample: dict[str, Any],
    *,
    top_k: int,
    pad_token_id: int,
) -> tuple[list[list[float]], list[list[int]]]:
    device = next(model.parameters()).device
    input_ids = torch.tensor([sample["teacher_input_ids"]], dtype=torch.long, device=device)
    attention_mask = (input_ids != pad_token_id).long()
    start, end = sample["teacher_response_start_end"]
    outputs = model(input_ids=input_ids, attention_mask=attention_mask, use_cache=False)
    # Token at position p is predicted by logits at p - 1.
    logits = outputs.logits[0, start - 1 : end - 1].float()
    logp = torch.log_softmax(logits, dim=-1)
    vals, inds = torch.topk(logp, k=min(top_k, logp.shape[-1]), dim=-1)
    return vals.cpu().tolist(), inds.cpu().tolist()


if __name__ == "__main__":
    main()

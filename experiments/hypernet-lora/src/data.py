"""FineWeb-Edu (ctx, cont) pair stream and NIAH probe generator."""

from __future__ import annotations

import hashlib
import random
from typing import Iterator

from datasets import load_dataset
from datasets.distributed import split_dataset_by_node
from transformers import PreTrainedTokenizerBase

_NAMES = [
    "Aria", "Beck", "Cleo", "Dax", "Esme", "Finn", "Gus", "Hana",
    "Iris", "Jax", "Kai", "Lior", "Mira", "Nia", "Otis", "Pia",
    "Quinn", "Rai", "Sage", "Tora", "Uma", "Vex", "Wren", "Xiu",
    "Yara", "Zev",
]


def fineweb_pairs(
    tokenizer: PreTrainedTokenizerBase,
    snapshot: str = "CC-MAIN-2025-26",
    context_len: int = 4096,
    continuation_len: int = 512,
    min_doc_tokens: int = 4608,
    held_out: bool = False,
    rank: int = 0,
    world_size: int = 1,
) -> Iterator[dict]:
    """Stream FineWeb-Edu and yield (ctx, cont) token-id pairs.

    Train/held-out split: hash(doc_id) % 100, bucket 99 is held-out.
    Multi-rank shards via datasets.distributed.split_dataset_by_node.
    """
    ds = load_dataset(
        "HuggingFaceFW/fineweb-edu", snapshot, split="train", streaming=True,
    )
    ds = ds.filter(lambda ex: ex.get("token_count", 0) >= min_doc_tokens)
    if world_size > 1:
        ds = split_dataset_by_node(ds, rank=rank, world_size=world_size)

    need = context_len + continuation_len
    for ex in ds:
        bucket = int(hashlib.md5(ex["id"].encode()).hexdigest(), 16) % 100
        if held_out != (bucket == 99):
            continue
        ids = tokenizer(ex["text"], add_special_tokens=False).input_ids
        if len(ids) < need:
            continue
        yield {"ctx": ids[:context_len], "cont": ids[context_len:need]}


def niah_pairs(
    tokenizer: PreTrainedTokenizerBase,
    n: int = 1000,
    filler_snapshot: str = "CC-MAIN-2025-21",
    context_len: int = 4096,
    depths: tuple[float, ...] = (0.1, 0.5, 0.9),
    seed: int = 0,
) -> list[dict]:
    """RULER-style NIAH probes: needle inserted at fixed depths in filler text."""
    rng = random.Random(seed)
    filler_stream = iter(load_dataset(
        "HuggingFaceFW/fineweb-edu", filler_snapshot, split="train", streaming=True,
    ))

    out: list[dict] = []
    while len(out) < n:
        ex = next(filler_stream)
        filler_ids = tokenizer(ex["text"], add_special_tokens=False).input_ids
        if len(filler_ids) < context_len:
            continue

        name = rng.choice(_NAMES)
        code = f"{rng.randint(10_000, 99_999)}"
        needle = f" The secret code for {name} is {code}. "
        needle_ids = tokenizer(needle, add_special_tokens=False).input_ids

        depth = rng.choice(depths)
        budget = context_len - len(needle_ids)
        cut = int(budget * depth)
        ctx = (filler_ids[:cut] + needle_ids + filler_ids[cut:budget])[:context_len]

        question = f"\n\nQuestion: What is the secret code for {name}?\nAnswer:"
        cont = tokenizer(question, add_special_tokens=False).input_ids

        out.append({"ctx": ctx, "cont": cont, "answer": code, "depth": depth})
    return out

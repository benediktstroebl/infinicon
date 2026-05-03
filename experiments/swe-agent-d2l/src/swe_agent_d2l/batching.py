"""Batch collation helpers for variable-length tokenized samples."""

from __future__ import annotations

from typing import Any

import torch


def pad_1d(values: list[list[int]], *, pad_value: int, dtype=torch.long) -> torch.Tensor:
    max_len = max(len(v) for v in values)
    out = torch.full((len(values), max_len), pad_value, dtype=dtype)
    for i, row in enumerate(values):
        if row:
            out[i, : len(row)] = torch.tensor(row, dtype=dtype)
    return out


def attention_mask(ids: torch.Tensor, *, pad_token_id: int) -> torch.Tensor:
    return (ids != pad_token_id).long()


def collate_train(batch: list[dict[str, Any]], *, pad_token_id: int) -> dict[str, Any]:
    out: dict[str, Any] = {
        "ctx_ids": pad_1d([b["ctx_ids"] for b in batch], pad_value=pad_token_id),
        "input_ids": pad_1d([b["input_ids"] for b in batch], pad_value=pad_token_id),
        "labels": pad_1d([b["labels"] for b in batch], pad_value=-100),
        "response_start_end": torch.tensor(
            [b["response_start_end"] for b in batch], dtype=torch.long
        ),
    }
    out["ctx_attention_mask"] = attention_mask(out["ctx_ids"], pad_token_id=pad_token_id)
    out["attention_mask"] = attention_mask(out["input_ids"], pad_token_id=pad_token_id)

    if "logprobs_vals" in batch[0] and batch[0]["logprobs_vals"] is not None:
        # Teacher top-k arrays are variable by target length but fixed by top-k.
        max_target = max(len(b["logprobs_vals"]) for b in batch)
        top_k = len(batch[0]["logprobs_vals"][0]) if batch[0]["logprobs_vals"] else 0
        vals = torch.zeros((len(batch), max_target, top_k), dtype=torch.float32)
        inds = torch.zeros((len(batch), max_target, top_k), dtype=torch.long)
        mask = torch.zeros((len(batch), max_target), dtype=torch.bool)
        for i, b in enumerate(batch):
            lp_vals = b["logprobs_vals"]
            lp_inds = b["logprobs_indices"]
            if not lp_vals:
                continue
            n = len(lp_vals)
            vals[i, :n] = torch.tensor(lp_vals, dtype=torch.float32)
            inds[i, :n] = torch.tensor(lp_inds, dtype=torch.long)
            mask[i, :n] = True
        out["logprobs_vals"] = vals
        out["logprobs_indices"] = inds
        out["logprobs_mask"] = mask

    return out


def collate_eval(batch: list[dict[str, Any]], *, pad_token_id: int) -> dict[str, Any]:
    out = collate_train(batch, pad_token_id=pad_token_id)
    if "teacher_input_ids" in batch[0]:
        out["teacher_input_ids"] = pad_1d(
            [b["teacher_input_ids"] for b in batch],
            pad_value=pad_token_id,
        )
        out["teacher_attention_mask"] = attention_mask(
            out["teacher_input_ids"],
            pad_token_id=pad_token_id,
        )
        out["teacher_response_start_end"] = torch.tensor(
            [b["teacher_response_start_end"] for b in batch], dtype=torch.long
        )
    for key in (
        "session_id",
        "repo_id",
        "target_kind",
        "target_tools",
        "context_length_bucket",
        "prompt_intent",
        "session_position",
    ):
        if key in batch[0]:
            out[key] = [b[key] for b in batch]
    return out

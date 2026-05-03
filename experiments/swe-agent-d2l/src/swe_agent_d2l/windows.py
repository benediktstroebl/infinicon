"""Causal resume-window construction."""

from __future__ import annotations

import hashlib
import random
from collections import Counter
from dataclasses import dataclass
from typing import Any

from .trajectory import ParsedMessage, ParsedSession, message_depends_on_excluded_result


@dataclass(frozen=True)
class ResumeWindow:
    session_id: str
    repo_id: str
    ctx_messages: list[dict[str, Any]]
    prompt_messages: list[dict[str, Any]]
    target_message: dict[str, Any]
    target_turn_ids: list[str]
    target_turn_numbers: list[int]
    target_kind: str
    target_tools: list[str]
    target_index: int
    cut_index: int
    message_count: int


def split_for_repo(repo_id: str, *, train_pct: int = 90, val_pct: int = 5) -> str:
    bucket = int(hashlib.md5(repo_id.encode("utf-8")).hexdigest(), 16) % 100
    if bucket < train_pct:
        return "train"
    if bucket < train_pct + val_pct:
        return "validation"
    return "test"


def build_resume_windows(
    session: ParsedSession,
    *,
    prompt_message_counts: tuple[int, ...] = (2, 4, 8),
    max_windows_per_session: int = 32,
    min_context_messages: int = 1,
    seed: int = 0,
) -> tuple[list[ResumeWindow], Counter[str]]:
    """Build causal windows from parsed messages.

    For each target assistant message at index t, choose several cut points k:
    messages[:k] become internalized context and messages[k:t] remain visible.
    """
    dropped: Counter[str] = Counter()
    candidates: list[ResumeWindow] = []

    messages = session.messages
    for target_index, target in enumerate(messages):
        if not target.targetable:
            continue
        if message_depends_on_excluded_result(session, target_index):
            dropped["target_after_excluded_tool"] += 1
            continue

        for tail_count in prompt_message_counts:
            cut_index = max(0, target_index - tail_count)
            if cut_index < min_context_messages:
                dropped["not_enough_context"] += 1
                continue
            if cut_index >= target_index:
                dropped["empty_prompt"] += 1
                continue

            ctx = [m.to_chat_message() for m in messages[:cut_index]]
            prompt = [m.to_chat_message() for m in messages[cut_index:target_index]]
            candidates.append(
                ResumeWindow(
                    session_id=session.session_id,
                    repo_id=session.repo_id,
                    ctx_messages=ctx,
                    prompt_messages=prompt,
                    target_message=target.to_chat_message(),
                    target_turn_ids=list(target.source_turn_ids),
                    target_turn_numbers=list(target.source_turn_numbers),
                    target_kind=target.target_kind,
                    target_tools=list(target.tool_names),
                    target_index=target_index,
                    cut_index=cut_index,
                    message_count=len(messages),
                )
            )

    rng_seed = int(hashlib.md5(f"{seed}:{session.session_id}".encode()).hexdigest(), 16)
    rng = random.Random(rng_seed)
    if len(candidates) > max_windows_per_session:
        candidates = rng.sample(candidates, max_windows_per_session)
    candidates.sort(key=lambda w: (w.target_index, w.cut_index))
    return candidates, dropped

"""Qwen chat-template tokenization for resume windows."""

from __future__ import annotations

from collections.abc import Mapping
from collections import Counter
from dataclasses import dataclass
from typing import Any

from transformers import PreTrainedTokenizerBase

from .tools import TOOL_SCHEMAS
from .windows import ResumeWindow


SYSTEM_PROMPT = (
    "You are a coding agent continuing an existing software engineering session. "
    "Use the prior tool results and messages to produce the next assistant message."
)


@dataclass(frozen=True)
class TokenBudgets:
    ctx_tokens: int = 8192
    prompt_tokens: int = 1024
    response_tokens: int = 384


def chat_template_kwargs(enable_thinking: bool | None) -> dict[str, Any]:
    kwargs: dict[str, Any] = {}
    if enable_thinking is not None:
        kwargs["enable_thinking"] = enable_thinking
    return kwargs


def with_system(messages: list[dict[str, Any]]) -> list[dict[str, Any]]:
    if messages and messages[0].get("role") == "system":
        return messages
    return [{"role": "system", "content": SYSTEM_PROMPT}] + messages


def apply_qwen_template(
    tokenizer: PreTrainedTokenizerBase,
    messages: list[dict[str, Any]],
    *,
    add_generation_prompt: bool,
    enable_thinking: bool | None,
) -> list[int]:
    rendered = tokenizer.apply_chat_template(
        with_system(messages),
        tools=TOOL_SCHEMAS,
        tokenize=True,
        add_generation_prompt=add_generation_prompt,
        add_special_tokens=False,
        return_attention_mask=False,
        **chat_template_kwargs(enable_thinking),
    )
    if isinstance(rendered, str):
        encoded = tokenizer(rendered, add_special_tokens=False, return_attention_mask=False)
        rendered = encoded["input_ids"]
    elif isinstance(rendered, Mapping) and "input_ids" in rendered:
        rendered = rendered["input_ids"]
    if hasattr(rendered, "tolist"):
        rendered = rendered.tolist()
    if rendered and isinstance(rendered[0], list):
        if len(rendered) != 1:
            raise ValueError("chat template returned a batch of multiple sequences")
        rendered = rendered[0]
    return list(rendered)


def tokenize_window(
    tokenizer: PreTrainedTokenizerBase,
    window: ResumeWindow,
    *,
    budgets: TokenBudgets,
    enable_thinking: bool | None,
) -> tuple[dict[str, Any] | None, str | None]:
    """Tokenize one causal window.

    Returns a pre-tokenized D2L-style sample plus teacher oracle ids. The stored
    `input_ids` are the student-side prompt plus target; labels mask everything
    except the target span.
    """
    try:
        ctx_ids = apply_qwen_template(
            tokenizer,
            window.ctx_messages,
            add_generation_prompt=False,
            enable_thinking=enable_thinking,
        )
        prompt_ids = apply_qwen_template(
            tokenizer,
            window.prompt_messages,
            add_generation_prompt=True,
            enable_thinking=enable_thinking,
        )
        input_ids = apply_qwen_template(
            tokenizer,
            window.prompt_messages + [window.target_message],
            add_generation_prompt=False,
            enable_thinking=enable_thinking,
        )
        teacher_prompt_ids = apply_qwen_template(
            tokenizer,
            window.ctx_messages + window.prompt_messages,
            add_generation_prompt=True,
            enable_thinking=enable_thinking,
        )
        teacher_input_ids = apply_qwen_template(
            tokenizer,
            window.ctx_messages + window.prompt_messages + [window.target_message],
            add_generation_prompt=False,
            enable_thinking=enable_thinking,
        )
    except Exception as exc:  # noqa: BLE001 - malformed dataset rows should be skipped.
        return None, f"template_error:{type(exc).__name__}"

    if not _has_prefix(input_ids, prompt_ids):
        return None, "student_prefix_mismatch"
    if not _has_prefix(teacher_input_ids, teacher_prompt_ids):
        return None, "teacher_prefix_mismatch"

    response_start = len(prompt_ids)
    response_end = len(input_ids)
    teacher_response_start = len(teacher_prompt_ids)
    teacher_response_end = len(teacher_input_ids)

    if len(ctx_ids) > budgets.ctx_tokens:
        return None, "ctx_too_long"
    if len(prompt_ids) > budgets.prompt_tokens:
        return None, "prompt_too_long"
    if response_end <= response_start:
        return None, "empty_response"
    if response_end - response_start > budgets.response_tokens:
        return None, "response_too_long"

    labels = [-100] * len(input_ids)
    labels[response_start:response_end] = input_ids[response_start:response_end]

    sample = {
        "ctx_ids": ctx_ids,
        "input_ids": input_ids,
        "labels": labels,
        "response_start_end": [response_start, response_end],
        "teacher_input_ids": teacher_input_ids,
        "teacher_response_start_end": [teacher_response_start, teacher_response_end],
        "session_id": window.session_id,
        "repo_id": window.repo_id,
        "target_turn_ids": window.target_turn_ids,
        "target_turn_numbers": window.target_turn_numbers,
        "target_kind": window.target_kind,
        "target_tools": window.target_tools,
        "target_index": window.target_index,
        "cut_index": window.cut_index,
        "message_count": window.message_count,
        "ctx_token_count": len(ctx_ids),
        "prompt_token_count": len(prompt_ids),
        "response_token_count": response_end - response_start,
        "context_length_bucket": context_length_bucket(len(ctx_ids)),
        "session_position": session_position(window.target_index, window.message_count),
        "prompt_intent": prompt_intent(window.prompt_messages),
    }
    return sample, None


def tokenize_windows(
    tokenizer: PreTrainedTokenizerBase,
    windows: list[ResumeWindow],
    *,
    budgets: TokenBudgets,
    enable_thinking: bool | None,
) -> tuple[list[dict[str, Any]], Counter[str]]:
    samples: list[dict[str, Any]] = []
    dropped: Counter[str] = Counter()
    for window in windows:
        sample, reason = tokenize_window(
            tokenizer,
            window,
            budgets=budgets,
            enable_thinking=enable_thinking,
        )
        if sample is None:
            dropped[reason or "unknown"] += 1
            continue
        samples.append(sample)
    return samples, dropped


def _has_prefix(values: list[int], prefix: list[int]) -> bool:
    return len(values) >= len(prefix) and values[: len(prefix)] == prefix


def context_length_bucket(token_count: int) -> str:
    if token_count < 4096:
        return "ctx_lt_4k"
    if token_count < 8192:
        return "ctx_4k_8k"
    if token_count < 16384:
        return "ctx_8k_16k"
    if token_count < 32768:
        return "ctx_16k_32k"
    return "ctx_ge_32k"


def session_position(target_index: int, message_count: int) -> str:
    if message_count <= 0:
        return "unknown"
    ratio = target_index / message_count
    if ratio < 0.33:
        return "early"
    if ratio < 0.67:
        return "middle"
    return "late"


def prompt_intent(messages: list[dict[str, Any]]) -> str:
    text = "\n".join(str(msg.get("content", "")) for msg in messages[-4:]).lower()
    if any(word in text for word in ("test", "pytest", "failure", "failing", "assert")):
        return "test_or_failure"
    if any(word in text for word in ("bug", "fix", "error", "exception", "traceback")):
        return "bug_fix"
    if any(word in text for word in ("read", "inspect", "look", "find", "search")):
        return "inspect"
    if any(word in text for word in ("edit", "modify", "update", "change", "implement")):
        return "edit_or_implement"
    if any(word in text for word in ("run", "execute", "command", "shell")):
        return "run_command"
    return "other"

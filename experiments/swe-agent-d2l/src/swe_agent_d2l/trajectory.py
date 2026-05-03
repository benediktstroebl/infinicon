"""SWE-chat trajectory parsing into structured Qwen chat messages."""

from __future__ import annotations

from collections import Counter
from collections.abc import Iterable, Mapping
from dataclasses import dataclass, field
from typing import Any

from .tools import (
    canonical_tool_args_json,
    is_excluded_tool,
    is_known_tool,
    normalize_tool_arguments,
    parse_tool_input,
    tool_call,
)


SKIPPED_TURN_TYPES = {
    "assistant_thinking",
    "summary",
    "system_event",
    "file_snapshot",
    "progress",
    "queue_operation",
    "metadata",
}


@dataclass
class ParsedMessage:
    role: str
    content: str = ""
    tool_calls: list[dict[str, Any]] = field(default_factory=list)
    source_turn_ids: list[str] = field(default_factory=list)
    source_turn_numbers: list[int] = field(default_factory=list)
    source_turn_types: list[str] = field(default_factory=list)
    tool_names: list[str] = field(default_factory=list)

    def to_chat_message(self) -> dict[str, Any]:
        msg: dict[str, Any] = {"role": self.role, "content": self.content}
        if self.tool_calls:
            msg["tool_calls"] = self.tool_calls
        return msg

    @property
    def first_turn_number(self) -> int:
        return min(self.source_turn_numbers)

    @property
    def last_turn_number(self) -> int:
        return max(self.source_turn_numbers)

    @property
    def targetable(self) -> bool:
        return self.role == "assistant" and bool(self.content.strip() or self.tool_calls)

    @property
    def target_kind(self) -> str:
        if self.role != "assistant":
            return self.role
        if self.content.strip() and self.tool_calls:
            return "assistant_text_and_tool_call"
        if self.tool_calls:
            return "assistant_tool_call"
        return "assistant_response"


@dataclass
class ParsedSession:
    session_id: str
    repo_id: str
    messages: list[ParsedMessage]
    dropped_tool_counts: Counter[str]
    excluded_turn_numbers: set[int]


def parse_session_rows(
    rows: Iterable[Mapping[str, Any]],
    *,
    max_tool_result_chars: int = 6000,
) -> ParsedSession:
    """Parse one SWE-chat session into structured chat messages.

    Unknown, excluded, and malformed tools are dropped. Their turn numbers are
    retained so window generation can avoid targets that immediately depend on
    missing tool results.
    """
    sorted_rows = sorted(rows, key=lambda r: int(_get(r, "turn_number", 0) or 0))
    if not sorted_rows:
        raise ValueError("empty session")

    session_id = str(_get(sorted_rows[0], "session_id", ""))
    repo_id = str(_get(sorted_rows[0], "repo_id", ""))
    messages: list[ParsedMessage] = []
    dropped_tool_counts: Counter[str] = Counter()
    excluded_turn_numbers: set[int] = set()
    excluded_call_ids: set[str] = set()
    current_assistant: ParsedMessage | None = None

    def flush_assistant() -> None:
        nonlocal current_assistant
        if current_assistant and current_assistant.targetable:
            messages.append(current_assistant)
        current_assistant = None

    for row in sorted_rows:
        turn_type = str(_get(row, "turn_type", "") or "")
        turn_number = int(_get(row, "turn_number", 0) or 0)
        turn_id = str(_get(row, "turn_id", f"{session_id}#{turn_number}") or "")

        if turn_type in SKIPPED_TURN_TYPES:
            flush_assistant()
            continue

        if turn_type in {"user_prompt", "system_injected"}:
            flush_assistant()
            content = _content(row)
            if content.strip():
                messages.append(
                    ParsedMessage(
                        role="user",
                        content=content,
                        source_turn_ids=[turn_id],
                        source_turn_numbers=[turn_number],
                        source_turn_types=[turn_type],
                    )
                )
            continue

        if turn_type == "assistant_response":
            content = _content(row)
            if current_assistant is None:
                current_assistant = ParsedMessage(role="assistant")
            if content.strip():
                if current_assistant.content:
                    current_assistant.content += "\n\n"
                current_assistant.content += content
            current_assistant.source_turn_ids.append(turn_id)
            current_assistant.source_turn_numbers.append(turn_number)
            current_assistant.source_turn_types.append(turn_type)
            continue

        if turn_type == "tool_use":
            tool_name = str(_get(row, "tool_name", "") or "")
            call_id = str(_get(row, "tool_call_id", "") or "")
            if not is_known_tool(tool_name):
                flush_assistant()
                dropped_tool_counts[tool_name or "<missing>"] += 1
                excluded_turn_numbers.add(turn_number)
                if call_id:
                    excluded_call_ids.add(call_id)
                continue

            raw_args = parse_tool_input(_get(row, "tool_input_json", None))
            args = normalize_tool_arguments(tool_name, raw_args, row)
            if args is None:
                flush_assistant()
                dropped_tool_counts[f"{tool_name}:malformed"] += 1
                excluded_turn_numbers.add(turn_number)
                if call_id:
                    excluded_call_ids.add(call_id)
                continue

            if current_assistant is None:
                current_assistant = ParsedMessage(role="assistant")
            current_assistant.tool_calls.append(tool_call(tool_name, args))
            current_assistant.tool_names.append(tool_name)
            current_assistant.source_turn_ids.append(turn_id)
            current_assistant.source_turn_numbers.append(turn_number)
            current_assistant.source_turn_types.append(turn_type)
            continue

        if turn_type == "tool_result":
            flush_assistant()
            tool_name = str(_get(row, "tool_name", "") or "")
            call_id = str(_get(row, "tool_call_id", "") or "")
            if call_id and call_id in excluded_call_ids:
                dropped_tool_counts[f"{tool_name or '<missing>'}:result_after_excluded"] += 1
                excluded_turn_numbers.add(turn_number)
                continue
            if not is_known_tool(tool_name):
                dropped_tool_counts[tool_name or "<missing>"] += 1
                excluded_turn_numbers.add(turn_number)
                continue
            content = normalize_tool_result(_content(row), max_chars=max_tool_result_chars)
            messages.append(
                ParsedMessage(
                    role="tool",
                    content=content,
                    source_turn_ids=[turn_id],
                    source_turn_numbers=[turn_number],
                    source_turn_types=[turn_type],
                    tool_names=[tool_name],
                )
            )
            continue

        flush_assistant()

    flush_assistant()
    return ParsedSession(
        session_id=session_id,
        repo_id=repo_id,
        messages=messages,
        dropped_tool_counts=dropped_tool_counts,
        excluded_turn_numbers=excluded_turn_numbers,
    )


def normalize_tool_result(content: str, *, max_chars: int) -> str:
    content = content or ""
    if len(content) <= max_chars:
        return content
    head = max_chars // 2
    tail = max_chars - head
    return (
        content[:head]
        + "\n\n[... tool result truncated ...]\n\n"
        + content[-tail:]
    )


def message_depends_on_excluded_result(
    session: ParsedSession,
    target_index: int,
) -> bool:
    """Return True if excluded raw events sit immediately before target."""
    if target_index <= 0:
        previous_last = -1
    else:
        previous_last = session.messages[target_index - 1].last_turn_number
    target_first = session.messages[target_index].first_turn_number
    return any(previous_last < n < target_first for n in session.excluded_turn_numbers)


def target_identity(message: ParsedMessage) -> str:
    if message.tool_calls:
        names = [
            call["function"]["name"]
            for call in message.tool_calls
            if "function" in call and "name" in call["function"]
        ]
        args = [
            canonical_tool_args_json(call["function"].get("arguments", {}))
            for call in message.tool_calls
            if "function" in call
        ]
        return "|".join(f"{n}:{a}" for n, a in zip(names, args))
    return message.content.strip()[:200]


def _get(row: Mapping[str, Any], key: str, default: Any = None) -> Any:
    value = row.get(key, default)
    # pandas uses NaN for missing values; NaN is not equal to itself.
    if value != value:
        return default
    return value


def _content(row: Mapping[str, Any]) -> str:
    value = _get(row, "content", "")
    return value if isinstance(value, str) else str(value or "")

"""Known Claude-style tool schemas and normalization helpers.

Qwen's chat template accepts OpenAI-style tool definitions and assistant
messages with a `tool_calls` field. We keep tool calls structured until
`tokenizer.apply_chat_template` renders them.
"""

from __future__ import annotations

import json
from collections.abc import Mapping
from typing import Any


KNOWN_TOOL_NAMES = {
    "Read",
    "Write",
    "Edit",
    "MultiEdit",
    "Bash",
    "Grep",
    "Glob",
    "LS",
    "WebFetch",
    "WebSearch",
}

EXCLUDED_TOOL_NAMES = {
    "NotebookEdit",
    # Gemini-style aliases are intentionally filtered, not mapped.
    "read_file",
    "write_file",
    "edit_file",
    "run_command",
    "grep",
    "glob",
    "list_directory",
}


def _schema(
    name: str,
    description: str,
    properties: dict[str, dict[str, Any]],
    required: list[str],
) -> dict[str, Any]:
    return {
        "type": "function",
        "function": {
            "name": name,
            "description": description,
            "parameters": {
                "type": "object",
                "properties": properties,
                "required": required,
                "additionalProperties": False,
            },
        },
    }


TOOL_SCHEMAS: list[dict[str, Any]] = [
    _schema(
        "Read",
        "Read a file from the workspace.",
        {
            "file_path": {"type": "string"},
            "offset": {"type": "integer"},
            "limit": {"type": "integer"},
        },
        ["file_path"],
    ),
    _schema(
        "Write",
        "Write complete content to a file.",
        {
            "file_path": {"type": "string"},
            "content": {"type": "string"},
        },
        ["file_path", "content"],
    ),
    _schema(
        "Edit",
        "Replace text in a file.",
        {
            "file_path": {"type": "string"},
            "old_string": {"type": "string"},
            "new_string": {"type": "string"},
            "replace_all": {"type": "boolean"},
        },
        ["file_path", "old_string", "new_string"],
    ),
    _schema(
        "MultiEdit",
        "Apply multiple text replacements to a file.",
        {
            "file_path": {"type": "string"},
            "edits": {
                "type": "array",
                "items": {
                    "type": "object",
                    "properties": {
                        "old_string": {"type": "string"},
                        "new_string": {"type": "string"},
                        "replace_all": {"type": "boolean"},
                    },
                    "required": ["old_string", "new_string"],
                    "additionalProperties": False,
                },
            },
        },
        ["file_path", "edits"],
    ),
    _schema(
        "Bash",
        "Run a shell command.",
        {
            "command": {"type": "string"},
            "timeout": {"type": "number"},
            "description": {"type": "string"},
        },
        ["command"],
    ),
    _schema(
        "Grep",
        "Search text in files.",
        {
            "pattern": {"type": "string"},
            "path": {"type": "string"},
            "glob": {"type": "string"},
            "output_mode": {"type": "string"},
            "case_insensitive": {"type": "boolean"},
            "multiline": {"type": "boolean"},
            "head_limit": {"type": "integer"},
        },
        ["pattern"],
    ),
    _schema(
        "Glob",
        "Find files by glob pattern.",
        {
            "pattern": {"type": "string"},
            "path": {"type": "string"},
        },
        ["pattern"],
    ),
    _schema(
        "LS",
        "List files in a directory.",
        {
            "path": {"type": "string"},
            "ignore": {"type": "array", "items": {"type": "string"}},
        },
        ["path"],
    ),
    _schema(
        "WebFetch",
        "Fetch a URL and process it with a prompt.",
        {
            "url": {"type": "string"},
            "prompt": {"type": "string"},
        },
        ["url", "prompt"],
    ),
    _schema(
        "WebSearch",
        "Search the web.",
        {
            "query": {"type": "string"},
            "allowed_domains": {"type": "array", "items": {"type": "string"}},
            "blocked_domains": {"type": "array", "items": {"type": "string"}},
        },
        ["query"],
    ),
]

_SCHEMA_BY_NAME = {tool["function"]["name"]: tool for tool in TOOL_SCHEMAS}


def is_known_tool(name: str | None) -> bool:
    return name in KNOWN_TOOL_NAMES


def is_excluded_tool(name: str | None) -> bool:
    return name in EXCLUDED_TOOL_NAMES


def required_args(tool_name: str) -> set[str]:
    schema = _SCHEMA_BY_NAME[tool_name]["function"]["parameters"]
    return set(schema["required"])


def allowed_args(tool_name: str) -> set[str]:
    schema = _SCHEMA_BY_NAME[tool_name]["function"]["parameters"]
    return set(schema["properties"])


def parse_tool_input(raw: Any) -> dict[str, Any]:
    """Parse SWE-chat's tool_input_json-like value into a dict."""
    if raw is None:
        return {}
    if isinstance(raw, Mapping):
        return dict(raw)
    if not isinstance(raw, str):
        return {}
    raw = raw.strip()
    if not raw:
        return {}
    try:
        parsed = json.loads(raw)
    except json.JSONDecodeError:
        return {}
    return dict(parsed) if isinstance(parsed, Mapping) else {}


def normalize_tool_arguments(
    tool_name: str,
    raw_args: Mapping[str, Any],
    row: Mapping[str, Any] | None = None,
) -> dict[str, Any] | None:
    """Keep only schema fields and fill common extracted SWE-chat columns."""
    if tool_name not in KNOWN_TOOL_NAMES:
        return None

    row = row or {}
    args = {k: v for k, v in dict(raw_args).items() if k in allowed_args(tool_name)}

    for key in ("file_path", "command", "pattern"):
        if key in allowed_args(tool_name) and key not in args and row.get(key):
            args[key] = row[key]
    if "path" in allowed_args(tool_name) and "path" not in args and row.get("file_path"):
        args["path"] = row["file_path"]

    for key in list(args):
        if args[key] is None:
            args.pop(key)

    missing = required_args(tool_name) - set(args)
    if missing:
        return None
    return _stable_jsonish(args)


def tool_call(tool_name: str, arguments: Mapping[str, Any]) -> dict[str, Any]:
    """Return the shape expected by Qwen's Jinja chat template."""
    return {
        "type": "function",
        "function": {
            "name": tool_name,
            "arguments": _stable_jsonish(arguments),
        },
    }


def _stable_jsonish(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {str(k): _stable_jsonish(value[k]) for k in sorted(value)}
    if isinstance(value, list):
        return [_stable_jsonish(v) for v in value]
    return value


def canonical_tool_args_json(arguments: Mapping[str, Any]) -> str:
    return json.dumps(_stable_jsonish(arguments), sort_keys=True, separators=(",", ":"))

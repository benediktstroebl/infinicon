"""Prepare SWE-chat causal resume windows for Qwen/D2L training."""

from __future__ import annotations

import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

from datasets import Dataset, DatasetDict, Features, Sequence, Value, load_dataset
from tqdm import tqdm
from transformers import AutoTokenizer

from .env import load_dotenv_from_parents
from .tokenization import TokenBudgets, tokenize_windows
from .trajectory import parse_session_rows
from .windows import build_resume_windows, split_for_repo


KEEP_COLUMNS = [
    "turn_id",
    "session_id",
    "repo_id",
    "turn_number",
    "turn_type",
    "content",
    "tool_name",
    "tool_call_id",
    "tool_input_json",
    "file_path",
    "command",
    "pattern",
    "language",
]

SAMPLE_FEATURES = Features(
    {
        "ctx_ids": Sequence(Value("int64")),
        "input_ids": Sequence(Value("int64")),
        "labels": Sequence(Value("int64")),
        "response_start_end": Sequence(Value("int64")),
        "teacher_input_ids": Sequence(Value("int64")),
        "teacher_response_start_end": Sequence(Value("int64")),
        "session_id": Value("string"),
        "repo_id": Value("string"),
        "target_turn_ids": Sequence(Value("string")),
        "target_turn_numbers": Sequence(Value("int64")),
        "target_kind": Value("string"),
        "target_tools": Sequence(Value("string")),
        "target_index": Value("int64"),
        "cut_index": Value("int64"),
        "message_count": Value("int64"),
        "ctx_token_count": Value("int64"),
        "prompt_token_count": Value("int64"),
        "response_token_count": Value("int64"),
        "context_length_bucket": Value("string"),
        "session_position": Value("string"),
        "prompt_intent": Value("string"),
    }
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", default="SALT-NLP/SWE-chat")
    parser.add_argument("--config", default="conversations")
    parser.add_argument("--split", default="train")
    parser.add_argument("--model", default="Qwen/Qwen3-4B-Instruct-2507")
    parser.add_argument("--out", required=True)
    parser.add_argument("--ctx-token-budget", type=int, default=8192)
    parser.add_argument("--prompt-token-budget", type=int, default=1024)
    parser.add_argument("--response-token-budget", type=int, default=384)
    parser.add_argument("--max-tool-result-chars", type=int, default=6000)
    parser.add_argument("--max-windows-per-session", type=int, default=32)
    parser.add_argument("--prompt-message-counts", default="2,4,8")
    parser.add_argument("--max-sessions", type=int, default=None)
    parser.add_argument("--max-rows", type=int, default=None)
    parser.add_argument(
        "--no-sort",
        action="store_true",
        help="Assume rows are already grouped by session_id and sorted by turn_number.",
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--enable-thinking",
        choices=["true", "false", "none"],
        default="none",
        help="Pass enable_thinking to Qwen chat template. Use false for Qwen3-0.6B smoke.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    load_dotenv_from_parents()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    enable_thinking = _parse_enable_thinking(args.enable_thinking)
    prompt_counts = tuple(
        int(x.strip()) for x in args.prompt_message_counts.split(",") if x.strip()
    )
    budgets = TokenBudgets(
        ctx_tokens=args.ctx_token_budget,
        prompt_tokens=args.prompt_token_budget,
        response_tokens=args.response_token_budget,
    )

    tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
    ds = load_dataset(args.dataset, args.config, split=args.split)
    cols = [c for c in KEEP_COLUMNS if c in ds.column_names]
    ds = ds.select_columns(cols)
    if args.max_rows is not None:
        ds = ds.select(range(min(args.max_rows, len(ds))))
    if not args.no_sort:
        ds = ds.sort(["session_id", "turn_number"])

    jsonl_dir = out / "jsonl"
    jsonl_dir.mkdir(parents=True, exist_ok=True)
    writers = {
        split: (jsonl_dir / f"{split}.jsonl").open("w")
        for split in ("train", "validation", "test")
    }
    stats: Counter[str] = Counter()
    split_sizes: Counter[str] = Counter()
    dropped_tools: Counter[str] = Counter()
    dropped_windows: Counter[str] = Counter()
    dropped_tokenization: Counter[str] = Counter()

    for rows in tqdm(_iter_session_rows(ds), desc="sessions"):
        if args.max_sessions is not None and stats["sessions_seen"] >= args.max_sessions:
            break
        stats["sessions_seen"] += 1
        if _has_non_english_user_prompt(rows):
            stats["non_english_sessions_dropped"] += 1
            continue
        try:
            parsed = parse_session_rows(
                rows,
                max_tool_result_chars=args.max_tool_result_chars,
            )
        except ValueError:
            stats["empty_session"] += 1
            continue

        dropped_tools.update(parsed.dropped_tool_counts)
        windows, dropped = build_resume_windows(
            parsed,
            prompt_message_counts=prompt_counts,
            max_windows_per_session=args.max_windows_per_session,
            seed=args.seed,
        )
        dropped_windows.update(dropped)
        tokenized, token_dropped = tokenize_windows(
            tokenizer,
            windows,
            budgets=budgets,
            enable_thinking=enable_thinking,
        )
        dropped_tokenization.update(token_dropped)

        split_name = split_for_repo(parsed.repo_id)
        for sample in tokenized:
            writers[split_name].write(json.dumps(sample, separators=(",", ":")) + "\n")
            split_sizes[split_name] += 1
        stats["sessions"] += 1
        stats["messages"] += len(parsed.messages)
        stats["windows_before_token_filter"] += len(windows)
        stats["windows_after_token_filter"] += len(tokenized)

    for writer in writers.values():
        writer.close()

    loaded_splits: dict[str, Dataset] = {}
    for split_name in ("train", "validation", "test"):
        path = jsonl_dir / f"{split_name}.jsonl"
        if split_sizes[split_name]:
            loaded_splits[split_name] = load_dataset(
                "json",
                data_files=str(path),
                split="train",
                features=SAMPLE_FEATURES,
            )
    if not loaded_splits:
        raise RuntimeError(
            "No tokenized samples were produced. Inspect metadata counters by rerunning "
            "with wider token budgets or fewer excluded tools."
        )
    dataset_dict = DatasetDict(
        {
            split_name: loaded_splits[split_name]
            for split_name in ("train", "validation", "test")
            if split_name in loaded_splits
        }
    )
    dataset_dict.save_to_disk(out)

    metadata = {
        "args": vars(args),
        "enable_thinking": enable_thinking,
        "tokenizer": args.model,
        "stats": dict(stats),
        "split_sizes": dict(split_sizes),
        "dropped_tools": dict(dropped_tools),
        "dropped_windows": dict(dropped_windows),
        "dropped_tokenization": dict(dropped_tokenization),
    }
    (out / "metadata.json").write_text(json.dumps(metadata, indent=2, sort_keys=True))
    print(json.dumps(metadata, indent=2, sort_keys=True))


def _parse_enable_thinking(value: str) -> bool | None:
    if value == "true":
        return True
    if value == "false":
        return False
    return None


def _iter_session_rows(ds) -> Any:
    current_session_id = None
    rows: list[dict[str, Any]] = []
    for row in ds:
        session_id = row["session_id"]
        if current_session_id is None:
            current_session_id = session_id
        if session_id != current_session_id:
            yield rows
            rows = []
            current_session_id = session_id
        rows.append(row)
    if rows:
        yield rows


def _has_non_english_user_prompt(rows: list[dict[str, Any]]) -> bool:
    for row in rows:
        if row.get("turn_type") != "user_prompt":
            continue
        language = row.get("language")
        if language and language == language and language != "english":
            return True
    return False


if __name__ == "__main__":
    main()

from __future__ import annotations

from swe_agent_d2l.tools import EXCLUDED_TOOL_NAMES, KNOWN_TOOL_NAMES, normalize_tool_arguments
from swe_agent_d2l.trajectory import parse_session_rows
from swe_agent_d2l.windows import build_resume_windows


def row(n, turn_type, **kw):
    out = {
        "turn_id": f"s#${n}",
        "session_id": "s",
        "repo_id": "owner/repo",
        "turn_number": n,
        "turn_type": turn_type,
        "content": "",
        "tool_name": None,
        "tool_call_id": None,
        "tool_input_json": None,
    }
    out.update(kw)
    return out


def test_known_and_excluded_tool_sets_are_disjoint():
    assert "NotebookEdit" in EXCLUDED_TOOL_NAMES
    assert "read_file" in EXCLUDED_TOOL_NAMES
    assert "grep" in EXCLUDED_TOOL_NAMES
    assert "Read" in KNOWN_TOOL_NAMES
    assert "Grep" in KNOWN_TOOL_NAMES
    assert not (KNOWN_TOOL_NAMES & EXCLUDED_TOOL_NAMES)


def test_normalize_tool_arguments_uses_schema_and_row_fallbacks():
    args = normalize_tool_arguments("Read", {"offset": 10, "ignored": "x"}, {"file_path": "a.py"})
    assert args == {"file_path": "a.py", "offset": 10}
    assert normalize_tool_arguments("Read", {}, {}) is None
    assert normalize_tool_arguments("NotebookEdit", {"notebook_path": "x.ipynb"}, {}) is None


def test_parse_combines_assistant_text_and_known_tool_call():
    parsed = parse_session_rows(
        [
            row(0, "user_prompt", content="fix tests"),
            row(1, "assistant_response", content="I will inspect the failure."),
            row(
                2,
                "tool_use",
                tool_name="Read",
                tool_call_id="call_1",
                tool_input_json='{"file_path":"tests/test_api.py"}',
            ),
            row(3, "tool_result", tool_name="Read", tool_call_id="call_1", content="assert 1 == 2"),
        ]
    )
    assert [m.role for m in parsed.messages] == ["user", "assistant", "tool"]
    assistant = parsed.messages[1].to_chat_message()
    assert assistant["content"] == "I will inspect the failure."
    assert assistant["tool_calls"][0]["function"]["name"] == "Read"
    assert assistant["tool_calls"][0]["function"]["arguments"] == {"file_path": "tests/test_api.py"}


def test_excluded_tool_is_dropped_and_blocks_immediate_target():
    parsed = parse_session_rows(
        [
            row(0, "user_prompt", content="edit notebook"),
            row(
                1,
                "tool_use",
                tool_name="NotebookEdit",
                tool_call_id="bad",
                tool_input_json='{"notebook_path":"nb.ipynb","new_source":"x"}',
            ),
            row(2, "tool_result", tool_name="NotebookEdit", tool_call_id="bad", content="ok"),
            row(3, "assistant_response", content="Done."),
            row(4, "user_prompt", content="now inspect file"),
            row(
                5,
                "tool_use",
                tool_name="Read",
                tool_call_id="good",
                tool_input_json='{"file_path":"a.py"}',
            ),
        ]
    )
    assert parsed.dropped_tool_counts["NotebookEdit"] == 1
    assert 1 in parsed.excluded_turn_numbers
    assert 2 in parsed.excluded_turn_numbers
    windows, dropped = build_resume_windows(
        parsed,
        prompt_message_counts=(1, 2),
        max_windows_per_session=10,
        min_context_messages=1,
    )
    assert dropped["target_after_excluded_tool"] >= 1
    assert windows
    assert all(w.target_message.get("tool_calls") for w in windows)

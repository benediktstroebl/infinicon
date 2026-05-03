from __future__ import annotations

from swe_agent_d2l.tokenization import TokenBudgets, tokenize_window
from swe_agent_d2l.windows import ResumeWindow


class FakeQwenTokenizer:
    def __init__(self):
        self.calls = []

    def apply_chat_template(self, messages, tools, tokenize, add_generation_prompt, **kwargs):
        self.calls.append(
            {
                "messages": messages,
                "tools": tools,
                "add_generation_prompt": add_generation_prompt,
            }
        )
        text = ""
        for msg in messages:
            text += f"<{msg['role']}>"
            text += msg.get("content", "")
            for call in msg.get("tool_calls", []) or []:
                text += f"<tool_call>{call['function']['name']}"
            text += f"</{msg['role']}>"
        if add_generation_prompt:
            text += "<assistant>"
        return [ord(ch) for ch in text]


def test_tokenize_window_uses_structured_messages_and_target_span():
    tokenizer = FakeQwenTokenizer()
    window = ResumeWindow(
        session_id="s",
        repo_id="owner/repo",
        ctx_messages=[{"role": "user", "content": "old"}],
        prompt_messages=[{"role": "user", "content": "recent"}],
        target_message={
            "role": "assistant",
            "content": "",
            "tool_calls": [
                {
                    "type": "function",
                    "function": {"name": "Read", "arguments": {"file_path": "a.py"}},
                }
            ],
        },
        target_turn_ids=["s#2"],
        target_turn_numbers=[2],
        target_kind="assistant_tool_call",
        target_tools=["Read"],
        target_index=2,
        cut_index=1,
        message_count=3,
    )
    sample, reason = tokenize_window(
        tokenizer,
        window,
        budgets=TokenBudgets(ctx_tokens=1000, prompt_tokens=1000, response_tokens=1000),
        enable_thinking=None,
    )
    assert reason is None
    assert sample is not None
    start, end = sample["response_start_end"]
    assert sample["labels"][:start] == [-100] * start
    assert sample["labels"][start:end] == sample["input_ids"][start:end]
    # The target is a structured assistant message with tool_calls, not XML in user content.
    rendered_target = tokenizer.calls[2]["messages"][-1]
    assert rendered_target["role"] == "assistant"
    assert rendered_target["tool_calls"][0]["function"]["name"] == "Read"

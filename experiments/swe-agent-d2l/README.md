# SWE-Agent Doc-to-LoRA

Fresh experiment for training a Doc-to-LoRA-style hypernetwork on SWE-chat
coding-agent trajectories.

The central invariant is that SWE-chat transcripts are reconstructed into
structured Qwen chat messages before tokenization. Tool calls are represented
with `tool_calls` and tool results with role `tool`; Qwen's
`tokenizer.apply_chat_template` renders `<tool_call>` and `<tool_response>`.
The code never hand-writes Qwen chat delimiters.

## Setup

```bash
cd experiments/swe-agent-d2l
uv sync --extra test
```

SWE-chat is gated. Accept access on Hugging Face before running data prep.

## Data Prep

```bash
uv run -m swe_agent_d2l.prepare_swechat \
  --model Qwen/Qwen3-4B-Instruct-2507 \
  --out data/swechat_qwen3_4b \
  --ctx-token-budget 32768 \
  --prompt-token-budget 4096 \
  --response-token-budget 512 \
  --max-windows-per-session 32
```

Qwen's tool-aware chat template includes the tool schema in the rendered prompt,
so the visible prompt budget must leave room for roughly 1k schema tokens before
the recent conversation tail.

For a local parser smoke run without the gated dataset:

```bash
uv run pytest
```

## Teacher Logprobs

```bash
uv run -m swe_agent_d2l.precompute_teacher \
  --data data/swechat_qwen3_4b \
  --out data/swechat_qwen3_4b_teacher \
  --model Qwen/Qwen3-4B-Instruct-2507 \
  --top-k 64 \
  --split train
```

## Train

```bash
accelerate launch -m swe_agent_d2l.train \
  --config configs/qwen3_4b_pilot.yaml \
  --data data/swechat_qwen3_4b_teacher
```

## Evaluate

```bash
uv run -m swe_agent_d2l.eval_next_action \
  --config configs/qwen3_4b_pilot.yaml \
  --data data/swechat_qwen3_4b_teacher \
  --checkpoint outputs/qwen3_4b_swechat_d2l/best.pt \
  --split test
```

The primary metric is oracle-gap closure:

```text
(NLL_tail_only - NLL_adapter) / (NLL_tail_only - NLL_full_context_oracle)
```

# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project

`swe-agent-d2l` trains a Doc-to-LoRA-style hypernetwork on SWE-chat
(`SALT-NLP/SWE-chat`) coding-agent trajectories. The base Qwen model stays
frozen; the hypernetwork sees a long internalized context and emits per-module
LoRA factors that adapt the base for the visible prompt + next-action target.

The full pipeline is four stages, each a `python -m` entrypoint:

1. `swe_agent_d2l.prepare_swechat` — flatten SWE-chat session rows into
   structured Qwen chat messages, build causal resume windows, tokenize, write
   a `DatasetDict` to disk.
2. `swe_agent_d2l.precompute_teacher` — for one split, run the *base* model on
   the full-context teacher prompt and store top-k logprobs over the target
   span.
3. `swe_agent_d2l.train` — train `TrajectoryHyperNetwork` with teacher KL or
   plain CE on the target tokens.
4. `swe_agent_d2l.eval_next_action` — compute tail-only / adapter / oracle /
   mismatched-context NLLs and the oracle-gap-closure ratio.

## Central invariant (do not break)

SWE-chat transcripts are reconstructed into structured chat messages and only
rendered to text by `tokenizer.apply_chat_template`. **Never hand-write Qwen
chat delimiters or `<tool_call>`/`<tool_response>` strings.**

- Assistant tool calls live in the `tool_calls` field of an assistant message
  (OpenAI shape: `{"type": "function", "function": {"name", "arguments"}}`).
- Tool results are messages with `role: "tool"`.
- The tool schema list (`TOOL_SCHEMAS` in `tools.py`) is passed via the
  `tools=` kwarg to `apply_chat_template`. Qwen's template inlines it into the
  rendered prompt, so prompt budgets must leave ~1k tokens of headroom for the
  schema.
- Tools outside `KNOWN_TOOL_NAMES` are dropped (not mapped). Lowercase
  Gemini-style aliases (`read_file`, `grep`, …) are intentionally in
  `EXCLUDED_TOOL_NAMES` — do not add aliasing logic.

When a tool call/result is dropped, its turn number is recorded in
`excluded_turn_numbers` so `windows.build_resume_windows` can skip target
messages that depend on a missing immediately-prior tool result
(`message_depends_on_excluded_result`).

## Commands

Setup (uses `uv`):

```bash
uv sync --extra test
```

Run all tests:

```bash
uv run pytest
```

Run a single test or test function:

```bash
uv run pytest tests/test_tokenization.py
uv run pytest tests/test_tools_and_trajectory.py::test_excluded_tool_is_dropped_and_blocks_immediate_target
```

The tests use a `FakeQwenTokenizer` and parser fixtures, so they run without
HF credentials or the gated dataset. Data prep, teacher precompute, and train
require `HF_TOKEN` (loaded automatically from `../../.env` via
`env.load_dotenv_from_parents`) and SWE-chat access on Hugging Face.

Pilot data prep / train / eval (Qwen3-4B):

```bash
uv run -m swe_agent_d2l.prepare_swechat --model Qwen/Qwen3-4B-Instruct-2507 \
  --out data/swechat_qwen3_4b --ctx-token-budget 32768 --prompt-token-budget 4096 \
  --response-token-budget 512 --max-windows-per-session 32
uv run -m swe_agent_d2l.precompute_teacher --data data/swechat_qwen3_4b \
  --out data/swechat_qwen3_4b_teacher --model Qwen/Qwen3-4B-Instruct-2507 --top-k 64 --split train
accelerate launch -m swe_agent_d2l.train --config configs/qwen3_4b_pilot.yaml \
  --data data/swechat_qwen3_4b_teacher
uv run -m swe_agent_d2l.eval_next_action --config configs/qwen3_4b_pilot.yaml \
  --data data/swechat_qwen3_4b_teacher --checkpoint outputs/qwen3_4b_swechat_d2l/best.pt --split test
```

Smoke run (Qwen3-0.6B): use `configs/qwen3_06b_smoke.yaml` and pass
`--enable-thinking false` to `prepare_swechat` — the 0.6B template requires an
explicit `enable_thinking=False` kwarg, while Qwen3-4B-Instruct-2507 wants the
kwarg omitted (`--enable-thinking none`, the default).

## Architecture notes

### Resume windows (`windows.py`)

Each session yields multiple `(target_index, cut_index)` candidates. For each
target assistant message at index `t` and a tail size `k` (default 2/4/8):

- `messages[:cut_index]` becomes `ctx_messages` (internalized via the
  hypernet).
- `messages[cut_index:t]` becomes `prompt_messages` (visible to the student
  forward).
- `messages[t]` is the `target_message` whose tokens are the only unmasked
  labels.

Train/val/test split is deterministic by `md5(repo_id) % 100` (90/5/5). Per-
session candidate count is capped by `max_windows_per_session` using a per-
session-seeded RNG.

### Tokenization (`tokenization.py`)

`tokenize_window` renders four token sequences with `apply_chat_template`:

- `ctx_ids` — context-only, no generation prompt (used for hypernet input).
- `prompt_ids` — prompt-only with generation prompt (used to locate the
  student response span).
- `input_ids` — prompt + target, no generation prompt (student labels).
- `teacher_input_ids` — ctx + prompt + target (oracle baseline / teacher
  logprob source).

`labels` is `[-100]` everywhere except the target span. Prefix-mismatch checks
catch any chat-template inconsistency and drop the sample. Budgets
(`TokenBudgets`) drop windows that exceed `ctx`, `prompt`, or `response` token
caps.

The system prompt (`SYSTEM_PROMPT`) is auto-prepended whenever messages
start with non-`system`.

### Hypernet + generated LoRA (`hyper_lora.py`)

- `inject_generated_lora` walks `nn.Linear` children whose name matches the
  target list (default `down_proj`) and wraps each in a
  `GeneratedLoRALinear`. The wrapper computes
  `out + (alpha/rank) * (x @ Aᵀ @ Bᵀ)` when `A`/`B` are set.
- `TrajectoryHyperNetwork.encode_context` runs the *frozen base model's*
  `.model` submodule under `no_grad` and mean-pools over the attention mask.
- The trunk + per-module embedding produces `delta` of shape `(rank, rank)`
  per module per example. `A` is a learned per-module parameter (shared
  across the batch); `B` is `B_base @ delta` per example. Only the hypernet
  parameters train.
- After each forward, the trainer **must** `clear_lora(registry)` so cached
  `A`/`B` don't leak into the next sample.

### Loss (`losses.py`)

`shifted_target_logits` aligns logits at position `p-1` with label at
position `p`, then masks to unmasked label positions. `selected_kl_loss`
implements forward KL on the teacher's top-k support only — softmax denominator
is computed over the full vocab logits (`logsumexp`) but the cross-entropy
sum is restricted to teacher's top-k indices. `cross_entropy_target_loss`
is the fallback when teacher logprobs are absent.

### Eval (`eval_next_action.py`)

Reports per-row: `tail_nll` (no ctx), `adapter_nll` (hypernet LoRA),
`oracle_nll` (full context in prompt), `mismatch_nll` (negative control: ctx
batch is `torch.roll(..., shifts=1)` so each row gets a wrong sibling's
context), and the closure ratio
`(tail_nll - adapter_nll) / (tail_nll - oracle_nll)`. Results are bucketed
by `target_kind`, `tool_name`, `context_length_bucket`, `prompt_intent`, and
`session_position` (these labels are computed at prep time in `tokenization.py`
and survive into the saved DatasetDict).

## Conventions

- Use the four-stage flow above; don't add new chat-template rendering paths
  outside `tokenization.apply_qwen_template`.
- New tool support: add to `TOOL_SCHEMAS` and `KNOWN_TOOL_NAMES` in
  `tools.py`. Don't aliasing-map foreign tool names — drop them.
- Targets that immediately follow excluded events are skipped at window
  construction. If you change the parser to keep more turns, update
  `message_depends_on_excluded_result` accordingly.
- The base model is frozen. Anything that needs gradients must live on
  `TrajectoryHyperNetwork`. The optimizer in `train.py` only sees
  `hypernet.parameters()`.
- `attn_implementation: flash_attention_2` requires `uv sync --extra flash`.
  Default smoke / CPU runs use `sdpa`.

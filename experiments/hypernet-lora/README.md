# hypernet-lora

MVP for the doc-to-LoRA experiment. A hypernetwork takes a 4096-token context, emits an all-layer rank-16 LoRA for a frozen Qwen3 base, and is trained by forward-KL distillation against the same base conditioned on the context in-prompt.

## Setup

```bash
pip install -e .
wandb login
```

## Run (MVP — 1× H100, Qwen3-0.6B-Base)

```bash
accelerate launch --config_file configs/ddp.yaml --num_processes=1 \
    -m src.train --config configs/qwen3_06b.yaml
```

## Run (scaled — multi-GPU FSDP, Qwen3-8B)

```bash
accelerate launch --config_file configs/fsdp.yaml --num_processes=4 \
    -m src.train --config configs/qwen3_8b.yaml
```

## Layout

- `src/data.py` — FineWeb-Edu streaming + NIAH generator
- `src/lora.py` — LoRALinear wrapper, injection, set/clear of hypernet-emitted weights
- `src/hypernet.py` — context encoder + trunk + factorized per-module heads
- `src/train.py` — main loop
- `src/eval.py` — held-out KL, NIAH, baselines

## Plan

See `~/.claude/plans/golden-herding-willow.md` for the full experiment plan.

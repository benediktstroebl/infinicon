"""Train a generated-LoRA hypernetwork on prepared SWE-chat windows."""

from __future__ import annotations

import argparse
import json
from functools import partial
from pathlib import Path

import torch
from accelerate import Accelerator
from datasets import load_from_disk
from torch.utils.data import DataLoader
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer, get_cosine_schedule_with_warmup

from .batching import collate_train
from .config import load_yaml, torch_dtype
from .hyper_lora import (
    HyperLoRAConfig,
    TrajectoryHyperNetwork,
    clear_lora,
    hyper_lora_state_dict,
    inject_generated_lora,
    set_lora,
)
from .losses import cross_entropy_target_loss, selected_kl_loss


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    parser.add_argument("--data", required=True)
    parser.add_argument("--split", default="train")
    parser.add_argument("--max-steps", type=int, default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    cfg = load_yaml(args.config)
    train_cfg = cfg["train"]
    if args.max_steps is not None:
        train_cfg["max_steps"] = args.max_steps

    accelerator = Accelerator(
        mixed_precision="bf16",
        gradient_accumulation_steps=int(train_cfg.get("grad_accum", 1)),
    )
    torch.manual_seed(int(train_cfg.get("seed", 42)))

    tokenizer = AutoTokenizer.from_pretrained(cfg["model"]["name"], trust_remote_code=True)
    pad_token_id = tokenizer.pad_token_id
    if pad_token_id is None:
        pad_token_id = tokenizer.eos_token_id

    base = AutoModelForCausalLM.from_pretrained(
        cfg["model"]["name"],
        torch_dtype=torch_dtype(cfg["model"].get("torch_dtype", "bfloat16")),
        attn_implementation=cfg["model"].get("attn_implementation", "sdpa"),
        trust_remote_code=True,
    )
    base.requires_grad_(False)
    registry = inject_generated_lora(
        base,
        target_modules=tuple(cfg["lora"]["target_modules"]),
        alpha=float(cfg["lora"]["alpha"]),
    )
    if not registry:
        raise RuntimeError(f"No target modules matched {cfg['lora']['target_modules']}")

    hypernet = TrajectoryHyperNetwork(
        registry,
        HyperLoRAConfig(
            hidden_size=base.config.hidden_size,
            rank=int(cfg["lora"]["rank"]),
            alpha=float(cfg["lora"]["alpha"]),
            trunk_dim=int(cfg["hypernet"]["trunk_dim"]),
            trunk_layers=int(cfg["hypernet"]["trunk_layers"]),
        ),
    )
    base.to(accelerator.device)
    base.eval()

    ds = load_from_disk(args.data)[args.split]
    loader = DataLoader(
        ds,
        batch_size=int(train_cfg["batch_size"]),
        shuffle=True,
        collate_fn=partial(collate_train, pad_token_id=pad_token_id),
    )
    optimizer = torch.optim.AdamW(hypernet.parameters(), lr=float(train_cfg["lr"]))
    scheduler = get_cosine_schedule_with_warmup(
        optimizer,
        num_warmup_steps=int(train_cfg["warmup_steps"]),
        num_training_steps=int(train_cfg["max_steps"]),
    )
    hypernet, optimizer, scheduler, loader = accelerator.prepare(
        hypernet,
        optimizer,
        scheduler,
        loader,
    )

    output_dir = Path(train_cfg["output_dir"])
    if accelerator.is_main_process:
        output_dir.mkdir(parents=True, exist_ok=True)
        (output_dir / "config.json").write_text(json.dumps(cfg, indent=2, sort_keys=True))
        accelerator.print(f"Injected {len(registry)} generated-LoRA modules.")

    step = 0
    progress = tqdm(total=int(train_cfg["max_steps"]), disable=not accelerator.is_main_process)
    while step < int(train_cfg["max_steps"]):
        for batch in loader:
            if step >= int(train_cfg["max_steps"]):
                break
            with accelerator.accumulate(hypernet):
                lora = hypernet(
                    base,
                    batch["ctx_ids"].to(accelerator.device),
                    batch["ctx_attention_mask"].to(accelerator.device),
                )
                set_lora(registry, lora)
                outputs = base(
                    input_ids=batch["input_ids"].to(accelerator.device),
                    attention_mask=batch["attention_mask"].to(accelerator.device),
                    use_cache=False,
                )
                clear_lora(registry)

                if train_cfg.get("use_teacher_kl", True) and "logprobs_vals" in batch:
                    loss = selected_kl_loss(
                        outputs.logits,
                        batch["labels"].to(accelerator.device),
                        batch["logprobs_vals"].to(accelerator.device),
                        batch["logprobs_indices"].to(accelerator.device),
                        batch["logprobs_mask"].to(accelerator.device),
                    )
                else:
                    loss = cross_entropy_target_loss(
                        outputs.logits,
                        batch["labels"].to(accelerator.device),
                    )

                accelerator.backward(loss)
                if accelerator.sync_gradients:
                    accelerator.clip_grad_norm_(
                        hypernet.parameters(),
                        float(train_cfg["grad_clip"]),
                    )
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad()

            if accelerator.sync_gradients:
                step += 1
                progress.update(1)
                if step % int(train_cfg["log_every"]) == 0:
                    accelerator.print(
                        json.dumps(
                            {
                                "step": step,
                                "loss": float(loss.detach().cpu()),
                                "lr": scheduler.get_last_lr()[0],
                            },
                            sort_keys=True,
                        )
                    )
                if step % int(train_cfg["save_every"]) == 0:
                    _save(accelerator, hypernet, output_dir / f"step_{step}.pt", cfg)

    _save(accelerator, hypernet, output_dir / "best.pt", cfg)
    progress.close()


def _save(accelerator: Accelerator, hypernet: TrajectoryHyperNetwork, path: Path, cfg: dict) -> None:
    if not accelerator.is_main_process:
        return
    unwrapped = accelerator.unwrap_model(hypernet)
    torch.save(hyper_lora_state_dict(unwrapped, extra={"config": cfg}), path)


if __name__ == "__main__":
    main()

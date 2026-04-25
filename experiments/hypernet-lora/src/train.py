"""Train the hypernetwork by forward-KL distillation against base+context-in-prompt."""

from __future__ import annotations

import argparse
import dataclasses
from pathlib import Path

import torch
import torch.nn.functional as F
import wandb
import yaml
from accelerate import Accelerator
from torch.utils.data import DataLoader, IterableDataset
from transformers import AutoModelForCausalLM, AutoTokenizer, get_cosine_schedule_with_warmup

from .data import fineweb_pairs
from .hypernet import Hypernet
from .lora import clear_lora, inject_lora, mean_BA_norm, set_lora


@dataclasses.dataclass
class Config:
    model: dict
    data: dict
    lora: dict
    hypernet: dict
    train: dict
    logging: dict


def load_config(path: str) -> Config:
    with open(path) as f:
        return Config(**yaml.safe_load(f))


class FinewebDataset(IterableDataset):
    def __init__(self, tokenizer, **kwargs):
        super().__init__()
        self.tokenizer = tokenizer
        self.kwargs = kwargs

    def __iter__(self):
        return iter(fineweb_pairs(self.tokenizer, **self.kwargs))


def collate(batch):
    return {
        "ctx": torch.tensor([b["ctx"] for b in batch], dtype=torch.long),
        "cont": torch.tensor([b["cont"] for b in batch], dtype=torch.long),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--max_steps", type=int, default=None)
    args = parser.parse_args()

    cfg = load_config(args.config)
    if args.max_steps is not None:
        cfg.train["max_steps"] = args.max_steps

    accelerator = Accelerator(
        mixed_precision="bf16",
        gradient_accumulation_steps=cfg.train.get("grad_accum", 1),
    )
    torch.manual_seed(cfg.train["seed"])

    if accelerator.is_main_process:
        wandb.init(
            project=cfg.logging["wandb_project"],
            name=cfg.logging["wandb_run_name"],
            config=dataclasses.asdict(cfg),
        )

    tokenizer = AutoTokenizer.from_pretrained(cfg.model["name"])
    base = AutoModelForCausalLM.from_pretrained(
        cfg.model["name"],
        torch_dtype=torch.bfloat16,
        attn_implementation=cfg.model.get("attn_impl", "sdpa"),
    )
    base.requires_grad_(False)
    if cfg.model.get("activation_checkpoint"):
        base.gradient_checkpointing_enable()

    registry = inject_lora(base, tuple(cfg.lora["targets"]))
    accelerator.print(f"Injected {len(registry)} LoRALinear modules.")

    hypernet = Hypernet(
        registry=registry,
        d_model=base.config.hidden_size,
        rank=cfg.lora["rank"],
        alpha=cfg.lora["alpha"],
        trunk_dim=cfg.hypernet["trunk_dim"],
        trunk_layers=cfg.hypernet["trunk_layers"],
    )

    n_trainable = sum(p.numel() for p in hypernet.parameters() if p.requires_grad)
    accelerator.print(f"Hypernet trainable params: {n_trainable / 1e6:.1f}M")

    opt = torch.optim.AdamW(hypernet.parameters(), lr=cfg.train["lr"])
    sched = get_cosine_schedule_with_warmup(
        opt,
        num_warmup_steps=cfg.train["warmup_steps"],
        num_training_steps=cfg.train["max_steps"],
    )

    train_ds = FinewebDataset(
        tokenizer,
        snapshot=cfg.data["snapshot"],
        context_len=cfg.data["context_len"],
        continuation_len=cfg.data["continuation_len"],
        min_doc_tokens=cfg.data["min_doc_tokens"],
        held_out=False,
        rank=accelerator.process_index,
        world_size=accelerator.num_processes,
    )
    train_loader = DataLoader(
        train_ds, batch_size=cfg.train["batch_size"], collate_fn=collate, num_workers=0,
    )

    base, hypernet, opt, sched = accelerator.prepare(base, hypernet, opt, sched)
    base.eval()
    hypernet.train()

    step = 0
    log_buf: dict = {}
    for batch in train_loader:
        if step >= cfg.train["max_steps"]:
            break

        with accelerator.accumulate(hypernet):
            ctx = batch["ctx"].to(accelerator.device)
            cont = batch["cont"].to(accelerator.device)
            ctx_len = ctx.size(1)

            with torch.no_grad():
                full = torch.cat([ctx, cont], dim=1)
                teacher_logits = base(full).logits[:, ctx_len:-1]   # predicts cont[1:]

            A_dict, B_dict = hypernet(ctx, base)
            set_lora(registry, A_dict, B_dict)
            student_logits = base(cont).logits[:, :-1]              # predicts cont[1:]
            clear_lora(registry)

            loss = F.kl_div(
                F.log_softmax(student_logits.float(), dim=-1),
                F.log_softmax(teacher_logits.float(), dim=-1),
                reduction="batchmean",
                log_target=True,
            )
            accelerator.backward(loss)

            grad_norm = None
            if accelerator.sync_gradients:
                grad_norm = accelerator.clip_grad_norm_(
                    hypernet.parameters(), cfg.train["grad_clip"],
                )
            opt.step()
            sched.step()
            opt.zero_grad()

        if accelerator.sync_gradients:
            step += 1
            log_buf = {
                "kl_loss": loss.item(),
                "mean_BA_norm": mean_BA_norm(registry).item(),
                "lr": sched.get_last_lr()[0],
            }
            if grad_norm is not None:
                log_buf["grad_norm"] = grad_norm.item()

            if accelerator.is_main_process and step % 10 == 0:
                wandb.log(log_buf, step=step)

            if step % cfg.train["checkpoint_every"] == 0 and accelerator.is_main_process:
                ckpt_dir = Path("checkpoints") / cfg.logging["wandb_run_name"]
                ckpt_dir.mkdir(parents=True, exist_ok=True)
                torch.save(
                    accelerator.unwrap_model(hypernet).state_dict(),
                    ckpt_dir / f"step_{step}.pt",
                )

    if accelerator.is_main_process:
        wandb.finish()


if __name__ == "__main__":
    main()

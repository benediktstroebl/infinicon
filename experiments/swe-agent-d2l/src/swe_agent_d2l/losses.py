"""Loss and likelihood helpers."""

from __future__ import annotations

import torch
import torch.nn.functional as F


def shifted_target_logits(
    logits: torch.Tensor,
    labels: torch.Tensor,
) -> tuple[list[torch.Tensor], list[torch.Tensor]]:
    """Return per-row logits and labels for unmasked target tokens."""
    shift_logits = logits[:, :-1, :]
    shift_labels = labels[:, 1:]
    row_logits: list[torch.Tensor] = []
    row_labels: list[torch.Tensor] = []
    for i in range(labels.shape[0]):
        mask = shift_labels[i] != -100
        row_logits.append(shift_logits[i][mask])
        row_labels.append(shift_labels[i][mask])
    return row_logits, row_labels


def cross_entropy_target_loss(logits: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
    row_logits, row_labels = shifted_target_logits(logits, labels)
    losses = []
    for logit, label in zip(row_logits, row_labels):
        if label.numel() == 0:
            continue
        losses.append(F.cross_entropy(logit.float(), label, reduction="mean"))
    if not losses:
        return logits.sum() * 0.0
    return torch.stack(losses).mean()


def selected_kl_loss(
    logits: torch.Tensor,
    labels: torch.Tensor,
    logprobs_vals: torch.Tensor,
    logprobs_indices: torch.Tensor,
    logprobs_mask: torch.Tensor,
) -> torch.Tensor:
    row_logits, row_labels = shifted_target_logits(logits, labels)
    losses = []
    for i, logit in enumerate(row_logits):
        n = logit.shape[0]
        if n == 0:
            continue
        valid = logprobs_mask[i, :n]
        if valid.sum() == 0:
            continue
        logit = logit[valid].float()
        indices = logprobs_indices[i, :n][valid]
        teacher_logp = logprobs_vals[i, :n][valid].to(logit.device)
        logq_denom = torch.logsumexp(logit, dim=-1, keepdim=True)
        logq_selected = logit.gather(1, indices.to(logit.device)) - logq_denom
        teacher_p = teacher_logp.exp()
        losses.append(-(teacher_p * logq_selected).sum(dim=-1).mean())
    if not losses:
        return logits.sum() * 0.0
    return torch.stack(losses).mean()


@torch.no_grad()
def target_nll(logits: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
    row_logits, row_labels = shifted_target_logits(logits, labels)
    nlls = []
    for logit, label in zip(row_logits, row_labels):
        if label.numel() == 0:
            continue
        logp = F.log_softmax(logit.float(), dim=-1)
        nll = -logp.gather(1, label.unsqueeze(1)).squeeze(1).mean()
        nlls.append(nll)
    if not nlls:
        return torch.tensor(float("nan"), device=logits.device)
    return torch.stack(nlls)

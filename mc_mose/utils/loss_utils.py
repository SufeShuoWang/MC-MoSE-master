from __future__ import annotations

import torch
import torch.nn.functional as F


def masked_pairwise_rank_loss(
    pred: torch.Tensor,
    target: torch.Tensor,
    mask: torch.Tensor,
    alpha: float = 1.0,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Monthly loss over funds with a valid target."""

    pred = pred.reshape(-1).float()
    target = target.reshape(-1).float()
    valid = mask.reshape(-1).bool()

    if not (
        pred.numel()
        == target.numel()
        == valid.numel()
    ):
        raise ValueError(
            "pred, target, and mask must contain the same number "
            "of elements."
        )

    if not torch.any(valid):
        zero = pred.sum() * 0.0
        return zero, zero, zero

    pred_valid = pred[valid].reshape(-1, 1)
    target_valid = target[valid].reshape(-1, 1)

    reg_loss = F.mse_loss(
        pred_valid,
        target_valid,
        reduction="mean",
    )

    num_valid = pred_valid.shape[0]
    if num_valid < 2:
        rank_loss = pred_valid.sum() * 0.0
    else:
        pred_diff = pred_valid - pred_valid.t()
        target_diff = target_valid.t() - target_valid

        rank_loss = F.relu(
            pred_diff * target_diff
        ).sum() / float(
            num_valid * (num_valid - 1)
        )

    loss = reg_loss + float(alpha) * rank_loss
    return loss, reg_loss, rank_loss


def loss_parts_to_float(
    loss: torch.Tensor,
    reg_loss: torch.Tensor,
    rank_loss: torch.Tensor,
) -> dict[str, float]:
    return {
        "loss": float(loss.detach().cpu()),
        "reg_loss": float(reg_loss.detach().cpu()),
        "rank_loss": float(rank_loss.detach().cpu()),
    }

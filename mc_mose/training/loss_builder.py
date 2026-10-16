from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any

import torch

from mc_mose.utils.loss_utils import masked_pairwise_rank_loss
from mc_mose.losses import (
    LoadBalanceTracker,
    entropy_sparse_loss,
    pairwise_kl_diversity_loss,
)


@dataclass(frozen=True)
class MCMoSELossOutput:
    loss: torch.Tensor
    base_loss: torch.Tensor
    reg_loss: torch.Tensor
    rank_loss: torch.Tensor
    sparse_loss: torch.Tensor
    diversity_loss: torch.Tensor
    load_balance_loss: torch.Tensor

    def as_tensor_dict(self) -> dict[str, torch.Tensor]:
        return {
            "loss": self.loss,
            "base_loss": self.base_loss,
            "reg_loss": self.reg_loss,
            "rank_loss": self.rank_loss,
            "sparse_loss": self.sparse_loss,
            "diversity_loss": self.diversity_loss,
            "load_balance_loss": self.load_balance_loss,
        }

    def as_log_dict(self) -> dict[str, float]:
        return {
            key: float(value.detach().cpu())
            for key, value in self.as_tensor_dict().items()
        }


class MCMoSELossBuilder:
    """Build MC-MoSE losses from predictions, month batches, and MoE aux."""

    def __init__(
        self,
        rank_loss_alpha: float = 1.0,
        lambda_sparse: float = 0.0,
        lambda_diversity: float = 0.0,
        lambda_lb: float = 0.0,
        balance_threshold: int = 2,
        num_routed_experts: int | None = None,
        topk_routed: int | None = None,
    ) -> None:
        if rank_loss_alpha < 0.0:
            raise ValueError("rank_loss_alpha must be non-negative.")
        if lambda_sparse < 0.0:
            raise ValueError("lambda_sparse must be non-negative.")
        if lambda_diversity < 0.0:
            raise ValueError("lambda_diversity must be non-negative.")
        if lambda_lb < 0.0:
            raise ValueError("lambda_lb must be non-negative.")
        if balance_threshold <= 0:
            raise ValueError("balance_threshold must be positive.")

        self.rank_loss_alpha = float(rank_loss_alpha)
        self.lambda_sparse = float(lambda_sparse)
        self.lambda_diversity = float(lambda_diversity)
        self.lambda_lb = float(lambda_lb)
        self.balance_threshold = int(balance_threshold)

        self.load_balance_tracker: LoadBalanceTracker | None = None
        if self.lambda_lb > 0.0:
            if num_routed_experts is None:
                raise ValueError("num_routed_experts is required when lambda_lb > 0.")
            if topk_routed is None:
                raise ValueError("topk_routed is required when lambda_lb > 0.")
            self.load_balance_tracker = LoadBalanceTracker(
                num_routed_experts=int(num_routed_experts),
                topk_routed=int(topk_routed),
                threshold=self.balance_threshold,
                alpha=1.0,
            )

    def __call__(
        self,
        prediction: torch.Tensor,
        batch: Mapping[str, torch.Tensor],
        aux: Mapping[str, Any] | None = None,
        update_balance: bool = False,
    ) -> MCMoSELossOutput:
        base_loss, reg_loss, rank_loss = compute_mc_mose_base_loss(
            prediction=prediction,
            batch=batch,
            rank_loss_alpha=self.rank_loss_alpha,
        )

        zero = base_loss.new_tensor(0.0)
        sparse_loss = zero
        diversity_loss = zero
        load_balance_loss = zero

        moe_aux = _extract_moe_aux(aux)
        needs_importance = (
            self.lambda_sparse > 0.0
            or self.lambda_diversity > 0.0
            or _has_importance_tensors(moe_aux)
        )
        if needs_importance:
            feature_importance = _require_aux_tensor(moe_aux, "feature_importance")
            scale_importance = _require_aux_tensor(moe_aux, "scale_importance")
            sparse_loss = entropy_sparse_loss(
                feature_importance=feature_importance,
                scale_importance=scale_importance,
            )
            diversity_loss = pairwise_kl_diversity_loss(
                feature_importance=feature_importance,
                scale_importance=scale_importance,
            )

        if self.lambda_lb > 0.0 and update_balance:
            if self.load_balance_tracker is None:
                raise RuntimeError("load_balance_tracker is not initialized.")
            topk_local = _require_aux_tensor(moe_aux, "topk_local")
            routed_raw_scores = _require_aux_tensor(moe_aux, "routed_raw_scores")
            self.load_balance_tracker.update_trusted(topk_local)
            load_balance_loss = self.load_balance_tracker.loss(routed_raw_scores)

        total_loss = (
            base_loss
            + self.lambda_sparse * sparse_loss
            + self.lambda_diversity * diversity_loss
            + self.lambda_lb * load_balance_loss
        )

        return MCMoSELossOutput(
            loss=total_loss,
            base_loss=base_loss,
            reg_loss=reg_loss,
            rank_loss=rank_loss,
            sparse_loss=sparse_loss,
            diversity_loss=diversity_loss,
            load_balance_loss=load_balance_loss,
        )

    def state_dict(self) -> dict[str, Any]:
        return {
            "rank_loss_alpha": self.rank_loss_alpha,
            "lambda_sparse": self.lambda_sparse,
            "lambda_diversity": self.lambda_diversity,
            "lambda_lb": self.lambda_lb,
            "balance_threshold": self.balance_threshold,
            "load_balance_tracker": (
                self.load_balance_tracker.state_dict()
                if self.load_balance_tracker is not None
                else None
            ),
        }

    def load_state_dict(self, state: Mapping[str, Any]) -> None:
        if "load_balance_tracker" in state and state["load_balance_tracker"] is not None:
            if self.load_balance_tracker is None:
                raise RuntimeError("load_balance_tracker is not initialized.")
            self.load_balance_tracker.load_state_dict(state["load_balance_tracker"])

    def reset_balance_tracker(self) -> None:
        if self.load_balance_tracker is not None:
            self.load_balance_tracker.reset()


def compute_mc_mose_base_loss(
    prediction: torch.Tensor,
    batch: Mapping[str, torch.Tensor],
    rank_loss_alpha: float = 1.0,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Compute V1 base monthly cross-section loss shared with V0."""

    if rank_loss_alpha < 0.0:
        raise ValueError("rank_loss_alpha must be non-negative.")
    if not isinstance(prediction, torch.Tensor):
        raise TypeError("prediction must be a torch.Tensor.")

    target = _require_tensor(batch, "target")
    mask = _require_tensor(batch, "mask")

    pred = prediction.reshape(-1)
    target = target.reshape(-1).to(device=pred.device, dtype=pred.dtype)
    mask = mask.reshape(-1).to(device=pred.device, dtype=pred.dtype)

    if target.numel() != pred.numel():
        raise ValueError(
            f"prediction and target must have the same length, got "
            f"{pred.numel()} and {target.numel()}."
        )
    if mask.numel() != pred.numel():
        raise ValueError(
            f"prediction and mask must have the same length, got "
            f"{pred.numel()} and {mask.numel()}."
        )

    return masked_pairwise_rank_loss(
        pred=pred,
        target=target,
        mask=mask,
        alpha=float(rank_loss_alpha),
    )


def build_loss_builder_from_args(args: Any) -> MCMoSELossBuilder:
    return MCMoSELossBuilder(
        rank_loss_alpha=float(_get_arg(args, "rank_loss_alpha", 1.0)),
        lambda_sparse=float(_get_arg(args, "lambda_sparse", 0.0)),
        lambda_diversity=float(_get_arg(args, "lambda_diversity", 0.0)),
        lambda_lb=float(_get_arg(args, "lambda_lb", 0.0)),
        balance_threshold=int(_get_arg(args, "balance_threshold", 2)),
        num_routed_experts=int(_get_arg(args, "num_routed_experts", 0)),
        topk_routed=int(_get_arg(args, "topk_routed", 0)),
    )


def _extract_moe_aux(aux: Mapping[str, Any] | None) -> Mapping[str, Any]:
    if aux is None:
        return {}
    if not isinstance(aux, Mapping):
        raise TypeError("aux must be a mapping when provided.")
    nested = aux.get("moe_aux")
    if nested is None:
        return aux
    if not isinstance(nested, Mapping):
        raise TypeError("aux['moe_aux'] must be a mapping.")
    return nested


def _has_importance_tensors(aux: Mapping[str, Any]) -> bool:
    return "feature_importance" in aux and "scale_importance" in aux


def _require_aux_tensor(aux: Mapping[str, Any], key: str) -> torch.Tensor:
    if key not in aux:
        raise KeyError(f"moe_aux must contain '{key}'.")
    value = aux[key]
    if not isinstance(value, torch.Tensor):
        raise TypeError(f"moe_aux['{key}'] must be a torch.Tensor.")
    return value


def _require_tensor(
    batch: Mapping[str, torch.Tensor],
    key: str,
) -> torch.Tensor:
    if key not in batch:
        raise KeyError(f"batch must contain '{key}'.")
    value = batch[key]
    if not isinstance(value, torch.Tensor):
        raise TypeError(f"batch['{key}'] must be a torch.Tensor.")
    return value


def _get_arg(args: Any, name: str, default: Any = None) -> Any:
    if isinstance(args, Mapping):
        return args.get(name, default)
    return getattr(args, name, default)

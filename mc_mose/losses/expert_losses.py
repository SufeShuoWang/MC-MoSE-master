from __future__ import annotations

from collections.abc import Mapping
from typing import Any

import torch
import torch.nn.functional as F


def entropy_sparse_loss(
    feature_importance: torch.Tensor,
    scale_importance: torch.Tensor,
    eps: float = 1e-8,
) -> torch.Tensor:
    """Encourage each expert to focus on fewer features and scales.

    Parameters
    ----------
    feature_importance:
        Softmax-normalized tensor with shape (E, D_f).
    scale_importance:
        Softmax-normalized tensor with shape (E, S).
    eps:
        Numerical floor used before log.

    Returns
    -------
    torch.Tensor
        Mean expert entropy over feature and scale distributions.
    """

    _validate_importance_pair(feature_importance, scale_importance)
    if eps <= 0.0:
        raise ValueError("eps must be positive.")

    feature_entropy = _entropy(feature_importance, eps=eps)
    scale_entropy = _entropy(scale_importance, eps=eps)
    return (feature_entropy + scale_entropy).mean()


def pairwise_kl_diversity_loss(
    feature_importance: torch.Tensor,
    scale_importance: torch.Tensor,
    eps: float = 1e-8,
) -> torch.Tensor:
    """Encourage experts to learn different feature/scale distributions.

    The returned value is negative mean ordered-pair KL divergence. Minimizing
    this loss increases pairwise diversity across experts.
    """

    _validate_importance_pair(feature_importance, scale_importance)
    if eps <= 0.0:
        raise ValueError("eps must be positive.")

    num_experts = int(feature_importance.shape[0])
    if num_experts < 2:
        return feature_importance.new_tensor(0.0)

    feature_kl = _mean_ordered_pairwise_kl(feature_importance, eps=eps)
    scale_kl = _mean_ordered_pairwise_kl(scale_importance, eps=eps)
    return -(feature_kl + scale_kl)


class LoadBalanceTracker:
    """Track routed expert usage and compute cumulative load-balance loss.

    The tracker stores discrete TopK selection counts outside the gradient
    graph. Call ``update`` for validated external indices, or
    ``update_trusted`` for model-produced TopK indices, before calling
    ``loss`` when training.
    """

    def __init__(
        self,
        num_routed_experts: int,
        topk_routed: int,
        threshold: int,
        alpha: float = 1.0,
    ) -> None:
        if num_routed_experts <= 0:
            raise ValueError("num_routed_experts must be positive.")
        if topk_routed <= 0:
            raise ValueError("topk_routed must be positive.")
        if topk_routed > num_routed_experts:
            raise ValueError("topk_routed must not exceed num_routed_experts.")
        if threshold <= 0:
            raise ValueError("threshold must be positive.")
        if alpha < 0.0:
            raise ValueError("alpha must be non-negative.")

        self.num_routed_experts = int(num_routed_experts)
        self.topk_routed = int(topk_routed)
        self.threshold = int(threshold)
        self.alpha = float(alpha)
        self.balance_count = 0
        self.select_count = torch.zeros(self.num_routed_experts, dtype=torch.float32)

    @torch.no_grad()
    def update(self, topk_local: torch.Tensor) -> None:
        """Accumulate routed experts with full public input validation."""

        self._update(topk_local, validate_indices=True)

    @torch.no_grad()
    def update_trusted(self, topk_local: torch.Tensor) -> None:
        """Accumulate model-produced TopK indices without a device sync.

        This fast path is intended only for indices returned directly by
        ``torch.topk`` over the configured routed experts. Call :meth:`update`
        for untrusted inputs that still require an explicit range check.
        """

        self._update(topk_local, validate_indices=False)

    def _update(
        self,
        topk_local: torch.Tensor,
        *,
        validate_indices: bool,
    ) -> None:
        """Accumulate one TopK selection on the input tensor's device."""

        if not isinstance(topk_local, torch.Tensor):
            raise TypeError("topk_local must be a torch.Tensor.")
        topk = topk_local.detach().reshape(-1).to(dtype=torch.long)
        if topk.numel() != self.topk_routed:
            raise ValueError(
                f"Expected topk_local to contain {self.topk_routed} indices, "
                f"got {topk.numel()}."
            )
        if validate_indices and bool(
            torch.any((topk < 0) | (topk >= self.num_routed_experts)).item()
        ):
            raise ValueError("topk_local contains an out-of-range routed expert index.")

        if self.select_count.device != topk.device:
            self.select_count = self.select_count.to(device=topk.device)
        counts = torch.bincount(topk, minlength=self.num_routed_experts).to(
            dtype=self.select_count.dtype,
        )
        self.select_count += counts
        self.balance_count += 1

    def loss(self, routed_raw_scores: torch.Tensor) -> torch.Tensor:
        """Compute load-balance loss from current raw router scores."""

        scores = self._validate_routed_raw_scores(routed_raw_scores)
        if self.balance_count < self.threshold:
            return scores.new_tensor(0.0)

        probs = F.softmax(scores, dim=-1)
        frequency = self.select_count.to(device=scores.device, dtype=scores.dtype)
        frequency = frequency * (
            self.num_routed_experts / (self.topk_routed * self.balance_count)
        )
        return scores.new_tensor(self.alpha) * torch.sum(frequency.detach() * probs)

    def state_dict(self) -> dict[str, Any]:
        return {
            "num_routed_experts": self.num_routed_experts,
            "topk_routed": self.topk_routed,
            "threshold": self.threshold,
            "alpha": self.alpha,
            "balance_count": self.balance_count,
            "select_count": self.select_count.detach().to(device="cpu").clone(),
        }

    def load_state_dict(self, state: Mapping[str, Any]) -> None:
        for key in (
            "num_routed_experts",
            "topk_routed",
            "threshold",
            "alpha",
            "balance_count",
            "select_count",
        ):
            if key not in state:
                raise KeyError(f"state must contain '{key}'.")

        if int(state["num_routed_experts"]) != self.num_routed_experts:
            raise ValueError("state num_routed_experts does not match tracker.")
        if int(state["topk_routed"]) != self.topk_routed:
            raise ValueError("state topk_routed does not match tracker.")

        threshold = int(state["threshold"])
        alpha = float(state["alpha"])
        balance_count = int(state["balance_count"])
        select_count = torch.as_tensor(
            state["select_count"],
            dtype=torch.float32,
            device="cpu",
        ).reshape(-1)

        if threshold <= 0:
            raise ValueError("state threshold must be positive.")
        if alpha < 0.0:
            raise ValueError("state alpha must be non-negative.")
        if balance_count < 0:
            raise ValueError("state balance_count must be non-negative.")
        if select_count.numel() != self.num_routed_experts:
            raise ValueError(
                f"state select_count must have length {self.num_routed_experts}."
            )

        self.threshold = threshold
        self.alpha = alpha
        self.balance_count = balance_count
        self.select_count = select_count.clone()

    def reset(self) -> None:
        self.balance_count = 0
        self.select_count.zero_()

    def _validate_routed_raw_scores(self, routed_raw_scores: torch.Tensor) -> torch.Tensor:
        if not isinstance(routed_raw_scores, torch.Tensor):
            raise TypeError("routed_raw_scores must be a torch.Tensor.")
        scores = routed_raw_scores.reshape(-1)
        if scores.numel() != self.num_routed_experts:
            raise ValueError(
                f"Expected routed_raw_scores length {self.num_routed_experts}, "
                f"got {scores.numel()}."
            )
        return scores


def _entropy(probabilities: torch.Tensor, eps: float) -> torch.Tensor:
    probs = probabilities.clamp_min(eps)
    return -(probabilities * probs.log()).sum(dim=-1)


def _mean_ordered_pairwise_kl(probabilities: torch.Tensor, eps: float) -> torch.Tensor:
    probs = probabilities.clamp_min(eps)
    log_probs = probs.log()
    pairwise_kl = (
        probabilities[:, None, :] * (log_probs[:, None, :] - log_probs[None, :, :])
    ).sum(dim=-1)

    num_experts = int(probabilities.shape[0])
    mask = ~torch.eye(num_experts, dtype=torch.bool, device=probabilities.device)
    return pairwise_kl[mask].mean()


def _validate_importance_pair(
    feature_importance: torch.Tensor,
    scale_importance: torch.Tensor,
) -> None:
    if not isinstance(feature_importance, torch.Tensor):
        raise TypeError("feature_importance must be a torch.Tensor.")
    if not isinstance(scale_importance, torch.Tensor):
        raise TypeError("scale_importance must be a torch.Tensor.")
    if feature_importance.ndim != 2:
        raise ValueError(
            "Expected feature_importance to have shape (E, D_f), "
            f"got {tuple(feature_importance.shape)}."
        )
    if scale_importance.ndim != 2:
        raise ValueError(
            "Expected scale_importance to have shape (E, S), "
            f"got {tuple(scale_importance.shape)}."
        )
    if feature_importance.shape[0] != scale_importance.shape[0]:
        raise ValueError(
            "feature_importance and scale_importance must have the same number "
            "of experts."
        )
    if feature_importance.shape[0] < 1:
        raise ValueError("importance tensors must contain at least one expert.")
    if feature_importance.shape[1] < 1:
        raise ValueError("feature_importance must contain at least one feature.")
    if scale_importance.shape[1] < 1:
        raise ValueError("scale_importance must contain at least one scale.")

from __future__ import annotations

from collections.abc import Mapping, Sequence

import torch
import torch.nn as nn

from .csfi import CSFI
from .expert_moe import ExpertMoE
from .market_feature_encoder import MarketFeatureSequenceEncoder
from .mhfe import MHFE


class MCMoSE(nn.Module):
    """End-to-end MC-MoSE model."""

    def __init__(
        self,
        fund_dim: int,
        market_dim: int,
        lookback: int,
        patch_scales: Sequence[int],
        hidden_dim: int,
        market_embed_dim: int,
        num_shared_experts: int,
        num_routed_experts: int,
        topk_routed: int,
        num_heads: int = 4,
        mlp_hidden_dim: int | None = None,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        if fund_dim <= 0:
            raise ValueError("fund_dim must be positive.")
        if market_dim <= 0:
            raise ValueError("market_dim must be positive.")
        if lookback <= 0:
            raise ValueError("lookback must be positive.")
        if hidden_dim <= 0:
            raise ValueError("hidden_dim must be positive.")
        if market_embed_dim <= 0:
            raise ValueError("market_embed_dim must be positive.")
        if num_heads <= 0:
            raise ValueError("num_heads must be positive.")
        if hidden_dim % num_heads != 0:
            raise ValueError("hidden_dim must be divisible by num_heads.")
        if num_shared_experts <= 0:
            raise ValueError("num_shared_experts must be positive.")
        if num_routed_experts <= 0:
            raise ValueError("num_routed_experts must be positive.")
        if topk_routed <= 0:
            raise ValueError("topk_routed must be positive.")
        if topk_routed > num_routed_experts:
            raise ValueError("topk_routed must not exceed num_routed_experts.")
        if dropout < 0.0 or dropout >= 1.0:
            raise ValueError("dropout must be in [0, 1).")

        scales = [int(scale) for scale in patch_scales]
        if not scales:
            raise ValueError("patch_scales must not be empty.")
        if any(scale <= 0 for scale in scales):
            raise ValueError("all patch_scales must be positive.")

        head_hidden_dim = int(mlp_hidden_dim) if mlp_hidden_dim is not None else int(hidden_dim)
        if head_hidden_dim <= 0:
            raise ValueError("mlp_hidden_dim must be positive when provided.")

        self.fund_dim = int(fund_dim)
        self.market_dim = int(market_dim)
        self.lookback = int(lookback)
        self.patch_scales = scales
        self.hidden_dim = int(hidden_dim)
        self.market_embed_dim = int(market_embed_dim)
        self.num_shared_experts = int(num_shared_experts)
        self.num_routed_experts = int(num_routed_experts)
        self.topk_routed = int(topk_routed)
        self.num_experts = self.num_shared_experts + self.num_routed_experts
        self.num_heads = int(num_heads)
        self.mlp_hidden_dim = head_hidden_dim

        self.mhfe = MHFE(
            num_features=self.fund_dim,
            patch_scales=self.patch_scales,
            hidden_dim=self.hidden_dim,
            num_heads=self.num_heads,
        )
        self.csfi = CSFI(
            hidden_dim=self.hidden_dim,
            num_heads=self.num_heads,
        )
        self.market_encoder = MarketFeatureSequenceEncoder(
            market_dim=self.market_dim,
            lookback=self.lookback,
            embed_dim=self.market_embed_dim,
        )
        self.expert_moe = ExpertMoE(
            num_features=self.fund_dim,
            num_scales=len(self.patch_scales),
            hidden_dim=self.hidden_dim,
            market_embed_dim=self.market_embed_dim,
            num_shared_experts=self.num_shared_experts,
            num_routed_experts=self.num_routed_experts,
            topk_routed=self.topk_routed,
        )
        self.prediction_head = nn.Sequential(
            nn.Linear(self.num_experts * self.hidden_dim, self.mlp_hidden_dim),
            nn.ReLU(),
            nn.Dropout(p=float(dropout)),
            nn.Linear(self.mlp_hidden_dim, 1),
        )

    def forward(
        self,
        batch: Mapping[str, torch.Tensor],
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor | dict[str, torch.Tensor]]]:
        """Run a complete V1 forward pass.

        Parameters
        ----------
        batch:
            A month batch with ``fund_seq`` shaped (N, L, D_f) and
            ``market_seq`` shaped (N, L, D_m).

        Returns
        -------
        tuple[torch.Tensor, dict[str, torch.Tensor | dict[str, torch.Tensor]]]
            Prediction shaped (N,) and auxiliary tensors from the model path.
        """

        fund_seq = self._require_tensor(batch, "fund_seq")
        market_seq = self._require_tensor(batch, "market_seq")

        self._validate_fund_seq(fund_seq)
        self._validate_market_seq(market_seq)

        fund_seq = torch.nan_to_num(
            fund_seq.float(),
            nan=0.0,
            posinf=0.0,
            neginf=0.0,
        )
        market_seq = torch.nan_to_num(
            market_seq.float(),
            nan=0.0,
            posinf=0.0,
            neginf=0.0,
        )

        mhfe_output = self.mhfe(fund_seq, layout="NLD")
        fund_encoded_output = self.csfi(mhfe_output)
        market_embedding = self.market_encoder(market_seq)
        prediction_input, moe_aux = self.expert_moe(
            feature_bank=fund_encoded_output,
            market_embedding=market_embedding,
        )
        prediction = self.prediction_head(prediction_input).squeeze(-1)

        aux: dict[str, torch.Tensor | dict[str, torch.Tensor]] = {
            "mhfe_output": mhfe_output,
            "fund_encoded_output": fund_encoded_output,
            "market_embedding": market_embedding,
            "prediction_input": prediction_input,
            "moe_aux": moe_aux,
        }
        return prediction, aux

    def _require_tensor(
        self,
        batch: Mapping[str, torch.Tensor],
        key: str,
    ) -> torch.Tensor:
        if key not in batch:
            raise KeyError(f"batch must contain '{key}'.")
        value = batch[key]
        if not isinstance(value, torch.Tensor):
            raise TypeError(f"batch['{key}'] must be a torch.Tensor.")
        return value

    def _validate_fund_seq(self, fund_seq: torch.Tensor) -> None:
        if fund_seq.ndim != 3:
            raise ValueError(
                "Expected fund_seq to have shape (N, L, D_f), "
                f"got {tuple(fund_seq.shape)}."
            )
        if fund_seq.shape[1] != self.lookback:
            raise ValueError(
                f"Expected fund_seq lookback {self.lookback}, got {fund_seq.shape[1]}."
            )
        if fund_seq.shape[2] != self.fund_dim:
            raise ValueError(
                f"Expected fund_seq feature dim {self.fund_dim}, got {fund_seq.shape[2]}."
            )

    def _validate_market_seq(self, market_seq: torch.Tensor) -> None:
        if market_seq.ndim != 3:
            raise ValueError(
                "Expected market_seq to have shape (N, L, D_m), "
                f"got {tuple(market_seq.shape)}."
            )
        if market_seq.shape[1] != self.lookback:
            raise ValueError(
                f"Expected market_seq lookback {self.lookback}, got {market_seq.shape[1]}."
            )
        if market_seq.shape[2] != self.market_dim:
            raise ValueError(
                f"Expected market_seq feature dim {self.market_dim}, got {market_seq.shape[2]}."
            )

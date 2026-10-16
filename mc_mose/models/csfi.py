from __future__ import annotations

import torch
import torch.nn as nn


class CSFI(nn.Module):
    """Cross-feature Self-attention Interaction.

    The token axis is the feature axis of one sample at one scale.
    """

    def __init__(self, hidden_dim: int, num_heads: int = 4) -> None:
        super().__init__()
        if hidden_dim <= 0:
            raise ValueError("hidden_dim must be positive.")

        self.hidden_dim = int(hidden_dim)
        self.attn = nn.MultiheadAttention(
            embed_dim=hidden_dim,
            num_heads=num_heads,
            batch_first=True,
        )

    def forward(self, feature_scale_bank: torch.Tensor) -> torch.Tensor:
        """Refine MHFE outputs with per-scale feature self-attention.

        Parameters
        ----------
        feature_scale_bank:
            Tensor with shape (N, D_f, S, H).

        Returns
        -------
        torch.Tensor
            Tensor with shape (N, D_f, S, H).
        """

        if feature_scale_bank.ndim != 4:
            raise ValueError(
                "Expected feature_scale_bank to have shape (N, D_f, S, H), "
                f"got {tuple(feature_scale_bank.shape)}."
            )
        if feature_scale_bank.shape[-1] != self.hidden_dim:
            raise ValueError(
                f"Expected hidden dimension {self.hidden_dim}, "
                f"got {feature_scale_bank.shape[-1]}."
            )

        x = torch.nan_to_num(
            feature_scale_bank.float(),
            nan=0.0,
            posinf=0.0,
            neginf=0.0,
        )

        num_funds, num_features, num_scales, hidden_dim = x.shape
        feature_tokens = x.permute(0, 2, 1, 3).reshape(
            num_funds * num_scales,
            num_features,
            hidden_dim,
        )
        refined, _ = self.attn(
            feature_tokens,
            feature_tokens,
            feature_tokens,
            need_weights=False,
        )
        return (
            (refined + feature_tokens)
            .reshape(num_funds, num_scales, num_features, hidden_dim)
            .permute(0, 2, 1, 3)
            .contiguous()
        )

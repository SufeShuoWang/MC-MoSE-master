from __future__ import annotations

import torch
import torch.nn as nn


class MarketFeatureSequenceEncoder(nn.Module):
    """Encode the shared monthly market feature sequence.

    The input data loader repeats the same market sequence for every fund in a
    month. This encoder uses the first row and returns one shared embedding.
    """

    def __init__(self, market_dim: int, lookback: int, embed_dim: int) -> None:
        super().__init__()
        if market_dim <= 0:
            raise ValueError("market_dim must be positive.")
        if lookback <= 0:
            raise ValueError("lookback must be positive.")
        if embed_dim <= 0:
            raise ValueError("embed_dim must be positive.")

        self.market_dim = int(market_dim)
        self.lookback = int(lookback)
        self.embed_dim = int(embed_dim)
        self.projection = nn.Linear(self.lookback * self.market_dim, self.embed_dim)

    def forward(self, market_seq: torch.Tensor) -> torch.Tensor:
        """Project market history into a shared embedding.

        Parameters
        ----------
        market_seq:
            Shape (N, L, D_m) from the data loader, or (L, D_m) for direct use.

        Returns
        -------
        torch.Tensor
            Shape (H_m,).
        """

        if market_seq.ndim == 3:
            if market_seq.shape[0] < 1:
                raise ValueError("market_seq must contain at least one fund row.")
            x = market_seq[0]
        elif market_seq.ndim == 2:
            x = market_seq
        else:
            raise ValueError(
                "Expected market_seq to have shape (N, L, D_m) or (L, D_m), "
                f"got {tuple(market_seq.shape)}."
            )

        if x.shape != (self.lookback, self.market_dim):
            raise ValueError(
                f"Expected market sequence shape ({self.lookback}, {self.market_dim}), "
                f"got {tuple(x.shape)}."
            )

        x = torch.nan_to_num(x.float(), nan=0.0, posinf=0.0, neginf=0.0)
        return self.projection(x.reshape(1, -1)).squeeze(0)

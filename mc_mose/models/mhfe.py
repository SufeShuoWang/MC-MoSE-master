from __future__ import annotations

import torch
import torch.nn as nn


_ATTENTION_BATCH_CHUNK_SIZE = 32_768


def _chunked_self_attention(
    attention: nn.MultiheadAttention,
    patch_sequence: torch.Tensor,
) -> torch.Tensor:
    """Run independent attention batches below CUDA SDPA's grid limit."""

    if patch_sequence.shape[0] <= _ATTENTION_BATCH_CHUNK_SIZE:
        attended, _ = attention(
            patch_sequence,
            patch_sequence,
            patch_sequence,
            need_weights=False,
        )
        return attended

    attended_chunks: list[torch.Tensor] = []
    for chunk in patch_sequence.split(_ATTENTION_BATCH_CHUNK_SIZE, dim=0):
        attended_chunk, _ = attention(
            chunk,
            chunk,
            chunk,
            need_weights=False,
        )
        attended_chunks.append(attended_chunk)
    return torch.cat(attended_chunks, dim=0)


class MHFE(nn.Module):
    """Multi-scale Historical Fund Encoder.

    The attention token axis is the patch sequence of one feature path.
    """

    def __init__(
        self,
        num_features: int,
        patch_scales: list[int] | tuple[int, ...],
        hidden_dim: int,
        num_heads: int = 4,
    ) -> None:
        super().__init__()
        if num_features <= 0:
            raise ValueError("num_features must be positive.")
        if hidden_dim <= 0:
            raise ValueError("hidden_dim must be positive.")
        if not patch_scales:
            raise ValueError("patch_scales must not be empty.")

        scales = [int(scale) for scale in patch_scales]
        if any(scale <= 0 for scale in scales):
            raise ValueError("all patch_scales must be positive.")

        self.num_features = int(num_features)
        self.patch_scales = scales
        self.hidden_dim = int(hidden_dim)

        self.full_projections = nn.ModuleDict()
        self.short_projections = nn.ModuleDict()
        self.attentions = nn.ModuleDict()

        for scale in self.patch_scales:
            key = str(scale)
            self.full_projections[key] = nn.ModuleList(
                [nn.Linear(scale, hidden_dim) for _ in range(self.num_features)]
            )

            short_layers = nn.ModuleDict()
            for length in range(1, scale):
                short_layers[str(length)] = nn.ModuleList(
                    [nn.Linear(length, hidden_dim) for _ in range(self.num_features)]
                )
            self.short_projections[key] = short_layers

            self.attentions[key] = nn.MultiheadAttention(
                embed_dim=hidden_dim,
                num_heads=num_heads,
                batch_first=True,
            )

    def forward(self, x: torch.Tensor, layout: str = "NLD") -> torch.Tensor:
        """Encode fund history.

        Parameters
        ----------
        x:
            Fund history tensor. By default, shape is (N, L, D_f), matching the
            data loader. Use layout="NDL" when passing (N, D_f, L).
        layout:
            "NLD" or "NDL".

        Returns
        -------
        torch.Tensor
            Shape (N, D_f, S, H).
        """

        if x.ndim != 3:
            raise ValueError(f"Expected x to have 3 dimensions, got shape {tuple(x.shape)}.")

        if layout == "NLD":
            x = x.transpose(1, 2)
        elif layout == "NDL":
            pass
        else:
            raise ValueError("layout must be 'NLD' or 'NDL'.")

        x = torch.nan_to_num(x.float(), nan=0.0, posinf=0.0, neginf=0.0)

        num_funds, num_features, seq_len = x.shape
        if num_features != self.num_features:
            raise ValueError(
                f"Expected {self.num_features} features, got {num_features}."
            )
        if seq_len <= 0:
            raise ValueError("sequence length must be positive.")

        scale_outputs: list[torch.Tensor] = []
        for scale in self.patch_scales:
            scale_key = str(scale)
            full_patch_count, tail_length = divmod(seq_len, scale)

            patch_embeddings: torch.Tensor | None = None
            if full_patch_count > 0:
                full_length = full_patch_count * scale
                full_patches = x[:, :, :full_length].reshape(
                    num_funds,
                    num_features,
                    full_patch_count,
                    scale,
                )
                full_layers = self.full_projections[scale_key]
                full_weight = torch.stack(
                    [layer.weight for layer in full_layers],
                    dim=0,
                )
                full_bias = torch.stack(
                    [layer.bias for layer in full_layers],
                    dim=0,
                )
                patch_embeddings = torch.einsum(
                    "ndpk,dhk->ndph",
                    full_patches,
                    full_weight,
                )
                patch_embeddings = patch_embeddings + full_bias[None, :, None, :]

            if tail_length > 0:
                tail_patch = x[:, :, full_patch_count * scale :]
                tail_layers = self.short_projections[scale_key][str(tail_length)]
                tail_weight = torch.stack(
                    [layer.weight for layer in tail_layers],
                    dim=0,
                )
                tail_bias = torch.stack(
                    [layer.bias for layer in tail_layers],
                    dim=0,
                )
                tail_embedding = torch.einsum(
                    "ndk,dhk->ndh",
                    tail_patch,
                    tail_weight,
                )
                tail_embedding = tail_embedding + tail_bias[None, :, :]
                tail_embedding = tail_embedding.unsqueeze(2)
                patch_embeddings = (
                    tail_embedding
                    if patch_embeddings is None
                    else torch.cat([patch_embeddings, tail_embedding], dim=2)
                )

            if patch_embeddings is None:
                raise RuntimeError("Failed to construct patch embeddings.")

            num_patches = patch_embeddings.shape[2]
            patch_sequence = patch_embeddings.reshape(
                num_funds * num_features,
                num_patches,
                self.hidden_dim,
            )
            attended = _chunked_self_attention(
                self.attentions[scale_key],
                patch_sequence,
            )
            scale_outputs.append(
                attended.mean(dim=1).reshape(
                    num_funds,
                    num_features,
                    self.hidden_dim,
                )
            )

        return torch.stack(scale_outputs, dim=2)

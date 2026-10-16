from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F


class ExpertMoE(nn.Module):
    """Expert mixture module for MC-MoSE.

    The fund-side feature bank uses ``hidden_dim`` (H_f), while market routing
    uses ``market_embed_dim`` (H_m). The router produces scalar gate scores for
    experts; it does not change the H_f dimension of expert vectors.
    """

    def __init__(
        self,
        num_features: int,
        num_scales: int,
        hidden_dim: int,
        market_embed_dim: int,
        num_shared_experts: int,
        num_routed_experts: int,
        topk_routed: int,
    ) -> None:
        super().__init__()
        if num_features <= 0:
            raise ValueError("num_features must be positive.")
        if num_scales <= 0:
            raise ValueError("num_scales must be positive.")
        if hidden_dim <= 0:
            raise ValueError("hidden_dim must be positive.")
        if market_embed_dim <= 0:
            raise ValueError("market_embed_dim must be positive.")
        if num_shared_experts <= 0:
            raise ValueError("num_shared_experts must be positive.")
        if num_routed_experts <= 0:
            raise ValueError("num_routed_experts must be positive.")
        if topk_routed <= 0:
            raise ValueError("topk_routed must be positive.")
        if topk_routed > num_routed_experts:
            raise ValueError("topk_routed must not exceed num_routed_experts.")

        self.num_features = int(num_features)
        self.num_scales = int(num_scales)
        self.hidden_dim = int(hidden_dim)
        self.market_embed_dim = int(market_embed_dim)
        self.num_shared_experts = int(num_shared_experts)
        self.num_routed_experts = int(num_routed_experts)
        self.topk_routed = int(topk_routed)
        self.num_experts = self.num_shared_experts + self.num_routed_experts

        self.feature_logits = nn.Parameter(
            torch.empty(self.num_experts, self.num_features)
        )
        self.scale_logits = nn.Parameter(torch.empty(self.num_experts, self.num_scales))
        self.shared_gate_bias = nn.Parameter(torch.zeros(self.num_shared_experts))
        self.routed_route_embeddings = nn.Parameter(
            torch.empty(self.num_routed_experts, self.market_embed_dim)
        )

        self.route_query = nn.Linear(
            self.market_embed_dim,
            self.market_embed_dim,
            bias=False,
        )
        self.route_key = nn.Linear(
            self.market_embed_dim,
            self.market_embed_dim,
            bias=False,
        )

        self.reset_parameters()

    def reset_parameters(self) -> None:
        nn.init.xavier_uniform_(self.feature_logits)
        nn.init.xavier_uniform_(self.scale_logits)
        nn.init.zeros_(self.shared_gate_bias)
        nn.init.xavier_uniform_(self.routed_route_embeddings)
        self.route_query.reset_parameters()
        self.route_key.reset_parameters()

    def forward(
        self,
        feature_bank: torch.Tensor,
        market_embedding: torch.Tensor,
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        """Aggregate feature-scale banks and route experts.

        Parameters
        ----------
        feature_bank:
            Tensor with shape (N, D_f, S, H_f).
        market_embedding:
            Tensor with shape (H_m,).

        Returns
        -------
        tuple[torch.Tensor, dict[str, torch.Tensor]]
            Prediction input with shape (N, E * H_f), plus diagnostic tensors.
        """

        self._validate_feature_bank(feature_bank)
        market_embedding = self._prepare_market_embedding(market_embedding)

        feature_bank = torch.nan_to_num(
            feature_bank.to(dtype=self.feature_logits.dtype),
            nan=0.0,
            posinf=0.0,
            neginf=0.0,
        )
        market_embedding = torch.nan_to_num(
            market_embedding.to(dtype=self.route_query.weight.dtype),
            nan=0.0,
            posinf=0.0,
            neginf=0.0,
        )

        feature_importance = F.softmax(self.feature_logits, dim=-1)
        scale_importance = F.softmax(self.scale_logits, dim=-1)

        expert_vectors = torch.einsum(
            "ef,es,nfsh->neh",
            feature_importance,
            scale_importance,
            feature_bank,
        )

        route_query = self.route_query(market_embedding.reshape(1, -1))
        route_keys = self.route_key(self.routed_route_embeddings)
        routed_raw_scores = (
            (route_query @ route_keys.T).reshape(-1) / math.sqrt(self.market_embed_dim)
        )
        routed_probs = F.softmax(routed_raw_scores, dim=-1)

        topk_scores, topk_local = torch.topk(
            routed_raw_scores,
            k=self.topk_routed,
            dim=-1,
        )
        routed_scores = routed_raw_scores.new_zeros(self.num_routed_experts)
        routed_scores = routed_scores.scatter(0, topk_local, topk_scores)

        shared_indices = torch.arange(
            self.num_shared_experts,
            device=feature_bank.device,
        )
        routed_global_indices = topk_local + self.num_shared_experts
        active_indices = torch.cat([shared_indices, routed_global_indices], dim=0)

        active_scores = torch.cat(
            [self.shared_gate_bias, routed_scores[topk_local]],
            dim=0,
        )
        active_gate_weights = F.softmax(active_scores, dim=-1)

        full_gate_weights = expert_vectors.new_zeros(self.num_experts)
        full_gate_weights = full_gate_weights.scatter(
            0,
            active_indices,
            active_gate_weights,
        )

        active_expert_vectors = expert_vectors[:, active_indices, :]
        gated_vectors = expert_vectors * full_gate_weights.reshape(1, -1, 1)
        prediction_input = gated_vectors.reshape(
            feature_bank.shape[0],
            self.num_experts * self.hidden_dim,
        )

        aux = {
            "feature_importance": feature_importance,
            "scale_importance": scale_importance,
            "expert_vectors": expert_vectors,
            "shared_indices": shared_indices,
            "routed_global_indices": routed_global_indices,
            "active_expert_indices": active_indices,
            "active_scores": active_scores,
            "active_gate_weights": active_gate_weights,
            "full_gate_weights": full_gate_weights,
            "active_expert_vectors": active_expert_vectors,
            "gated_vectors": gated_vectors,
            "topk_local": topk_local,
            "topk_scores": topk_scores,
            "routed_raw_scores": routed_raw_scores,
            "routed_scores": routed_scores,
            "routed_probs": routed_probs,
            "prediction_input": prediction_input,
        }
        return prediction_input, aux

    def _validate_feature_bank(self, feature_bank: torch.Tensor) -> None:
        if not isinstance(feature_bank, torch.Tensor):
            raise TypeError("feature_bank must be a torch.Tensor.")
        if feature_bank.ndim != 4:
            raise ValueError(
                "Expected feature_bank to have shape (N, D_f, S, H_f), "
                f"got {tuple(feature_bank.shape)}."
            )
        if feature_bank.shape[0] < 1:
            raise ValueError("feature_bank must contain at least one fund.")
        if feature_bank.shape[1] != self.num_features:
            raise ValueError(
                f"Expected num_features {self.num_features}, "
                f"got {feature_bank.shape[1]}."
            )
        if feature_bank.shape[2] != self.num_scales:
            raise ValueError(
                f"Expected num_scales {self.num_scales}, got {feature_bank.shape[2]}."
            )
        if feature_bank.shape[3] != self.hidden_dim:
            raise ValueError(
                f"Expected hidden_dim {self.hidden_dim}, got {feature_bank.shape[3]}."
            )

    def _prepare_market_embedding(self, market_embedding: torch.Tensor) -> torch.Tensor:
        if not isinstance(market_embedding, torch.Tensor):
            raise TypeError("market_embedding must be a torch.Tensor.")
        market_embedding = market_embedding.reshape(-1)
        if market_embedding.numel() != self.market_embed_dim:
            raise ValueError(
                f"Expected market_embed_dim {self.market_embed_dim}, "
                f"got {market_embedding.numel()}."
            )
        return market_embedding

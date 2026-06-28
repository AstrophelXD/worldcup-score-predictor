from __future__ import annotations

import torch
from torch import nn


class TeamPlayerGraphEncoder(nn.Module):
    """Encode player sets with intra-team attention and explicit matchup edges."""

    def __init__(
        self,
        player_dim: int,
        d_model: int,
        edge_dim: int,
        max_players: int = 11,
        max_edges: int = 24,
        n_heads: int = 2,
    ) -> None:
        super().__init__()
        self.player_proj = nn.Linear(player_dim, d_model)
        self.team_fallback = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.ReLU(),
            nn.Linear(d_model, d_model),
        )
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=n_heads,
            dim_feedforward=d_model * 2,
            batch_first=True,
            dropout=0.1,
        )
        self.intra_team = nn.TransformerEncoder(encoder_layer, num_layers=1)
        self.matchup_msg = nn.Sequential(
            nn.Linear(d_model * 2 + edge_dim, d_model),
            nn.ReLU(),
            nn.Linear(d_model, d_model),
        )
        self.cross_attn = nn.MultiheadAttention(d_model, n_heads, batch_first=True, dropout=0.1)

    def _side_embeddings(
        self,
        players: torch.Tensor,
        mask: torch.Tensor,
    ) -> torch.Tensor:
        x = self.player_proj(players)
        if not mask.any():
            return x
        return self.intra_team(x, src_key_padding_mask=~mask)

    def _apply_matchup_edges(
        self,
        target_emb: torch.Tensor,
        opponent_emb: torch.Tensor,
        edge_target_idx: torch.Tensor,
        edge_opponent_idx: torch.Tensor,
        edge_feats: torch.Tensor,
        edge_mask: torch.Tensor,
    ) -> torch.Tensor:
        updated = target_emb.clone()
        batch_size = target_emb.shape[0]
        for batch_idx in range(batch_size):
            active = edge_mask[batch_idx]
            if not active.any():
                continue
            for edge_idx in torch.where(active)[0]:
                t_idx = int(edge_target_idx[batch_idx, edge_idx].item())
                o_idx = int(edge_opponent_idx[batch_idx, edge_idx].item())
                msg = self.matchup_msg(
                    torch.cat(
                        [
                            target_emb[batch_idx, t_idx],
                            opponent_emb[batch_idx, o_idx],
                            edge_feats[batch_idx, edge_idx],
                        ],
                        dim=-1,
                    )
                )
                updated[batch_idx, t_idx] = updated[batch_idx, t_idx] + msg
        return updated

    def _pool_side(
        self,
        embeddings: torch.Tensor,
        mask: torch.Tensor,
        team_context: torch.Tensor,
    ) -> torch.Tensor:
        if not mask.any():
            return self.team_fallback(team_context)
        weights = mask.float().unsqueeze(-1)
        pooled = (embeddings * weights).sum(dim=1) / weights.sum(dim=1).clamp(min=1.0)
        return pooled + self.team_fallback(team_context)

    def _encode_batch_item(
        self,
        home_players: torch.Tensor,
        home_mask: torch.Tensor,
        away_players: torch.Tensor,
        away_mask: torch.Tensor,
        home_context: torch.Tensor,
        away_context: torch.Tensor,
        edge_home_idx: torch.Tensor,
        edge_away_idx: torch.Tensor,
        edge_feats: torch.Tensor,
        edge_mask: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        home_base = self._side_embeddings(home_players, home_mask)
        away_base = self._side_embeddings(away_players, away_mask)
        home_emb = self._apply_matchup_edges(
            home_base, away_base, edge_home_idx, edge_away_idx, edge_feats, edge_mask
        )
        away_emb = self._apply_matchup_edges(
            away_base, home_base, edge_away_idx, edge_home_idx, edge_feats, edge_mask
        )
        home_pooled = self._pool_side(home_emb, home_mask, home_context)
        away_pooled = self._pool_side(away_emb, away_mask, away_context)
        return home_pooled, away_pooled

    def forward(
        self,
        home_players: torch.Tensor,
        home_mask: torch.Tensor,
        away_players: torch.Tensor,
        away_mask: torch.Tensor,
        home_context: torch.Tensor,
        away_context: torch.Tensor,
        edge_home_idx: torch.Tensor,
        edge_away_idx: torch.Tensor,
        edge_feats: torch.Tensor,
        edge_mask: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        batch_size = home_players.shape[0]
        home_outputs: list[torch.Tensor] = []
        away_outputs: list[torch.Tensor] = []
        for idx in range(batch_size):
            home_pooled, away_pooled = self._encode_batch_item(
                home_players[idx : idx + 1],
                home_mask[idx : idx + 1],
                away_players[idx : idx + 1],
                away_mask[idx : idx + 1],
                home_context[idx : idx + 1],
                away_context[idx : idx + 1],
                edge_home_idx[idx : idx + 1],
                edge_away_idx[idx : idx + 1],
                edge_feats[idx : idx + 1],
                edge_mask[idx : idx + 1],
            )
            home_outputs.append(home_pooled)
            away_outputs.append(away_pooled)

        home_pooled = torch.cat(home_outputs, dim=0)
        away_pooled = torch.cat(away_outputs, dim=0)
        home_cross, _ = self.cross_attn(
            home_pooled.unsqueeze(1),
            away_pooled.unsqueeze(1),
            away_pooled.unsqueeze(1),
        )
        away_cross, _ = self.cross_attn(
            away_pooled.unsqueeze(1),
            home_pooled.unsqueeze(1),
            home_pooled.unsqueeze(1),
        )
        return home_cross.squeeze(1), away_cross.squeeze(1)

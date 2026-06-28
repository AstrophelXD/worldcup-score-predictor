from __future__ import annotations

import torch
from torch import nn


class TeamPlayerGraphEncoder(nn.Module):
    """Encode home/away player sets with cross-team interaction; degrades without players."""

    def __init__(
        self,
        player_dim: int,
        d_model: int,
        max_players: int = 11,
        n_heads: int = 2,
    ) -> None:
        super().__init__()
        self.max_players = max_players
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
        self.cross_attn = nn.MultiheadAttention(d_model, n_heads, batch_first=True, dropout=0.1)

    def _encode_side(
        self,
        players: torch.Tensor,
        mask: torch.Tensor,
        team_context: torch.Tensor,
    ) -> torch.Tensor:
        batch_size = players.shape[0]
        outputs: list[torch.Tensor] = []
        for idx in range(batch_size):
            if not mask[idx].any():
                outputs.append(self.team_fallback(team_context[idx : idx + 1]))
                continue
            x = self.player_proj(players[idx : idx + 1])
            side_mask = mask[idx : idx + 1]
            encoded = self.intra_team(x, src_key_padding_mask=~side_mask)
            weights = side_mask.float().unsqueeze(-1)
            pooled = (encoded * weights).sum(dim=1) / weights.sum(dim=1).clamp(min=1.0)
            outputs.append(pooled + self.team_fallback(team_context[idx : idx + 1]))
        return torch.cat(outputs, dim=0)

    def forward(
        self,
        home_players: torch.Tensor,
        home_mask: torch.Tensor,
        away_players: torch.Tensor,
        away_mask: torch.Tensor,
        home_context: torch.Tensor,
        away_context: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        home_emb = self._encode_side(home_players, home_mask, home_context)
        away_emb = self._encode_side(away_players, away_mask, away_context)

        home_cross, _ = self.cross_attn(
            home_emb.unsqueeze(1),
            away_emb.unsqueeze(1),
            away_emb.unsqueeze(1),
        )
        away_cross, _ = self.cross_attn(
            away_emb.unsqueeze(1),
            home_emb.unsqueeze(1),
            home_emb.unsqueeze(1),
        )
        return home_cross.squeeze(1), away_cross.squeeze(1)

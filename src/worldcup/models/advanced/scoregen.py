from __future__ import annotations

import torch
from torch import nn

from worldcup.models.advanced.bivariate_score_head import BivariateMixtureScoreHead
from worldcup.models.advanced.match_context_transformer import (
    MatchContextTransformer,
    TeamSequenceEncoder,
)
from worldcup.models.advanced.team_player_graph import TeamPlayerGraphEncoder


class ScoreGenFootballTransformer(nn.Module):
    """ScoreGen: tabular + sequence + player graph + odds -> mixture bivariate score head."""

    def __init__(
        self,
        *,
        tabular_dim: int,
        seq_dim: int,
        player_dim: int,
        odds_dim: int,
        edge_dim: int,
        d_model: int = 64,
        n_heads: int = 4,
        n_layers: int = 2,
        max_players: int = 11,
        max_edges: int = 24,
        grid_max_goal: int = 7,
        n_components: int = 3,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        self.grid_max_goal = grid_max_goal
        self.tabular_proj = nn.Sequential(
            nn.Linear(tabular_dim, d_model),
            nn.ReLU(),
            nn.Dropout(dropout),
        )
        self.odds_proj = nn.Sequential(
            nn.Linear(odds_dim, d_model),
            nn.ReLU(),
        )
        self.home_seq_encoder = TeamSequenceEncoder(
            seq_dim, d_model, n_heads=n_heads, n_layers=n_layers, dropout=dropout
        )
        self.away_seq_encoder = TeamSequenceEncoder(
            seq_dim, d_model, n_heads=n_heads, n_layers=n_layers, dropout=dropout
        )
        self.graph_encoder = TeamPlayerGraphEncoder(
            player_dim,
            d_model,
            edge_dim=edge_dim,
            max_players=max_players,
            max_edges=max_edges,
            n_heads=n_heads,
        )
        self.match_transformer = MatchContextTransformer(
            d_model, n_heads=n_heads, n_layers=n_layers, dropout=dropout
        )
        self.score_head = BivariateMixtureScoreHead(
            d_model,
            grid_max_goal=grid_max_goal,
            n_components=n_components,
            hidden_dim=d_model,
        )

    def enable_gradient_checkpointing(self) -> None:
        self.home_seq_encoder.gradient_checkpointing = True
        self.away_seq_encoder.gradient_checkpointing = True
        self.match_transformer.gradient_checkpointing = True

    def _match_context(
        self,
        tabular: torch.Tensor,
        home_seq: torch.Tensor,
        home_seq_mask: torch.Tensor,
        away_seq: torch.Tensor,
        away_seq_mask: torch.Tensor,
        home_players: torch.Tensor,
        home_player_mask: torch.Tensor,
        away_players: torch.Tensor,
        away_player_mask: torch.Tensor,
        odds: torch.Tensor,
        odds_mask: torch.Tensor,
        edge_home_idx: torch.Tensor,
        edge_away_idx: torch.Tensor,
        edge_feats: torch.Tensor,
        edge_mask: torch.Tensor,
    ) -> torch.Tensor:
        tabular_token = self.tabular_proj(tabular).unsqueeze(1)
        home_ctx = tabular_token.squeeze(1)
        away_ctx = tabular_token.squeeze(1)
        home_seq_emb = self.home_seq_encoder(home_seq, home_seq_mask).unsqueeze(1)
        away_seq_emb = self.away_seq_encoder(away_seq, away_seq_mask).unsqueeze(1)
        home_graph, away_graph = self.graph_encoder(
            home_players,
            home_player_mask,
            away_players,
            away_player_mask,
            home_ctx,
            away_ctx,
            edge_home_idx,
            edge_away_idx,
            edge_feats,
            edge_mask,
        )
        tokens = torch.cat(
            [
                tabular_token,
                home_seq_emb,
                away_seq_emb,
                home_graph.unsqueeze(1),
                away_graph.unsqueeze(1),
                self.odds_proj(odds).unsqueeze(1),
            ],
            dim=1,
        )
        token_mask = torch.cat(
            [
                torch.ones(tabular.shape[0], 1, dtype=torch.bool, device=tabular.device),
                home_seq_mask.any(dim=1, keepdim=True),
                away_seq_mask.any(dim=1, keepdim=True),
                home_player_mask.any(dim=1, keepdim=True),
                away_player_mask.any(dim=1, keepdim=True),
                odds_mask.unsqueeze(1),
            ],
            dim=1,
        )
        return self.match_transformer(tokens, token_mask)

    def forward(
        self,
        tabular: torch.Tensor,
        home_seq: torch.Tensor,
        home_seq_mask: torch.Tensor,
        away_seq: torch.Tensor,
        away_seq_mask: torch.Tensor,
        home_players: torch.Tensor,
        home_player_mask: torch.Tensor,
        away_players: torch.Tensor,
        away_player_mask: torch.Tensor,
        odds: torch.Tensor,
        odds_mask: torch.Tensor,
        edge_home_idx: torch.Tensor,
        edge_away_idx: torch.Tensor,
        edge_feats: torch.Tensor,
        edge_mask: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        context = self._match_context(
            tabular,
            home_seq,
            home_seq_mask,
            away_seq,
            away_seq_mask,
            home_players,
            home_player_mask,
            away_players,
            away_player_mask,
            odds,
            odds_mask,
            edge_home_idx,
            edge_away_idx,
            edge_feats,
            edge_mask,
        )
        return self.score_head(context)

    def predict_matrix(
        self,
        tabular: torch.Tensor,
        home_seq: torch.Tensor,
        home_seq_mask: torch.Tensor,
        away_seq: torch.Tensor,
        away_seq_mask: torch.Tensor,
        home_players: torch.Tensor,
        home_player_mask: torch.Tensor,
        away_players: torch.Tensor,
        away_player_mask: torch.Tensor,
        odds: torch.Tensor,
        odds_mask: torch.Tensor,
        edge_home_idx: torch.Tensor,
        edge_away_idx: torch.Tensor,
        edge_feats: torch.Tensor,
        edge_mask: torch.Tensor,
        temperature: float = 1.0,
    ) -> tuple[torch.Tensor, float]:
        context = self._match_context(
            tabular,
            home_seq,
            home_seq_mask,
            away_seq,
            away_seq_mask,
            home_players,
            home_player_mask,
            away_players,
            away_player_mask,
            odds,
            odds_mask,
            edge_home_idx,
            edge_away_idx,
            edge_feats,
            edge_mask,
        )
        return self.score_head.predict_matrix(context, temperature=temperature)

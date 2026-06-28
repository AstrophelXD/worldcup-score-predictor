from __future__ import annotations

import torch
from torch import nn

from worldcup.models.score_matrix_torch import matrix_to_logits, mixture_score_matrix


class BivariateMixtureScoreHead(nn.Module):
    """Mixture of Dixon-Coles bivariate distributions (not flat 64-way classifier)."""

    def __init__(
        self,
        input_dim: int,
        *,
        grid_max_goal: int = 7,
        n_components: int = 3,
        hidden_dim: int = 64,
    ) -> None:
        super().__init__()
        self.grid_max_goal = grid_max_goal
        self.n_components = n_components
        self.context = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
        )
        self.weight_head = nn.Linear(hidden_dim, n_components)
        self.home_rate_head = nn.Linear(hidden_dim, n_components)
        self.away_rate_head = nn.Linear(hidden_dim, n_components)
        self.rho_head = nn.Linear(hidden_dim, 1)

    def forward(self, context: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        encoded = self.context(context)
        weights = torch.softmax(self.weight_head(encoded), dim=-1)
        lambda_home = torch.nn.functional.softplus(self.home_rate_head(encoded)) + 0.05
        lambda_away = torch.nn.functional.softplus(self.away_rate_head(encoded)) + 0.05
        rho = torch.tanh(self.rho_head(encoded)).squeeze(-1) * 0.2

        matrix = mixture_score_matrix(
            weights,
            lambda_home,
            lambda_away,
            rho,
            grid_max_goal=self.grid_max_goal,
        )
        logits = matrix_to_logits(matrix)
        return matrix, logits

    def predict_matrix(
        self,
        context: torch.Tensor,
        temperature: float = 1.0,
    ) -> tuple[torch.Tensor, float]:
        matrix, logits = self.forward(context)
        if temperature != 1.0:
            scaled = torch.softmax(logits / max(temperature, 1e-6), dim=-1)
            grid = self.grid_max_goal + 1
            matrix = scaled.view(-1, grid, grid)
        overflow = max(0.0, 1.0 - float(matrix.sum().item()))
        return matrix, overflow

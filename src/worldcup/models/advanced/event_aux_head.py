from __future__ import annotations

import torch
from torch import nn


class MatchEventAuxHead(nn.Module):
    """Auxiliary regression head for match-level xG and shot totals."""

    TARGET_NAMES = ("home_xg", "away_xg", "home_shots", "away_shots")

    def __init__(self, input_dim: int, hidden_dim: int | None = None) -> None:
        super().__init__()
        width = hidden_dim or max(32, input_dim // 2)
        self.net = nn.Sequential(
            nn.Linear(input_dim, width),
            nn.ReLU(),
            nn.Linear(width, len(self.TARGET_NAMES)),
        )

    def forward(self, context: torch.Tensor) -> torch.Tensor:
        return torch.nn.functional.softplus(self.net(context)) + 1e-4

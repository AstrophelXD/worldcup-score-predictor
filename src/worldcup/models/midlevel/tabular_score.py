from __future__ import annotations

import torch
from torch import nn


class TabularScoreModel(nn.Module):
    """Mid-level tabular encoder that outputs 8x8 score matrix logits."""

    def __init__(
        self,
        input_dim: int,
        hidden_dims: list[int],
        grid_max_goal: int = 7,
        dropout: float = 0.2,
    ) -> None:
        super().__init__()
        self.grid_max_goal = grid_max_goal
        output_dim = (grid_max_goal + 1) ** 2

        layers: list[nn.Module] = []
        prev_dim = input_dim
        for hidden_dim in hidden_dims:
            layers.extend(
                [
                    nn.Linear(prev_dim, hidden_dim),
                    nn.ReLU(),
                    nn.Dropout(dropout),
                ]
            )
            prev_dim = hidden_dim
        layers.append(nn.Linear(prev_dim, output_dim))
        self.network = nn.Sequential(*layers)

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        return self.network(features)

    def predict_matrix(
        self,
        features: torch.Tensor,
        temperature: float = 1.0,
    ) -> tuple[torch.Tensor, float]:
        logits = self.forward(features) / max(temperature, 1e-6)
        probs = torch.softmax(logits, dim=-1)
        grid_size = self.grid_max_goal + 1
        matrix = probs.view(-1, grid_size, grid_size)
        overflow_prob = max(0.0, 1.0 - float(matrix.sum().item()))
        if overflow_prob > 0:
            matrix = matrix / matrix.sum()
        return matrix[0].detach().cpu().numpy(), overflow_prob

from __future__ import annotations

import torch


def _poisson_pmf(lam: torch.Tensor, max_k: int) -> torch.Tensor:
    """Batch Poisson PMF for k = 0..max_k."""
    ks = torch.arange(max_k + 1, device=lam.device, dtype=lam.dtype)
    ks = ks.view(1, -1)
    lam = lam.unsqueeze(-1)
    log_pmf = ks * torch.log(lam.clamp(min=1e-8)) - lam - torch.lgamma(ks + 1.0)
    return torch.exp(log_pmf)


def dixon_coles_matrix(
    lambda_home: torch.Tensor,
    lambda_away: torch.Tensor,
    rho: torch.Tensor,
    *,
    grid_max_goal: int = 7,
    max_support: int = 20,
) -> torch.Tensor:
    """Differentiable 8x8 score matrix with Dixon-Coles low-score correction."""
    home_pmf = _poisson_pmf(lambda_home, max_support)
    away_pmf = _poisson_pmf(lambda_away, max_support)
    joint = home_pmf.unsqueeze(-1) * away_pmf.unsqueeze(-2)
    grid = joint[:, : grid_max_goal + 1, : grid_max_goal + 1]

    rho = rho.unsqueeze(-1).unsqueeze(-1)
    lam_h = lambda_home.unsqueeze(-1).unsqueeze(-1)
    lam_a = lambda_away.unsqueeze(-1).unsqueeze(-1)
    tau = torch.ones_like(grid)
    tau[:, 0, 0] = 1.0 - (lam_h[:, 0, 0] * lam_a[:, 0, 0] * rho[:, 0, 0]).squeeze(-1)
    tau[:, 0, 1] = 1.0 + (lam_h[:, 0, 0] * rho[:, 0, 0]).squeeze(-1)
    tau[:, 1, 0] = 1.0 + (lam_a[:, 0, 0] * rho[:, 0, 0]).squeeze(-1)
    tau[:, 1, 1] = 1.0 - rho[:, 0, 0].squeeze(-1)

    adjusted = grid * tau
    normalized = adjusted / adjusted.sum(dim=(1, 2), keepdim=True).clamp(min=1e-8)
    return normalized


def mixture_score_matrix(
    weights: torch.Tensor,
    lambda_home: torch.Tensor,
    lambda_away: torch.Tensor,
    rho: torch.Tensor,
    *,
    grid_max_goal: int = 7,
) -> torch.Tensor:
    """Weighted mixture of Dixon-Coles components -> (batch, grid, grid)."""
    batch_size, n_components = weights.shape
    matrices = []
    for component in range(n_components):
        matrices.append(
            dixon_coles_matrix(
                lambda_home[:, component],
                lambda_away[:, component],
                rho if rho.ndim == 1 else rho[:, component],
                grid_max_goal=grid_max_goal,
            )
        )
    stacked = torch.stack(matrices, dim=1)
    w = weights.view(batch_size, n_components, 1, 1)
    mixed = (stacked * w).sum(dim=1)
    return mixed / mixed.sum(dim=(1, 2), keepdim=True).clamp(min=1e-8)


def matrix_to_logits(matrix: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    flat = matrix.view(matrix.shape[0], -1).clamp(min=eps)
    return torch.log(flat)

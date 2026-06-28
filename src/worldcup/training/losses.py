from __future__ import annotations

import torch
import torch.nn.functional as F


def flatten_score_index(
    home_goals: torch.Tensor,
    away_goals: torch.Tensor,
    grid_size: int,
) -> torch.Tensor:
    return home_goals * grid_size + away_goals


def score_nll_loss(
    logits: torch.Tensor,
    home_goals: torch.Tensor,
    away_goals: torch.Tensor,
    grid_max_goal: int = 7,
) -> torch.Tensor:
    grid_size = grid_max_goal + 1
    clamped_home = torch.clamp(home_goals, 0, grid_max_goal)
    clamped_away = torch.clamp(away_goals, 0, grid_max_goal)
    targets = flatten_score_index(clamped_home, clamped_away, grid_size)
    return F.cross_entropy(logits, targets)


def marginal_goal_nll(
    logits: torch.Tensor,
    goals: torch.Tensor,
    axis: str,
    grid_max_goal: int = 7,
) -> torch.Tensor:
    grid_size = grid_max_goal + 1
    probs = torch.softmax(logits, dim=-1).view(-1, grid_size, grid_size)
    if axis == "home":
        marginal = probs.sum(dim=2)
    elif axis == "away":
        marginal = probs.sum(dim=1)
    else:
        raise ValueError("axis must be 'home' or 'away'")
    clamped = torch.clamp(goals, 0, grid_max_goal)
    return F.nll_loss(marginal.log(), clamped)


def combined_midlevel_loss(
    logits: torch.Tensor,
    home_goals: torch.Tensor,
    away_goals: torch.Tensor,
    *,
    grid_max_goal: int,
    aux_weight: float,
) -> torch.Tensor:
    total = score_nll_loss(logits, home_goals, away_goals, grid_max_goal)
    if aux_weight <= 0:
        return total
    total = total + aux_weight * marginal_goal_nll(logits, home_goals, "home", grid_max_goal)
    total = total + aux_weight * marginal_goal_nll(logits, away_goals, "away", grid_max_goal)
    return total

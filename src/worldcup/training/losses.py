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


def score_nll_from_log_probs(
    log_probs: torch.Tensor,
    home_goals: torch.Tensor,
    away_goals: torch.Tensor,
    grid_max_goal: int = 7,
) -> torch.Tensor:
    grid_size = grid_max_goal + 1
    clamped_home = torch.clamp(home_goals, 0, grid_max_goal)
    clamped_away = torch.clamp(away_goals, 0, grid_max_goal)
    targets = flatten_score_index(clamped_home, clamped_away, grid_size)
    return F.nll_loss(log_probs, targets)


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
    return F.nll_loss(marginal.clamp(min=1e-8).log(), clamped)


def marginal_goal_nll_from_probs(
    probs: torch.Tensor,
    goals: torch.Tensor,
    axis: str,
    grid_max_goal: int = 7,
) -> torch.Tensor:
    grid_size = grid_max_goal + 1
    matrix = probs.view(-1, grid_size, grid_size)
    if axis == "home":
        marginal = matrix.sum(dim=2)
    elif axis == "away":
        marginal = matrix.sum(dim=1)
    else:
        raise ValueError("axis must be 'home' or 'away'")
    clamped = torch.clamp(goals, 0, grid_max_goal)
    return F.nll_loss(marginal.clamp(min=1e-8).log(), clamped)


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


def combined_scoregen_loss(
    log_probs: torch.Tensor,
    home_goals: torch.Tensor,
    away_goals: torch.Tensor,
    *,
    grid_max_goal: int,
    aux_weight: float,
    event_preds: torch.Tensor | None = None,
    event_targets: torch.Tensor | None = None,
    event_mask: torch.Tensor | None = None,
    event_aux_weight: float = 0.0,
) -> torch.Tensor:
    total = score_nll_from_log_probs(log_probs, home_goals, away_goals, grid_max_goal)
    if aux_weight > 0:
        probs = log_probs.exp()
        total = total + aux_weight * marginal_goal_nll_from_probs(
            probs, home_goals, "home", grid_max_goal
        )
        total = total + aux_weight * marginal_goal_nll_from_probs(
            probs, away_goals, "away", grid_max_goal
        )
    if (
        event_aux_weight > 0
        and event_preds is not None
        and event_targets is not None
        and event_mask is not None
        and event_mask.any()
    ):
        masked_preds = event_preds[event_mask]
        masked_targets = event_targets[event_mask]
        total = total + event_aux_weight * torch.nn.functional.mse_loss(
            torch.log1p(masked_preds),
            torch.log1p(masked_targets),
        )
    return total

from __future__ import annotations

from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path

import pandas as pd
from scipy.optimize import minimize

from worldcup.models.baseline.likelihood import expected_lambdas, match_log_likelihood
from worldcup.models.registry import BaselineCheckpoint, save_checkpoint


@dataclass
class TrainSplit:
    train: pd.DataFrame
    val: pd.DataFrame


@dataclass
class TrainResult:
    checkpoint_path: Path
    train_nll: float
    val_nll: float


def temporal_train_val_split(
    features: pd.DataFrame,
    train_cutoff: str,
    val_ratio: float,
) -> TrainSplit:
    features = features.sort_values("kickoff_ts").reset_index(drop=True)
    cutoff = pd.Timestamp(train_cutoff, tz="UTC")
    train_all = features.loc[pd.to_datetime(features["kickoff_ts"], utc=True) <= cutoff].copy()
    if train_all.empty:
        raise ValueError("no training rows before train_cutoff")

    val_size = max(1, int(len(train_all) * val_ratio)) if len(train_all) > 1 else 0
    if val_size >= len(train_all):
        val_size = max(1, len(train_all) // 5)

    train = train_all.iloc[:-val_size].copy() if val_size else train_all.copy()
    val = train_all.iloc[-val_size:].copy() if val_size else train_all.iloc[0:0].copy()
    return TrainSplit(train=train, val=val)


def _encode_teams(train_df: pd.DataFrame) -> tuple[list[str], dict[str, int]]:
    teams = sorted(set(train_df["home_team_id"]) | set(train_df["away_team_id"]))
    index = {team_id: idx for idx, team_id in enumerate(teams)}
    return teams, index


def _negative_log_likelihood(
    params: list[float],
    matches: pd.DataFrame,
    team_index: dict[str, int],
    rho: float,
) -> float:
    home_adv = params[0]
    attack = params[1 : 1 + len(team_index)]
    defense = params[1 + len(team_index) : 1 + 2 * len(team_index)]
    attack_map = {team_id: attack[idx] for team_id, idx in team_index.items()}
    defense_map = {team_id: defense[idx] for team_id, idx in team_index.items()}

    total = 0.0
    for row in matches.itertuples(index=False):
        if pd.isna(row.home_score_ft) or pd.isna(row.away_score_ft):
            continue
        lambda_home, lambda_away = expected_lambdas(
            row.home_team_id,
            row.away_team_id,
            attack_map,
            defense_map,
            home_adv,
        )
        total -= match_log_likelihood(
            int(row.home_score_ft),
            int(row.away_score_ft),
            lambda_home,
            lambda_away,
            rho,
        )
    return total


def _split_nll(
    matches: pd.DataFrame,
    team_index: dict[str, int],
    home_adv: float,
    attack_map: dict[str, float],
    defense_map: dict[str, float],
    rho: float,
) -> float:
    total = 0.0
    count = 0
    for row in matches.itertuples(index=False):
        if pd.isna(row.home_score_ft) or pd.isna(row.away_score_ft):
            continue
        lambda_home, lambda_away = expected_lambdas(
            row.home_team_id,
            row.away_team_id,
            attack_map,
            defense_map,
            home_adv,
        )
        total -= match_log_likelihood(
            int(row.home_score_ft),
            int(row.away_score_ft),
            lambda_home,
            lambda_away,
            rho,
        )
        count += 1
    return total / max(count, 1)


def train_baseline_dixon_coles(
    *,
    features: pd.DataFrame,
    train_cutoff: str,
    val_ratio: float,
    home_advantage_init: float,
    rho: float,
    grid_max_goal: int,
    model_name: str,
    model_version: str,
    checkpoint_dir: Path,
) -> TrainResult:
    split = temporal_train_val_split(features, train_cutoff, val_ratio)
    teams, team_index = _encode_teams(split.train)
    n_teams = len(teams)
    x0 = [home_advantage_init] + [0.0] * (2 * n_teams)

    bounds = [(-2.0, 2.0)] + [(-3.0, 3.0)] * (2 * n_teams)
    result = minimize(
        _negative_log_likelihood,
        x0=x0,
        args=(split.train, team_index, rho),
        method="L-BFGS-B",
        bounds=bounds,
    )
    if not result.success:
        raise RuntimeError(f"baseline optimization failed: {result.message}")

    home_adv = float(result.x[0])
    attack_vals = result.x[1 : 1 + n_teams]
    defense_vals = result.x[1 + n_teams : 1 + 2 * n_teams]
    attack_map = {team_id: float(attack_vals[idx]) for team_id, idx in team_index.items()}
    defense_map = {team_id: float(defense_vals[idx]) for team_id, idx in team_index.items()}

    train_nll = _split_nll(split.train, team_index, home_adv, attack_map, defense_map, rho)
    val_nll = _split_nll(split.val, team_index, home_adv, attack_map, defense_map, rho)

    checkpoint = BaselineCheckpoint(
        model_name=model_name,
        model_version=model_version,
        home_advantage=home_adv,
        rho=rho,
        grid_max_goal=grid_max_goal,
        attack=attack_map,
        defense=defense_map,
        train_cutoff=train_cutoff,
        trained_at=datetime.now(UTC).isoformat(),
        train_match_count=len(split.train),
    )
    checkpoint_path = save_checkpoint(checkpoint, checkpoint_dir)
    return TrainResult(checkpoint_path=checkpoint_path, train_nll=train_nll, val_nll=val_nll)

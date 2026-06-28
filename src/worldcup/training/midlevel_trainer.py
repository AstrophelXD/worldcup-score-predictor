from __future__ import annotations

import random
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, TensorDataset

from worldcup.features.tabular import (
    extract_labels,
    fit_feature_spec,
    vectorize_features,
)
from worldcup.models.midlevel.tabular_score import TabularScoreModel
from worldcup.models.registry import MidlevelCheckpoint, save_midlevel_checkpoint
from worldcup.training.baseline_trainer import TrainResult, temporal_train_val_split
from worldcup.training.losses import combined_midlevel_loss


@dataclass
class MidlevelTrainConfig:
    hidden_dims: list[int]
    dropout: float
    learning_rate: float
    weight_decay: float
    epochs: int
    patience: int
    batch_size: int
    aux_loss_weight: float
    seed: int
    mixed_precision: str | None


def _set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _scored_rows(df: pd.DataFrame) -> pd.DataFrame:
    return df.loc[df["home_score_ft"].notna() & df["away_score_ft"].notna()].copy()


def _batch_nll(
    model: TabularScoreModel,
    features: torch.Tensor,
    home_goals: torch.Tensor,
    away_goals: torch.Tensor,
    *,
    grid_max_goal: int,
    aux_loss_weight: float,
) -> float:
    model.eval()
    with torch.no_grad():
        logits = model(features)
        loss = combined_midlevel_loss(
            logits,
            home_goals,
            away_goals,
            grid_max_goal=grid_max_goal,
            aux_weight=aux_loss_weight,
        )
    return float(loss.item())


def train_midlevel_tabular(
    *,
    features: pd.DataFrame,
    train_cutoff: str,
    val_ratio: float,
    grid_max_goal: int,
    model_name: str,
    model_version: str,
    checkpoint_dir: Path,
    train_cfg: MidlevelTrainConfig,
) -> TrainResult:
    _set_seed(train_cfg.seed)
    split = temporal_train_val_split(features, train_cutoff, val_ratio)
    train_df = _scored_rows(split.train)
    val_df = _scored_rows(split.val)
    if train_df.empty:
        raise ValueError("no scored training rows before train_cutoff")

    spec = fit_feature_spec(train_df)
    x_train = vectorize_features(train_df, spec)
    home_train, away_train = extract_labels(train_df)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    use_amp = (
        device.type == "cuda"
        and train_cfg.mixed_precision in {"bf16", "bfloat16"}
        and torch.cuda.is_bf16_supported()
    )

    model = TabularScoreModel(
        input_dim=len(spec.feature_columns),
        hidden_dims=train_cfg.hidden_dims,
        grid_max_goal=grid_max_goal,
        dropout=train_cfg.dropout,
    ).to(device)

    train_tensors = TensorDataset(
        torch.from_numpy(x_train),
        torch.from_numpy(home_train.astype(np.int64)),
        torch.from_numpy(away_train.astype(np.int64)),
    )
    loader = DataLoader(
        train_tensors,
        batch_size=min(train_cfg.batch_size, len(train_tensors)),
        shuffle=True,
    )

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=train_cfg.learning_rate,
        weight_decay=train_cfg.weight_decay,
    )

    x_val = vectorize_features(val_df, spec) if not val_df.empty else None
    home_val, away_val = extract_labels(val_df) if not val_df.empty else (None, None)

    best_val = float("inf")
    best_state: dict[str, torch.Tensor] | None = None
    stale_epochs = 0

    for _epoch in range(train_cfg.epochs):
        model.train()
        for batch_x, batch_home, batch_away in loader:
            batch_x = batch_x.to(device)
            batch_home = batch_home.to(device)
            batch_away = batch_away.to(device)
            optimizer.zero_grad(set_to_none=True)

            if use_amp:
                with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                    logits = model(batch_x)
                    loss = combined_midlevel_loss(
                        logits,
                        batch_home,
                        batch_away,
                        grid_max_goal=grid_max_goal,
                        aux_weight=train_cfg.aux_loss_weight,
                    )
                loss.backward()
            else:
                logits = model(batch_x)
                loss = combined_midlevel_loss(
                    logits,
                    batch_home,
                    batch_away,
                    grid_max_goal=grid_max_goal,
                    aux_weight=train_cfg.aux_loss_weight,
                )
                loss.backward()

            optimizer.step()

        if x_val is not None and home_val is not None and away_val is not None:
            val_features = torch.from_numpy(x_val).to(device)
            val_home = torch.from_numpy(home_val.astype(np.int64)).to(device)
            val_away = torch.from_numpy(away_val.astype(np.int64)).to(device)
            val_loss = _batch_nll(
                model,
                val_features,
                val_home,
                val_away,
                grid_max_goal=grid_max_goal,
                aux_loss_weight=train_cfg.aux_loss_weight,
            )
        else:
            val_loss = _batch_nll(
                model,
                torch.from_numpy(x_train).to(device),
                torch.from_numpy(home_train.astype(np.int64)).to(device),
                torch.from_numpy(away_train.astype(np.int64)).to(device),
                grid_max_goal=grid_max_goal,
                aux_loss_weight=train_cfg.aux_loss_weight,
            )

        if val_loss < best_val:
            best_val = val_loss
            best_state = {
                key: value.detach().cpu().clone()
                for key, value in model.state_dict().items()
            }
            stale_epochs = 0
        else:
            stale_epochs += 1
            if stale_epochs >= train_cfg.patience:
                break

    if best_state is not None:
        model.load_state_dict(best_state)

    train_features = torch.from_numpy(x_train).to(device)
    train_home_t = torch.from_numpy(home_train.astype(np.int64)).to(device)
    train_away_t = torch.from_numpy(away_train.astype(np.int64)).to(device)
    train_nll = _batch_nll(
        model,
        train_features,
        train_home_t,
        train_away_t,
        grid_max_goal=grid_max_goal,
        aux_loss_weight=train_cfg.aux_loss_weight,
    )
    val_nll = best_val

    checkpoint = MidlevelCheckpoint(
        model_name=model_name,
        model_version=model_version,
        grid_max_goal=grid_max_goal,
        hidden_dims=list(train_cfg.hidden_dims),
        dropout=train_cfg.dropout,
        feature_spec=spec.to_dict(),
        train_cutoff=train_cutoff,
        trained_at=datetime.now(UTC).isoformat(),
        train_match_count=len(train_df),
        train_nll=train_nll,
        val_nll=val_nll,
    )
    checkpoint_path = save_midlevel_checkpoint(checkpoint, model.state_dict(), checkpoint_dir)
    return TrainResult(checkpoint_path=checkpoint_path, train_nll=train_nll, val_nll=val_nll)

from __future__ import annotations

import random
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, Dataset

from worldcup.features.event_features import event_targets_for_match
from worldcup.features.scoregen import (
    PLAYER_FEATURE_DIM,
    PlayerContext,
    ScoreGenFeatureSpec,
    build_scoregen_batch_item,
    fit_scoregen_spec,
    load_odds_table,
    load_player_context,
)
from worldcup.features.tabular import extract_labels
from worldcup.models.advanced.scoregen import ScoreGenFootballTransformer
from worldcup.models.registry import ScoregenCheckpoint, save_scoregen_checkpoint
from worldcup.training.baseline_trainer import TrainResult, temporal_train_val_split
from worldcup.training.losses import combined_scoregen_loss


@dataclass
class ScoregenTrainConfig:
    d_model: int
    n_heads: int
    n_layers: int
    n_components: int
    seq_len: int
    player_slots: int
    dropout: float
    learning_rate: float
    weight_decay: float
    epochs: int
    patience: int
    batch_size: int
    aux_loss_weight: float
    seed: int
    mixed_precision: str | None
    odds_path: Path | None
    curated_dir: Path | None = None
    grad_accum_steps: int = 1
    gradient_checkpointing: bool = False
    torch_compile: bool = False
    num_workers: int = 0
    pin_memory: bool = False
    event_aux_weight: float = 0.0


def _resolve_amp(
    mixed_precision: str | None,
    device: torch.device,
) -> tuple[bool, torch.dtype | None, torch.cuda.amp.GradScaler | None]:
    if device.type != "cuda" or not mixed_precision:
        return False, None, None
    mode = mixed_precision.lower()
    if mode in {"fp16", "float16"}:
        return True, torch.float16, torch.cuda.amp.GradScaler()
    if mode in {"bf16", "bfloat16"} and torch.cuda.is_bf16_supported():
        return True, torch.bfloat16, None
    return False, None, None


def _compute_batch_loss(
    model: ScoreGenFootballTransformer,
    batch: dict[str, torch.Tensor],
    *,
    grid_max_goal: int,
    aux_loss_weight: float,
    event_aux_weight: float,
    amp_dtype: torch.dtype | None,
) -> torch.Tensor:
    def _loss_from_outputs(log_probs: torch.Tensor, event_preds: torch.Tensor | None) -> torch.Tensor:
        return combined_scoregen_loss(
            log_probs,
            batch["home_goals"],
            batch["away_goals"],
            grid_max_goal=grid_max_goal,
            aux_weight=aux_loss_weight,
            event_preds=event_preds,
            event_targets=batch.get("event_targets"),
            event_mask=batch.get("event_mask"),
            event_aux_weight=event_aux_weight,
        )

    if event_aux_weight > 0:
        forward = lambda: model.forward_with_events(
            batch["tabular"],
            batch["home_seq"],
            batch["home_seq_mask"],
            batch["away_seq"],
            batch["away_seq_mask"],
            batch["home_players"],
            batch["home_player_mask"],
            batch["away_players"],
            batch["away_player_mask"],
            batch["odds"],
            batch["odds_mask"],
            batch["edge_home_idx"],
            batch["edge_away_idx"],
            batch["edge_feats"],
            batch["edge_mask"],
        )
    else:
        forward = lambda: _model_forward(model, batch)

    if amp_dtype is not None:
        with torch.autocast(device_type="cuda", dtype=amp_dtype):
            outputs = forward()
            if event_aux_weight > 0:
                _, log_probs, event_preds = outputs
            else:
                _, log_probs = outputs
                event_preds = None
            return _loss_from_outputs(log_probs, event_preds)
    outputs = forward()
    if event_aux_weight > 0:
        _, log_probs, event_preds = outputs
    else:
        _, log_probs = outputs
        event_preds = None
    return _loss_from_outputs(log_probs, event_preds)


def _model_forward(model: ScoreGenFootballTransformer, batch: dict[str, torch.Tensor]):
    return model(
        batch["tabular"],
        batch["home_seq"],
        batch["home_seq_mask"],
        batch["away_seq"],
        batch["away_seq_mask"],
        batch["home_players"],
        batch["home_player_mask"],
        batch["away_players"],
        batch["away_player_mask"],
        batch["odds"],
        batch["odds_mask"],
        batch["edge_home_idx"],
        batch["edge_away_idx"],
        batch["edge_feats"],
        batch["edge_mask"],
    )


class ScoreGenMatchDataset(Dataset):
    def __init__(
        self,
        features: pd.DataFrame,
        matches: pd.DataFrame,
        spec: ScoreGenFeatureSpec,
        odds_df: pd.DataFrame,
        player_context: PlayerContext,
        team_stats: pd.DataFrame,
    ) -> None:
        self.features = features.reset_index(drop=True)
        self.matches = matches
        self.spec = spec
        self.odds_df = odds_df
        self.player_context = player_context
        self.team_stats = team_stats

    def __len__(self) -> int:
        return len(self.features)

    def __getitem__(self, index: int) -> dict[str, torch.Tensor | bool]:
        row = self.features.iloc[index]
        item = build_scoregen_batch_item(
            row,
            self.matches,
            self.spec,
            self.odds_df,
            self.player_context,
        )
        home_goals, away_goals = extract_labels(pd.DataFrame([row]))
        event_targets, event_available = event_targets_for_match(
            self.team_stats,
            str(row["match_id"]),
            str(row["home_team_id"]),
            str(row["away_team_id"]),
        )
        return {
            "tabular": torch.from_numpy(item["tabular"]),
            "home_seq": torch.from_numpy(item["home_seq"]),
            "home_seq_mask": torch.from_numpy(item["home_seq_mask"]),
            "away_seq": torch.from_numpy(item["away_seq"]),
            "away_seq_mask": torch.from_numpy(item["away_seq_mask"]),
            "home_players": torch.from_numpy(item["home_players"]),
            "home_player_mask": torch.from_numpy(item["home_player_mask"]),
            "away_players": torch.from_numpy(item["away_players"]),
            "away_player_mask": torch.from_numpy(item["away_player_mask"]),
            "edge_home_idx": torch.from_numpy(item["edge_home_idx"]),
            "edge_away_idx": torch.from_numpy(item["edge_away_idx"]),
            "edge_feats": torch.from_numpy(item["edge_feats"]),
            "edge_mask": torch.from_numpy(item["edge_mask"]),
            "odds": torch.from_numpy(item["odds"]),
            "odds_mask": torch.tensor(item["odds_mask"], dtype=torch.bool),
            "home_goals": torch.tensor(int(home_goals[0]), dtype=torch.long),
            "away_goals": torch.tensor(int(away_goals[0]), dtype=torch.long),
            "event_targets": torch.tensor(event_targets, dtype=torch.float32),
            "event_mask": torch.tensor(event_available, dtype=torch.bool),
        }


def _collate(batch: list[dict]) -> dict[str, torch.Tensor]:
    keys = batch[0].keys()
    output: dict[str, torch.Tensor] = {}
    for key in keys:
        values = [item[key] for item in batch]
        if key == "odds_mask":
            output[key] = torch.stack(values)
        elif key == "event_mask":
            output[key] = torch.stack(values)
        elif values[0].dtype == torch.bool:
            output[key] = torch.stack(values)
        else:
            output[key] = torch.stack(values)
    return output


def _set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _scored_rows(df: pd.DataFrame) -> pd.DataFrame:
    return df.loc[df["home_score_ft"].notna() & df["away_score_ft"].notna()].copy()


def _evaluate(
    model: ScoreGenFootballTransformer,
    loader: DataLoader,
    *,
    grid_max_goal: int,
    aux_loss_weight: float,
    device: torch.device,
    mixed_precision: str | None = None,
) -> float:
    model.eval()
    total = 0.0
    count = 0
    _, amp_dtype, _ = _resolve_amp(mixed_precision, device)
    with torch.no_grad():
        for batch in loader:
            batch = {key: value.to(device) for key, value in batch.items()}
            if amp_dtype is not None:
                with torch.autocast(device_type="cuda", dtype=amp_dtype):
                    _, log_probs = _model_forward(model, batch)
                    loss = combined_scoregen_loss(
                        log_probs,
                        batch["home_goals"],
                        batch["away_goals"],
                        grid_max_goal=grid_max_goal,
                        aux_weight=aux_loss_weight,
                    )
            else:
                _, log_probs = _model_forward(model, batch)
                loss = combined_scoregen_loss(
                    log_probs,
                    batch["home_goals"],
                    batch["away_goals"],
                    grid_max_goal=grid_max_goal,
                    aux_weight=aux_loss_weight,
                )
            total += float(loss.item()) * len(batch["home_goals"])
            count += len(batch["home_goals"])
    return total / max(count, 1)


def train_scoregen_football_transformer(
    *,
    features: pd.DataFrame,
    matches: pd.DataFrame,
    train_cutoff: str,
    val_ratio: float,
    grid_max_goal: int,
    model_name: str,
    model_version: str,
    checkpoint_dir: Path,
    train_cfg: ScoregenTrainConfig,
) -> TrainResult:
    _set_seed(train_cfg.seed)
    split = temporal_train_val_split(features, train_cutoff, val_ratio)
    train_df = _scored_rows(split.train)
    val_df = _scored_rows(split.val)
    if train_df.empty:
        raise ValueError("no scored training rows before train_cutoff")

    odds_df = load_odds_table(train_cfg.odds_path)
    player_context = load_player_context(train_cfg.curated_dir)
    team_stats_path = (
        train_cfg.curated_dir / "team_match_stats.parquet"
        if train_cfg.curated_dir
        else None
    )
    team_stats = (
        pd.read_parquet(team_stats_path)
        if team_stats_path and team_stats_path.exists()
        else pd.DataFrame()
    )
    spec = fit_scoregen_spec(
        train_df,
        matches,
        seq_len=train_cfg.seq_len,
        player_slots=train_cfg.player_slots,
        odds_path=train_cfg.odds_path,
        player_context=player_context,
    )

    train_ds = ScoreGenMatchDataset(train_df, matches, spec, odds_df, player_context, team_stats)
    val_ds = ScoreGenMatchDataset(
        val_df if not val_df.empty else train_df.iloc[0:0],
        matches,
        spec,
        odds_df,
        player_context,
        team_stats,
    )
    train_loader = DataLoader(
        train_ds,
        batch_size=min(train_cfg.batch_size, max(1, len(train_ds))),
        shuffle=True,
        collate_fn=_collate,
        num_workers=train_cfg.num_workers,
        pin_memory=train_cfg.pin_memory and torch.cuda.is_available(),
    )
    val_loader = (
        DataLoader(
            val_ds,
            batch_size=train_cfg.batch_size,
            shuffle=False,
            collate_fn=_collate,
            num_workers=train_cfg.num_workers,
            pin_memory=train_cfg.pin_memory and torch.cuda.is_available(),
        )
        if len(val_ds) > 0
        else train_loader
    )

    tabular_dim = len(spec.tabular.feature_columns)
    model = ScoreGenFootballTransformer(
        tabular_dim=tabular_dim,
        seq_dim=len(spec.seq_means),
        player_dim=PLAYER_FEATURE_DIM,
        odds_dim=spec.odds_dim,
        edge_dim=spec.edge_dim,
        d_model=train_cfg.d_model,
        n_heads=train_cfg.n_heads,
        n_layers=train_cfg.n_layers,
        max_players=train_cfg.player_slots,
        max_edges=spec.max_matchup_edges,
        grid_max_goal=grid_max_goal,
        n_components=train_cfg.n_components,
        dropout=train_cfg.dropout,
    )
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    if train_cfg.gradient_checkpointing:
        model.enable_gradient_checkpointing()
    if train_cfg.torch_compile and hasattr(torch, "compile"):
        model = torch.compile(model)

    use_amp, amp_dtype, scaler = _resolve_amp(train_cfg.mixed_precision, device)
    grad_accum = max(1, train_cfg.grad_accum_steps)

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=train_cfg.learning_rate,
        weight_decay=train_cfg.weight_decay,
    )

    best_val = float("inf")
    best_state: dict[str, torch.Tensor] | None = None
    stale_epochs = 0

    for _epoch in range(train_cfg.epochs):
        model.train()
        optimizer.zero_grad(set_to_none=True)
        for step, batch in enumerate(train_loader):
            batch = {key: value.to(device) for key, value in batch.items()}
            loss = _compute_batch_loss(
                model,
                batch,
                grid_max_goal=grid_max_goal,
                aux_loss_weight=train_cfg.aux_loss_weight,
                event_aux_weight=train_cfg.event_aux_weight,
                amp_dtype=amp_dtype if use_amp else None,
            )
            loss = loss / grad_accum
            if scaler is not None:
                scaler.scale(loss).backward()
            else:
                loss.backward()
            if (step + 1) % grad_accum == 0 or (step + 1) == len(train_loader):
                if scaler is not None:
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    optimizer.step()
                optimizer.zero_grad(set_to_none=True)

        val_loss = _evaluate(
            model,
            val_loader,
            grid_max_goal=grid_max_goal,
            aux_loss_weight=train_cfg.aux_loss_weight,
            device=device,
            mixed_precision=train_cfg.mixed_precision,
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

    train_nll = _evaluate(
        model,
        train_loader,
        grid_max_goal=grid_max_goal,
        aux_loss_weight=train_cfg.aux_loss_weight,
        device=device,
        mixed_precision=train_cfg.mixed_precision,
    )
    checkpoint = ScoregenCheckpoint(
        model_type="scoregen",
        model_name=model_name,
        model_version=model_version,
        grid_max_goal=grid_max_goal,
        n_components=train_cfg.n_components,
        d_model=train_cfg.d_model,
        n_heads=train_cfg.n_heads,
        n_layers=train_cfg.n_layers,
        seq_len=train_cfg.seq_len,
        player_slots=train_cfg.player_slots,
        dropout=train_cfg.dropout,
        feature_spec=spec.to_dict(),
        train_cutoff=train_cutoff,
        trained_at=datetime.now(UTC).isoformat(),
        train_match_count=len(train_df),
        train_nll=train_nll,
        val_nll=best_val,
    )
    checkpoint_path = save_scoregen_checkpoint(checkpoint, model.state_dict(), checkpoint_dir)
    return TrainResult(checkpoint_path=checkpoint_path, train_nll=train_nll, val_nll=best_val)

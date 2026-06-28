import pandas as pd

from worldcup.features.player_state import (
    PLAYER_FEATURE_DIM,
    lineup_entries_for_match,
    player_availability,
)
from worldcup.utils.paths import project_root


def test_lineup_prefers_historical_over_projected():
    lineups = pd.read_csv(project_root() / "data" / "samples" / "lineups.csv")
    as_of = pd.Timestamp("2022-12-10T20:00:00Z", tz="UTC")
    fra = lineup_entries_for_match(lineups, "wc2022_fra_eng_qf", "team_fra", as_of)
    assert not fra.empty
    assert fra["lineup_status"].str.lower().eq("projected").all()


def test_lineup_uses_historical_for_final():
    lineups = pd.read_csv(project_root() / "data" / "samples" / "lineups.csv")
    as_of = pd.Timestamp("2022-12-18T15:00:00Z", tz="UTC")
    arg = lineup_entries_for_match(lineups, "wc2022_arg_fra_final", "team_arg", as_of)
    assert arg["lineup_status"].str.lower().eq("historical").all()


def test_projected_lineup_respects_snapshot_ts():
    lineups = pd.read_csv(project_root() / "data" / "samples" / "lineups.csv")
    before_snapshot = pd.Timestamp("2022-12-10T08:00:00Z", tz="UTC")
    fra = lineup_entries_for_match(lineups, "wc2022_fra_eng_qf", "team_fra", before_snapshot)
    assert fra.empty


def test_injured_player_has_zero_availability():
    injuries = pd.read_csv(project_root() / "data" / "samples" / "injuries.csv")
    injuries["team_id"] = injuries["team_id"].astype(str)
    as_of = pd.Timestamp("2022-12-10T20:00:00Z", tz="UTC")
    assert player_availability(injuries, "pla_benzema", "team_fra", as_of) == 0.0
    assert player_availability(injuries, "pla_kane", "team_eng", as_of) == 1.0
    assert 0.0 < player_availability(injuries, "pla_dybala", "team_arg", as_of) < 1.0


def test_player_feature_dim_is_seven():
    assert PLAYER_FEATURE_DIM == 7

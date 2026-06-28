import shutil

import pytest

from worldcup.data_ingestion.pipeline import run_ingest
from worldcup.data_ingestion.team_resolver import TeamResolver
from worldcup.data_ingestion.validate import validate_curated
from worldcup.utils.paths import project_root


@pytest.fixture
def sample_paths():
    root = project_root()
    tmp_root = root / "tests" / "tmp" / "ingest"
    if tmp_root.exists():
        shutil.rmtree(tmp_root)
    tmp_root.mkdir(parents=True)
    paths = {
        "raw_dir": tmp_root / "raw",
        "curated_dir": tmp_root / "curated",
        "mappings_dir": root / "data" / "external_mappings",
        "matches_csv": root / "data" / "samples" / "matches.csv",
        "elo_csv": root / "data" / "samples" / "elo.csv",
        "fifa_csv": root / "data" / "samples" / "fifa_rankings.csv",
    }
    yield paths
    shutil.rmtree(tmp_root, ignore_errors=True)


def test_team_resolver_uses_aliases():
    resolver = TeamResolver(aliases={"brazil": "team_bra"})
    assert resolver.resolve("Brazil") == "team_bra"
    assert resolver.resolve("Germany") == "team_germany"


def test_run_ingest_builds_curated_tables(sample_paths):
    import pandas as pd

    expected_matches = len(pd.read_csv(sample_paths["matches_csv"]))
    result = run_ingest(
        raw_dir=sample_paths["raw_dir"],
        curated_dir=sample_paths["curated_dir"],
        mappings_dir=sample_paths["mappings_dir"],
        matches_csv=sample_paths["matches_csv"],
        elo_csv=sample_paths["elo_csv"],
        fifa_csv=sample_paths["fifa_csv"],
        source_systems={
            "matches": "test_matches",
            "elo": "test_elo",
            "fifa_rankings": "test_fifa",
        },
    )
    assert result.match_count == expected_matches
    assert result.team_count >= 41
    assert result.elo_count >= 123
    assert result.fifa_count >= 123
    assert (sample_paths["curated_dir"] / "matches.parquet").exists()
    assert (sample_paths["curated_dir"] / "teams.parquet").exists()

    report = validate_curated(sample_paths["curated_dir"])
    assert report.ok
    assert report.stats["world_cup_matches"] >= 128

def test_world_cup_final_has_ft_label_not_aet_only(sample_paths):
    curated_dir = sample_paths["curated_dir"].parent / "curated_matches_only"
    raw_dir = sample_paths["raw_dir"].parent / "raw_matches_only"
    run_ingest(
        raw_dir=raw_dir,
        curated_dir=curated_dir,
        mappings_dir=sample_paths["mappings_dir"],
        matches_csv=sample_paths["matches_csv"],
        elo_csv=None,
        fifa_csv=None,
        source_systems={"matches": "test"},
    )
    import pandas as pd

    matches = pd.read_parquet(curated_dir / "matches.parquet")
    final_2022 = matches[matches["match_id"] == "wc2022_arg_fra_final"].iloc[0]
    assert final_2022["home_score_ft"] == 3
    assert final_2022["away_score_ft"] == 3
    assert final_2022["pen_score_home"] == 4

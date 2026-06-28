"""Download public datasets into data/external/downloads/ for the external data profile."""

from __future__ import annotations

import shutil
import urllib.request
from pathlib import Path

import pandas as pd

from worldcup.data_ingestion.sources.elo_from_matches import compute_elo_from_matches
from worldcup.data_ingestion.sources.transformers import (
    transform_eloratings_world_tsv,
    transform_fifa_rankings,
    transform_kaggle_international,
)
from worldcup.utils.paths import project_root

DOWNLOAD_URLS = {
    "international_results.csv": (
        "https://raw.githubusercontent.com/martj42/international_results/master/results.csv"
    ),
    "fifa_rankings_history.csv": (
        "https://raw.githubusercontent.com/Dato-Futbol/fifa-ranking/refs/heads/master/"
        "ranking_fifa_historical.csv"
    ),
    "elo_snapshot.tsv": "https://www.eloratings.net/World.tsv",
}


def _download(url: str, dest: Path) -> None:
    dest.parent.mkdir(parents=True, exist_ok=True)
    print(f"Downloading {url} -> {dest}")
    urllib.request.urlretrieve(url, dest)


def bootstrap_external_downloads(
    downloads_dir: Path | None = None,
    seeds_dir: Path | None = None,
) -> dict[str, int | str]:
    root = project_root()
    downloads = downloads_dir or root / "data" / "external" / "downloads"
    seeds = seeds_dir or root / "data" / "external" / "seeds"
    downloads.mkdir(parents=True, exist_ok=True)

    for filename, url in DOWNLOAD_URLS.items():
        _download(url, downloads / filename)

    international = pd.read_csv(downloads / "international_results.csv")
    canonical_matches = transform_kaggle_international(international)
    canonical_matches.to_csv(downloads / "international_matches_canonical.csv", index=False)

    elo_derived = compute_elo_from_matches(canonical_matches)
    elo_derived.to_csv(downloads / "elo_from_results.csv", index=False)

    snapshot_raw = pd.read_csv(downloads / "elo_snapshot.tsv", sep="\t", header=None)
    snapshot = transform_eloratings_world_tsv(snapshot_raw)
    snapshot.to_csv(downloads / "elo_snapshot.csv", index=False)

    fifa_raw = pd.read_csv(downloads / "fifa_rankings_history.csv")
    fifa = transform_fifa_rankings(fifa_raw)
    fifa.to_csv(downloads / "fifa_rankings_canonical.csv", index=False)

    seed_files = [
        "wc_matches.csv",
        "players.csv",
        "lineups.csv",
        "player_match_stats.csv",
        "injuries.csv",
        "odds.csv",
    ]
    copied = 0
    for name in seed_files:
        src = seeds / name
        if not src.exists():
            raise FileNotFoundError(
                f"Missing seed file {src}. Run: python -m scripts.export_external_seeds"
            )
        shutil.copy2(src, downloads / name)
        copied += 1

    return {
        "international_results_rows": len(international),
        "canonical_match_rows": len(canonical_matches),
        "elo_derived_rows": len(elo_derived),
        "elo_snapshot_rows": len(snapshot),
        "fifa_rows": len(fifa),
        "seed_files_copied": copied,
        "downloads_dir": str(downloads),
    }


def main() -> None:
    summary = bootstrap_external_downloads()
    print("Bootstrap complete:")
    for key, value in summary.items():
        print(f"  {key}: {value}")


if __name__ == "__main__":
    main()

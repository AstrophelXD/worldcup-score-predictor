"""Seed wc2026_results.csv with played matches through a cutoff date."""

from __future__ import annotations

import argparse

from worldcup.data_ingestion.sources.world_cup_2026_results import write_wc2026_results_csv


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate WC2026 played results CSV")
    parser.add_argument(
        "--cutoff",
        default="2026-06-28",
        help="Include matches on or before this date (YYYY-MM-DD)",
    )
    args = parser.parse_args()
    count = write_wc2026_results_csv(cutoff=args.cutoff)
    print(f"Wrote {count} results through {args.cutoff} -> data/samples/wc2026_results.csv")


if __name__ == "__main__":
    main()

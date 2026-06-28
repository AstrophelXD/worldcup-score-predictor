from __future__ import annotations

from datetime import UTC, datetime
from typing import Any

import pandas as pd


def utc_now() -> datetime:
    return datetime.now(UTC)


def stamp_source_metadata(
    records: list[dict[str, Any]],
    source_system: str,
    source_record_id_field: str | None = None,
) -> list[dict[str, Any]]:
    """Attach required traceability fields to raw records before curation."""
    now = utc_now()
    stamped: list[dict[str, Any]] = []
    for record in records:
        row = dict(record)
        row["source_system"] = source_system
        row["source_record_id"] = (
            str(row.get(source_record_id_field)) if source_record_id_field else None
        )
        row["ingested_at"] = now
        row["updated_at"] = now
        stamped.append(row)
    return stamped


def write_parquet(df: pd.DataFrame, path: str) -> None:
    df.to_parquet(path, index=False)


def read_parquet(path: str) -> pd.DataFrame:
    return pd.read_parquet(path)

from __future__ import annotations

from pathlib import Path

import pandas as pd

from worldcup.data_ingestion.base import stamp_source_metadata, write_parquet
from worldcup.utils.paths import ensure_dir


def load_csv_to_raw(
    csv_path: Path,
    raw_dir: Path,
    source_system: str,
    output_name: str,
) -> Path:
    """Load a CSV into raw layer with source metadata columns."""
    df = pd.read_csv(csv_path)
    records = stamp_source_metadata(df.to_dict(orient="records"), source_system=source_system)
    out_df = pd.DataFrame(records)
    ensure_dir(raw_dir)
    out_path = raw_dir / f"{output_name}.parquet"
    write_parquet(out_df, str(out_path))
    return out_path

from datetime import UTC, datetime

import pandas as pd

from worldcup.features.point_in_time import filter_as_of


def test_filter_as_of_excludes_future_rows():
    df = pd.DataFrame(
        {
            "kickoff_ts": [
                "2022-11-20T18:00:00Z",
                "2022-12-18T18:00:00Z",
            ],
            "value": [1, 2],
        }
    )
    cutoff = datetime(2022, 12, 1, tzinfo=UTC)
    filtered = filter_as_of(df, cutoff, "kickoff_ts")
    assert len(filtered) == 1
    assert filtered.iloc[0]["value"] == 1

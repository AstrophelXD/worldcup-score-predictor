from datetime import date

from pydantic import BaseModel, ConfigDict

from worldcup.entities.base import ManualOverrideMetadata, SourceMetadata


class EloRating(BaseModel):
    model_config = ConfigDict(extra="forbid")

    elo_id: str
    team_id: str
    rating: float
    rating_date: date
    rating_system: str = "elo"
    rank: int | None = None
    metadata: SourceMetadata


class FifaRanking(BaseModel):
    model_config = ConfigDict(extra="forbid")

    fifa_ranking_id: str
    team_id: str
    ranking_date: date
    rank: int
    points: float
    metadata: SourceMetadata


class Injury(BaseModel):
    model_config = ConfigDict(extra="forbid")

    injury_id: str
    player_id: str
    team_id: str
    injury_type: str | None = None
    status: str
    start_date: date
    expected_return_date: date | None = None
    confidence: float | None = None
    notes: str | None = None
    metadata: SourceMetadata
    override: ManualOverrideMetadata = ManualOverrideMetadata()


class LineupEntry(BaseModel):
    model_config = ConfigDict(extra="forbid")

    lineup_id: str
    match_id: str
    team_id: str
    player_id: str
    is_starting: bool
    bench_order: int | None = None
    position_code: str | None = None
    formation_slot: str | None = None
    minutes_played: int | None = None
    captain_flag: bool = False
    lineup_status: str  # projected / official / historical
    projection_prob: float | None = None
    metadata: SourceMetadata
    override: ManualOverrideMetadata = ManualOverrideMetadata()

from datetime import date, datetime

from pydantic import BaseModel, ConfigDict

from worldcup.entities.base import SourceMetadata


class Match(BaseModel):
    """90-minute regular time score is the primary label."""

    model_config = ConfigDict(extra="forbid")

    match_id: str
    competition_name: str
    season_name: str | None = None
    stage_name: str | None = None
    match_date: date
    kickoff_ts: datetime | None = None
    venue: str | None = None
    city: str | None = None
    country: str | None = None
    home_team_id: str
    away_team_id: str
    home_score_ft: int | None = None
    away_score_ft: int | None = None
    home_score_ht: int | None = None
    away_score_ht: int | None = None
    aet_score_home: int | None = None
    aet_score_away: int | None = None
    pen_score_home: int | None = None
    pen_score_away: int | None = None
    status: str = "scheduled"
    is_world_cup: bool = False
    is_knockout: bool = False
    metadata: SourceMetadata


class MatchContextFeatures(BaseModel):
    """Pre-match context features bound to a prediction cutoff."""

    model_config = ConfigDict(extra="forbid")

    match_id: str
    as_of_time: datetime
    stage_type: str | None = None
    is_knockout: bool = False
    is_world_cup: bool = False
    must_win_flag: bool = False
    draw_acceptable_flag: bool = True

from datetime import date

from pydantic import BaseModel, ConfigDict

from worldcup.entities.base import SourceMetadata


class Player(BaseModel):
    model_config = ConfigDict(extra="forbid")

    player_id: str
    full_name: str
    dob: date | None = None
    national_team_id: str | None = None
    primary_position: str | None = None
    secondary_positions: list[str] | None = None
    preferred_foot: str | None = None
    height_cm: int | None = None
    market_value_eur: float | None = None
    current_club: str | None = None
    metadata: SourceMetadata

from pydantic import BaseModel, ConfigDict

from worldcup.entities.base import SourceMetadata


class Team(BaseModel):
    model_config = ConfigDict(extra="forbid")

    team_id: str
    team_name: str
    country_code: str | None = None
    confederation: str | None = None
    fifa_team_id: str | None = None
    statsbomb_team_id: str | None = None
    is_national_team: bool = True
    metadata: SourceMetadata

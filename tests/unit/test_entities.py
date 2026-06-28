from datetime import datetime

from worldcup.entities.base import SourceMetadata
from worldcup.entities.match import Match
from worldcup.entities.team import Team


def test_team_schema_requires_metadata():
    meta = SourceMetadata(source_system="test", source_record_id="t1")
    team = Team(team_id="team_bra", team_name="Brazil", metadata=meta)
    assert team.team_name == "Brazil"


def test_match_primary_label_is_ft_score():
    meta = SourceMetadata(source_system="test")
    match = Match(
        match_id="m1",
        competition_name="FIFA World Cup",
        match_date=datetime(2022, 12, 18).date(),
        home_team_id="team_arg",
        away_team_id="team_fra",
        home_score_ft=3,
        away_score_ft=3,
        aet_score_home=4,
        aet_score_away=2,
        is_world_cup=True,
        metadata=meta,
    )
    assert match.home_score_ft == 3
    assert match.aet_score_home == 4

from worldcup.dashboard.flags import (
    is_known_team,
    team_flag,
    team_flag_image_url,
    team_label_html,
    team_label_plain,
)
from worldcup.dashboard.world_cup_2026_schedule import (
    WORLD_CUP_2026_GROUPS,
    WORLD_CUP_2026_SCHEDULE,
)


def test_world_cup_2026_schedule_has_full_tournament():
    assert len(WORLD_CUP_2026_SCHEDULE) == 104
    assert len(WORLD_CUP_2026_GROUPS) == 12
    assert sum(len(v) for v in WORLD_CUP_2026_GROUPS.values()) == 48


def test_world_cup_2026_group_stage_count():
    group_matches = [m for m in WORLD_CUP_2026_SCHEDULE if m.stage_name == "Group stage"]
    assert len(group_matches) == 72


def test_team_flags_cover_all_group_teams():
    teams = {team for group in WORLD_CUP_2026_GROUPS.values() for team in group}
    assert len(teams) == 48
    for team in teams:
        assert is_known_team(team)
        url = team_flag_image_url(team)
        assert url is not None
        assert "jsdelivr.net" in url
        assert url.endswith(".svg")
        assert team in team_label_plain(team)


def test_team_id_resolves_to_country_flag():
    assert team_flag_image_url("team_bra").endswith("/br.svg")
    assert team_flag_image_url("team_arg").endswith("/ar.svg")
    assert team_label_plain("team_bra") == "Brazil"


def test_team_label_html_renders_image_tag():
    html = team_label_html("England")
    assert "<img" in html
    assert "gb-eng.svg" in html
    assert "England" in html


def test_scotland_uses_subdivision_flag():
    assert team_flag("Scotland").endswith("/gb-sct.svg")

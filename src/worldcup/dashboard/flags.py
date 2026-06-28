"""Country flag rendering for dashboard (HTML images; Windows-safe)."""

from __future__ import annotations

from worldcup.data_ingestion.sources.world_cup_squads import TEAM_ALIASES

_FLAGCDN_BASE = "https://flagcdn.com"

_TEAM_ID_TO_NAME: dict[str, str] = {team_id: name for name, team_id in TEAM_ALIASES.items()}
_TEAM_ID_TO_NAME.setdefault("team_usa", "United States")

_TEAM_ISO: dict[str, str] = {
    "Algeria": "dz",
    "Argentina": "ar",
    "Australia": "au",
    "Austria": "at",
    "Belgium": "be",
    "Bosnia and Herzegovina": "ba",
    "Brazil": "br",
    "Canada": "ca",
    "Cape Verde": "cv",
    "Cabo Verde": "cv",
    "Colombia": "co",
    "Congo DR": "cd",
    "DR Congo": "cd",
    "Croatia": "hr",
    "Curaçao": "cw",
    "Curacao": "cw",
    "Czechia": "cz",
    "Czech Republic": "cz",
    "Ecuador": "ec",
    "Egypt": "eg",
    "England": "gb-eng",
    "France": "fr",
    "Germany": "de",
    "Ghana": "gh",
    "Haiti": "ht",
    "Iran": "ir",
    "Iraq": "iq",
    "Ivory Coast": "ci",
    "Côte d'Ivoire": "ci",
    "Japan": "jp",
    "Jordan": "jo",
    "Korea Republic": "kr",
    "South Korea": "kr",
    "Mexico": "mx",
    "Morocco": "ma",
    "Netherlands": "nl",
    "New Zealand": "nz",
    "Norway": "no",
    "Panama": "pa",
    "Paraguay": "py",
    "Portugal": "pt",
    "Qatar": "qa",
    "Saudi Arabia": "sa",
    "Scotland": "gb-sct",
    "Senegal": "sn",
    "South Africa": "za",
    "Spain": "es",
    "Sweden": "se",
    "Switzerland": "ch",
    "Tunisia": "tn",
    "Türkiye": "tr",
    "Turkey": "tr",
    "United States": "us",
    "USA": "us",
    "Uruguay": "uy",
    "Uzbekistan": "uz",
}


def resolve_team_name(name_or_id: str) -> str:
    cleaned = name_or_id.strip()
    if not cleaned:
        return cleaned
    if cleaned in _TEAM_ID_TO_NAME:
        return _TEAM_ID_TO_NAME[cleaned]
    if cleaned in _TEAM_ISO:
        return cleaned
    return cleaned


def team_iso_slug(team_name: str) -> str | None:
    name = resolve_team_name(team_name)
    return _TEAM_ISO.get(name)


def team_flag_image_url(team_name: str, *, width: int = 40) -> str | None:
    slug = team_iso_slug(team_name)
    if not slug:
        return None
    return f"{_FLAGCDN_BASE}/w{width}/{slug}.png"


def team_flag_html(team_name: str, *, width: int = 20, alt: str | None = None) -> str:
    display = resolve_team_name(team_name)
    url = team_flag_image_url(display, width=width)
    if not url:
        return "⚪"
    label = alt or display or "flag"
    height = max(12, int(width * 0.75))
    return (
        f'<img src="{url}" width="{width}" height="{height}" '
        f'style="vertical-align:middle;display:inline-block;border-radius:2px;" '
        f'alt="{label} flag"/>'
    )


def team_label_html(team_name: str, *, width: int = 20) -> str:
    display = resolve_team_name(team_name)
    return f'{team_flag_html(display, width=width, alt=display)}&nbsp;{display}'


def matchup_label_html(home_team: str, away_team: str, *, width: int = 20) -> str:
    return f"{team_label_html(home_team, width=width)} vs {team_label_html(away_team, width=width)}"


def team_label_plain(team_name: str) -> str:
    return resolve_team_name(team_name)


def team_label(team_name: str) -> str:
    """Plain-text label for widgets that cannot render HTML."""
    return team_label_plain(team_name)


def team_flag(team_name: str) -> str:
    """Return flag image URL when available (legacy helper name)."""
    url = team_flag_image_url(team_name)
    return url or "🏳️"


def matchup_label(home_team: str, away_team: str, *, separator: str = " vs ") -> str:
    home = team_label_plain(home_team)
    away = team_label_plain(away_team)
    return f"{home}{separator}{away}"


def is_known_team(team_name: str) -> bool:
    name = resolve_team_name(team_name)
    return name in _TEAM_ISO

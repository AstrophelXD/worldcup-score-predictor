"""Country flag rendering for dashboard (Streamlit-native + HTML fallback)."""

from __future__ import annotations

import html

from worldcup.data_ingestion.sources.world_cup_squads import TEAM_ALIASES

_FLAGCDN_BASE = "https://flagcdn.com"
_JSdelivr_BASE = "https://cdn.jsdelivr.net/gh/lipis/flag-icons@7.2.3/flags/4x3"

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


def team_iso_code(team_name: str) -> str | None:
    slug = team_iso_slug(team_name)
    if not slug:
        return None
    return slug.split("-")[-1].upper()


def team_flag_image_url(team_name: str, *, width: int = 40) -> str | None:
    """Primary image URL (SVG via jsDelivr; reliable in Streamlit st.image)."""
    slug = team_iso_slug(team_name)
    if not slug:
        return None
    return f"{_JSdelivr_BASE}/{slug}.svg"


def team_flag_image_url_png(team_name: str, *, width: int = 40) -> str | None:
    slug = team_iso_slug(team_name)
    if not slug:
        return None
    return f"{_FLAGCDN_BASE}/w{width}/{slug}.png"


def _iso_badge_html(team_name: str) -> str:
    code = team_iso_code(team_name) or "?"
    safe = html.escape(code)
    return (
        f'<span style="display:inline-flex;align-items:center;justify-content:center;'
        f"min-width:2rem;height:1.5rem;padding:0 0.35rem;border-radius:4px;"
        f'background:#e2e8f0;color:#334155;font-weight:700;font-size:0.75rem;">'
        f"{safe}</span>"
    )


def team_flag_html(team_name: str, *, width: int = 28, alt: str | None = None) -> str:
    display = resolve_team_name(team_name)
    url = team_flag_image_url(display)
    if not url:
        return _iso_badge_html(display)
    label = html.escape(alt or display or "flag")
    height = max(18, int(width * 0.75))
    png = team_flag_image_url_png(display, width=width) or url
    return (
        f'<img src="{url}" width="{width}" height="{height}" '
        f'style="vertical-align:middle;display:inline-block;border-radius:3px;'
        f'object-fit:cover;box-shadow:0 1px 2px rgba(0,0,0,0.12);" '
        f'alt="{label}" '
        f"onerror=\"this.onerror=null;this.src='{png}';\"/>"
    )


def team_label_html(team_name: str, *, width: int = 28) -> str:
    display = html.escape(resolve_team_name(team_name))
    return (
        f'<span style="display:inline-flex;align-items:center;gap:0.45rem;'
        f'font-size:1.05rem;font-weight:600;">'
        f"{team_flag_html(team_name, width=width, alt=display)}"
        f"<span>{display}</span></span>"
    )


def matchup_label_html(home_team: str, away_team: str, *, width: int = 32) -> str:
    return (
        f'<div style="display:flex;align-items:center;justify-content:center;'
        f'gap:1rem;flex-wrap:wrap;font-size:1.35rem;font-weight:700;">'
        f"{team_label_html(home_team, width=width)}"
        f'<span style="color:#94a3b8;font-weight:600;font-size:1.1rem;">vs</span>'
        f"{team_label_html(away_team, width=width)}"
        f"</div>"
    )


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

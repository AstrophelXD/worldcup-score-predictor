"""Country flag emoji helpers for dashboard display."""

from __future__ import annotations

# ISO 3166-1 alpha-2 → regional indicator flag sequence.
_TEAM_ISO: dict[str, str] = {
    "Algeria": "DZ",
    "Argentina": "AR",
    "Australia": "AU",
    "Austria": "AT",
    "Belgium": "BE",
    "Bosnia and Herzegovina": "BA",
    "Brazil": "BR",
    "Canada": "CA",
    "Cape Verde": "CV",
    "Cabo Verde": "CV",
    "Colombia": "CO",
    "Congo DR": "CD",
    "DR Congo": "CD",
    "Croatia": "HR",
    "Curaçao": "CW",
    "Curacao": "CW",
    "Czechia": "CZ",
    "Czech Republic": "CZ",
    "Ecuador": "EC",
    "Egypt": "EG",
    "England": "GB",
    "France": "FR",
    "Germany": "DE",
    "Ghana": "GH",
    "Haiti": "HT",
    "Iran": "IR",
    "Iraq": "IQ",
    "Ivory Coast": "CI",
    "Côte d'Ivoire": "CI",
    "Japan": "JP",
    "Jordan": "JO",
    "Korea Republic": "KR",
    "South Korea": "KR",
    "Mexico": "MX",
    "Morocco": "MA",
    "Netherlands": "NL",
    "New Zealand": "NZ",
    "Norway": "NO",
    "Panama": "PA",
    "Paraguay": "PY",
    "Portugal": "PT",
    "Qatar": "QA",
    "Saudi Arabia": "SA",
    "Scotland": "GB",
    "Senegal": "SN",
    "South Africa": "ZA",
    "Spain": "ES",
    "Sweden": "SE",
    "Switzerland": "CH",
    "Tunisia": "TN",
    "Türkiye": "TR",
    "Turkey": "TR",
    "United States": "US",
    "USA": "US",
    "Uruguay": "UY",
    "Uzbekistan": "UZ",
}

# Teams that share GB ISO but deserve distinct badges in the UI.
_SPECIAL_FLAGS: dict[str, str] = {
    "England": "🏴󠁧󠁢󠁥󠁮󠁧󠁿",
    "Scotland": "🏴󠁧󠁢󠁳󠁣󠁴󠁿",
}


def iso_to_flag(iso_code: str) -> str:
    code = iso_code.upper()
    if len(code) != 2 or not code.isalpha():
        return "🏳️"
    return "".join(chr(0x1F1E6 + ord(char) - ord("A")) for char in code)


def team_flag(team_name: str) -> str:
    name = team_name.strip()
    if not name:
        return "🏳️"
    if name in _SPECIAL_FLAGS:
        return _SPECIAL_FLAGS[name]
    iso = _TEAM_ISO.get(name)
    if iso:
        return iso_to_flag(iso)
    return "🏳️"


def team_label(team_name: str) -> str:
    flag = team_flag(team_name)
    return f"{flag} {team_name}"


def matchup_label(home_team: str, away_team: str, *, separator: str = " vs ") -> str:
    return f"{team_label(home_team)}{separator}{team_label(away_team)}"


def is_known_team(team_name: str) -> bool:
    name = team_name.strip()
    return name in _SPECIAL_FLAGS or name in _TEAM_ISO

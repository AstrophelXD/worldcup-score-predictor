"""Realistic June 2026 Elo / FIFA strength for World Cup 48 teams (sample data)."""

from __future__ import annotations

# elo ~1500 weak, ~1900 elite; fifa_rank lower is stronger
WC2026_TEAM_STRENGTH: dict[str, dict[str, float | int]] = {
    "Argentina": {"elo": 1985, "fifa_rank": 1, "fifa_points": 1860},
    "France": {"elo": 1975, "fifa_rank": 2, "fifa_points": 1850},
    "England": {"elo": 1945, "fifa_rank": 3, "fifa_points": 1820},
    "Brazil": {"elo": 1935, "fifa_rank": 4, "fifa_points": 1810},
    "Spain": {"elo": 1920, "fifa_rank": 5, "fifa_points": 1795},
    "Portugal": {"elo": 1910, "fifa_rank": 6, "fifa_points": 1785},
    "Germany": {"elo": 1905, "fifa_rank": 7, "fifa_points": 1780},
    "Netherlands": {"elo": 1895, "fifa_rank": 8, "fifa_points": 1770},
    "Belgium": {"elo": 1885, "fifa_rank": 9, "fifa_points": 1760},
    "Croatia": {"elo": 1875, "fifa_rank": 10, "fifa_points": 1750},
    "Norway": {"elo": 1865, "fifa_rank": 11, "fifa_points": 1740},
    "Colombia": {"elo": 1855, "fifa_rank": 12, "fifa_points": 1730},
    "Uruguay": {"elo": 1850, "fifa_rank": 13, "fifa_points": 1725},
    "United States": {"elo": 1840, "fifa_rank": 14, "fifa_points": 1715},
    "Mexico": {"elo": 1835, "fifa_rank": 15, "fifa_points": 1710},
    "Japan": {"elo": 1825, "fifa_rank": 16, "fifa_points": 1700},
    "Switzerland": {"elo": 1820, "fifa_rank": 17, "fifa_points": 1695},
    "Morocco": {"elo": 1815, "fifa_rank": 18, "fifa_points": 1690},
    "Senegal": {"elo": 1810, "fifa_rank": 19, "fifa_points": 1685},
    "Austria": {"elo": 1805, "fifa_rank": 20, "fifa_points": 1680},
    "Korea Republic": {"elo": 1795, "fifa_rank": 22, "fifa_points": 1670},
    "Ecuador": {"elo": 1785, "fifa_rank": 24, "fifa_points": 1660},
    "Australia": {"elo": 1780, "fifa_rank": 25, "fifa_points": 1655},
    "Paraguay": {"elo": 1775, "fifa_rank": 26, "fifa_points": 1650},
    "Türkiye": {"elo": 1770, "fifa_rank": 27, "fifa_points": 1645},
    "Sweden": {"elo": 1765, "fifa_rank": 28, "fifa_points": 1640},
    "Canada": {"elo": 1760, "fifa_rank": 29, "fifa_points": 1635},
    "Egypt": {"elo": 1755, "fifa_rank": 30, "fifa_points": 1630},
    "Algeria": {"elo": 1745, "fifa_rank": 32, "fifa_points": 1620},
    "Ivory Coast": {"elo": 1740, "fifa_rank": 33, "fifa_points": 1615},
    "Iran": {"elo": 1735, "fifa_rank": 34, "fifa_points": 1610},
    "Tunisia": {"elo": 1725, "fifa_rank": 36, "fifa_points": 1600},
    "Ghana": {"elo": 1720, "fifa_rank": 37, "fifa_points": 1595},
    "Czechia": {"elo": 1715, "fifa_rank": 38, "fifa_points": 1590},
    "Scotland": {"elo": 1710, "fifa_rank": 39, "fifa_points": 1585},
    "Bosnia and Herzegovina": {"elo": 1700, "fifa_rank": 41, "fifa_points": 1575},
    "Jordan": {"elo": 1690, "fifa_rank": 43, "fifa_points": 1565},
    "Panama": {"elo": 1685, "fifa_rank": 44, "fifa_points": 1560},
    "Uzbekistan": {"elo": 1680, "fifa_rank": 45, "fifa_points": 1555},
    "Congo DR": {"elo": 1675, "fifa_rank": 46, "fifa_points": 1550},
    "Iraq": {"elo": 1620, "fifa_rank": 58, "fifa_points": 1490},
    "Saudi Arabia": {"elo": 1665, "fifa_rank": 48, "fifa_points": 1540},
    "South Africa": {"elo": 1660, "fifa_rank": 49, "fifa_points": 1535},
    "Haiti": {"elo": 1580, "fifa_rank": 72, "fifa_points": 1450},
    "Curaçao": {"elo": 1560, "fifa_rank": 78, "fifa_points": 1430},
    "Cape Verde": {"elo": 1655, "fifa_rank": 50, "fifa_points": 1530},
    "New Zealand": {"elo": 1590, "fifa_rank": 68, "fifa_points": 1460},
    "Qatar": {"elo": 1600, "fifa_rank": 65, "fifa_points": 1470},
}


def strength_for(team_name: str, *, default_elo: float, default_rank: int) -> dict[str, float | int]:
    if team_name in WC2026_TEAM_STRENGTH:
        return dict(WC2026_TEAM_STRENGTH[team_name])
    return {"elo": default_elo, "fifa_rank": default_rank, "fifa_points": 1100 + default_elo / 3}

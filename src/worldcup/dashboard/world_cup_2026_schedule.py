"""Official-style FIFA World Cup 2026 schedule (104 matches, 48 teams)."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class ScheduleMatch:
    match_number: int
    match_date: str
    kickoff_et: str
    home_team: str
    away_team: str
    stage_name: str
    group: str = ""
    venue: str = ""
    city: str = ""


def _gs(
    num: int,
    date: str,
    time_et: str,
    home: str,
    away: str,
    group: str,
    venue: str,
    city: str,
) -> ScheduleMatch:
    return ScheduleMatch(
        match_number=num,
        match_date=date,
        kickoff_et=time_et,
        home_team=home,
        away_team=away,
        stage_name="Group stage",
        group=group,
        venue=venue,
        city=city,
    )


def _ko(
    num: int,
    date: str,
    time_et: str,
    home: str,
    away: str,
    stage: str,
    venue: str,
    city: str,
) -> ScheduleMatch:
    return ScheduleMatch(
        match_number=num,
        match_date=date,
        kickoff_et=time_et,
        home_team=home,
        away_team=away,
        stage_name=stage,
        venue=venue,
        city=city,
    )


WORLD_CUP_2026_GROUPS: dict[str, list[str]] = {
    "A": ["Mexico", "South Africa", "Korea Republic", "Czechia"],
    "B": ["Canada", "Switzerland", "Qatar", "Bosnia and Herzegovina"],
    "C": ["Brazil", "Morocco", "Haiti", "Scotland"],
    "D": ["United States", "Paraguay", "Australia", "Türkiye"],
    "E": ["Germany", "Curaçao", "Ivory Coast", "Ecuador"],
    "F": ["Netherlands", "Japan", "Sweden", "Tunisia"],
    "G": ["Belgium", "Egypt", "Iran", "New Zealand"],
    "H": ["Spain", "Cape Verde", "Saudi Arabia", "Uruguay"],
    "I": ["France", "Senegal", "Norway", "Iraq"],
    "J": ["Argentina", "Algeria", "Austria", "Jordan"],
    "K": ["Portugal", "Congo DR", "Uzbekistan", "Colombia"],
    "L": ["England", "Croatia", "Ghana", "Panama"],
}

WORLD_CUP_2026_SCHEDULE: list[ScheduleMatch] = [
    _gs(1, "2026-06-11", "15:00", "Mexico", "South Africa", "A", "Estadio Azteca", "Mexico City"),
    _gs(2, "2026-06-11", "22:00", "Korea Republic", "Czechia", "A", "Estadio Akron", "Guadalajara"),
    _gs(3, "2026-06-12", "15:00", "Canada", "Bosnia and Herzegovina", "B", "BMO Field", "Toronto"),
    _gs(4, "2026-06-12", "21:00", "United States", "Paraguay", "D", "SoFi Stadium", "Los Angeles"),
    _gs(5, "2026-06-13", "21:00", "Haiti", "Scotland", "C", "Gillette Stadium", "Boston"),
    _gs(6, "2026-06-13", "24:00", "Australia", "Türkiye", "D", "BC Place", "Vancouver"),
    _gs(7, "2026-06-13", "18:00", "Brazil", "Morocco", "C", "MetLife Stadium", "New York/New Jersey"),
    _gs(8, "2026-06-13", "15:00", "Qatar", "Switzerland", "B", "Levi's Stadium", "San Francisco Bay Area"),
    _gs(9, "2026-06-14", "19:00", "Ivory Coast", "Ecuador", "E", "Lincoln Financial Field", "Philadelphia"),
    _gs(10, "2026-06-14", "13:00", "Germany", "Curaçao", "E", "NRG Stadium", "Houston"),
    _gs(11, "2026-06-14", "16:00", "Netherlands", "Japan", "F", "AT&T Stadium", "Dallas"),
    _gs(12, "2026-06-14", "22:00", "Sweden", "Tunisia", "F", "Estadio BBVA", "Monterrey"),
    _gs(13, "2026-06-15", "18:00", "Saudi Arabia", "Uruguay", "H", "Hard Rock Stadium", "Miami"),
    _gs(14, "2026-06-15", "12:00", "Spain", "Cape Verde", "H", "Mercedes-Benz Stadium", "Atlanta"),
    _gs(15, "2026-06-15", "21:00", "Iran", "New Zealand", "G", "SoFi Stadium", "Los Angeles"),
    _gs(16, "2026-06-15", "15:00", "Belgium", "Egypt", "G", "Lumen Field", "Seattle"),
    _gs(17, "2026-06-16", "15:00", "France", "Senegal", "I", "MetLife Stadium", "New York/New Jersey"),
    _gs(18, "2026-06-16", "18:00", "Iraq", "Norway", "I", "Gillette Stadium", "Boston"),
    _gs(19, "2026-06-16", "21:00", "Argentina", "Algeria", "J", "Arrowhead Stadium", "Kansas City"),
    _gs(20, "2026-06-16", "24:00", "Austria", "Jordan", "J", "Levi's Stadium", "San Francisco Bay Area"),
    _gs(21, "2026-06-17", "19:00", "Ghana", "Panama", "L", "BMO Field", "Toronto"),
    _gs(22, "2026-06-17", "16:00", "England", "Croatia", "L", "AT&T Stadium", "Dallas"),
    _gs(23, "2026-06-17", "13:00", "Portugal", "Congo DR", "K", "NRG Stadium", "Houston"),
    _gs(24, "2026-06-17", "22:00", "Uzbekistan", "Colombia", "K", "Estadio Azteca", "Mexico City"),
    _gs(25, "2026-06-18", "12:00", "Czechia", "South Africa", "A", "Mercedes-Benz Stadium", "Atlanta"),
    _gs(26, "2026-06-18", "15:00", "Switzerland", "Bosnia and Herzegovina", "B", "SoFi Stadium", "Los Angeles"),
    _gs(27, "2026-06-18", "18:00", "Canada", "Qatar", "B", "BC Place", "Vancouver"),
    _gs(28, "2026-06-18", "21:00", "Mexico", "Korea Republic", "A", "Estadio Akron", "Guadalajara"),
    _gs(29, "2026-06-19", "21:00", "Brazil", "Haiti", "C", "Lincoln Financial Field", "Philadelphia"),
    _gs(30, "2026-06-19", "18:00", "Scotland", "Morocco", "C", "Gillette Stadium", "Boston"),
    _gs(31, "2026-06-19", "23:00", "Türkiye", "Paraguay", "D", "Levi's Stadium", "San Francisco Bay Area"),
    _gs(32, "2026-06-19", "15:00", "United States", "Australia", "D", "Lumen Field", "Seattle"),
    _gs(33, "2026-06-20", "16:00", "Germany", "Ivory Coast", "E", "BMO Field", "Toronto"),
    _gs(34, "2026-06-20", "20:00", "Ecuador", "Curaçao", "E", "Arrowhead Stadium", "Kansas City"),
    _gs(35, "2026-06-20", "13:00", "Netherlands", "Sweden", "F", "NRG Stadium", "Houston"),
    _gs(36, "2026-06-20", "24:00", "Tunisia", "Japan", "F", "Estadio BBVA", "Monterrey"),
    _gs(37, "2026-06-21", "18:00", "Uruguay", "Cape Verde", "H", "Hard Rock Stadium", "Miami"),
    _gs(38, "2026-06-21", "12:00", "Spain", "Saudi Arabia", "H", "Mercedes-Benz Stadium", "Atlanta"),
    _gs(39, "2026-06-21", "15:00", "Belgium", "Iran", "G", "SoFi Stadium", "Los Angeles"),
    _gs(40, "2026-06-21", "21:00", "New Zealand", "Egypt", "G", "BC Place", "Vancouver"),
    _gs(41, "2026-06-22", "20:00", "Norway", "Senegal", "I", "MetLife Stadium", "New York/New Jersey"),
    _gs(42, "2026-06-22", "17:00", "France", "Iraq", "I", "Lincoln Financial Field", "Philadelphia"),
    _gs(43, "2026-06-22", "13:00", "Argentina", "Austria", "J", "AT&T Stadium", "Dallas"),
    _gs(44, "2026-06-22", "23:00", "Jordan", "Algeria", "J", "Levi's Stadium", "San Francisco Bay Area"),
    _gs(45, "2026-06-23", "16:00", "England", "Ghana", "L", "Gillette Stadium", "Boston"),
    _gs(46, "2026-06-23", "19:00", "Panama", "Croatia", "L", "BMO Field", "Toronto"),
    _gs(47, "2026-06-23", "13:00", "Portugal", "Uzbekistan", "K", "NRG Stadium", "Houston"),
    _gs(48, "2026-06-23", "22:00", "Colombia", "Congo DR", "K", "Estadio Akron", "Guadalajara"),
    _gs(49, "2026-06-24", "18:00", "Scotland", "Brazil", "C", "Hard Rock Stadium", "Miami"),
    _gs(50, "2026-06-24", "18:00", "Morocco", "Haiti", "C", "Mercedes-Benz Stadium", "Atlanta"),
    _gs(51, "2026-06-24", "15:00", "Switzerland", "Canada", "B", "BC Place", "Vancouver"),
    _gs(52, "2026-06-24", "15:00", "Bosnia and Herzegovina", "Qatar", "B", "Lumen Field", "Seattle"),
    _gs(53, "2026-06-24", "21:00", "Czechia", "Mexico", "A", "Estadio Azteca", "Mexico City"),
    _gs(54, "2026-06-24", "21:00", "South Africa", "Korea Republic", "A", "Estadio BBVA", "Monterrey"),
    _gs(55, "2026-06-25", "16:00", "Curaçao", "Ivory Coast", "E", "Lincoln Financial Field", "Philadelphia"),
    _gs(56, "2026-06-25", "16:00", "Ecuador", "Germany", "E", "MetLife Stadium", "New York/New Jersey"),
    _gs(57, "2026-06-25", "19:00", "Japan", "Sweden", "F", "AT&T Stadium", "Dallas"),
    _gs(58, "2026-06-25", "19:00", "Tunisia", "Netherlands", "F", "Arrowhead Stadium", "Kansas City"),
    _gs(59, "2026-06-25", "22:00", "Türkiye", "United States", "D", "SoFi Stadium", "Los Angeles"),
    _gs(60, "2026-06-25", "22:00", "Paraguay", "Australia", "D", "Levi's Stadium", "San Francisco Bay Area"),
    _gs(61, "2026-06-26", "15:00", "Norway", "France", "I", "Gillette Stadium", "Boston"),
    _gs(62, "2026-06-26", "15:00", "Senegal", "Iraq", "I", "BMO Field", "Toronto"),
    _gs(63, "2026-06-26", "23:00", "Egypt", "Iran", "G", "Lumen Field", "Seattle"),
    _gs(64, "2026-06-26", "23:00", "New Zealand", "Belgium", "G", "BC Place", "Vancouver"),
    _gs(65, "2026-06-26", "20:00", "Cape Verde", "Saudi Arabia", "H", "NRG Stadium", "Houston"),
    _gs(66, "2026-06-26", "20:00", "Uruguay", "Spain", "H", "Estadio Akron", "Guadalajara"),
    _gs(67, "2026-06-27", "17:00", "Panama", "England", "L", "MetLife Stadium", "New York/New Jersey"),
    _gs(68, "2026-06-27", "17:00", "Croatia", "Ghana", "L", "Lincoln Financial Field", "Philadelphia"),
    _gs(69, "2026-06-27", "22:00", "Algeria", "Austria", "J", "Arrowhead Stadium", "Kansas City"),
    _gs(70, "2026-06-27", "22:00", "Jordan", "Argentina", "J", "AT&T Stadium", "Dallas"),
    _gs(71, "2026-06-27", "19:30", "Colombia", "Portugal", "K", "Hard Rock Stadium", "Miami"),
    _gs(72, "2026-06-27", "19:30", "Congo DR", "Uzbekistan", "K", "Mercedes-Benz Stadium", "Atlanta"),
    _ko(73, "2026-06-28", "15:00", "South Africa", "Canada", "Round of 32", "SoFi Stadium", "Los Angeles"),
    _ko(74, "2026-06-29", "16:30", "Germany", "Paraguay", "Round of 32", "Gillette Stadium", "Boston"),
    _ko(75, "2026-06-29", "21:00", "Netherlands", "Morocco", "Round of 32", "Estadio BBVA", "Monterrey"),
    _ko(76, "2026-06-29", "13:00", "Brazil", "Japan", "Round of 32", "NRG Stadium", "Houston"),
    _ko(77, "2026-06-30", "17:00", "France", "Group C/D/F/G/H 3rd Place", "Round of 32", "MetLife Stadium", "New York/New Jersey"),
    _ko(78, "2026-06-30", "13:00", "Ivory Coast", "Norway", "Round of 32", "AT&T Stadium", "Dallas"),
    _ko(79, "2026-06-30", "21:00", "Mexico", "Group C/E/F/H/I 3rd Place", "Round of 32", "Estadio Azteca", "Mexico City"),
    _ko(80, "2026-07-01", "12:00", "Group L Winners", "Group E/H/I/J/K 3rd Place", "Round of 32", "Mercedes-Benz Stadium", "Atlanta"),
    _ko(81, "2026-07-01", "20:00", "United States", "Bosnia and Herzegovina", "Round of 32", "Levi's Stadium", "San Francisco Bay Area"),
    _ko(82, "2026-07-01", "16:00", "Group G Winners", "Group A/E/H/I/J 3rd Place", "Round of 32", "Lumen Field", "Seattle"),
    _ko(83, "2026-07-02", "19:00", "Group K Runners Up", "Group L Runners Up", "Round of 32", "BMO Field", "Toronto"),
    _ko(84, "2026-07-02", "15:00", "Group H Winners", "Group J Runners Up", "Round of 32", "SoFi Stadium", "Los Angeles"),
    _ko(85, "2026-07-02", "23:00", "Switzerland", "Group E/F/G/I/J 3rd Place", "Round of 32", "BC Place", "Vancouver"),
    _ko(86, "2026-07-03", "18:00", "Argentina", "Group H Runners Up", "Round of 32", "Hard Rock Stadium", "Miami"),
    _ko(87, "2026-07-03", "21:30", "Group K Winners", "Group D/E/I/J/L 3rd Place", "Round of 32", "Arrowhead Stadium", "Kansas City"),
    _ko(88, "2026-07-03", "14:00", "Australia", "Group G Runners Up", "Round of 32", "AT&T Stadium", "Dallas"),
    _ko(89, "2026-07-04", "17:00", "Match 74 Winner", "Match 77 Winner", "Round of 16", "Lincoln Financial Field", "Philadelphia"),
    _ko(90, "2026-07-04", "13:00", "Match 73 Winner", "Match 75 Winner", "Round of 16", "NRG Stadium", "Houston"),
    _ko(91, "2026-07-05", "16:00", "Match 76 Winner", "Match 78 Winner", "Round of 16", "MetLife Stadium", "New York/New Jersey"),
    _ko(92, "2026-07-05", "20:00", "Match 79 Winner", "Match 80 Winner", "Round of 16", "Estadio Azteca", "Mexico City"),
    _ko(93, "2026-07-06", "15:00", "Match 83 Winner", "Match 84 Winner", "Round of 16", "AT&T Stadium", "Dallas"),
    _ko(94, "2026-07-06", "20:00", "Match 81 Winner", "Match 82 Winner", "Round of 16", "Lumen Field", "Seattle"),
    _ko(95, "2026-07-07", "12:00", "Match 86 Winner", "Match 88 Winner", "Round of 16", "Mercedes-Benz Stadium", "Atlanta"),
    _ko(96, "2026-07-07", "16:00", "Match 85 Winner", "Match 87 Winner", "Round of 16", "BC Place", "Vancouver"),
    _ko(97, "2026-07-09", "16:00", "Match 89 Winner", "Match 90 Winner", "Quarter-finals", "Gillette Stadium", "Boston"),
    _ko(98, "2026-07-10", "15:00", "Match 93 Winner", "Match 94 Winner", "Quarter-finals", "SoFi Stadium", "Los Angeles"),
    _ko(99, "2026-07-11", "17:00", "Match 91 Winner", "Match 92 Winner", "Quarter-finals", "Hard Rock Stadium", "Miami"),
    _ko(100, "2026-07-11", "21:00", "Match 95 Winner", "Match 96 Winner", "Quarter-finals", "Arrowhead Stadium", "Kansas City"),
    _ko(101, "2026-07-14", "15:00", "Match 97 Winner", "Match 98 Winner", "Semi-finals", "AT&T Stadium", "Dallas"),
    _ko(102, "2026-07-15", "15:00", "Match 99 Winner", "Match 100 Winner", "Semi-finals", "Mercedes-Benz Stadium", "Atlanta"),
    _ko(103, "2026-07-18", "17:00", "Match 101 Loser", "Match 102 Loser", "Third place", "Hard Rock Stadium", "Miami"),
    _ko(104, "2026-07-19", "15:00", "Match 101 Winner", "Match 102 Winner", "Final", "MetLife Stadium", "New York/New Jersey"),
]

TOURNAMENT_START = "2026-06-11"
TOURNAMENT_END = "2026-07-19"

STAGE_ORDER = [
    "Group stage",
    "Round of 32",
    "Round of 16",
    "Quarter-finals",
    "Semi-finals",
    "Third place",
    "Final",
]

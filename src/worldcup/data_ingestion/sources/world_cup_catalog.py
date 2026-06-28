"""Canonical World Cup 2018 / 2022 match results (90-minute scores as primary label)."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class WorldCupMatch:
    year: int
    stage_name: str
    match_date: str
    kickoff_ts: str
    home_team: str
    away_team: str
    home_score_ft: int
    away_score_ft: int
    home_score_ht: int | None = None
    away_score_ht: int | None = None
    aet_score_home: int | None = None
    aet_score_away: int | None = None
    pen_score_home: int | None = None
    pen_score_away: int | None = None
    venue: str = ""
    city: str = ""
    country: str = ""


def _m(
    year: int,
    stage: str,
    date: str,
    home: str,
    away: str,
    h_ft: int,
    a_ft: int,
    *,
    h_ht: int | None = None,
    a_ht: int | None = None,
    aet_h: int | None = None,
    aet_a: int | None = None,
    pen_h: int | None = None,
    pen_a: int | None = None,
    venue: str = "",
    city: str = "",
    country: str = "",
) -> WorldCupMatch:
    return WorldCupMatch(
        year=year,
        stage_name=stage,
        match_date=date,
        kickoff_ts=f"{date}T15:00:00Z",
        home_team=home,
        away_team=away,
        home_score_ft=h_ft,
        away_score_ft=a_ft,
        home_score_ht=h_ht,
        away_score_ht=a_ht,
        aet_score_home=aet_h,
        aet_score_away=aet_a,
        pen_score_home=pen_h,
        pen_score_away=pen_a,
        venue=venue,
        city=city,
        country=country or ("Russia" if year == 2018 else "Qatar"),
    )


WORLD_CUP_2018: list[WorldCupMatch] = [
    _m(2018, "Group stage", "2018-06-14", "Russia", "Saudi Arabia", 5, 0, h_ht=2, a_ht=0, venue="Luzhniki Stadium", city="Moscow"),
    _m(2018, "Group stage", "2018-06-15", "Egypt", "Uruguay", 0, 1, venue="Ekaterinburg Arena", city="Yekaterinburg"),
    _m(2018, "Group stage", "2018-06-19", "Russia", "Egypt", 3, 1, h_ht=0, a_ht=0, venue="Krestovsky Stadium", city="Saint Petersburg"),
    _m(2018, "Group stage", "2018-06-20", "Uruguay", "Saudi Arabia", 1, 0, venue="Rostov Arena", city="Rostov-on-Don"),
    _m(2018, "Group stage", "2018-06-25", "Uruguay", "Russia", 3, 0, h_ht=2, a_ht=0, venue="Samara Arena", city="Samara"),
    _m(2018, "Group stage", "2018-06-25", "Saudi Arabia", "Egypt", 2, 1, venue="Volgograd Arena", city="Volgograd"),
    _m(2018, "Group stage", "2018-06-15", "Portugal", "Spain", 3, 3, h_ht=2, a_ht=1, venue="Fisht Olympic Stadium", city="Sochi"),
    _m(2018, "Group stage", "2018-06-15", "Morocco", "Iran", 0, 1, venue="Krestovsky Stadium", city="Saint Petersburg"),
    _m(2018, "Group stage", "2018-06-20", "Portugal", "Morocco", 1, 0, venue="Luzhniki Stadium", city="Moscow"),
    _m(2018, "Group stage", "2018-06-20", "Iran", "Spain", 0, 1, venue="Kazan Arena", city="Kazan"),
    _m(2018, "Group stage", "2018-06-25", "Iran", "Portugal", 1, 1, venue="Saransk Arena", city="Saransk"),
    _m(2018, "Group stage", "2018-06-25", "Spain", "Morocco", 2, 2, venue="Kaliningrad Stadium", city="Kaliningrad"),
    _m(2018, "Group stage", "2018-06-16", "France", "Australia", 2, 1, h_ht=0, a_ht=0, venue="Kazan Arena", city="Kazan"),
    _m(2018, "Group stage", "2018-06-16", "Peru", "Denmark", 0, 1, venue="Mordovia Arena", city="Saransk"),
    _m(2018, "Group stage", "2018-06-21", "Denmark", "Australia", 1, 1, venue="Samara Arena", city="Samara"),
    _m(2018, "Group stage", "2018-06-21", "France", "Peru", 1, 0, venue="Ekaterinburg Arena", city="Yekaterinburg"),
    _m(2018, "Group stage", "2018-06-26", "Denmark", "France", 0, 0, venue="Luzhniki Stadium", city="Moscow"),
    _m(2018, "Group stage", "2018-06-26", "Australia", "Peru", 0, 2, venue="Fisht Olympic Stadium", city="Sochi"),
    _m(2018, "Group stage", "2018-06-16", "Argentina", "Iceland", 1, 1, h_ht=1, a_ht=1, venue="Spartak Stadium", city="Moscow"),
    _m(2018, "Group stage", "2018-06-16", "Croatia", "Nigeria", 2, 0, venue="Kaliningrad Stadium", city="Kaliningrad"),
    _m(2018, "Group stage", "2018-06-21", "Argentina", "Croatia", 0, 3, venue="Nizhny Novgorod Stadium", city="Nizhny Novgorod"),
    _m(2018, "Group stage", "2018-06-22", "Nigeria", "Iceland", 2, 0, venue="Volgograd Arena", city="Volgograd"),
    _m(2018, "Group stage", "2018-06-26", "Nigeria", "Argentina", 1, 2, venue="Krestovsky Stadium", city="Saint Petersburg"),
    _m(2018, "Group stage", "2018-06-26", "Iceland", "Croatia", 1, 2, venue="Rostov Arena", city="Rostov-on-Don"),
    _m(2018, "Group stage", "2018-06-17", "Costa Rica", "Serbia", 0, 1, venue="Samara Arena", city="Samara"),
    _m(2018, "Group stage", "2018-06-17", "Brazil", "Switzerland", 1, 1, h_ht=1, a_ht=0, venue="Rostov Arena", city="Rostov-on-Don"),
    _m(2018, "Group stage", "2018-06-22", "Brazil", "Costa Rica", 2, 0, venue="Krestovsky Stadium", city="Saint Petersburg"),
    _m(2018, "Group stage", "2018-06-22", "Serbia", "Switzerland", 1, 2, venue="Kaliningrad Stadium", city="Kaliningrad"),
    _m(2018, "Group stage", "2018-06-27", "Serbia", "Brazil", 0, 2, venue="Spartak Stadium", city="Moscow"),
    _m(2018, "Group stage", "2018-06-27", "Switzerland", "Costa Rica", 2, 2, venue="Nizhny Novgorod Stadium", city="Nizhny Novgorod"),
    _m(2018, "Group stage", "2018-06-17", "Germany", "Mexico", 0, 1, venue="Luzhniki Stadium", city="Moscow"),
    _m(2018, "Group stage", "2018-06-18", "Sweden", "South Korea", 1, 0, venue="Nizhny Novgorod Stadium", city="Nizhny Novgorod"),
    _m(2018, "Group stage", "2018-06-23", "Germany", "Sweden", 2, 1, venue="Fisht Olympic Stadium", city="Sochi"),
    _m(2018, "Group stage", "2018-06-23", "South Korea", "Mexico", 1, 2, venue="Rostov Arena", city="Rostov-on-Don"),
    _m(2018, "Group stage", "2018-06-27", "South Korea", "Germany", 2, 0, venue="Kazan Arena", city="Kazan"),
    _m(2018, "Group stage", "2018-06-27", "Mexico", "Sweden", 0, 3, venue="Ekaterinburg Arena", city="Yekaterinburg"),
    _m(2018, "Group stage", "2018-06-18", "Belgium", "Panama", 3, 0, venue="Fisht Olympic Stadium", city="Sochi"),
    _m(2018, "Group stage", "2018-06-18", "Tunisia", "England", 1, 2, venue="Volgograd Arena", city="Volgograd"),
    _m(2018, "Group stage", "2018-06-23", "Belgium", "Tunisia", 5, 2, venue="Otkrytie Arena", city="Moscow"),
    _m(2018, "Group stage", "2018-06-24", "England", "Panama", 6, 1, h_ht=5, a_ht=0, venue="Nizhny Novgorod Stadium", city="Nizhny Novgorod"),
    _m(2018, "Group stage", "2018-06-28", "England", "Belgium", 0, 1, venue="Kaliningrad Stadium", city="Kaliningrad"),
    _m(2018, "Group stage", "2018-06-28", "Panama", "Tunisia", 1, 2, venue="Saransk Arena", city="Saransk"),
    _m(2018, "Group stage", "2018-06-19", "Colombia", "Japan", 1, 2, venue="Mordovia Arena", city="Saransk"),
    _m(2018, "Group stage", "2018-06-19", "Poland", "Senegal", 1, 2, venue="Spartak Stadium", city="Moscow"),
    _m(2018, "Group stage", "2018-06-24", "Japan", "Senegal", 2, 2, venue="Ekaterinburg Arena", city="Yekaterinburg"),
    _m(2018, "Group stage", "2018-06-24", "Poland", "Colombia", 0, 3, venue="Kazan Arena", city="Kazan"),
    _m(2018, "Group stage", "2018-06-28", "Japan", "Poland", 0, 1, venue="Volgograd Arena", city="Volgograd"),
    _m(2018, "Group stage", "2018-06-28", "Senegal", "Colombia", 0, 1, venue="Samara Arena", city="Samara"),
    _m(2018, "Round of 16", "2018-06-30", "France", "Argentina", 4, 3, h_ht=1, a_ht=1, venue="Kazan Arena", city="Kazan"),
    _m(2018, "Round of 16", "2018-06-30", "Uruguay", "Portugal", 2, 1, venue="Fisht Olympic Stadium", city="Sochi"),
    _m(2018, "Round of 16", "2018-07-01", "Spain", "Russia", 1, 1, h_ht=1, a_ht=0, pen_h=3, pen_a=4, venue="Luzhniki Stadium", city="Moscow"),
    _m(2018, "Round of 16", "2018-07-01", "Croatia", "Denmark", 1, 1, h_ht=0, a_ht=0, pen_h=3, pen_a=2, venue="Nizhny Novgorod Stadium", city="Nizhny Novgorod"),
    _m(2018, "Round of 16", "2018-07-02", "Brazil", "Mexico", 2, 0, venue="Samara Arena", city="Samara"),
    _m(2018, "Round of 16", "2018-07-02", "Belgium", "Japan", 3, 2, h_ht=0, a_ht=2, venue="Rostov Arena", city="Rostov-on-Don"),
    _m(2018, "Round of 16", "2018-07-03", "Sweden", "Switzerland", 1, 0, venue="Krestovsky Stadium", city="Saint Petersburg"),
    _m(2018, "Round of 16", "2018-07-03", "Colombia", "England", 1, 1, h_ht=1, a_ht=1, pen_h=3, pen_a=4, venue="Spartak Stadium", city="Moscow"),
    _m(2018, "Quarter-finals", "2018-07-06", "Uruguay", "France", 0, 2, h_ht=0, a_ht=1, venue="Nizhny Novgorod Stadium", city="Nizhny Novgorod"),
    _m(2018, "Quarter-finals", "2018-07-06", "Brazil", "Belgium", 1, 2, venue="Kazan Arena", city="Kazan"),
    _m(2018, "Quarter-finals", "2018-07-07", "Sweden", "England", 0, 2, venue="Samara Arena", city="Samara"),
    _m(2018, "Quarter-finals", "2018-07-07", "Russia", "Croatia", 2, 2, h_ht=1, a_ht=1, aet_h=2, aet_a=2, pen_h=3, pen_a=4, venue="Fisht Olympic Stadium", city="Sochi"),
    _m(2018, "Semi-finals", "2018-07-10", "France", "Belgium", 1, 0, venue="Krestovsky Stadium", city="Saint Petersburg"),
    _m(2018, "Semi-finals", "2018-07-11", "Croatia", "England", 2, 1, h_ht=1, a_ht=1, aet_h=2, aet_a=1, venue="Luzhniki Stadium", city="Moscow"),
    _m(2018, "Third place", "2018-07-14", "Belgium", "England", 2, 0, venue="Krestovsky Stadium", city="Saint Petersburg"),
    _m(2018, "Final", "2018-07-15", "France", "Croatia", 4, 2, h_ht=2, a_ht=1, venue="Luzhniki Stadium", city="Moscow"),
]

WORLD_CUP_2022: list[WorldCupMatch] = [
    _m(2022, "Group stage", "2022-11-20", "Qatar", "Ecuador", 0, 2, venue="Al Bayt Stadium", city="Al Khor"),
    _m(2022, "Group stage", "2022-11-21", "Senegal", "Netherlands", 0, 2, venue="Al Thumama Stadium", city="Doha"),
    _m(2022, "Group stage", "2022-11-25", "Qatar", "Senegal", 1, 3, venue="Al Thumama Stadium", city="Doha"),
    _m(2022, "Group stage", "2022-11-25", "Netherlands", "Ecuador", 1, 1, venue="Khalifa International Stadium", city="Al Rayyan"),
    _m(2022, "Group stage", "2022-11-29", "Netherlands", "Qatar", 2, 0, venue="Al Bayt Stadium", city="Al Khor"),
    _m(2022, "Group stage", "2022-11-29", "Ecuador", "Senegal", 1, 2, venue="Khalifa International Stadium", city="Al Rayyan"),
    _m(2022, "Group stage", "2022-11-21", "England", "Iran", 6, 2, h_ht=3, a_ht=0, venue="Khalifa International Stadium", city="Al Rayyan"),
    _m(2022, "Group stage", "2022-11-21", "USA", "Wales", 1, 1, venue="Ahmad bin Ali Stadium", city="Al Rayyan"),
    _m(2022, "Group stage", "2022-11-25", "Wales", "Iran", 0, 2, venue="Ahmad bin Ali Stadium", city="Al Rayyan"),
    _m(2022, "Group stage", "2022-11-25", "England", "USA", 0, 0, venue="Al Bayt Stadium", city="Al Khor"),
    _m(2022, "Group stage", "2022-11-29", "England", "Wales", 3, 0, venue="Ahmad bin Ali Stadium", city="Al Rayyan"),
    _m(2022, "Group stage", "2022-11-29", "Iran", "USA", 0, 1, venue="Al Thumama Stadium", city="Doha"),
    _m(2022, "Group stage", "2022-11-22", "Argentina", "Saudi Arabia", 1, 2, h_ht=1, a_ht=0, venue="Lusail Stadium", city="Lusail"),
    _m(2022, "Group stage", "2022-11-22", "Mexico", "Poland", 0, 0, venue="Stadium 974", city="Doha"),
    _m(2022, "Group stage", "2022-11-26", "Poland", "Saudi Arabia", 2, 0, venue="Education City Stadium", city="Al Rayyan"),
    _m(2022, "Group stage", "2022-11-26", "Argentina", "Mexico", 2, 0, venue="Lusail Stadium", city="Lusail"),
    _m(2022, "Group stage", "2022-11-30", "Poland", "Argentina", 0, 2, venue="Stadium 974", city="Doha"),
    _m(2022, "Group stage", "2022-11-30", "Saudi Arabia", "Mexico", 1, 2, venue="Lusail Stadium", city="Lusail"),
    _m(2022, "Group stage", "2022-11-22", "Denmark", "Tunisia", 0, 0, venue="Education City Stadium", city="Al Rayyan"),
    _m(2022, "Group stage", "2022-11-22", "France", "Australia", 4, 1, h_ht=2, a_ht=1, venue="Al Janoub Stadium", city="Al Wakrah"),
    _m(2022, "Group stage", "2022-11-26", "Tunisia", "Australia", 0, 1, venue="Al Janoub Stadium", city="Al Wakrah"),
    _m(2022, "Group stage", "2022-11-26", "France", "Denmark", 2, 1, venue="Stadium 974", city="Doha"),
    _m(2022, "Group stage", "2022-11-30", "Australia", "Denmark", 1, 0, venue="Al Janoub Stadium", city="Al Wakrah"),
    _m(2022, "Group stage", "2022-11-30", "Tunisia", "France", 1, 0, venue="Education City Stadium", city="Al Rayyan"),
    _m(2022, "Group stage", "2022-11-23", "Germany", "Japan", 1, 2, venue="Khalifa International Stadium", city="Al Rayyan"),
    _m(2022, "Group stage", "2022-11-23", "Spain", "Costa Rica", 7, 0, venue="Al Thumama Stadium", city="Doha"),
    _m(2022, "Group stage", "2022-11-27", "Japan", "Costa Rica", 0, 1, venue="Ahmad bin Ali Stadium", city="Al Rayyan"),
    _m(2022, "Group stage", "2022-11-27", "Spain", "Germany", 1, 1, venue="Al Bayt Stadium", city="Al Khor"),
    _m(2022, "Group stage", "2022-12-01", "Japan", "Spain", 2, 1, venue="Khalifa International Stadium", city="Al Rayyan"),
    _m(2022, "Group stage", "2022-12-01", "Costa Rica", "Germany", 2, 4, venue="Al Bayt Stadium", city="Al Khor"),
    _m(2022, "Group stage", "2022-11-23", "Morocco", "Croatia", 0, 0, venue="Al Bayt Stadium", city="Al Khor"),
    _m(2022, "Group stage", "2022-11-23", "Belgium", "Canada", 1, 0, venue="Ahmad bin Ali Stadium", city="Al Rayyan"),
    _m(2022, "Group stage", "2022-11-27", "Belgium", "Morocco", 0, 2, venue="Al Thumama Stadium", city="Doha"),
    _m(2022, "Group stage", "2022-11-27", "Croatia", "Canada", 4, 1, venue="Khalifa International Stadium", city="Al Rayyan"),
    _m(2022, "Group stage", "2022-12-01", "Croatia", "Belgium", 0, 0, venue="Ahmad bin Ali Stadium", city="Al Rayyan"),
    _m(2022, "Group stage", "2022-12-01", "Canada", "Morocco", 1, 2, venue="Al Thumama Stadium", city="Doha"),
    _m(2022, "Group stage", "2022-11-24", "Switzerland", "Cameroon", 1, 0, venue="Al Janoub Stadium", city="Al Wakrah"),
    _m(2022, "Group stage", "2022-11-24", "Brazil", "Serbia", 2, 0, venue="Lusail Stadium", city="Lusail"),
    _m(2022, "Group stage", "2022-11-28", "Cameroon", "Serbia", 3, 3, venue="Stadium 974", city="Doha"),
    _m(2022, "Group stage", "2022-11-28", "Brazil", "Switzerland", 1, 0, venue="Stadium 974", city="Doha"),
    _m(2022, "Group stage", "2022-12-02", "Serbia", "Switzerland", 2, 3, h_ht=0, a_ht=2, venue="Stadium 974", city="Doha"),
    _m(2022, "Group stage", "2022-12-02", "Cameroon", "Brazil", 1, 0, venue="Lusail Stadium", city="Lusail"),
    _m(2022, "Group stage", "2022-11-24", "Uruguay", "South Korea", 0, 0, venue="Education City Stadium", city="Al Rayyan"),
    _m(2022, "Group stage", "2022-11-24", "Portugal", "Ghana", 3, 2, venue="Stadium 974", city="Doha"),
    _m(2022, "Group stage", "2022-11-28", "South Korea", "Ghana", 2, 3, venue="Education City Stadium", city="Al Rayyan"),
    _m(2022, "Group stage", "2022-11-28", "Portugal", "Uruguay", 2, 0, venue="Lusail Stadium", city="Lusail"),
    _m(2022, "Group stage", "2022-12-02", "Ghana", "Uruguay", 0, 2, venue="Al Janoub Stadium", city="Al Wakrah"),
    _m(2022, "Group stage", "2022-12-02", "South Korea", "Portugal", 2, 1, venue="Education City Stadium", city="Al Rayyan"),
    _m(2022, "Round of 16", "2022-12-03", "Netherlands", "USA", 3, 1, venue="Ahmad bin Ali Stadium", city="Al Rayyan"),
    _m(2022, "Round of 16", "2022-12-03", "Argentina", "Australia", 2, 1, h_ht=1, a_ht=0, venue="Ahmad bin Ali Stadium", city="Al Rayyan"),
    _m(2022, "Round of 16", "2022-12-04", "France", "Poland", 3, 1, venue="Al Thumama Stadium", city="Doha"),
    _m(2022, "Round of 16", "2022-12-04", "England", "Senegal", 3, 0, venue="Al Bayt Stadium", city="Al Khor"),
    _m(2022, "Round of 16", "2022-12-05", "Japan", "Croatia", 1, 1, h_ht=1, a_ht=0, pen_h=1, pen_a=3, venue="Al Janoub Stadium", city="Al Wakrah"),
    _m(2022, "Round of 16", "2022-12-05", "Brazil", "South Korea", 4, 1, venue="Stadium 974", city="Doha"),
    _m(2022, "Round of 16", "2022-12-06", "Morocco", "Spain", 0, 0, pen_h=3, pen_a=0, venue="Education City Stadium", city="Al Rayyan"),
    _m(2022, "Round of 16", "2022-12-06", "Portugal", "Switzerland", 6, 1, venue="Lusail Stadium", city="Lusail"),
    _m(2022, "Quarter-finals", "2022-12-09", "Croatia", "Brazil", 1, 1, pen_h=4, pen_a=2, venue="Education City Stadium", city="Al Rayyan"),
    _m(2022, "Quarter-finals", "2022-12-09", "Netherlands", "Argentina", 2, 2, h_ht=1, a_ht=1, pen_h=3, pen_a=4, venue="Lusail Stadium", city="Lusail"),
    _m(2022, "Quarter-finals", "2022-12-10", "Morocco", "Portugal", 1, 0, venue="Al Thumama Stadium", city="Doha"),
    _m(2022, "Quarter-finals", "2022-12-10", "England", "France", 1, 2, h_ht=0, a_ht=1, venue="Al Bayt Stadium", city="Al Khor"),
    _m(2022, "Semi-finals", "2022-12-13", "Argentina", "Croatia", 3, 0, venue="Lusail Stadium", city="Lusail"),
    _m(2022, "Semi-finals", "2022-12-14", "France", "Morocco", 2, 0, venue="Al Bayt Stadium", city="Al Khor"),
    _m(2022, "Third place", "2022-12-17", "Croatia", "Morocco", 2, 1, venue="Khalifa International Stadium", city="Al Rayyan"),
    _m(2022, "Final", "2022-12-18", "Argentina", "France", 3, 3, h_ht=2, a_ht=0, pen_h=4, pen_a=2, venue="Lusail Stadium", city="Lusail"),
]

FRIENDLY_MATCHES: list[WorldCupMatch] = [
    _m(2017, "Friendly", "2017-06-09", "Brazil", "Argentina", 1, 0, country="Australia", venue="Melbourne Cricket Ground", city="Melbourne"),
    _m(2017, "Friendly", "2017-06-13", "France", "England", 3, 2, h_ht=2, a_ht=1, country="France", venue="Stade de France", city="Paris"),
    _m(2017, "Friendly", "2017-10-08", "England", "Croatia", 0, 0, country="England", venue="Wembley Stadium", city="London"),
    _m(2018, "Friendly", "2018-03-23", "France", "Serbia", 2, 0, h_ht=1, a_ht=0, country="France", venue="Stade de France", city="Paris"),
    _m(2019, "Friendly", "2019-11-15", "Argentina", "Uruguay", 2, 2, h_ht=1, a_ht=0, country="Saudi Arabia", venue="King Saud University Stadium", city="Riyadh"),
    _m(2020, "Friendly", "2020-10-14", "England", "Croatia", 2, 1, h_ht=1, a_ht=0, country="England", venue="Wembley Stadium", city="London"),
    _m(2022, "Friendly", "2022-06-01", "Argentina", "Italy", 3, 0, h_ht=2, a_ht=0, country="England", venue="Wembley Stadium", city="London"),
    _m(2022, "Friendly", "2022-09-23", "Brazil", "Ghana", 3, 0, h_ht=1, a_ht=0, country="France", venue="Stade Océane", city="Le Havre"),
    _m(2023, "Friendly", "2023-03-23", "France", "Croatia", 0, 0, country="France", venue="Stade de France", city="Paris"),
    _m(2023, "Friendly", "2023-03-26", "England", "Serbia", 2, 0, h_ht=1, a_ht=0, country="England", venue="Wembley Stadium", city="London"),
    _m(2023, "Friendly", "2023-06-08", "Brazil", "Argentina", 0, 1, country="Chile", venue="Estadio Nacional", city="Santiago"),
]

ALL_WORLD_CUP_MATCHES: list[WorldCupMatch] = WORLD_CUP_2018 + WORLD_CUP_2022

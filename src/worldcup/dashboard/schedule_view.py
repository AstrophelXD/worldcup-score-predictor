"""Streamlit helpers for rendering the 2026 World Cup schedule."""

from __future__ import annotations

from datetime import date
from typing import Any, Callable

import pandas as pd
import streamlit as st

from worldcup.dashboard.flags import (
    is_known_team,
    team_label_html,
    team_label_plain,
)
from worldcup.dashboard.prediction_view import format_pct, render_compact_prediction
from worldcup.data_ingestion.sources.world_cup_2026_catalog import wc2026_match_id
from worldcup.dashboard.world_cup_2026_schedule import (
    STAGE_ORDER,
    TOURNAMENT_END,
    TOURNAMENT_START,
    WORLD_CUP_2026_GROUPS,
    WORLD_CUP_2026_SCHEDULE,
    ScheduleMatch,
)

_API_NAME_ALIASES: dict[str, str] = {
    "USA": "United States",
    "South Korea": "Korea Republic",
    "Turkey": "Türkiye",
    "Curacao": "Curaçao",
    "Côte d'Ivoire": "Ivory Coast",
    "Cabo Verde": "Cape Verde",
    "DR Congo": "Congo DR",
}


def normalize_team_name(name: str) -> str:
    cleaned = name.strip()
    return _API_NAME_ALIASES.get(cleaned, cleaned)


def schedule_status(match_date: str, *, today: date | None = None) -> str:
    ref = today or date.today()
    day = date.fromisoformat(match_date)
    if day < ref:
        return "已结束"
    if day == ref:
        return "今日"
    return "未开始"


def _display_team_html(name: str) -> str:
    if is_known_team(name):
        return team_label_html(name)
    return f"⚪ {name}"


def _display_matchup_html(match: ScheduleMatch) -> str:
    home = _display_team_html(match.home_team)
    away = _display_team_html(match.away_team)
    return f"{home} vs {away}"


def _display_matchup_plain(match: ScheduleMatch) -> str:
    home = team_label_plain(match.home_team)
    away = team_label_plain(match.away_team)
    return f"{home} vs {away}"


def schedule_to_dataframe(
    rows: list[ScheduleMatch],
    *,
    today: date | None = None,
) -> pd.DataFrame:
    records: list[dict[str, Any]] = []
    for match in rows:
        records.append(
            {
                "#": match.match_number,
                "日期": match.match_date,
                "开球 (ET)": match.kickoff_et,
                "对阵": _display_matchup_plain(match),
                "阶段": match.stage_name,
                "小组": match.group or "—",
                "球场": match.venue,
                "城市": match.city,
                "状态": schedule_status(match.match_date, today=today),
            }
        )
    return pd.DataFrame(records)


def build_api_match_id_index(api_matches: list[dict[str, Any]]) -> dict[str, dict]:
    return {item["match_id"]: item for item in api_matches}


def build_api_match_index(api_matches: list[dict[str, Any]]) -> dict[tuple[str, str, str], str]:
    index: dict[tuple[str, str, str], str] = {}
    for item in api_matches:
        kickoff = str(item.get("kickoff_ts", ""))[:10]
        home = normalize_team_name(str(item.get("home_team_name") or item.get("home_team_id", "")))
        away = normalize_team_name(str(item.get("away_team_name") or item.get("away_team_id", "")))
        if kickoff and home and away:
            index[(kickoff, home, away)] = item["match_id"]
            index[(kickoff, away, home)] = item["match_id"]
    return index


def find_api_match_id(
    match: ScheduleMatch,
    api_index: dict[tuple[str, str, str], str],
    api_by_id: dict[str, dict] | None = None,
) -> str | None:
    stable_id = wc2026_match_id(match)
    if api_by_id and stable_id in api_by_id:
        return stable_id
    if not is_known_team(match.home_team) or not is_known_team(match.away_team):
        return None
    home = normalize_team_name(match.home_team)
    away = normalize_team_name(match.away_team)
    return api_index.get((match.match_date, home, away))


def schedule_meta(match: ScheduleMatch) -> dict[str, str]:
    return {
        "match_number": str(match.match_number),
        "venue": match.venue,
        "city": match.city,
        "group": match.group,
        "kickoff_et": match.kickoff_et,
        "stage_name": match.stage_name,
    }


def _current_tournament_matches(
    *,
    today: date,
    limit: int = 12,
) -> list[ScheduleMatch]:
    ref = today.isoformat()
    active = [
        m
        for m in WORLD_CUP_2026_SCHEDULE
        if is_known_team(m.home_team)
        and is_known_team(m.away_team)
        and m.match_date >= ref
    ]
    active.sort(key=lambda m: (m.match_date, m.kickoff_et, m.match_number))
    today_matches = [m for m in active if m.match_date == ref]
    if today_matches:
        return today_matches[:limit]
    return active[:limit]


def _lookup_prediction(
    match_id: str | None,
    *,
    predictions_by_id: dict[str, dict] | None,
    predict_fn: Callable[[str], dict | None] | None,
) -> dict | None:
    if not match_id:
        return None
    if predictions_by_id and match_id in predictions_by_id:
        return predictions_by_id[match_id]
    if predict_fn:
        try:
            return predict_fn(match_id)
        except Exception:  # noqa: BLE001
            return None
    return None


def render_current_predictions(
    api_matches: list[dict[str, Any]] | None,
    *,
    predict_fn: Callable[[str], dict | None] | None = None,
    predictions_by_id: dict[str, dict] | None = None,
    today: date | None = None,
) -> str | None:
    ref = today or date.today()
    if ref.isoformat() < TOURNAMENT_START or ref.isoformat() > TOURNAMENT_END:
        return None

    st.subheader("当前世界杯 · 场次预测")
    current = _current_tournament_matches(today=ref)
    if not current:
        st.info("当前筛选条件下没有可预测的未来场次。")
        return None

    api_index = build_api_match_index(api_matches or [])
    api_by_id = build_api_match_id_index(api_matches or [])
    selected: str | None = None

    for match in current:
        api_id = find_api_match_id(match, api_index, api_by_id)
        status = schedule_status(match.match_date, today=ref)
        with st.container(border=True):
            top = st.columns([4, 2, 2])
            with top[0]:
                st.markdown(_display_matchup_html(match), unsafe_allow_html=True)
                st.caption(
                    f"#{match.match_number} · {match.match_date} {match.kickoff_et} ET · "
                    f"{match.stage_name}"
                    + (f" · Group {match.group}" if match.group else "")
                    + f" · {match.venue}, {match.city}"
                )
            with top[1]:
                st.write(status)
            with top[2]:
                pred = _lookup_prediction(
                    api_id,
                    predictions_by_id=predictions_by_id,
                    predict_fn=predict_fn,
                )
                if pred:
                    render_compact_prediction(
                        pred,
                        home_team=match.home_team,
                        away_team=match.away_team,
                    )
                elif api_id:
                    st.caption("模型预测加载中…")
                else:
                    st.caption("尚未入库")

            if api_id and st.button("查看完整预测", key=f"full_pred_{match.match_number}"):
                selected = api_id

    return selected


def render_groups_overview() -> None:
    st.subheader("小组一览")
    cols = st.columns(4)
    for idx, (group, teams) in enumerate(sorted(WORLD_CUP_2026_GROUPS.items())):
        with cols[idx % 4]:
            lines = [f"**Group {group}**"]
            for team in teams:
                lines.append(team_label_html(team))
            st.markdown("<br>".join(lines), unsafe_allow_html=True)


def render_schedule_tab(
    api_matches: list[dict[str, Any]] | None = None,
    *,
    predict_fn: Callable[[str], dict | None] | None = None,
    predictions_by_id: dict[str, dict] | None = None,
    today: date | None = None,
) -> str | None:
    """Render the 2026 schedule page. Returns selected API match_id if any."""
    ref = today or date.today()
    st.subheader("2026 世界杯赛程")
    st.caption(
        f"美加墨 · {TOURNAMENT_START} ～ {TOURNAMENT_END} · "
        f"48 队 · {len(WORLD_CUP_2026_SCHEDULE)} 场 · 今日 {ref.isoformat()}"
    )

    c1, c2, c3, c4 = st.columns(4)
    finished = sum(1 for m in WORLD_CUP_2026_SCHEDULE if m.match_date < ref.isoformat())
    upcoming = len(WORLD_CUP_2026_SCHEDULE) - finished
    c1.metric("总场次", len(WORLD_CUP_2026_SCHEDULE))
    c2.metric("已结束", finished)
    c3.metric("待踢", upcoming)
    c4.metric("小组数", len(WORLD_CUP_2026_GROUPS))

    picked_current = render_current_predictions(
        api_matches,
        predict_fn=predict_fn,
        predictions_by_id=predictions_by_id,
        today=ref,
    )

    render_groups_overview()

    st.divider()
    st.subheader("比赛列表")

    stage_options = ["全部阶段", *STAGE_ORDER]
    group_options = ["全部小组", *sorted(WORLD_CUP_2026_GROUPS.keys())]
    status_options = ["全部状态", "今日", "未开始", "已结束"]

    f1, f2, f3, f4 = st.columns(4)
    stage_filter = f1.selectbox("阶段", stage_options, key="wc26_stage")
    group_filter = f2.selectbox("小组", group_options, key="wc26_group")
    status_filter = f3.selectbox("状态", status_options, key="wc26_status")
    team_query = f4.text_input("球队筛选", placeholder="例如 Brazil、墨西哥 host…")

    filtered = list(WORLD_CUP_2026_SCHEDULE)
    if stage_filter != "全部阶段":
        filtered = [m for m in filtered if m.stage_name == stage_filter]
    if group_filter != "全部小组":
        filtered = [m for m in filtered if m.group == group_filter]
    if status_filter != "全部状态":
        filtered = [
            m for m in filtered if schedule_status(m.match_date, today=ref) == status_filter
        ]
    if team_query.strip():
        query = team_query.strip().lower()
        filtered = [
            m
            for m in filtered
            if query in m.home_team.lower()
            or query in m.away_team.lower()
            or query in m.city.lower()
        ]

    api_index = build_api_match_index(api_matches or [])
    api_by_id = build_api_match_id_index(api_matches or [])
    predictable = sum(
        1 for m in filtered if find_api_match_id(m, api_index, api_by_id)
    )

    st.caption(f"筛选结果 {len(filtered)} 场 · 可跳转预测 {predictable} 场")

    if not filtered:
        st.info("没有符合筛选条件的比赛。")
        return None

    df = schedule_to_dataframe(filtered, today=ref)
    st.dataframe(
        df,
        use_container_width=True,
        hide_index=True,
        column_config={
            "对阵": st.column_config.TextColumn("对阵", width="large"),
            "状态": st.column_config.TextColumn("状态", width="small"),
        },
    )

    st.markdown("#### 按日浏览")
    by_date: dict[str, list[ScheduleMatch]] = {}
    for match in filtered:
        by_date.setdefault(match.match_date, []).append(match)

    selected_match_id: str | None = picked_current
    for match_date in sorted(by_date):
        day_matches = by_date[match_date]
        status = schedule_status(match_date, today=ref)
        header = f"{match_date} · {status} · {len(day_matches)} 场"
        with st.expander(header, expanded=(status == "今日")):
            for match in day_matches:
                cols = st.columns([1, 5, 2, 2])
                cols[0].write(f"#{match.match_number}")
                cols[1].markdown(_display_matchup_html(match), unsafe_allow_html=True)
                meta = match.stage_name
                if match.group:
                    meta += f" · Group {match.group}"
                cols[2].caption(f"{match.kickoff_et} ET · {meta}")
                cols[3].caption(f"{match.venue}, {match.city}")

                api_id = find_api_match_id(match, api_index, api_by_id)
                pred = _lookup_prediction(
                    api_id,
                    predictions_by_id=predictions_by_id,
                    predict_fn=predict_fn,
                )
                if pred:
                    probs = pred["result_probs"]
                    top = pred["top3_scorelines"][0]
                    st.caption(
                        f"模型预测：主胜 {format_pct(probs['home_win'])} · "
                        f"平 {format_pct(probs['draw'])} · "
                        f"客胜 {format_pct(probs['away_win'])} · "
                        f"最可能 {top['home_goals']}-{top['away_goals']}"
                    )
                if api_id and st.button("查看完整预测", key=f"predict_{match.match_number}"):
                    selected_match_id = api_id

    if predictable == 0 and api_matches is not None:
        st.info(
            "2026 赛程尚未入库或 API 未重启。请运行："
            "`export_sample_data` → `ingest` → `build_features` → `export_model_odds` → 重启 `serve`。"
        )

    return selected_match_id

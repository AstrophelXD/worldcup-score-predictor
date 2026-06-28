"""Streamlit dashboard for WorldCup predictor (API-backed)."""

from __future__ import annotations

import os

import httpx
import pandas as pd
import streamlit as st

from worldcup.dashboard.flags import team_label_plain
from worldcup.dashboard.prediction_view import render_full_prediction_panel, render_heatmap
from worldcup.dashboard.schedule_view import render_schedule_tab, schedule_meta
from worldcup.dashboard.world_cup_2026_schedule import WORLD_CUP_2026_SCHEDULE
from worldcup.data_ingestion.sources.world_cup_2026_catalog import wc2026_match_id

DEFAULT_API_URL = os.getenv("WORLDCUP_API_URL", "http://127.0.0.1:8000")

SCHEDULE_BY_ID = {wc2026_match_id(m): m for m in WORLD_CUP_2026_SCHEDULE}


def fetch_json(client: httpx.Client, path: str) -> dict:
    response = client.get(path, timeout=30.0)
    response.raise_for_status()
    return response.json()


def fetch_json_or_none(client: httpx.Client, path: str) -> dict | None:
    try:
        return fetch_json(client, path)
    except httpx.HTTPStatusError as exc:
        if exc.response.status_code == 404:
            return None
        raise


def post_json(client: httpx.Client, path: str, payload: dict) -> dict:
    response = client.post(path, json=payload, timeout=60.0)
    response.raise_for_status()
    return response.json()


def load_openapi_paths(client: httpx.Client) -> set[str]:
    try:
        payload = fetch_json(client, "/openapi.json")
    except httpx.HTTPError:
        return set()
    return set(payload.get("paths", {}).keys())


def match_from_list(matches: list[dict], match_id: str) -> dict:
    for item in matches:
        if item["match_id"] == match_id:
            return item
    raise KeyError(match_id)


def load_match_detail(
    client: httpx.Client,
    match_id: str,
    matches: list[dict],
    openapi_paths: set[str],
) -> dict:
    if "/matches/{match_id}" in openapi_paths:
        detail = fetch_json_or_none(client, f"/matches/{match_id}")
        if detail is not None:
            return detail
    return match_from_list(matches, match_id)


def match_label(item: dict) -> str:
    home = item.get("home_team_name") or item["home_team_id"]
    away = item.get("away_team_name") or item["away_team_id"]
    stage = item.get("stage") or "n/a"
    home = team_label_plain(str(home))
    away = team_label_plain(str(away))
    prefix = "⭐ " if str(item.get("match_id", "")).startswith("wc2026_") else ""
    return f"{prefix}{item['kickoff_ts']} · {home} vs {away} ({stage})"


def sort_matches_for_picker(matches: list[dict]) -> list[dict]:
    wc2026 = [m for m in matches if str(m.get("match_id", "")).startswith("wc2026_")]
    wc_hist = [m for m in matches if m.get("is_world_cup") and m not in wc2026]
    others = [m for m in matches if m not in wc2026 and m not in wc_hist]
    wc2026.sort(key=lambda m: m.get("kickoff_ts", ""))
    wc_hist.sort(key=lambda m: m.get("kickoff_ts", ""), reverse=True)
    others.sort(key=lambda m: m.get("kickoff_ts", ""), reverse=True)
    return wc2026 + wc_hist + others


def load_wc2026_predictions(client: httpx.Client, matches: list[dict]) -> dict[str, dict]:
    wc_ids = [m["match_id"] for m in matches if str(m["match_id"]).startswith("wc2026_")]
    if not wc_ids:
        return {}
    try:
        payload = post_json(client, "/predict/batch", {"match_ids": wc_ids})
    except httpx.HTTPStatusError:
        return {}
    return {item["match_id"]: item for item in payload.get("items", [])}


def main() -> None:
    st.set_page_config(
        page_title="WorldCup Predictor",
        layout="wide",
        page_icon="⚽",
    )
    st.title("⚽ WorldCup 2026 比分预测 Dashboard")

    api_url = st.sidebar.text_input("API URL", value=DEFAULT_API_URL)
    st.sidebar.caption("Dashboard 通过 API 获取预测；2026 赛程内置展示。")
    world_cup_only = st.sidebar.checkbox("仅世界杯比赛", value=True)

    try:
        with httpx.Client(base_url=api_url) as client:
            health = fetch_json(client, "/health")
            openapi_paths = load_openapi_paths(client)
            query = f"/matches?limit=500&world_cup_only={'true' if world_cup_only else 'false'}"
            matches_payload = fetch_json(client, query)
            freshness = fetch_json(client, "/data/freshness")
    except httpx.HTTPError as exc:
        st.error(f"无法连接 API：{exc}")
        st.info(
            "请先运行 `python -m scripts.serve` 或 `scripts/start_local.ps1`，"
            "并确保已完成 export / ingest / build_features / train。"
        )
        st.divider()
        st.warning("API 离线时仍可浏览 2026 赛程。")
        render_schedule_tab(None)
        return

    st.sidebar.success(f"API {health['status']} · v{health['version']}")
    if "/matches/{match_id}" not in openapi_paths:
        st.sidebar.warning(
            "API 进程较旧，请重启 `python -m scripts.serve` 以加载完整接口。"
        )

    matches = sort_matches_for_picker(matches_payload["items"])
    wc2026_count = sum(1 for m in matches if str(m.get("match_id", "")).startswith("wc2026_"))
    st.sidebar.metric("2026 可预测场次", wc2026_count)
    model_info = freshness.get("model") or {}
    if model_info.get("model_name"):
        st.sidebar.caption(
            f"当前模型: {model_info['model_name']} ({model_info.get('model_type', 'n/a')})"
        )

    with httpx.Client(base_url=api_url) as client:
        wc2026_predictions = load_wc2026_predictions(client, matches)

    if wc2026_predictions:
        st.sidebar.success(f"已加载 {len(wc2026_predictions)} 场 2026 模型预测")
    else:
        st.sidebar.info("2026 模型预测未加载（需重启 API 并运行 export_model_odds）")

    if "selected_match_id" not in st.session_state:
        st.session_state.selected_match_id = matches[0]["match_id"] if matches else None

    def predict_fn(match_id: str) -> dict | None:
        if match_id in wc2026_predictions:
            return wc2026_predictions[match_id]
        with httpx.Client(base_url=api_url) as client:
            return post_json(client, "/predict", {"match_id": match_id})

    tabs = st.tabs(
        [
            "2026 赛程",
            "场次预测",
            "比分矩阵",
            "特征详情",
            "回测",
            "数据 Freshness",
        ]
    )

    with tabs[0]:
        picked = render_schedule_tab(
            matches,
            predict_fn=predict_fn,
            predictions_by_id=wc2026_predictions,
        )
        if picked:
            st.session_state.selected_match_id = picked
            st.success("已选择比赛，请切换到「场次预测」查看完整结果。")

    if not matches:
        with tabs[1]:
            st.warning("API 未返回比赛列表，暂无法展示预测。")
        return

    labels = {item["match_id"]: match_label(item) for item in matches}
    match_ids = list(labels.keys())
    if st.session_state.selected_match_id not in match_ids:
        st.session_state.selected_match_id = match_ids[0]

    selected_id = st.selectbox(
        "选择比赛",
        options=match_ids,
        index=match_ids.index(st.session_state.selected_match_id),
        format_func=lambda x: labels[x],
        key="match_picker",
    )
    st.session_state.selected_match_id = selected_id

    schedule = SCHEDULE_BY_ID.get(selected_id)
    meta = schedule_meta(schedule) if schedule else None

    with httpx.Client(base_url=api_url) as client:
        match_detail = load_match_detail(client, selected_id, matches, openapi_paths)
        features = (
            fetch_json_or_none(client, f"/features/{selected_id}")
            if "/features/{match_id}" in openapi_paths
            else None
        )
        prediction = post_json(client, "/predict", {"match_id": selected_id})
        matrix_payload = fetch_json(client, f"/score-matrix/{selected_id}")

    with tabs[1]:
        render_full_prediction_panel(
            match_detail,
            prediction,
            features,
            matrix_payload,
            schedule_meta=meta,
        )

    with tabs[2]:
        if matrix_payload.get("overflow_prob", 0) > 0:
            st.caption(f"overflow_prob = {matrix_payload['overflow_prob']:.4f}")
        render_heatmap(matrix_payload["matrix"], matrix_payload["grid_max_goal"])

    with tabs[3]:
        if features is None:
            st.warning("当前 API 未提供 `/features/{match_id}`，请重启 serve。")
        else:
            st.json(features)

    with tabs[4]:
        if "/backtest/runs" not in openapi_paths:
            st.warning("当前 API 未提供 `/backtest/runs`，请重启 serve。")
        else:
            with httpx.Client(base_url=api_url) as client:
                backtests = fetch_json(client, "/backtest/runs")
            rows = backtests.get("items", [])
            if not rows:
                st.warning("暂无回测报告。请运行 `python -m scripts.backtest` 生成报告。")
            else:
                st.dataframe(pd.DataFrame(rows), use_container_width=True)

    with tabs[5]:
        st.subheader("数据 Freshness")
        st.dataframe(freshness["items"], use_container_width=True)
        if freshness.get("model"):
            st.subheader("模型信息")
            st.json(freshness["model"])
        st.caption(f"API: {api_url}")


if __name__ == "__main__":
    main()

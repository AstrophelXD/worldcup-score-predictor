"""Streamlit dashboard for WorldCup predictor (API-backed)."""

from __future__ import annotations

import os

import httpx
import matplotlib.pyplot as plt
import numpy as np
import streamlit as st

DEFAULT_API_URL = os.getenv("WORLDCUP_API_URL", "http://127.0.0.1:8000")


def fetch_json(client: httpx.Client, path: str) -> dict:
    response = client.get(path, timeout=30.0)
    response.raise_for_status()
    return response.json()


def post_json(client: httpx.Client, path: str, payload: dict) -> dict:
    response = client.post(path, json=payload, timeout=30.0)
    response.raise_for_status()
    return response.json()


def render_heatmap(matrix: list[list[float]], grid_max_goal: int) -> None:
    data = np.array(matrix)
    fig, ax = plt.subplots(figsize=(8, 6))
    im = ax.imshow(data, cmap="YlOrRd", origin="lower")
    ax.set_xlabel("Away goals")
    ax.set_ylabel("Home goals")
    ax.set_xticks(range(grid_max_goal + 1))
    ax.set_yticks(range(grid_max_goal + 1))
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    st.pyplot(fig)
    plt.close(fig)


def format_pct(value: float) -> str:
    return f"{100.0 * value:.1f}%"


def main() -> None:
    st.set_page_config(page_title="WorldCup Predictor", layout="wide")
    st.title("WorldCup 比分预测 Dashboard")

    api_url = st.sidebar.text_input("API URL", value=DEFAULT_API_URL)
    st.sidebar.caption("Dashboard 仅通过 API 访问预测结果。")

    try:
        with httpx.Client(base_url=api_url) as client:
            health = fetch_json(client, "/health")
            matches_payload = fetch_json(client, "/matches?limit=200")
    except httpx.HTTPError as exc:
        st.error(f"无法连接 API：{exc}")
        st.info(
            "请先运行 `python -m scripts.serve`，"
            "并确保已完成 ingest / build_features / train。"
        )
        return

    st.sidebar.success(f"API {health['status']} · v{health['version']}")
    matches = matches_payload["items"]
    if not matches:
        st.warning("API 未返回比赛列表。")
        return

    labels = {
        item["match_id"]: (
            f"{item['kickoff_ts']} · {item['home_team_id']} vs {item['away_team_id']} "
            f"({item.get('stage') or 'n/a'})"
        )
        for item in matches
    }
    selected_id = st.selectbox(
        "选择比赛",
        options=list(labels.keys()),
        format_func=lambda x: labels[x],
    )

    with httpx.Client(base_url=api_url) as client:
        prediction = post_json(client, "/predict", {"match_id": selected_id})
        matrix_payload = fetch_json(client, f"/score-matrix/{selected_id}")

    tab_summary, tab_matrix, tab_meta = st.tabs(
        ["Prediction Summary", "Score Matrix", "Data Freshness"]
    )

    with tab_summary:
        col1, col2, col3, col4 = st.columns(4)
        col1.metric("主胜", format_pct(prediction["result_probs"]["home_win"]))
        col2.metric("平局", format_pct(prediction["result_probs"]["draw"]))
        col3.metric("客胜", format_pct(prediction["result_probs"]["away_win"]))
        col4.metric("置信度", format_pct(prediction["uncertainty"]["confidence"]))

        st.subheader("Top 3 比分")
        for idx, scoreline in enumerate(prediction["top3_scorelines"], start=1):
            st.write(
                f"{idx}. {scoreline['home_goals']}-{scoreline['away_goals']} "
                f"({format_pct(scoreline['prob'])})"
            )

        st.subheader("大小球 / BTTS / 期望进球")
        c1, c2, c3 = st.columns(3)
        c1.metric("Over 2.5", format_pct(prediction["ou25_probs"]["over_2_5"]))
        c2.metric("BTTS Yes", format_pct(prediction["btts_probs"]["yes"]))
        c3.metric(
            "期望进球",
            f"{prediction['expected_goals']['home']:.2f} - "
            f"{prediction['expected_goals']['away']:.2f}",
        )

        st.caption(
            f"λ_home={prediction['lambda_home']:.3f}, "
            f"λ_away={prediction['lambda_away']:.3f}, "
            f"overflow={prediction['overflow_prob']:.4f}"
        )

    with tab_matrix:
        grid_max = matrix_payload["grid_max_goal"]
        if matrix_payload["overflow_prob"] > 0:
            st.info(f"尾部概率 overflow_prob = {matrix_payload['overflow_prob']:.4f}")
        render_heatmap(matrix_payload["matrix"], grid_max)

    with tab_meta:
        with httpx.Client(base_url=api_url) as client:
            freshness = fetch_json(client, "/data/freshness")
        st.subheader("数据 Freshness")
        st.dataframe(freshness["items"], use_container_width=True)
        st.caption(f"API: {api_url}")


if __name__ == "__main__":
    main()

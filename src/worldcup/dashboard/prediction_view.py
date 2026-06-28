"""Rich Streamlit components for single-match predictions."""

from __future__ import annotations

from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import streamlit as st

from worldcup.dashboard.flags import matchup_label_html, team_label_html


def format_pct(value: float) -> str:
    return f"{100.0 * value:.1f}%"


def _prob_bar(label: str, value: float, *, color: str = "#2563eb") -> None:
    pct = max(0.0, min(1.0, value))
    st.markdown(
        f"""
        <div style="margin-bottom:0.35rem;">
          <div style="display:flex;justify-content:space-between;font-size:0.9rem;">
            <span>{label}</span><span><strong>{format_pct(pct)}</strong></span>
          </div>
          <div style="background:#eef2f7;border-radius:999px;height:10px;overflow:hidden;">
            <div style="width:{pct * 100:.1f}%;background:{color};height:10px;"></div>
          </div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_match_header(
    match_detail: dict,
    *,
    schedule_meta: dict[str, Any] | None = None,
) -> None:
    home = match_detail.get("home_team_name") or match_detail["home_team_id"]
    away = match_detail.get("away_team_name") or match_detail["away_team_id"]
    stage = match_detail.get("stage") or match_detail.get("stage_name") or "n/a"
    kickoff = match_detail.get("kickoff_ts", "")

    left, center, right = st.columns([2, 3, 2])
    with center:
        st.markdown(
            matchup_label_html(str(home), str(away), width=36),
            unsafe_allow_html=True,
        )
        meta_bits = [str(kickoff), stage]
        if schedule_meta:
            if schedule_meta.get("venue"):
                meta_bits.append(schedule_meta["venue"])
            if schedule_meta.get("city"):
                meta_bits.append(schedule_meta["city"])
            if schedule_meta.get("group"):
                meta_bits.append(f"Group {schedule_meta['group']}")
        st.caption(" · ".join(meta_bits))

    if match_detail.get("home_score_ft") is not None:
        st.success(
            f"实际 90 分钟比分：{match_detail['home_score_ft']}-{match_detail['away_score_ft']}"
        )


def render_result_probabilities(prediction: dict) -> None:
    probs = prediction["result_probs"]
    st.subheader("胜平负")
    cols = st.columns(3)
    cols[0].metric("主胜", format_pct(probs["home_win"]))
    cols[1].metric("平局", format_pct(probs["draw"]))
    cols[2].metric("客胜", format_pct(probs["away_win"]))
    _prob_bar("主胜", probs["home_win"], color="#16a34a")
    _prob_bar("平局", probs["draw"], color="#64748b")
    _prob_bar("客胜", probs["away_win"], color="#dc2626")


def render_market_probabilities(prediction: dict) -> None:
    st.subheader("市场衍生概率")
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Over 2.5", format_pct(prediction["ou25_probs"]["over_2_5"]))
    c2.metric("Under 2.5", format_pct(prediction["ou25_probs"]["under_2_5"]))
    c3.metric("BTTS Yes", format_pct(prediction["btts_probs"]["yes"]))
    c4.metric("BTTS No", format_pct(prediction["btts_probs"]["no"]))
    xg = prediction["expected_goals"]
    st.metric(
        "期望进球 (主 - 客)",
        f"{xg['home']:.2f} - {xg['away']:.2f}",
        delta=f"总计 {xg.get('total', xg['home'] + xg['away']):.2f}",
    )


def render_top_scorelines(prediction: dict, *, top_n: int = 8) -> None:
    st.subheader(f"Top {top_n} 比分")
    rows = []
    for item in prediction.get("top3_scorelines", [])[:top_n]:
        rows.append(
            {
                "比分": f"{item['home_goals']}-{item['away_goals']}",
                "概率": item["prob"],
            }
        )
    if not rows:
        st.info("暂无比分分布数据。")
        return
    df = pd.DataFrame(rows)
    df["概率条"] = df["概率"]
    st.dataframe(
        df,
        use_container_width=True,
        hide_index=True,
        column_config={
            "概率": st.column_config.NumberColumn("概率", format="%.1%%"),
            "概率条": st.column_config.ProgressColumn(
                "分布",
                format="%.1%%",
                min_value=0,
                max_value=max(df["概率"].max(), 0.01),
            ),
        },
    )


def render_model_diagnostics(prediction: dict) -> None:
    st.subheader("模型诊断")
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("置信度", format_pct(prediction["uncertainty"]["confidence"]))
    c2.metric("熵", f"{prediction['uncertainty']['entropy']:.3f}")
    c3.metric("λ_home", f"{prediction['lambda_home']:.3f}")
    c4.metric("λ_away", f"{prediction['lambda_away']:.3f}")
    st.caption(
        f"lambda_scale={prediction.get('lambda_scale', 1.0):.3f}, "
        f"overflow={prediction['overflow_prob']:.4f}, "
        f"ood={prediction['uncertainty'].get('ood_score', 0.0):.3f}"
    )


def render_feature_summary(features: dict | None) -> None:
    st.subheader("特征摘要")
    if not features:
        st.info("特征接口不可用或未返回数据。")
        return

    strength = features.get("team_strength", {})
    form = features.get("recent_form", {})
    players = features.get("player_summary", {})
    odds = features.get("market_odds", {})
    events = features.get("match_events", {})

    c1, c2 = st.columns(2)
    with c1:
        st.markdown("**实力**")
        st.write(
            f"Elo: {strength.get('home_elo', '—')} vs {strength.get('away_elo', '—')}\n\n"
            f"FIFA 排名: {strength.get('home_fifa_rank', '—')} vs "
            f"{strength.get('away_fifa_rank', '—')}"
        )
        st.markdown("**近 5 场进球**")
        st.write(
            f"进: {form.get('home_goals_for_last5', '—')} vs {form.get('away_goals_for_last5', '—')}\n\n"
            f"失: {form.get('home_goals_against_last5', '—')} vs "
            f"{form.get('away_goals_against_last5', '—')}"
        )
    with c2:
        st.markdown("**阵容 / 伤停**")
        st.write(
            f"首发人数: {players.get('home_starter_count', '—')} vs "
            f"{players.get('away_starter_count', '—')}\n\n"
            f"伤停(out): {players.get('home_injured_out_count', '—')} vs "
            f"{players.get('away_injured_out_count', '—')}\n\n"
            f"预计首发占比: {players.get('home_lineup_projected_share', '—')} vs "
            f"{players.get('away_lineup_projected_share', '—')}"
        )
        if odds.get("available"):
            st.markdown("**赔率隐含概率**")
            st.write(
                f"1X2: {odds.get('home_implied', '—')} / {odds.get('draw_implied', '—')} / "
                f"{odds.get('away_implied', '—')}"
            )
        if events.get("available"):
            st.markdown("**事件 form (xG)**")
            st.write(
                f"xG 进: {events.get('home_xg_for_last5', '—')} vs "
                f"{events.get('away_xg_for_last5', '—')}"
            )


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


def render_compact_prediction(
    prediction: dict,
    *,
    home_team: str,
    away_team: str,
) -> None:
    """One-line mini card for schedule lists."""
    probs = prediction["result_probs"]
    top = prediction["top3_scorelines"][0]
    st.markdown(
        f"{team_label_html(home_team, width=18)} vs {team_label_html(away_team, width=18)}",
        unsafe_allow_html=True,
    )
    c1, c2, c3, c4 = st.columns(4)
    c1.caption(f"主胜 {format_pct(probs['home_win'])}")
    c2.caption(f"平 {format_pct(probs['draw'])}")
    c3.caption(f"客胜 {format_pct(probs['away_win'])}")
    c4.caption(f"最可能 {top['home_goals']}-{top['away_goals']} ({format_pct(top['prob'])})")


def render_full_prediction_panel(
    match_detail: dict,
    prediction: dict,
    features: dict | None,
    matrix_payload: dict | None,
    *,
    schedule_meta: dict[str, Any] | None = None,
) -> None:
    render_match_header(match_detail, schedule_meta=schedule_meta)
    st.divider()
    left, right = st.columns([1, 1])
    with left:
        render_result_probabilities(prediction)
        render_market_probabilities(prediction)
    with right:
        render_top_scorelines(prediction, top_n=8)
        render_model_diagnostics(prediction)
    st.divider()
    render_feature_summary(features)
    if matrix_payload:
        st.subheader("比分矩阵")
        if matrix_payload.get("overflow_prob", 0) > 0:
            st.caption(f"overflow_prob = {matrix_payload['overflow_prob']:.4f}")
        render_heatmap(matrix_payload["matrix"], matrix_payload["grid_max_goal"])

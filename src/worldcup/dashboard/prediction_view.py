"""Rich Streamlit components for single-match predictions."""

from __future__ import annotations

import html
from datetime import datetime
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import streamlit as st

from worldcup.dashboard.flags import resolve_team_name, team_flag_html

_DASHBOARD_CSS = """
<style>
.wc-meta {
  text-align: center;
  color: #64748b;
  font-size: 0.95rem;
  line-height: 1.7;
  margin: 0.25rem 0 1rem;
}
.wc-model-badge {
  text-align: center;
  color: #94a3b8;
  font-size: 0.85rem;
  margin-bottom: 1rem;
}
.wc-outcome-card {
  background: #f8fafc;
  border: 1px solid #e2e8f0;
  border-radius: 14px;
  padding: 1.25rem 1.5rem 1.1rem;
  margin-bottom: 1.5rem;
}
.wc-teams-row {
  display: flex;
  justify-content: space-between;
  align-items: center;
  gap: 1rem;
  margin-bottom: 1rem;
}
.wc-team {
  display: flex;
  align-items: center;
  gap: 0.65rem;
  font-size: 1.25rem;
  font-weight: 700;
  color: #0f172a;
}
.wc-team.away { flex-direction: row-reverse; text-align: right; }
.wc-bar {
  display: flex;
  height: 22px;
  border-radius: 999px;
  overflow: hidden;
  box-shadow: inset 0 0 0 1px rgba(15, 23, 42, 0.06);
}
.wc-bar-seg { min-width: 3px; }
.wc-labels-row {
  display: flex;
  margin-top: 0.65rem;
}
.wc-label {
  text-align: center;
  font-size: 0.9rem;
  color: #64748b;
  line-height: 1.45;
  padding: 0 0.15rem;
}
.wc-label strong {
  display: block;
  font-size: 1.15rem;
  color: #0f172a;
  margin-top: 0.1rem;
}
.wc-outcome-card.wc-schedule {
  margin-bottom: 0;
  padding: 1rem 1.15rem 0.95rem;
}
.wc-outcome-card.wc-schedule .wc-teams-row {
  margin-bottom: 0.7rem;
  gap: 2rem;
}
.wc-outcome-card.wc-schedule .wc-team {
  flex: 1 1 0;
  min-width: 0;
  max-width: 50%;
  font-size: 1.05rem;
}
.wc-outcome-card.wc-schedule .wc-team span {
  overflow: hidden;
  text-overflow: ellipsis;
  white-space: nowrap;
}
.wc-outcome-card.wc-schedule .wc-team.home { justify-content: flex-start; }
.wc-outcome-card.wc-schedule .wc-team.away { justify-content: flex-end; }
.wc-outcome-card.wc-schedule .wc-bar { height: 26px; }
.wc-outcome-card.wc-schedule .wc-label strong { font-size: 1.05rem; }
.wc-outcome-card.wc-schedule.wc-flags-only .wc-team span { display: none; }
.wc-outcome-card.wc-schedule.wc-flags-only .wc-teams-row { margin-bottom: 0.55rem; }
.wc-top-scores {
  margin: 0.65rem 0 0.85rem;
  display: flex;
  flex-wrap: wrap;
  align-items: center;
  gap: 0.45rem;
}
.wc-top-scores-label {
  font-weight: 600;
  color: #334155;
  margin-right: 0.25rem;
  font-size: 0.95rem;
}
.wc-score-chip {
  background: #eef2ff;
  border: 1px solid #c7d2fe;
  border-radius: 999px;
  padding: 0.22rem 0.7rem;
  font-size: 0.92rem;
  color: #1e293b;
  white-space: nowrap;
}
.wc-score-chip em {
  font-style: normal;
  color: #64748b;
  margin-left: 0.2rem;
  font-size: 0.88rem;
}
</style>
"""


def inject_dashboard_css() -> None:
    st.markdown(_DASHBOARD_CSS, unsafe_allow_html=True)


def format_pct(value: float) -> str:
    return f"{100.0 * value:.1f}%"


def _format_kickoff(kickoff: str) -> str:
    raw = str(kickoff).strip()
    if not raw:
        return "—"
    try:
        dt = datetime.fromisoformat(raw.replace("Z", "+00:00"))
        return dt.strftime("%Y-%m-%d %H:%M UTC")
    except ValueError:
        return raw


def render_market_sanity(prediction: dict, features: dict | None) -> None:
    comparison = prediction.get("market_comparison") or {}
    if not comparison.get("market_available"):
        if features and features.get("model_odds", {}).get("available"):
            st.caption("当前 odds 列为模型隐含概率（export_model_odds），非博彩公司盘口。")
        return

    market = comparison.get("market_result_probs") or {}
    raw_model = comparison.get("raw_result_probs") or prediction.get("raw_result_probs") or {}
    source = comparison.get("market_source") or "market"
    div = comparison.get("max_result_divergence")
    level = comparison.get("divergence_level", "ok")
    alerts = comparison.get("alerts") or []

    title = f"赛前市场对比（{source}）"
    if level == "critical" or "likely_team_mapping_bug" in alerts:
        st.error(f"{title} · 最大 1X2 偏差 {format_pct(float(div))} · 严重偏离/疑似映射错误")
    elif level == "warning":
        st.error(f"{title} · 最大 1X2 偏差 {format_pct(float(div))} · 模型观点较激进")
    elif level == "caution":
        st.warning(f"{title} · 最大 1X2 偏差 {format_pct(float(div))} · 模型观点较激进")
    else:
        st.info(f"{title} · 最大 1X2 偏差 {format_pct(float(div))}")

    cols = st.columns(3)
    cols[0].metric("市场主胜", format_pct(market.get("home_win", 0)))
    cols[1].metric("市场平局", format_pct(market.get("draw", 0)))
    cols[2].metric("市场客胜", format_pct(market.get("away_win", 0)))

    if raw_model:
        st.caption(
            f"原始模型：主 {format_pct(raw_model['home_win'])} · "
            f"平 {format_pct(raw_model['draw'])} · "
            f"客 {format_pct(raw_model['away_win'])}"
        )
    if alerts:
        st.caption("告警：" + " · ".join(alerts))
    blend_w = comparison.get("market_blend_weight")
    ou_w = comparison.get("ou_blend_weight")
    if blend_w:
        st.caption(
            f"市场校准权重 1X2={100.0 * float(blend_w):.0f}% · "
            f"O/U={100.0 * float(ou_w or 0):.0f}%"
        )
    applied = comparison.get("adjustments_applied") or []
    if applied:
        st.caption("已应用：" + " · ".join(applied))


def render_player_signal_debug(features: dict | None) -> None:
    if not features:
        return
    players = features.get("player_summary") or {}
    strength = features.get("team_strength") or {}
    hr = players.get("home_avg_player_rating")
    ar = players.get("away_avg_player_rating")
    if hr is None and ar is None:
        return
    with st.expander("球员层信号（debug）", expanded=False):
        c1, c2 = st.columns(2)
        with c1:
            st.markdown("**主队球员**")
            st.write(
                f"均分 {hr if hr is not None else '—'} · "
                f"首发 {players.get('home_starter_count', '—')} · "
                f"预计首发占比 {players.get('home_lineup_projected_share', '—')}"
            )
        with c2:
            st.markdown("**客队球员**")
            st.write(
                f"均分 {ar if ar is not None else '—'} · "
                f"首发 {players.get('away_starter_count', '—')} · "
                f"预计首发占比 {players.get('away_lineup_projected_share', '—')}"
            )
        for side, label in (("home_top_players", "主队关键球员"), ("away_top_players", "客队关键球员")):
            stars = players.get(side) or []
            if not stars:
                continue
            st.markdown(f"**{label}**")
            for p in stars:
                st.write(
                    f"{p.get('name', '—')} ({p.get('position', '')}): "
                    f"available={p.get('available')} · start_prob={p.get('start_prob')} · "
                    f"rating={p.get('rating')} · minutes={p.get('minutes_last5')} · "
                    f"contribution={p.get('contribution')}"
                )
        st.write(
            f"Elo {strength.get('home_elo', '—')} vs {strength.get('away_elo', '—')} · "
            f"FIFA #{strength.get('home_fifa_rank', '—')} vs #{strength.get('away_fifa_rank', '—')}"
        )
        if (
            hr is not None
            and ar is not None
            and abs(float(hr) - float(ar)) < 1.0
            and 70.0 <= float(hr) <= 74.0
        ):
            st.warning("两队球员均分接近 generic 72，可能未加载真实球星数据。")


def render_outcome_bar(
    home_team: str,
    away_team: str,
    prediction: dict,
    *,
    variant: str = "full",
    show_team_names: bool | None = None,
) -> None:
    """Single stacked bar: teams+flags on top, 1X2 bar, labels below."""
    inject_dashboard_css()
    probs = prediction["result_probs"]
    home_p = max(0.0, float(probs["home_win"]))
    draw_p = max(0.0, float(probs["draw"]))
    away_p = max(0.0, float(probs["away_win"]))
    total = home_p + draw_p + away_p
    if total <= 0:
        home_p, draw_p, away_p = 1 / 3, 1 / 3, 1 / 3
        total = 1.0
    home_p /= total
    draw_p /= total
    away_p /= total

    home_name = html.escape(resolve_team_name(str(home_team)))
    away_name = html.escape(resolve_team_name(str(away_team)))
    flag_width = 32 if variant == "schedule" else 40
    home_flag = team_flag_html(str(home_team), width=flag_width)
    away_flag = team_flag_html(str(away_team), width=flag_width)

    home_flex = max(1, round(home_p * 1000))
    draw_flex = max(1, round(draw_p * 1000))
    away_flex = max(1, round(away_p * 1000))

    if show_team_names is None:
        show_team_names = variant != "schedule"
    card_class = "wc-outcome-card"
    if variant == "schedule":
        card_class += " wc-schedule"
        if not show_team_names:
            card_class += " wc-flags-only"

    home_label = f"<span>{home_name}</span>" if show_team_names else ""
    away_label = f"<span>{away_name}</span>" if show_team_names else ""

    st.markdown(
        f"""
        <div class="{card_class}">
          <div class="wc-teams-row">
            <div class="wc-team home">{home_flag}{home_label}</div>
            <div class="wc-team away">{away_flag}{away_label}</div>
          </div>
          <div class="wc-bar">
            <div class="wc-bar-seg" style="flex:{home_flex};background:#16a34a;"></div>
            <div class="wc-bar-seg" style="flex:{draw_flex};background:#64748b;"></div>
            <div class="wc-bar-seg" style="flex:{away_flex};background:#dc2626;"></div>
          </div>
          <div class="wc-labels-row">
            <div class="wc-label" style="flex:{home_flex};">
              主胜<strong>{format_pct(home_p)}</strong>
            </div>
            <div class="wc-label" style="flex:{draw_flex};">
              平局<strong>{format_pct(draw_p)}</strong>
            </div>
            <div class="wc-label" style="flex:{away_flex};">
              客胜<strong>{format_pct(away_p)}</strong>
            </div>
          </div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_top_scorelines_compact(prediction: dict, *, top_n: int = 3) -> None:
    """Inline top scorelines for schedule/home cards."""
    inject_dashboard_css()
    items = prediction.get("top3_scorelines", [])[:top_n]
    if not items:
        return
    chips = []
    for item in items:
        score = f"{item['home_goals']}-{item['away_goals']}"
        prob = format_pct(float(item["prob"]))
        chips.append(f'<span class="wc-score-chip">{score} <em>{prob}</em></span>')
    st.markdown(
        f'<div class="wc-top-scores">'
        f'<span class="wc-top-scores-label">最可能比分</span> {"".join(chips)}'
        f"</div>",
        unsafe_allow_html=True,
    )


def render_match_meta(
    match_detail: dict,
    *,
    schedule_meta: dict[str, Any] | None = None,
    prediction: dict | None = None,
) -> None:
    inject_dashboard_css()
    stage = match_detail.get("stage") or match_detail.get("stage_name") or "n/a"
    kickoff = _format_kickoff(str(match_detail.get("kickoff_ts", "")))

    meta_bits = [kickoff, stage]
    if schedule_meta:
        if schedule_meta.get("venue"):
            meta_bits.append(str(schedule_meta["venue"]))
        if schedule_meta.get("city"):
            meta_bits.append(str(schedule_meta["city"]))
        if schedule_meta.get("group"):
            meta_bits.append(f"Group {schedule_meta['group']}")
    st.markdown(
        f"<div class='wc-meta'>{' · '.join(meta_bits)}</div>",
        unsafe_allow_html=True,
    )

    if prediction:
        model_name = prediction.get("model_name") or "unknown"
        model_type = prediction.get("model_type") or "model"
        version = prediction.get("model_version") or ""
        badge = f"模型预测 · {model_name} ({model_type})"
        if version:
            badge += f" · {version}"
        st.markdown(f"<div class='wc-model-badge'>{badge}</div>", unsafe_allow_html=True)

    if match_detail.get("home_score_ft") is not None:
        st.success(
            f"实际 90 分钟比分：{match_detail['home_score_ft']}-{match_detail['away_score_ft']}"
        )


def render_market_probabilities(prediction: dict) -> None:
    st.subheader("模型衍生概率")
    ou = prediction["ou25_probs"]
    btts = prediction["btts_probs"]
    xg = prediction["expected_goals"]

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Over 2.5", format_pct(ou["over_2_5"]))
    c2.metric("Under 2.5", format_pct(ou["under_2_5"]))
    c3.metric("BTTS Yes", format_pct(btts["yes"]))
    c4.metric("BTTS No", format_pct(btts["no"]))

    st.markdown("<div style='height:0.5rem'></div>", unsafe_allow_html=True)
    st.metric(
        "期望进球 (主 - 客)",
        f"{xg['home']:.2f} - {xg['away']:.2f}",
        delta=f"总计 {xg.get('total', xg['home'] + xg['away']):.2f}",
    )


def render_top_scorelines(prediction: dict, *, top_n: int = 5) -> None:
    st.subheader(f"Top {top_n} 比分")
    rows = []
    for item in prediction.get("top3_scorelines", [])[:top_n]:
        prob = float(item["prob"])
        rows.append(
            {
                "比分": f"{item['home_goals']}-{item['away_goals']}",
                "概率": format_pct(prob),
                "_bar": prob,
            }
        )
    if not rows:
        st.info("暂无比分分布数据。")
        return
    df = pd.DataFrame(rows)
    st.dataframe(
        df[["比分", "概率", "_bar"]],
        use_container_width=True,
        hide_index=True,
        column_config={
            "比分": st.column_config.TextColumn("比分", width="small"),
            "概率": st.column_config.TextColumn("概率", width="small"),
            "_bar": st.column_config.ProgressColumn(
                "分布",
                format="%.0f%%",
                min_value=0.0,
                max_value=1.0,
            ),
        },
    )


def render_model_diagnostics(prediction: dict) -> None:
    with st.expander("模型诊断（高级）", expanded=False):
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("置信度", format_pct(prediction["uncertainty"]["confidence"]))
        c2.metric("熵", f"{prediction['uncertainty']['entropy']:.3f}")
        c3.metric("λ_home", f"{prediction['lambda_home']:.3f}")
        c4.metric("λ_away", f"{prediction['lambda_away']:.3f}")
        st.caption(
            f"lambda_scale={prediction.get('lambda_scale', 1.0):.3f} · "
            f"overflow={prediction['overflow_prob']:.4f} · "
            f"ood={prediction['uncertainty'].get('ood_score', 0.0):.3f}"
        )


def render_feature_summary(
    features: dict | None,
    *,
    prediction: dict | None = None,
) -> None:
    with st.expander("特征摘要", expanded=False):
        if not features and not prediction:
            st.info("特征接口不可用或未返回数据。")
            return

        if prediction:
            result = prediction["result_probs"]
            ou = prediction["ou25_probs"]
            btts = prediction["btts_probs"]
            st.markdown("**模型输出**")
            st.write(
                f"1X2: {format_pct(result['home_win'])} / "
                f"{format_pct(result['draw'])} / {format_pct(result['away_win'])}  \n"
                f"Over 2.5: {format_pct(ou['over_2_5'])} · "
                f"BTTS: {format_pct(btts['yes'])}  \n"
                f"期望进球: {prediction['expected_goals']['home']:.2f} - "
                f"{prediction['expected_goals']['away']:.2f}"
            )

        if not features:
            return

        strength = features.get("team_strength", {})
        form = features.get("recent_form", {})
        players = features.get("player_summary", {})
        events = features.get("match_events", {})

        st.markdown("**输入特征（样例/PIT 数据）**")
        st.write(
            f"Elo: {strength.get('home_elo', '—')} vs {strength.get('away_elo', '—')}  \n"
            f"FIFA 排名: {strength.get('home_fifa_rank', '—')} vs "
            f"{strength.get('away_fifa_rank', '—')}  \n"
            f"近 5 场进/失: {form.get('home_goals_for_last5', '—')}/"
            f"{form.get('home_goals_against_last5', '—')} vs "
            f"{form.get('away_goals_for_last5', '—')}/"
            f"{form.get('away_goals_against_last5', '—')}  \n"
            f"首发: {players.get('home_starter_count', '—')} vs "
            f"{players.get('away_starter_count', '—')} · "
            f"伤停: {players.get('home_injured_out_count', '—')} vs "
            f"{players.get('away_injured_out_count', '—')}"
        )
        if events.get("available"):
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
    """Expanded 1X2 bar for schedule/home cards."""
    render_outcome_bar(
        home_team,
        away_team,
        prediction,
        variant="schedule",
        show_team_names=False,
    )


def render_full_prediction_panel(
    match_detail: dict,
    prediction: dict,
    features: dict | None,
    matrix_payload: dict | None,
    *,
    schedule_meta: dict[str, Any] | None = None,
) -> None:
    home = match_detail.get("home_team_name") or match_detail["home_team_id"]
    away = match_detail.get("away_team_name") or match_detail["away_team_id"]

    render_match_meta(match_detail, schedule_meta=schedule_meta, prediction=prediction)
    render_market_sanity(prediction, features)
    render_player_signal_debug(features)
    render_outcome_bar(str(home), str(away), prediction)

    render_market_probabilities(prediction)
    render_top_scorelines(prediction, top_n=5)
    render_model_diagnostics(prediction)
    render_feature_summary(features, prediction=prediction)

    if matrix_payload:
        with st.expander("比分矩阵热力图", expanded=False):
            if matrix_payload.get("overflow_prob", 0) > 0:
                st.caption(f"overflow_prob = {matrix_payload['overflow_prob']:.4f}")
            render_heatmap(matrix_payload["matrix"], matrix_payload["grid_max_goal"])

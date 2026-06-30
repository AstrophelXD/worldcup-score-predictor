"""2026 World Cup: compare model predictions with played results."""

from __future__ import annotations

import html
from datetime import date
from typing import Any, Callable

import pandas as pd
import streamlit as st

from worldcup.dashboard.flags import team_label_html, team_label_plain
from worldcup.dashboard.prediction_view import format_pct, inject_dashboard_css
from worldcup.dashboard.schedule_view import schedule_meta
from worldcup.dashboard.world_cup_2026_schedule import ScheduleMatch
from worldcup.data_ingestion.sources.world_cup_2026_results import load_wc2026_results
from worldcup.evaluation.match_eval import evaluate_played_match, slice_prediction_payload, summarize_evaluations
from worldcup.utils.paths import project_root

_FEATURE_MART = project_root() / "data" / "feature_mart" / "match_features.parquet"

_RESULT_LABELS = {
    "home_win": "主胜",
    "draw": "平局",
    "away_win": "客胜",
}


def _format_score(home: int | None, away: int | None) -> str:
    if home is None or away is None:
        return "—"
    return f"{home}-{away}"


def _top_scoreline_str(top3: list[dict[str, Any]]) -> str:
    if not top3:
        return "—"
    top = top3[0]
    return f"{top['home_goals']}-{top['away_goals']}"


def _resolve_scores(
    detail: dict[str, Any],
    *,
    fallback_results: dict[str, dict[str, int | str]],
) -> tuple[int | None, int | None]:
    home = detail.get("home_score_ft")
    away = detail.get("away_score_ft")
    if home is not None and away is not None:
        return int(home), int(away)
    match_id = str(detail.get("match_id") or "")
    overlay = fallback_results.get(match_id)
    if overlay:
        return int(overlay["home_score_ft"]), int(overlay["away_score_ft"])
    return None, None


def _comparison_bar_html(probs: dict[str, float], actual_key: str) -> str:
    keys = ["home_win", "draw", "away_win"]
    colors = {"home_win": "#16a34a", "draw": "#64748b", "away_win": "#dc2626"}
    labels = {"home_win": "主胜", "draw": "平", "away_win": "客胜"}
    segments = []
    label_cells = []
    for key in keys:
        pct = max(0.0, float(probs.get(key, 0.0)))
        flex = max(1, round(pct * 1000))
        border = "2px solid #0f172a" if key == actual_key else "1px solid transparent"
        segments.append(
            f'<div style="flex:{flex};background:{colors[key]};'
            f'border:{border};box-sizing:border-box;min-width:3px;"></div>'
        )
        mark = " ✓" if key == actual_key else ""
        label_cells.append(
            f'<div style="flex:{flex};text-align:center;font-size:0.82rem;color:#64748b;">'
            f"{labels[key]}{mark}<br>"
            f'<strong style="color:#0f172a;font-size:0.95rem;">{format_pct(pct)}</strong>'
            f"</div>"
        )
    return (
        '<div style="margin-top:0.35rem;">'
        '<div style="display:flex;height:18px;border-radius:999px;overflow:hidden;'
        'box-shadow:inset 0 0 0 1px rgba(15,23,42,0.08);">'
        f"{''.join(segments)}"
        "</div>"
        f'<div style="display:flex;margin-top:0.45rem;">{"".join(label_cells)}</div>'
        "</div>"
    )


def _load_feature_row(match_id: str) -> dict[str, Any] | None:
    if not _FEATURE_MART.exists():
        return None
    frame = pd.read_parquet(_FEATURE_MART)
    rows = frame.loc[frame["match_id"] == match_id]
    if rows.empty:
        return None
    return rows.iloc[0].to_dict()


def _pct(value: float | None) -> str:
    if value is None:
        return "—"
    return f"{100.0 * float(value):.1f}%"


def _build_display_row(
    *,
    detail: dict[str, Any],
    prediction: dict[str, Any],
    schedule: ScheduleMatch | None,
    home_score: int,
    away_score: int,
    eval_row: dict[str, Any],
    use_raw: bool,
) -> dict[str, Any]:
    pred = slice_prediction_payload(prediction, use_raw=use_raw)
    branch = eval_row["raw" if use_raw else "adjusted"]
    probs = pred["result_probs"]
    top3 = pred.get("top3_scorelines") or []
    top3_text = " · ".join(
        f"{s['home_goals']}-{s['away_goals']} ({format_pct(float(s['prob']))})"
        for s in top3[:3]
    )
    home_name = detail.get("home_team_name") or detail.get("home_team_id")
    away_name = detail.get("away_team_name") or detail.get("away_team_id")
    actual = eval_row["actual_outcome"]
    return {
        "match_id": detail.get("match_id"),
        "match_number": schedule.match_number if schedule else None,
        "match_date": schedule.match_date if schedule else str(detail.get("kickoff_ts", ""))[:10],
        "kickoff_ts": detail.get("kickoff_ts"),
        "stage": detail.get("stage") or (schedule.stage_name if schedule else ""),
        "home_team": home_name,
        "away_team": away_name,
        "matchup_plain": f"{team_label_plain(str(home_name))} vs {team_label_plain(str(away_name))}",
        "matchup_html": (
            f"{team_label_html(str(home_name), width=22)}"
            f'<span style="color:#94a3b8;margin:0 0.35rem;">vs</span>'
            f"{team_label_html(str(away_name), width=22)}"
        ),
        "actual_score": _format_score(home_score, away_score),
        "predicted_top1": _top_scoreline_str(top3),
        "top3_text": top3_text,
        "predicted_outcome": branch["predicted_outcome"],
        "predicted_outcome_label": _RESULT_LABELS[branch["predicted_outcome"]],
        "predicted_prob": probs[branch["predicted_outcome"]],
        "actual_outcome": actual,
        "actual_outcome_label": _RESULT_LABELS[actual],
        "hit_1x2": branch["hit_1x2"],
        "hit_top3": branch["hit_top3"],
        "brier": branch["brier"],
        "rps": branch["rps"],
        "probs": probs,
        "expected_goals": pred.get("expected_goals", {}),
        "comparison_html": _comparison_bar_html(probs, actual),
        "top1_agrees_1x2": branch.get("top1_agrees_1x2"),
        "hit_ou": (home_score + away_score >= 3)
        == (pred.get("ou25_probs", {}).get("over_2_5", 0.5) >= 0.5),
        "hit_btts": (home_score >= 1 and away_score >= 1)
        == (pred.get("btts_probs", {}).get("yes", 0.5) >= 0.5),
    }


def render_wc2026_results_compare_tab(
    matches: list[dict[str, Any]],
    *,
    predict_fn: Callable[[str], dict | None],
    match_detail_fn: Callable[[str], dict | None] | None = None,
    schedule_by_id: dict[str, ScheduleMatch] | None = None,
    predictions_by_id: dict[str, dict] | None = None,
    today: date | None = None,
) -> None:
    st.subheader("2026 赛果对照")
    st.caption(
        "对比赛前模型预测与已完赛比分（90 分钟）。"
        "模型质量请优先看 **原始 checkpoint**；**市场校准后**为对外展示口径。"
        "赛果来自 `data/samples/wc2026_results.csv`。"
    )

    fallback_results = load_wc2026_results()
    wc_matches = [m for m in matches if str(m.get("match_id", "")).startswith("wc2026_")]
    wc_matches.sort(key=lambda m: str(m.get("kickoff_ts", "")))

    eval_rows: list[dict[str, Any]] = []
    pending_count = 0

    for item in wc_matches:
        match_id = item["match_id"]
        detail = match_detail_fn(match_id) if match_detail_fn else item
        if detail is None:
            detail = item
        detail = {**item, **detail, "match_id": match_id}

        home_score, away_score = _resolve_scores(detail, fallback_results=fallback_results)
        if home_score is None or away_score is None:
            pending_count += 1
            continue

        prediction = None
        if predictions_by_id and match_id in predictions_by_id:
            prediction = predictions_by_id[match_id]
        if prediction is None:
            prediction = predict_fn(match_id)
        if prediction is None:
            continue

        feature_row = _load_feature_row(match_id) or detail
        eval_rows.append(
            evaluate_played_match(
                feature_row=feature_row,
                prediction=prediction,
                home_goals=home_score,
                away_goals=away_score,
            )
        )

    summary = summarize_evaluations(eval_rows)
    played_count = summary.get("count", 0)

    c1, c2, c3 = st.columns(3)
    c1.metric("2026 已入库", len(wc_matches))
    c2.metric("已完赛", played_count)
    c3.metric("待赛", pending_count)

    if not eval_rows:
        st.info(
            "暂无已完赛且可对照的 2026 场次。"
            "请更新 `data/samples/wc2026_results.csv` 或运行 `python -m scripts.seed_wc2026_results`。"
        )
        return

    st.markdown("#### 命中率对照（含 baseline）")
    st.info(
        "**1X2 命中率**：预测的最高概率胜平负是否与实际一致。"
        "**Top3 命中率**：实际精确比分是否落在模型概率最高的 3 个比分格内。"
        "两者均来自同一 `score_matrix`；若校准后 Top3 升高但 1X2 不变，说明市场校准改变了矩阵形状。"
    )

    raw = summary.get("raw", {})
    adj = summary.get("adjusted", {})
    base = summary.get("baselines", {})
    compare = pd.DataFrame(
        [
            {
                "方法": "模型（原始 checkpoint）",
                "1X2": _pct(raw.get("hit_1x2")),
                "Top3 比分": _pct(raw.get("hit_top3")),
                "Top1↔1X2 一致": _pct(raw.get("top1_agrees_1x2")),
                "平均 Brier": f"{raw.get('avg_brier', 0):.3f}" if raw.get("count") else "—",
            },
            {
                "方法": "模型（市场校准后）",
                "1X2": _pct(adj.get("hit_1x2")),
                "Top3 比分": _pct(adj.get("hit_top3")),
                "Top1↔1X2 一致": _pct(adj.get("top1_agrees_1x2")),
                "平均 Brier": f"{adj.get('avg_brier', 0):.3f}" if adj.get("count") else "—",
            },
            {
                "方法": "Elo favorite",
                "1X2": _pct(base.get("elo_favorite_hit")),
                "Top3 比分": "—",
                "Top1↔1X2 一致": "—",
                "平均 Brier": "—",
            },
            {
                "方法": "FIFA rank favorite",
                "1X2": _pct(base.get("fifa_favorite_hit")),
                "Top3 比分": "—",
                "Top1↔1X2 一致": "—",
                "平均 Brier": "—",
            },
            {
                "方法": "市场 favorite（有盘口场次）",
                "1X2": _pct(base.get("market_favorite_hit")),
                "Top3 比分": "—",
                "Top1↔1X2 一致": "—",
                "平均 Brier": "—",
            },
            {
                "方法": "常见比分 Top3（1-1/1-0/0-1）",
                "1X2": "—",
                "Top3 比分": _pct(base.get("naive_top3_hit")),
                "Top1↔1X2 一致": "—",
                "平均 Brier": "—",
            },
            {
                "方法": "常见比分 Top6",
                "1X2": "—",
                "Top3 比分": _pct(base.get("naive_top6_hit")),
                "Top1↔1X2 一致": "—",
                "平均 Brier": "—",
            },
            {
                "方法": "随机猜 1X2",
                "1X2": "33.3%",
                "Top3 比分": "—",
                "Top1↔1X2 一致": "—",
                "平均 Brier": "—",
            },
        ]
    )
    st.dataframe(compare, use_container_width=True, hide_index=True)

    if (
        raw.get("hit_1x2") is not None
        and adj.get("hit_top3") is not None
        and raw.get("hit_top3") is not None
        and adj["hit_top3"] - raw["hit_top3"] > 0.12
    ):
        st.warning(
            "市场校准后 Top3 命中率显著高于原始模型，但 1X2 提升有限。"
            "请优先用 **原始 checkpoint** 评估模型；校准后指标仅反映对外展示口径。"
        )
    if summary.get("adjusted_top1_consistent_rate", 1.0) is not None and (
        summary.get("adjusted_top1_consistent_rate") or 0
    ) < 0.5:
        st.warning(
            f"校准后 Top1 比分与 1X2 最可能结果一致率仅 "
            f"{_pct(summary.get('adjusted_top1_consistent_rate'))}。"
            "这是市场拟合 λ 后矩阵变形的正常现象，不代表 Top3 泄漏。"
        )

    metric_mode = st.radio(
        "下方逐场表使用的预测口径",
        options=["raw", "adjusted"],
        format_func=lambda x: "原始 checkpoint（推荐评估）" if x == "raw" else "市场校准后（对外展示）",
        horizontal=True,
        key="wc26_results_metric_mode",
    )
    use_raw = metric_mode == "raw"

    played_rows: list[dict[str, Any]] = []
    for item in wc_matches:
        match_id = item["match_id"]
        eval_row = next((e for e in eval_rows if e["match_id"] == match_id), None)
        if eval_row is None:
            continue
        detail = match_detail_fn(match_id) if match_detail_fn else item
        if detail is None:
            detail = item
        detail = {**item, **detail, "match_id": match_id}
        home_score, away_score = _resolve_scores(detail, fallback_results=fallback_results)
        if home_score is None or away_score is None:
            continue
        prediction = predictions_by_id.get(match_id) if predictions_by_id else None
        if prediction is None:
            prediction = predict_fn(match_id)
        if prediction is None:
            continue
        schedule = (schedule_by_id or {}).get(match_id)
        played_rows.append(
            _build_display_row(
                detail=detail,
                prediction=prediction,
                schedule=schedule,
                home_score=home_score,
                away_score=away_score,
                eval_row=eval_row,
                use_raw=use_raw,
            )
        )

    filter_col1, filter_col2, filter_col3 = st.columns([1, 1, 1])
    outcome_filter = filter_col1.selectbox(
        "1X2 筛选",
        ["全部", "命中", "未中"],
        key="wc26_results_outcome_filter",
    )
    stage_options = ["全部阶段", *sorted({str(r["stage"]) for r in played_rows if r["stage"]})]
    stage_filter = filter_col2.selectbox("阶段", stage_options, key="wc26_results_stage_filter")
    date_options = ["全部日期", *sorted({str(r["match_date"]) for r in played_rows if r["match_date"]})]
    date_filter = filter_col3.selectbox("日期", date_options, key="wc26_results_date_filter")

    filtered = played_rows
    if outcome_filter == "命中":
        filtered = [r for r in filtered if r["hit_1x2"]]
    elif outcome_filter == "未中":
        filtered = [r for r in filtered if not r["hit_1x2"]]
    if stage_filter != "全部阶段":
        filtered = [r for r in filtered if r["stage"] == stage_filter]
    if date_filter != "全部日期":
        filtered = [r for r in filtered if r["match_date"] == date_filter]

    st.caption(f"展示 {len(filtered)} / {len(played_rows)} 场已完赛对照")

    table = pd.DataFrame(
        [
            {
                "#": r.get("match_number"),
                "日期": r.get("match_date"),
                "对阵": r["matchup_plain"],
                "阶段": r["stage"],
                "预测1X2": f"{r['predicted_outcome_label']} ({format_pct(float(r['predicted_prob']))})",
                "实际1X2": r["actual_outcome_label"],
                "预测Top1": r["predicted_top1"],
                "实际比分": r["actual_score"],
                "1X2": "✓" if r["hit_1x2"] else "✗",
                "Top3": "✓" if r["hit_top3"] else "✗",
                "Brier": r["brier"],
            }
            for r in sorted(filtered, key=lambda x: (str(x.get("match_date")), x.get("match_number") or 0))
        ]
    )
    st.markdown("#### 对照汇总")
    st.dataframe(
        table,
        use_container_width=True,
        hide_index=True,
        column_config={
            "1X2": st.column_config.TextColumn("1X2", width="small"),
            "Top3": st.column_config.TextColumn("Top3", width="small"),
        },
    )

    st.markdown("#### 单场详情")
    options = sorted(filtered, key=lambda x: (str(x.get("match_date")), x.get("match_number") or 0))
    if not options:
        return
    labels = {
        str(r["match_id"]): (
            f"#{r.get('match_number')} · {r['match_date']} · {r['matchup_plain']} · "
            f"实际 {r['actual_score']} · 预测 {r['predicted_top1']}"
        )
        for r in options
    }
    default_id = str(options[-1]["match_id"])
    picked_id = st.selectbox(
        "选择场次查看预测分布",
        options=list(labels.keys()),
        index=list(labels.keys()).index(default_id),
        format_func=lambda mid: labels[mid],
        key="wc26_results_detail_pick",
    )
    row = next(r for r in options if str(r["match_id"]) == picked_id)
    schedule = (schedule_by_id or {}).get(str(row["match_id"]))
    meta = schedule_meta(schedule) if schedule else {}
    icon = "✅" if row["hit_1x2"] else "❌"

    inject_dashboard_css()
    st.markdown(f"### {icon} {row['matchup_plain']} · **{row['actual_score']}**")
    left, right = st.columns(2)
    with left:
        st.markdown("**模型预测**")
        st.markdown(row["matchup_html"], unsafe_allow_html=True)
        st.write(
            f"1X2 最可能：**{row['predicted_outcome_label']}** "
            f"({format_pct(float(row['predicted_prob']))})"
        )
        xg = row.get("expected_goals") or {}
        if xg:
            st.write(
                f"期望进球 {xg.get('home', 0):.2f} - {xg.get('away', 0):.2f} · "
                f"Top1 比分 **{row['predicted_top1']}**"
            )
        st.caption(f"Top3：{row['top3_text']}")
    with right:
        st.markdown("**实际赛果**")
        st.markdown(
            f"<div style='font-size:1.8rem;font-weight:700;color:#0f172a;"
            f"margin:0.4rem 0 0.6rem;'>{html.escape(row['actual_score'])}</div>",
            unsafe_allow_html=True,
        )
        st.write(f"赛果：**{row['actual_outcome_label']}**")
        st.write(
            f"1X2 {'命中' if row['hit_1x2'] else '未中'} · "
            f"Top3 {'命中' if row['hit_top3'] else '未中'} · "
            f"O/U {'命中' if row['hit_ou'] else '未中'} · "
            f"BTTS {'命中' if row['hit_btts'] else '未中'}"
        )
        st.caption(f"Brier {row['brier']} · RPS {row['rps']}")

    st.markdown("**预测分布 vs 实际结果（黑框 = 实际）**")
    st.markdown(row["comparison_html"], unsafe_allow_html=True)
    if meta:
        st.caption(
            f"{meta.get('stage_name', row['stage'])} · "
            f"{meta.get('kickoff_et', '')} ET · "
            f"{meta.get('venue', '')}, {meta.get('city', '')}"
        )

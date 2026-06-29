"""Dashboard view for post-match model evaluation."""

from __future__ import annotations

from typing import Any, Callable

import pandas as pd
import streamlit as st

from worldcup.backtesting.metrics import actual_result, brier_multiclass, ranked_probability_score


def _format_score(home: int | None, away: int | None) -> str:
    if home is None or away is None:
        return "—"
    return f"{home}-{away}"


def _top_scoreline_str(top3: list[dict[str, Any]]) -> str:
    if not top3:
        return "—"
    top = top3[0]
    return f"{top['home_goals']}-{top['away_goals']}"


def render_postmatch_backtest_tab(
    matches: list[dict[str, Any]],
    *,
    predict_fn: Callable[[str], dict | None],
    match_detail_fn: Callable[[str], dict | None] | None = None,
) -> None:
    st.subheader("赛后回测（模型 vs 实际）")
    st.caption("仅统计已有 90 分钟比分的比赛；使用 API 调整后预测。")

    rows: list[dict[str, Any]] = []
    for item in matches:
        match_id = item["match_id"]
        detail = match_detail_fn(match_id) if match_detail_fn else item
        if detail is None:
            continue
        home_score = detail.get("home_score_ft")
        away_score = detail.get("away_score_ft")
        if home_score is None or away_score is None:
            continue

        prediction = predict_fn(match_id)
        if prediction is None:
            continue

        actual = actual_result(int(home_score), int(away_score))
        predicted = max(
            prediction["result_probs"],
            key=prediction["result_probs"].get,
        )
        total_goals = int(home_score) + int(away_score)
        ou_hit = (total_goals >= 3) == (prediction["ou25_probs"]["over_2_5"] >= 0.5)
        btts_actual = int(home_score) >= 1 and int(away_score) >= 1
        btts_hit = btts_actual == (prediction["btts_probs"]["yes"] >= 0.5)
        top_hit = any(
            int(s["home_goals"]) == int(home_score) and int(s["away_goals"]) == int(away_score)
            for s in prediction.get("top3_scorelines", [])
        )

        rows.append(
            {
                "比赛": f"{detail.get('home_team_name', item.get('home_team_name'))} vs "
                f"{detail.get('away_team_name', item.get('away_team_name'))}",
                "实际比分": _format_score(home_score, away_score),
                "预测Top1": _top_scoreline_str(prediction.get("top3_scorelines", [])),
                "Top3命中": top_hit,
                "1X2预测": predicted,
                "1X2实际": actual,
                "1X2命中": predicted == actual,
                "Brier 1X2": round(
                    brier_multiclass(prediction["result_probs"], actual),
                    3,
                ),
                "RPS": round(
                    ranked_probability_score(prediction["result_probs"], actual),
                    3,
                ),
                "O/U方向": ou_hit,
                "BTTS方向": btts_hit,
            }
        )

    if not rows:
        st.info("暂无已完赛且可评估的比赛。请在 matches 数据写入实际比分后刷新 pipeline。")
        return

    df = pd.DataFrame(rows)
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("场次", len(df))
    c2.metric("1X2 命中率", f"{100.0 * df['1X2命中'].mean():.1f}%")
    c3.metric("Top3 命中率", f"{100.0 * df['Top3命中'].mean():.1f}%")
    c4.metric("平均 Brier", f"{df['Brier 1X2'].mean():.3f}")

    st.dataframe(df, use_container_width=True, hide_index=True)

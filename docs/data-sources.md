# 数据源 CSV 格式说明

项目内置样例位于 `data/samples/`，可直接用于本地测试与实验室首次跑通。替换为真实公开数据时，保持列名一致即可。

## matches.csv

| 列名 | 必填 | 说明 |
|------|------|------|
| match_id | 是 | 唯一比赛 ID |
| competition_name | 是 | 赛事名称 |
| season_name | 否 | 赛季，如 2018 / 2022 |
| stage_name | 否 | 阶段，如 Group stage / Final |
| match_date | 是 | 比赛日期 YYYY-MM-DD |
| kickoff_ts | 是 | 开球时间 ISO8601 |
| home_team_name | 是 | 主队名称 |
| away_team_name | 是 | 客队名称 |
| home_score_ft | 否 | **90 分钟**主队进球（主标签） |
| away_score_ft | 否 | **90 分钟**客队进球 |
| home_score_ht | 否 | 半场比分 |
| away_score_ht | 否 | 半场比分 |
| aet_score_home | 否 | 加时主队进球（不得混入主标签） |
| aet_score_away | 否 | 加时客队进球 |
| pen_score_home | 否 | 点球主队进球 |
| pen_score_away | 否 | 点球客队进球 |
| status | 否 | 默认 finished |
| is_world_cup | 否 | 是否世界杯 |
| is_knockout | 否 | 是否淘汰赛 |
| venue / city / country | 否 | 场地信息 |

## elo.csv

| 列名 | 必填 | 说明 |
|------|------|------|
| team_name | 是 | 球队名称 |
| rating | 是 | Elo 分值 |
| rating_date | 是 | 评级日期 |
| rating_system | 否 | 默认 elo |
| rank | 否 | 排名 |

## fifa_rankings.csv

| 列名 | 必填 | 说明 |
|------|------|------|
| team_name | 是 | 球队名称 |
| ranking_date | 是 | 排名发布日期 |
| rank | 是 | FIFA 排名 |
| points | 是 | FIFA 积分 |

## 球队名称对齐

在 `data/external_mappings/team_aliases.csv` 中维护别名到 `canonical_team_id` 的映射。未命中别名的球队会自动生成 `team_{slug}`。

## 配置

在 `configs/data/default.yaml` 中设置 `sources.*` 指向 CSV 路径。默认已指向 `data/samples/`。

## 运行

实验室 RTX 4090 主机（或本地轻量验证）：

```bash
python -m scripts.ingest
python -m scripts.validate_data
```

输出：

- `data/raw/*.parquet` — 带来源元数据的原始层
- `data/curated/*.parquet` — 标准化实体表

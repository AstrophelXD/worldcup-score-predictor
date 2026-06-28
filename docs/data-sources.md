# 数据源 CSV 格式说明

项目内置样例位于 `data/samples/`，可直接用于本地测试与首次跑通。

外部真实数据对接见 [`data/external/README.md`](../data/external/README.md) 与 `configs/data/external.yaml`。

## 标准 canonical 格式

以下列名是 `scripts.ingest` 直接消费的格式。外部 CSV 可先经 `scripts.prepare_data` 转换到 `data/staging/canonical/`。

### matches.csv

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

模板文件：`data/templates/matches.template.csv`

### elo.csv

| 列名 | 必填 | 说明 |
|------|------|------|
| team_name | 是 | 球队名称 |
| rating | 是 | Elo 分值 |
| rating_date | 是 | 评级日期 |
| rating_system | 否 | 默认 elo |
| rank | 否 | 排名 |

模板文件：`data/templates/elo.template.csv`

### fifa_rankings.csv

| 列名 | 必填 | 说明 |
|------|------|------|
| team_name | 是 | 球队名称 |
| ranking_date | 是 | 排名发布日期 |
| rank | 是 | FIFA 排名 |
| points | 是 | FIFA 积分 |

模板文件：`data/templates/fifa_rankings.template.csv`

## 支持的 external 输入格式

| format 值 | 典型来源 | 必需列（灵活匹配） |
|-----------|----------|-------------------|
| `kaggle_international` | Kaggle International Football Results | `date`, `home_team`, `away_team` |
| `elo_history` | Elo 历史 CSV | `date`/`rating_date`, `team`/`country`, `rating`/`elo` |
| `fifa_rankings` | FIFA 排名 CSV | `rank_date`/`ranking_date`, `country`, `rank`, `points`/`total_points` |
| `canonical` | 已符合上表 | 全部 canonical 列名 |

## 球队名称对齐

在 `data/external_mappings/team_aliases.csv` 中维护别名到 `canonical_team_id` 的映射。未命中别名的球队会自动生成 `team_{slug}`。

## 配置

| Profile | 用途 |
|---------|------|
| `configs/data/default.yaml` | 默认样例数据 `data/samples/` |
| `configs/data/external.yaml` | 外部数据 + staging canonical |

## 运行

### 样例数据（默认）

```bash
python -m scripts.ingest
python -m scripts.validate_data
```

### 外部真实数据

```bash
python -m scripts.prepare_data --config-name=config data=external
python -m scripts.ingest --config-name=config data=external
python -m scripts.validate_data
python -m scripts.build_features
```

将下载 CSV 放入 `data/external/downloads/`，并更新 `configs/data/external.yaml` 中 `external_inputs` 路径。

输出：

- `data/staging/canonical/*.csv` — 标准化中间 CSV
- `data/raw/*.parquet` — 带来源元数据的原始层
- `data/curated/*.parquet` — 标准化实体表

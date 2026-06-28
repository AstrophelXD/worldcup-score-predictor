# 世界杯指定场次比分预测系统设计方案

## Executive Summary

本项目目标是在本地机器上训练并运行一个可模块化扩展的世界杯指定场次比分预测系统。系统核心不只是输出胜平负，而是直接建模 **比分联合概率分布**，最终输出：

- 概率最高的 3 个具体比分及其概率
- 胜 / 平 / 负概率
- Over / Under 2.5 概率
- BTTS 概率
- 期望进球
- 模型不确定性 / 置信度
- `0-7 × 0-7` 的比分概率矩阵
- 可交互 dashboard

系统按分阶段能力建设：

- `v0`：公开数据 + 手工输入，跑通全链路
- `v1`：加入赔率、Elo、FIFA ranking
- `v2`：加入球员级特征、预计首发和伤停
- `v3`：加入事件数据和 xG
- `v4`：加入实时刷新和正式首发重跑

整体工程原则：

- 先构建稳定的 baseline 和可回测管线
- 再迭代中阶模型
- 最后引入高阶序列与图结构模型
- 所有业务输出统一从比分联合概率矩阵聚合而来

## 1. 需求澄清与假设

### 1.1 预测目标

- 预测目标为 **90 分钟常规时间比分**。
- 默认不包含加时赛与点球大战。
- 淘汰赛也以 90 分钟比分为主标签。

### 1.2 小组赛与淘汰赛区别

- 小组赛需要考虑积分驱动和“接受平局”行为。
- 淘汰赛需要考虑保守倾向和加时赛预期。
- 二者共用主标签，但应通过特征建模差异。

### 1.3 输出格式

建议统一响应对象：

```json
{
  "top3_scorelines": [
    {"home_goals": 1, "away_goals": 1, "prob": 0.146},
    {"home_goals": 2, "away_goals": 1, "prob": 0.118},
    {"home_goals": 1, "away_goals": 0, "prob": 0.097}
  ],
  "result_probs": {
    "home_win": 0.412,
    "draw": 0.301,
    "away_win": 0.287
  },
  "ou25_probs": {
    "over_2_5": 0.471,
    "under_2_5": 0.529
  },
  "btts_probs": {
    "yes": 0.503,
    "no": 0.497
  },
  "expected_goals": {
    "home": 1.38,
    "away": 1.09,
    "total": 2.47
  },
  "uncertainty": {
    "entropy": 3.41,
    "ensemble_var": 0.018,
    "ood_score": 0.11,
    "confidence": 0.68
  }
}
```

### 1.4 Dashboard 核心用户流程

1. 选择比赛。
2. 查看比赛背景、球队/球员特征、数据新鲜度。
3. 查看 top-3 比分、胜平负、大小球、BTTS、置信度。
4. 查看比分概率热力图。
5. 查看特征解释和关键驱动因子。
6. 如有临场变动，手动调整首发/伤停并重跑预测。

## 2. 总体系统架构

```text
[Public Data / Commercial APIs / Manual Inputs]
        |
        v
[Data Ingestion Layer]
  - API pull
  - file import
  - manual forms
  - schema validation
        |
        v
[Raw Data Storage]
  - raw_json/
  - raw_csv/
  - source snapshots
        |
        v
[Curated Storage + Feature Store]
  - normalized entities
  - point-in-time joins
  - training feature views
  - inference feature views
        |
   +----+--------------------+
   |                         |
   v                         v
[Training Pipeline]     [Inference Pipeline]
  - dataset build         - feature assembly
  - train/val/test split  - checkpoint load
  - model train           - score matrix decode
  - calibration           - explanation payload
  - backtest                    |
        |                       v
        v                 [FastAPI Service]
[Model Registry]               |
  - checkpoints                +--> /predict
  - metrics                    +--> /matches
  - run metadata               +--> /features
        |                      +--> /backtest
        v
[Monitoring / Evaluation]
  - backtest metrics
  - calibration drift
  - data freshness
        |
        v
[Dashboard]
```

### 2.1 模块边界

- `data_ingestion`：负责抓取、导入、解析、标准化。
- `feature_store`：负责 point-in-time 特征构建。
- `training`：负责训练、验证、回测、校准。
- `inference`：负责单场特征组装和模型调用。
- `api`：负责标准化接口暴露。
- `dashboard`：负责交互式可视化。

## 3. 数据模型与数据库设计

建议采用：

- 训练与离线分析：`Parquet + DuckDB`
- 在线 serving 元数据：`PostgreSQL` 或先用 `SQLite / DuckDB`

### 3.1 核心表

#### `teams`

| 字段 | 类型 | 说明 | 约束 |
|---|---|---|---|
| team_id | STRING | 主键 | PK |
| team_name | STRING | 队名 | NOT NULL |
| country_code | STRING | 国家代码 | NULL |
| confederation | STRING | 洲际足联 | NULL |
| fifa_team_id | STRING | FIFA 源 ID | NULL |
| statsbomb_team_id | STRING | StatsBomb 源 ID | NULL |
| is_national_team | BOOL | 是否国家队 | NOT NULL |
| source_system | STRING | 数据来源 | NOT NULL |
| ingested_at | TIMESTAMP | 入库时间 | NOT NULL |
| updated_at | TIMESTAMP | 更新时间 | NOT NULL |

#### `players`

| 字段 | 类型 | 说明 | 约束 |
|---|---|---|---|
| player_id | STRING | 主键 | PK |
| full_name | STRING | 姓名 | NOT NULL |
| dob | DATE | 生日 | NULL |
| national_team_id | STRING | 国家队 | FK teams.team_id |
| primary_position | STRING | 主位置 | NULL |
| secondary_positions | ARRAY/JSON | 次位置 | NULL |
| preferred_foot | STRING | 惯用脚 | NULL |
| height_cm | INT | 身高 | NULL |
| market_value_eur | FLOAT | 身价 | NULL |
| current_club | STRING | 俱乐部 | NULL |
| source_system | STRING | 来源 | NOT NULL |
| ingested_at | TIMESTAMP | 入库时间 | NOT NULL |
| updated_at | TIMESTAMP | 更新时间 | NOT NULL |

#### `matches`

| 字段 | 类型 | 说明 | 约束 |
|---|---|---|---|
| match_id | STRING | 主键 | PK |
| competition_name | STRING | 赛事名 | NOT NULL |
| season_name | STRING | 赛季 | NULL |
| stage_name | STRING | 阶段 | NULL |
| match_date | DATE | 比赛日期 | NOT NULL |
| kickoff_ts | TIMESTAMP | 开球时间 | NULL |
| venue | STRING | 场地 | NULL |
| city | STRING | 城市 | NULL |
| country | STRING | 国家 | NULL |
| home_team_id | STRING | 主队 | FK teams.team_id |
| away_team_id | STRING | 客队 | FK teams.team_id |
| home_score_ft | INT | 90 分钟主队进球 | NULL |
| away_score_ft | INT | 90 分钟客队进球 | NULL |
| home_score_ht | INT | 半场主队进球 | NULL |
| away_score_ht | INT | 半场客队进球 | NULL |
| aet_score_home | INT | 加时主队进球 | NULL |
| aet_score_away | INT | 加时客队进球 | NULL |
| pen_score_home | INT | 点球主队进球 | NULL |
| pen_score_away | INT | 点球客队进球 | NULL |
| status | STRING | 状态 | NOT NULL |
| is_world_cup | BOOL | 是否世界杯 | NOT NULL |
| is_knockout | BOOL | 是否淘汰赛 | NOT NULL |
| source_system | STRING | 来源 | NOT NULL |
| ingested_at | TIMESTAMP | 入库时间 | NOT NULL |
| updated_at | TIMESTAMP | 更新时间 | NOT NULL |

#### `squads`

| 字段 | 类型 | 说明 | 约束 |
|---|---|---|---|
| squad_id | STRING | 主键 | PK |
| tournament_key | STRING | 赛事标识 | NOT NULL |
| team_id | STRING | 球队 | FK teams.team_id |
| player_id | STRING | 球员 | FK players.player_id |
| is_final_roster | BOOL | 是否最终名单 | NOT NULL |
| joined_on | DATE | 加入时间 | NULL |
| left_on | DATE | 离开时间 | NULL |
| source_system | STRING | 来源 | NOT NULL |
| ingested_at | TIMESTAMP | 入库时间 | NOT NULL |
| updated_at | TIMESTAMP | 更新时间 | NOT NULL |

#### `lineups`

| 字段 | 类型 | 说明 | 约束 |
|---|---|---|---|
| lineup_id | STRING | 主键 | PK |
| match_id | STRING | 比赛 | FK matches.match_id |
| team_id | STRING | 球队 | FK teams.team_id |
| player_id | STRING | 球员 | FK players.player_id |
| is_starting | BOOL | 是否首发 | NOT NULL |
| bench_order | INT | 替补顺位 | NULL |
| position_code | STRING | 位置 | NULL |
| formation_slot | STRING | 阵型槽位 | NULL |
| minutes_played | INT | 出场分钟 | NULL |
| captain_flag | BOOL | 是否队长 | NOT NULL |
| lineup_status | STRING | projected / official / historical | NOT NULL |
| projection_prob | FLOAT | 预计首发概率 | NULL |
| source_system | STRING | 来源 | NOT NULL |
| ingested_at | TIMESTAMP | 入库时间 | NOT NULL |
| updated_at | TIMESTAMP | 更新时间 | NOT NULL |

#### `injuries`

| 字段 | 类型 | 说明 | 约束 |
|---|---|---|---|
| injury_id | STRING | 主键 | PK |
| player_id | STRING | 球员 | FK players.player_id |
| team_id | STRING | 球队 | FK teams.team_id |
| injury_type | STRING | 伤病类型 | NULL |
| status | STRING | out / doubtful / probable / fit | NOT NULL |
| start_date | DATE | 开始时间 | NOT NULL |
| expected_return_date | DATE | 预计回归 | NULL |
| confidence | FLOAT | 可信度 | NULL |
| notes | STRING | 备注 | NULL |
| source_system | STRING | 来源 | NOT NULL |
| ingested_at | TIMESTAMP | 入库时间 | NOT NULL |
| updated_at | TIMESTAMP | 更新时间 | NOT NULL |

#### `player_match_stats`

| 字段 | 类型 | 说明 | 约束 |
|---|---|---|---|
| player_match_stat_id | STRING | 主键 | PK |
| match_id | STRING | 比赛 | FK matches.match_id |
| player_id | STRING | 球员 | FK players.player_id |
| team_id | STRING | 球队 | FK teams.team_id |
| minutes | INT | 出场分钟 | NULL |
| goals | INT | 进球 | NULL |
| assists | INT | 助攻 | NULL |
| shots | INT | 射门 | NULL |
| xg | FLOAT | xG | NULL |
| xa | FLOAT | xA | NULL |
| key_passes | INT | 关键传球 | NULL |
| duels_won | INT | 对抗成功 | NULL |
| pressures | INT | 压迫 | NULL |
| yellow_cards | INT | 黄牌 | NULL |
| red_cards | INT | 红牌 | NULL |
| source_system | STRING | 来源 | NOT NULL |
| ingested_at | TIMESTAMP | 入库时间 | NOT NULL |
| updated_at | TIMESTAMP | 更新时间 | NOT NULL |

#### `team_match_stats`

| 字段 | 类型 | 说明 | 约束 |
|---|---|---|---|
| team_match_stat_id | STRING | 主键 | PK |
| match_id | STRING | 比赛 | FK matches.match_id |
| team_id | STRING | 球队 | FK teams.team_id |
| possession | FLOAT | 控球率 | NULL |
| shots | INT | 射门 | NULL |
| shots_on_target | INT | 射正 | NULL |
| xg | FLOAT | xG | NULL |
| passes_completed | INT | 成功传球 | NULL |
| ppda | FLOAT | PPDA | NULL |
| corners | INT | 角球 | NULL |
| fouls | INT | 犯规 | NULL |
| cards | INT | 牌数 | NULL |
| source_system | STRING | 来源 | NOT NULL |
| ingested_at | TIMESTAMP | 入库时间 | NOT NULL |
| updated_at | TIMESTAMP | 更新时间 | NOT NULL |

#### `odds`

| 字段 | 类型 | 说明 | 约束 |
|---|---|---|---|
| odds_id | STRING | 主键 | PK |
| match_id | STRING | 比赛 | FK matches.match_id |
| bookmaker | STRING | 博彩公司 | NULL |
| market_type | STRING | 盘口类型 | NOT NULL |
| selection | STRING | 选项 | NOT NULL |
| decimal_odds | FLOAT | 欧赔 | NOT NULL |
| implied_prob_raw | FLOAT | 原始隐含概率 | NULL |
| implied_prob_vig_removed | FLOAT | 去水后概率 | NULL |
| snapshot_ts | TIMESTAMP | 抓取时间 | NOT NULL |
| source_system | STRING | 来源 | NOT NULL |
| ingested_at | TIMESTAMP | 入库时间 | NOT NULL |
| updated_at | TIMESTAMP | 更新时间 | NOT NULL |

#### `elo_ratings`

| 字段 | 类型 | 说明 | 约束 |
|---|---|---|---|
| elo_id | STRING | 主键 | PK |
| team_id | STRING | 球队 | FK teams.team_id |
| rating | FLOAT | Elo 值 | NOT NULL |
| rating_date | DATE | 评级日期 | NOT NULL |
| rating_system | STRING | 评级体系 | NOT NULL |
| rank | INT | 排名 | NULL |
| source_system | STRING | 来源 | NOT NULL |
| ingested_at | TIMESTAMP | 入库时间 | NOT NULL |
| updated_at | TIMESTAMP | 更新时间 | NOT NULL |

#### `fifa_rankings`

| 字段 | 类型 | 说明 | 约束 |
|---|---|---|---|
| fifa_ranking_id | STRING | 主键 | PK |
| team_id | STRING | 球队 | FK teams.team_id |
| ranking_date | DATE | 排名日期 | NOT NULL |
| rank | INT | 排名 | NOT NULL |
| points | FLOAT | 积分 | NOT NULL |
| source_system | STRING | 来源 | NOT NULL |
| ingested_at | TIMESTAMP | 入库时间 | NOT NULL |
| updated_at | TIMESTAMP | 更新时间 | NOT NULL |

#### `prediction_runs`

| 字段 | 类型 | 说明 | 约束 |
|---|---|---|---|
| run_id | STRING | 主键 | PK |
| model_name | STRING | 模型名 | NOT NULL |
| model_version | STRING | 版本 | NOT NULL |
| data_cutoff_ts | TIMESTAMP | 数据截断时间 | NOT NULL |
| feature_view_version | STRING | 特征版本 | NOT NULL |
| checkpoint_path | STRING | 模型路径 | NOT NULL |
| calibration_version | STRING | 校准版本 | NULL |
| git_commit | STRING | 提交号 | NULL |
| status | STRING | 状态 | NOT NULL |
| created_at | TIMESTAMP | 创建时间 | NOT NULL |

#### `predictions`

| 字段 | 类型 | 说明 | 约束 |
|---|---|---|---|
| prediction_id | STRING | 主键 | PK |
| run_id | STRING | 预测运行 | FK prediction_runs.run_id |
| match_id | STRING | 比赛 | FK matches.match_id |
| prediction_ts | TIMESTAMP | 预测时间 | NOT NULL |
| score_matrix_json | JSON | 比分矩阵 | NOT NULL |
| top3_json | JSON | top 3 比分 | NOT NULL |
| home_win_prob | FLOAT | 主胜概率 | NOT NULL |
| draw_prob | FLOAT | 平局概率 | NOT NULL |
| away_win_prob | FLOAT | 客胜概率 | NOT NULL |
| over25_prob | FLOAT | 大 2.5 | NOT NULL |
| btts_yes_prob | FLOAT | BTTS 是 | NOT NULL |
| exp_home_goals | FLOAT | 主队期望进球 | NOT NULL |
| exp_away_goals | FLOAT | 客队期望进球 | NOT NULL |
| uncertainty_json | JSON | 不确定性 | NOT NULL |
| created_at | TIMESTAMP | 创建时间 | NOT NULL |

## 4. 特征工程设计

### 4.1 Match context features

- 比赛阶段
- 是否小组赛 / 淘汰赛
- 是否中立场
- 赛地、天气、海拔
- 小组赛积分压力
- 开球时间

### 4.2 Team strength features

- Elo rating
- FIFA ranking / points
- 过去 `N` 场胜平负
- 过去 `N` 场滚动净胜球
- 对手强度修正后的攻防效率

### 4.3 Recent form features

- 最近 3 / 5 / 10 场进球和失球
- 最近 3 / 5 / 10 场 xG / xGA
- 最近若干场射门、射正、控球、PPDA
- 指数衰减形式的近期表现分数

### 4.4 Player availability features

- 预计可出场球员数量
- 核心球员缺席标记
- 首发 11 人平均身价 / 经验
- 最近 30 天分钟负荷

### 4.5 Projected lineup features

- 预计首发概率分布
- 阵型稳定性
- 后防线默契度
- 中轴完整度
- 攻击线平均速度 / 对抗 / 创造力

### 4.6 Tactical matchup features

- 阵型对位
- 边路推进对边后卫防守能力
- 高空球与定位球对位
- 反击倾向与压迫倾向冲突

### 4.7 Betting market features

- 1X2 去水后隐含概率
- O/U 2.5 去水后隐含概率
- BTTS 去水后隐含概率
- correct score 盘口作为校准先验

### 4.8 Fatigue / rest / travel features

- 距上场比赛间隔天数
- 累计出场分钟
- 连续旅行距离
- 时差与恢复窗口

### 4.9 防泄漏原则

- 所有滚动统计以开球时间前数据为准。
- 所有赔率按 `snapshot_ts <= prediction_ts` 过滤。
- 官方首发只用于“官方首发后重跑”的推理版本。

## 5. 模型设计

### 5.1 Baseline：Elo + Dixon-Coles / Poisson

**输入**

- Elo / FIFA
- 主客场 / 中立场
- 近期进失球
- 简单阶段特征

**输出**

- `lambda_home`
- `lambda_away`
- 低比分修正参数
- 导出比分概率矩阵

**参数规模**

- 几百到几千

**训练成本**

- CPU 或单卡数分钟

**优点**

- 简洁、稳定、可解释
- 对数据需求低
- 很适合做长期基准线

**风险**

- 非线性表达有限
- 难以吸收球员级与对位级信号

**适用阶段**

- v0 起步
- 作为所有后续模型的回归基线和 fallback

### 5.2 Mid-level：Tabular model + player aggregation

**输入**

- tabular 比赛特征
- Elo / FIFA / odds
- 近期 form
- 预计首发或大名单球员 embedding 聚合

**输出**

- 比分矩阵 logits 或 goals 分布参数

**参数规模**

- 5M - 20M

**训练成本**

- RTX 4090 上 1 到 6 小时

**优点**

- 性能和复杂度平衡好
- 对球员可用性变化较敏感
- 工程难度可控

**风险**

- 预计首发误差会传导到预测
- 球员数据缺失时需要做好降级

**适用阶段**

- v1 / v2 主力模型

### 5.3 Advanced：Temporal Transformer + Player Set Transformer + Matchup Encoder + Mixture Score Head

**输入**

- 两队最近若干场比赛序列
- 球员级能力与状态
- 对位关系图结构
- 市场赔率与人工先验

**输出**

- 完整比分联合概率分布

**参数规模**

- 20M - 80M

**训练成本**

- RTX 4090 上 8 到 36 小时

**优点**

- 表达能力最强
- 可同时吸收序列、球员和对位信息
- 容易扩展事件数据和实时首发

**风险**

- 复杂度高
- 世界杯样本少，容易过拟合
- 需要稳定的数据基建和更严格的回测

**适用阶段**

- v3 / v4

## 6. 概率输出设计

设比分矩阵为 `P[i, j]`，其中 `i, j ∈ {0..7}`。

### 6.1 Score probability matrix

- 模型直接输出 64 个 logits。
- 经 `softmax` 转为 `P[i, j]`。
- 若存在 `>7` 进球尾部质量，则记录 `overflow_prob`。

### 6.2 Top 3 scorelines

- 将 `P[i, j]` 展平后排序，取概率最高前三项。

### 6.3 Win / Draw / Loss

- `P(home_win) = Σ P[i, j], i > j`
- `P(draw) = Σ P[i, j], i = j`
- `P(away_win) = Σ P[i, j], i < j`

### 6.4 Over / Under 2.5

- `P(over_2_5) = Σ P[i, j], i + j >= 3`
- `P(under_2_5) = 1 - P(over_2_5)`

### 6.5 BTTS

- `P(btts_yes) = Σ P[i, j], i >= 1 and j >= 1`

### 6.6 Expected goals

- `E[home] = Σ_i Σ_j i * P[i, j]`
- `E[away] = Σ_i Σ_j j * P[i, j]`

### 6.7 Uncertainty score

建议组合：

- 分布熵
- ensemble 方差
- OOD 分数

再映射为：

- `confidence = 1 - normalized_uncertainty`

## 7. 损失函数与评估指标

### 7.1 训练损失

建议总损失：

```text
L = w1 * score_nll
  + w2 * goal_marginal_nll_home
  + w3 * goal_marginal_nll_away
  + w4 * ce_1x2
  + w5 * bce_ou25
  + w6 * bce_btts
  + w7 * calibration_penalty
```

包含：

- exact score negative log likelihood
- goals marginal likelihood
- 1X2 cross entropy
- over/under binary loss
- BTTS binary loss
- calibration loss

### 7.2 评估指标

- score NLL
- top-3 scoreline hit rate
- Brier score
- ranked probability score
- calibration curve
- log loss
- backtest by tournament / year

## 8. 训练与回测计划

### 8.1 分阶段训练

1. 数据收集
2. 数据清洗与实体对齐
3. baseline 训练
4. advanced 模型预训练
5. fine-tuning
6. calibration
7. backtesting
8. error analysis

### 8.2 时间切分

- 必须按时间切分训练 / 验证 / 测试。
- 推荐：
  - train：早期国际比赛
  - val：较近时间窗
  - test：更新时间窗或世界杯专项

### 8.3 世界杯专项测试

- 2018 世界杯作为 out-of-time 测试集
- 2022 世界杯作为第二个独立测试集

### 8.4 反泄漏执行细则

- 每场回测都以该场开球前的 `as_of_time` 生成特征。
- 不得使用赛后修正值覆盖赛前记录。
- 使用 official lineup 的预测必须归类为单独版本。

## 9. Dashboard 设计

### 9.1 Match Selector 页面

**展示**

- 比赛列表
- 阶段、日期、球队、是否有官方首发

**API**

- `GET /matches`

**图表 / 交互**

- 筛选表格
- 搜索框

### 9.2 Prediction Summary 页面

**展示**

- top-3 比分
- 1X2
- OU2.5
- BTTS
- 置信度

**API**

- `GET /predictions/{match_id}`

**图表**

- 条形图
- KPI 卡片

### 9.3 Score Matrix Heatmap 页面

**展示**

- `0-7 × 0-7` 比分热力图

**API**

- `GET /score-matrix/{match_id}`

**图表**

- heatmap

### 9.4 Team Comparison 页面

**展示**

- Elo / FIFA
- 近期 form
- 攻防对比

**API**

- `GET /features/{match_id}`

### 9.5 Player Availability / Lineup 页面

**展示**

- 预计首发
- 伤停
- 核心缺席

**API**

- `GET /matches/{match_id}`
- `GET /features/{match_id}`
- `POST /predict`

### 9.6 Feature Importance / Explanation 页面

**展示**

- 关键驱动因子
- 局部解释

**API**

- `GET /explanations/{match_id}`

### 9.7 Backtesting 页面

**展示**

- 模型版本对比
- 年份 / 赛事 / 阶段回测指标

**API**

- `GET /backtest/runs`
- `GET /backtest/{run_id}`

### 9.8 Data Freshness 页面

**展示**

- 各数据源最近刷新时间
- 缺失率
- 异常告警

**API**

- `GET /data/freshness`

## 10. API 设计

### 10.1 `GET /matches`

**响应示例**

```json
{
  "items": [
    {
      "match_id": "wc2026_gs_bra_arg_001",
      "kickoff_ts": "2026-06-30T19:00:00Z",
      "home_team": "Brazil",
      "away_team": "Argentina",
      "stage": "Group",
      "has_official_lineup": false
    }
  ]
}
```

### 10.2 `GET /matches/{match_id}`

```json
{
  "match_id": "wc2026_gs_bra_arg_001",
  "competition": "FIFA World Cup 2026",
  "stage": "Group",
  "kickoff_ts": "2026-06-30T19:00:00Z",
  "home_team_id": "team_bra",
  "away_team_id": "team_arg"
}
```

### 10.3 `POST /predict`

```json
{
  "match_id": "wc2026_gs_bra_arg_001",
  "model_version": "advanced_v2",
  "prediction_mode": "pre_match",
  "use_official_lineup": false,
  "manual_overrides": {
    "home_absent_players": ["player_123"],
    "away_starting_xi": ["p1", "p2", "p3"],
    "notes": "away left back doubtful"
  }
}
```

### 10.4 `GET /predictions/{match_id}`

```json
{
  "match_id": "wc2026_gs_bra_arg_001",
  "run_id": "run_20260628_103015",
  "top3_scorelines": [
    {"home_goals": 1, "away_goals": 1, "prob": 0.146},
    {"home_goals": 2, "away_goals": 1, "prob": 0.118},
    {"home_goals": 1, "away_goals": 0, "prob": 0.097}
  ],
  "result_probs": {
    "home_win": 0.412,
    "draw": 0.301,
    "away_win": 0.287
  },
  "ou25_probs": {
    "over_2_5": 0.471,
    "under_2_5": 0.529
  },
  "btts_probs": {
    "yes": 0.503,
    "no": 0.497
  },
  "expected_goals": {
    "home": 1.38,
    "away": 1.09,
    "total": 2.47
  },
  "uncertainty": {
    "entropy": 3.41,
    "ensemble_var": 0.018,
    "ood_score": 0.11,
    "confidence": 0.68
  }
}
```

### 10.5 `GET /features/{match_id}`

```json
{
  "match_id": "wc2026_gs_bra_arg_001",
  "team_strength": {
    "home_elo": 1985.3,
    "away_elo": 1941.8,
    "home_fifa_rank": 2,
    "away_fifa_rank": 1
  },
  "recent_form": {
    "home_last5_points": 11,
    "away_last5_points": 10
  }
}
```

### 10.6 `GET /score-matrix/{match_id}`

```json
{
  "match_id": "wc2026_gs_bra_arg_001",
  "grid_max_goal": 7,
  "overflow_prob": 0.006,
  "matrix": [
    [0.051, 0.062, 0.041, 0.019, 0.007, 0.002, 0.001, 0.000],
    [0.071, 0.146, 0.083, 0.032, 0.011, 0.003, 0.001, 0.000],
    [0.058, 0.118, 0.076, 0.029, 0.010, 0.003, 0.001, 0.000]
  ]
}
```

### 10.7 `GET /backtest/runs`

```json
{
  "items": [
    {
      "run_id": "bt_20260628_baseline_v1",
      "model_version": "baseline_v1",
      "test_scope": "world_cup_2018",
      "score_nll": 2.41,
      "top3_hit_rate": 0.29
    }
  ]
}
```

### 10.8 `POST /train`

```json
{
  "model_name": "midlevel_ft_transformer",
  "config_name": "midlevel_v1",
  "train_cutoff": "2024-12-31T23:59:59Z"
}
```

### 10.9 `POST /data/refresh`

```json
{
  "sources": ["fifa_rankings", "elo_ratings", "matches"],
  "mode": "incremental"
}
```

## 11. 代码仓库结构

```text
worldcup-predictor/
├─ data/
│  ├─ raw/
│  ├─ curated/
│  ├─ feature_mart/
│  └─ external_mappings/
├─ artifacts/
│  ├─ checkpoints/
│  ├─ backtests/
│  └─ reports/
├─ configs/
│  ├─ data/
│  ├─ features/
│  ├─ models/
│  ├─ training/
│  └─ api/
├─ src/
│  ├─ data_ingestion/
│  ├─ data_validation/
│  ├─ entities/
│  ├─ features/
│  ├─ datasets/
│  ├─ models/
│  │  ├─ baseline/
│  │  ├─ midlevel/
│  │  └─ advanced/
│  ├─ training/
│  ├─ calibration/
│  ├─ backtesting/
│  ├─ inference/
│  ├─ api/
│  ├─ dashboard/
│  └─ utils/
├─ scripts/
│  ├─ ingest/
│  ├─ train/
│  ├─ backtest/
│  └─ serve/
├─ notebooks/
├─ tests/
│  ├─ unit/
│  ├─ integration/
│  └─ regression/
└─ docs/
```

## 12. 技术选型

- 数据存储：`Parquet + DuckDB` 起步，后续补 `PostgreSQL`
- 特征存储：自建 point-in-time feature builder
- 训练框架：`PyTorch 2.x`
- 训练组织：`PyTorch Lightning` 或轻量自写 trainer
- 配置管理：`Hydra`
- 实验跟踪：`MLflow`
- dashboard：`Streamlit`
- API：`FastAPI`
- 解释工具：`SHAP` + attention diagnostics + permutation importance
- 部署：本地 `uvicorn` / `streamlit run`，后续可 `Docker Compose`

## 13. 4090 本地训练配置

- mixed precision：优先 `bf16`
- baseline / tabular batch size：`512-4096`
- mid-level batch size：`64-256`
- advanced batch size：`16-64`
- gradient accumulation：按目标有效 batch 调整
- gradient checkpointing：advanced 默认开启
- `torch.compile`：调通后开启
- 参数规模建议：`20M-80M`
- 推荐起步：`30M-50M`

### 13.1 显存优化建议

- 控制序列长度为 `8-16`
- 控制球员槽位为 `18-26`
- embedding dim 从 `128/192/256` 起步
- attention heads `4-8`

### 13.2 时间估算方法

- 先做 500 step profiling
- 记录平均 step time
- 外推完整 epoch 和总训练时长

## 14. 4-6 周里程碑计划

### Week 1

- 数据源梳理
- baseline 训练
- schema 落地

### Week 2

- 特征工程初版
- dashboard 原型
- 推理 API

### Week 3

- 中阶模型
- player aggregation
- 初步回测

### Week 4

- 高阶模型
- Temporal / Set / Matchup Encoder
- Mixture Score Head

### Week 5

- 校准
- 回测分析
- 解释能力

### Week 6

- 实时更新
- 正式首发重跑
- 文档完善
- 系统打磨

## 15. 风险与替代方案

### 15.1 数据质量不足

**风险**

- 公共数据存在缺失和对齐问题。

**缓解**

- 保留 raw snapshot
- 做实体映射 QA
- 每次 refresh 输出异常报告

### 15.2 球员数据不完整

**风险**

- 高级模型依赖球员层信息，但公共数据未必齐全。

**缓解**

- 模型允许缺失
- 没有球员输入时退化为 team-level 分支

### 15.3 预计首发不准

**风险**

- 错误首发会直接影响预测。

**缓解**

- 用首发概率而非硬首发
- 增加敏感度分析

### 15.4 世界杯样本太少

**风险**

- 单独只用世界杯训练会过拟合。

**缓解**

- 先用所有国际比赛预训练
- 再世界杯微调

### 15.5 赔率授权问题

**风险**

- 商业赔率数据可能存在使用限制。

**缓解**

- 保证系统在无赔率下也能工作
- 将赔率定义为增强层

### 15.6 模型过拟合

**缓解**

- 严格时间切分
- 早停
- 正则化
- 多年份回测

### 15.7 概率未校准

**缓解**

- 单独 calibration 阶段
- 分阶段 / 分赛事类型校准

### 15.8 dashboard 与训练系统耦合过紧

**缓解**

- dashboard 仅调用 API
- 不直接触碰训练内部对象

## 第一阶段最小可行版本 MVP

1. 完成 `teams / players / matches` 基础 schema
2. 接入历史赛果、Elo、FIFA ranking
3. 建立 `Parquet + DuckDB` 数据层
4. 实现 point-in-time 特征构建
5. 实现 baseline：Elo + Dixon-Coles / Poisson
6. 输出比分矩阵、top-3、1X2、OU2.5、BTTS
7. 实现 2018 / 2022 世界杯严格回测
8. 增加 calibration
9. 搭建 FastAPI 预测接口
10. 搭建 Streamlit dashboard 原型

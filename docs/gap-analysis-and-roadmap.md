# 当前短板评估与实施路线图

本文档记录项目相对「2026 赛前决策系统」目标的差距，以及分阶段补齐计划（P0–P3）。

## 架构现状

```text
真实外部数据 (downloads/)     样例数据 (samples/) — 本地开发
         │                              │
         └──────── prepare_data ────────┘
                      │
              staging/canonical/
                      │
                 ingest → curated/
                      │
         ┌────────────┴────────────┐
         │                         │
  match_features.parquet     ScoreGen 训练时另读
  (Elo/FIFA/form/rest        players/lineups/stats/injuries
   + P1 球员/赔率摘要)        + matchup graph（不进 mart 前）
```

## 五大短板（评估基准：2026-06）

### 1. 真实数据接入 — 最大短板

| 维度 | 状态 |
|------|------|
| 接口位 | `prepare_data` / `transformers` / `external.yaml` 已具备 |
| 第一层（赛果/Elo/FIFA） | **P0 已接通**：bootstrap 拉取 Kaggle 国际赛 + FIFA 历史 + Elo 快照/赛果推导 |
| 高阶（球员/阵容/伤停/统计） | **P0 部分真实**：WC 2018/2022 名单来自公开 roster catalog（`seeds/`）；统计仍为赛果派生 |
| 赔率 | **P1 已接通配置位**：canonical odds 走 staging；完整历史博彩盘需第三方 CSV |

`configs/data/external.yaml` 在 P0/P1 后应指向 `data/external/downloads/`，而非仅 `examples/` + `samples/`。

### 2. 主特征 mart 偏传统 — P1 改善

Baseline / Midlevel 原先只消费 Elo、FIFA、form、rest。P1 在 `match_features.parquet` 增加：

- 球员摘要：首发人数、均分、可用率、伤停 OUT 人数、近期 form
- 赔率摘要：1X2 / O/U / BTTS 隐含概率与 `odds_available` 标志

ScoreGen 仍保留完整 7 维球员张量 + 对位图；mart 提供 Dashboard / API / 未来 Midlevel 扩展的统一 tabular 视图。

### 3. ScoreGen / Midlevel 模型偏小 — P2

| 模型 | 当前 | 目标 |
|------|------|------|
| ScoreGen | `d_model=64`, 2 层, ~10⁵ 参数 | 4090 profile：50M–100M 级 |
| Midlevel | MLP `[64, 32]` | 随 mart 列扩展后再放大 |

结构验证已完成；放大属于 P2，不阻塞 P0/P1。

### 4. Dashboard 仍是 MVP — P2

现有 Streamlit：预测摘要、矩阵、Features JSON、回测列表、freshness。

缺失：球队对比、球员可用性、伤停时间线、对位图、模型解释、赛前 checklist。依赖 P1 API 字段扩展后再做 UI。

### 5. 测试 — 可本地验证

仓库含 `tests/unit` + `tests/integration`。本地执行：

```bash
pytest
```

最近一次：**50 passed**（含 ingest、feature、ScoreGen smoke、API）。

---

## 优先级路线图

### P0 — 真实第一层 + 球员种子（本阶段实施）

1. **`scripts/bootstrap_external_downloads.py`**
   - 下载 [martj42/international_results](https://github.com/martj42/international_results) 全量赛果
   - 下载 [Dato-Futbol/fifa-ranking](https://github.com/Dato-Futbol/fifa-ranking) FIFA 历史
   - 抓取 [eloratings.net World.tsv](https://www.eloratings.net/World.tsv) 当前 Elo 快照
   - 由赛果推导历史 Elo 时间序列（PIT join）
   - 将 `data/external/seeds/` 中 WC  roster 复制到 `downloads/`

2. **`scripts/export_external_seeds.py`**
   - 从 `world_cup_catalog` + `world_cup_squads` 导出可提交的 WC 种子 CSV

3. **更新 `configs/data/external.yaml`**
   - 全部 `external_inputs` 指向 `downloads/`
   - `include_samples: false`（全外部模式）

4. **文档**：`data/external/README.md`、`data-sources.md` 同步

### P1 — 统一高阶 mart + 赔率（本阶段实施）

1. **`player_match_features.py`**：球员/伤停 PIT 摘要
2. **`builder.py`**：合并球员摘要 + 赔率进 mart
3. **`prepare.py`**：增加 `odds` 合并；`sources.odds` 走 staging
4. **API `/features`**：返回 `player_summary` 与 `market_odds` 块
5. **`transform_football_data_odds`**：支持 football-data.co.uk 格式（可选第三方 CSV）

### P2 — 大模型 + Dashboard 工作台

- `configs/models/scoregen_4090.yaml`
- Dashboard 多页：球队对比、伤停、对位、解释
- Midlevel 使用扩展 mart 列

### P3 — 事件级数据

- xG、射门、卡牌等新表
- 新预测 head 或 aux loss

---

## 推荐操作顺序

```bash
# 1. 生成并提交 WC 种子（一次性，仓库内）
python -m scripts.export_external_seeds

# 2. 下载真实第一层 + 复制种子到 downloads/
python -m scripts.bootstrap_external_downloads

# 3. 准备 canonical → ingest → features
python -m scripts.prepare_data --config-name=config data=external
python -m scripts.ingest --config-name=config data=external
python -m scripts.build_features --config-name=config data=external

# 4. 验证
pytest
python -m scripts.check_scoregen_coverage
```

全外部模式（不含 samples）：

```bash
python -m scripts.prepare_data data=external data.include_samples=false
```

---

## 数据源说明（P0 后）

| 表 | 来源 | 真实度 |
|----|------|--------|
| matches | Kaggle 国际赛 + WC catalog 种子 | 赛果真实 |
| elo_ratings | 赛果推导 + eloratings.net 快照 | 推导 + 快照 |
| fifa_rankings | Dato-Futbol 历史 CSV | 真实历史 |
| players / lineups | WC 2018/2022 roster catalog | 名单真实；非全职业生涯库 |
| player_match_stats | 由 WC 赛果分配进球 | 场次真实，进球归属简化 |
| injuries | 种子中的公开伤停案例 | 部分真实 |
| odds | WC 种子赔率 / football-data 格式 | 结构真实；历史盘需自备 CSV |

完整生产仍需要：Transfermarkt / FBref / API-Football / 博彩 API 等（见 `data-sources.md`）。

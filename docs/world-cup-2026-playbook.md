# 2026 世界杯预测：数据准备与赛中调参手册

本文档面向 **2026 年世界杯（美国 / 加拿大 / 墨西哥）** 的实战部署，说明需要准备哪些数据、何时更新、以及如何在赛中根据已结束比赛做调参，同时保证 **可复现、可回滚、可写进事后报告**。

与 [`constraints.md`](./constraints.md) 的关系：本文是操作层 playbook；所有数据与调参仍须遵守该文档中的 **90 分钟口径、PIT（Point-in-Time）反泄漏、矩阵唯一输出源** 等硬约束。

---

## 1. 总体原则

### 1.1 预测对象不变

- 赛前预测目标始终是 **90 分钟常规时间比分联合分布**（8×8 矩阵及衍生 1X2 / O/U / BTTS）。
- 2026 若扩军（48 队），数据与特征逻辑不变，仅实体规模变大；需提前扩展 `team_aliases` 与 squad 覆盖。

### 1.2 三条时间线

| 阶段 | 时间范围 | 目标 |
|------|----------|------|
| **赛前基线** | 开赛前数周～数天 | 固定一版「赛前模型 + 校准 + 特征 mart」，用于对外承诺的基准预测 |
| **赛中滚动** | 每场比赛结束后 | 用**已结束**比赛更新数据与可选调参，生成**下一批**赛前预测 |
| **赛后归档** | 整届结束 | 冻结 artifact 与预测台账，供报告与审计 |

### 1.3 赛中调参 ≠ 用未来信息

- 「根据实时比赛结果调参」仅指：在某场 **开球时刻 `kickoff_ts` 之前** 已知的赛果与统计，用于 **后续场次** 的特征与（可选）校准。
- **禁止** 用 A 场赛后信息去改 A 场开赛前已存档的预测；已发布预测只能标记为 superseded，不能静默覆盖。

---

## 2. 2026 需要准备的数据清单

按项目数据分层（`raw → curated → feature_mart`）与能力等级（[`constraints.md` §3.2](./constraints.md) v0～v4）组织。

### 2.1 核心表（所有模型必需）

| 表 / 文件 | 用途 | 2026 准备要点 | 建议来源 |
|-----------|------|---------------|----------|
| **matches** | 训练标签、球队关系、赛程 | ① 2026 官方 104 场赛程（含 `kickoff_ts`、阶段、是否淘汰赛）<br>② 2022-01-01 ～ 2026 开赛前所有相关国际赛（含洲际赛、友谊赛）<br>③ 已结束场次及时写入 **90 分钟** 比分；加时/点球单独字段 | FIFA 官网、Kaggle International Results、自建 CSV |
| **teams** | 实体对齐 | 48 支参赛队 + 预选赛对手；维护 `team_aliases.csv` | 与 matches 同步生成 |
| **elo_ratings** | 实力先验 | 至少两个快照：**2026 开赛前最近一期** + 历史滚动（按 `rating_date` PIT join） | eloratings.net、自建 Elo |
| **fifa_rankings** | 实力先验 | 同上，按 `ranking_date` 快照，禁止只用「最新排名」 | FIFA 排名发布 CSV |

**最低配置（Baseline Dixon-Coles）**：matches + teams + elo + fifa 即可跑通训练与预测。

### 2.2 ScoreGen / 球员图模型额外需要（v2，推荐）

| 表 / 文件 | 用途 | 2026 准备要点 |
|-----------|------|---------------|
| **players** | 球员静态属性 | 48 队大名单（或 26 人名单）：`player_id`、国籍、`primary_position`、可选 market_value / rating |
| **lineups** | 图编码输入 | ① **historical**：已结束比赛的实际首发（赛后录入）<br>② **projected**：下一场赛前预计首发，必须带 `snapshot_ts`（如赛前 12h）<br>③ 开赛前官方名单确认后，用 **official** 优先于 projected |
| **player_match_stats** | PIT 状态 / form | 每场已结束比赛的分钟、进球、助攻（可逐步扩展 xG、射门） |
| **injuries** | 可用性 | `status`（out / doubtful / fit）、`start_date`、`expected_return_date`、`confidence` |
| **odds**（可选 v1+） | 市场先验 | 每场 **`snapshot_ts` 早于 kickoff** 的 1X2 / O/U / BTTS 欧赔；赛中更新仅用于**后续场次**特征 |

模板见 `data/templates/*.template.csv`；完整样例生成可参考 `python -m scripts.export_sample_data`（2018/2022 结构示范）。

### 2.3 增强数据（v3 / v4，可选）

| 类型 | 价值 | 注意 |
|------|------|------|
| 事件 / xG（StatsBomb、Opta） | 提升 form 与 matchup 质量 | 需独立 adapter；仍按比赛结束时间 PIT 入库 |
| 官方 live 首发 API | 开赛前 1h 刷新 projected → official | 每次刷新保留 snapshot，不覆盖旧 snapshot |
| 裁判 / 天气 /  travel | 边际特征 | 需记录 `source_system` 与抓取时间 |

### 2.4 按时间节点的数据检查表

**开赛前 4～8 周**

- [ ] 2026 赛程 CSV 入库（赛果列空，`status=scheduled`）
- [ ] 48 队 `team_aliases` 定稿
- [ ] 2022 至今国际赛 matches 补全
- [ ] Elo / FIFA 最新快照
- [ ] 各队 players 大名单（可先用 club 赛季汇总）

**开赛前 24～72 小时（单场）**

- [ ] 该场 projected lineups + `snapshot_ts`
- [ ] 伤停表更新至 `as_of_time = 预测时刻`
- [ ] 赔率 snapshot（若使用）
- [ ] `build_features` 生成该场 PIT 特征行

**比赛结束后（用于后续场次）**

- [ ] matches 写入 **FT** 比分（90 分钟）
- [ ] lineups → historical
- [ ] player_match_stats 追加
- [ ] 可选：触发下一阶段的 ingest / features / 校准

**2026 扩军注意**

- 新军可能无历史 Elo：需预选赛赛果或手动初值，并在文档中记录初值假设。
- 样本外球队多 → Baseline 与 ScoreGen 都更依赖 **近期国际赛** 与 **赔率** 兜底。

---

## 3. 赛中「实时调参」应调什么

区分四层，避免混为一谈：

| 层级 | 内容 | 频率 | 风险 |
|------|------|------|------|
| **L0 数据刷新** | ingest 新赛果 → rebuild feature_mart | 每比赛日后 | 低（须 PIT 正确） |
| **L1 校准（Calibration）** | 温度 / λ 缩放，**不改** 网络权重 | 每 3～5 场或每个阶段 | 低，推荐赛中首选 |
| **L2 增量训练** | 用新数据 fine-tune ScoreGen | 仅阶段间隙（如小组赛结束） | 中（需 GPU、验证） |
| **L3 全量重训** | 从头训练 | 赛前一次 + 赛后复盘 | 高，不作为赛中默认 |

**推荐赛中策略**

1. **赛前冻结**一版权重 checkpoint（`wc2026_pre_tournament_v1`）。
2. 赛中只做 **L0 + L1**：新赛果入库后，在 **validation 窗口**（如已结束的 2026 小组赛）上重新拟合 calibration temperature，写入 **新 calibration 版本**，权重 JSON 不变。
3. L2 仅在数据漂移明显且有人力/GPU 时启用，且必须产出 **新 checkpoint 版本**，不得覆盖旧文件。

当前项目已支持：`scripts.calibrate` 对 baseline / midlevel / scoregen 做 temperature 缩放，且矩阵仍为唯一概率源（见 [`constraints.md` §2](./constraints.md)）。

---

## 4. 可回滚与可审计架构（不写代码的操作约定）

目标：任意一场对外公布的预测，事后能回答——**用的哪版权重、哪版校准、哪份特征、哪时刻的 as_of_time、输入数据快照 hash**。

### 4.1 不可变 Artifact 命名

约定所有产物 **只追加、不覆盖**：

```text
artifacts/
├── checkpoints/
│   └── scoregen_football_wc2026_pre_v1.json          # 权重 + 训练元数据
├── calibrations/
│   └── scoregen_football_wc2026_pre_v1_cal_20260625.json   # 仅 calibration 参数
├── predictions/                                        # 预测台账（建议新建）
│   └── wc2026/
│       └── 2026-06-25T18:00:00Z_wc2026_arg_mex_gs/
│           ├── manifest.json
│           ├── prediction.json
│           └── feature_row.parquet
├── data_snapshots/                                     # 可选：赛前 curated 冻结
│   └── wc2026_round1_end_20260628/
│       ├── matches.parquet
│       └── feature_mart.parquet
└── backtests/
    └── backtest_wc2026_live_val_20260628.json
```

**版本 ID 建议**：`{model_name}_{tournament}_{phase}_{date}`，例如 `scoregen_wc2026_group_stage_cal_20260628`。

### 4.2 预测台账（Prediction Manifest）

每场在 **首次对外发布预测前** 写一份 manifest（JSON 即可），最小字段：

| 字段 | 说明 |
|------|------|
| `prediction_id` | 唯一 ID，如 `{match_id}@{as_of_time}` |
| `match_id` | 比赛 ID |
| `as_of_time` | 特征与伤停/首发的截止时间（ISO8601 UTC） |
| `model_type` | baseline / midlevel / scoregen |
| `checkpoint_path` | 权重文件路径 + SHA256 |
| `calibration_path` | 校准文件路径 + temperature / lambda_scale |
| `feature_mart_path` | 特征行来源（文件 + row hash） |
| `data_snapshot_id` | 可选，指向 `data_snapshots/` |
| `config_hash` | Hydra 解析后配置的 hash |
| `git_commit` | 代码版本（若可获取） |
| `superseded_by` | 若后续更新预测，指向新 `prediction_id`；**不删除旧记录** |

API 或 Dashboard 展示时，应能显示 `prediction_id` 与 `as_of_time`，便于与赛后报告对照。

### 4.3 赛中更新流程（标准作业）

```text
[比赛结束]
    → ingest 赛果 + stats + historical lineups
    → build_features（全量或增量）
    → （可选）在「已结束 2026 比赛」上 calibrate → 新 cal_YYYYMMDD 文件
    → 对「未开赛比赛」生成新 prediction + 新 manifest
    → 旧 prediction 标记 superseded_by，保留文件

[回滚]
    → serving 层切换 checkpoint_path + calibration_path 指针（环境变量或 registry 配置）
    → 不删除新文件；报告可同时引用 v2 与回滚后的 v1 预测差异
```

**Serving 指针示例（概念）**

- `WORLDCUP_CHECKPOINT=artifacts/checkpoints/scoregen_football_wc2026_pre_v1.json`
- `WORLDCUP_CALIBRATION=artifacts/calibrations/scoregen_wc2026_cal_20260625.json`

回滚 = 改回上一组指针 + 重启 API，无需改历史 manifest。

### 4.4 与 Git 的分工

| 内容 | 是否进 Git | 说明 |
|------|------------|------|
| `team_aliases.csv`、赛程 template、Hydra config | 是 | 小文件、需 code review |
| `data/samples` 或 export 脚本 | 是 | 结构示范 |
| `data/curated`、`feature_mart`、parquet | 否 | 体积大，用 `data_snapshots/` + 对象存储或本地归档 |
| `artifacts/checkpoints` | 否（默认） | 用版本目录 + manifest 引用；发布报告时附 SHA256 |
| `artifacts/predictions/` | 视合规要求 | 报告必需则归档到独立存储，Git LFS 或 S3 |

### 4.5 报告引用规范

事后报告建议固定引用四元组：

1. **代码**：git commit
2. **权重**：checkpoint 文件名 + hash
3. **校准**：calibration 文件名 + temperature
4. **预测**：`prediction_id` 列表（按比赛日）

示例表述：「阿根廷 vs 墨西哥（2026-06-XX）赛前预测取自 `prediction_id=wc2026_arg_mex_gs@2026-06-XXT12:00:00Z`，模型 scoregen_wc2026_pre_v1，校准 cal_20260625（T=0.82）。」

---

## 5. 2026 赛前推荐工作流（汇总）

```text
1. 数据
   prepare_data / ingest（2022–2026 国际赛 + 2026 赛程）
   → export 或维护 48 队 players / aliases
   → build_features

2. 建模（开赛前一次）
   train（scoregen 或 baseline）→ checkpoint wc2026_pre_v1
   calibrate（2022 WC hold-out 或 2026 赛前友谊赛）→ cal_pre_v1

3. 预测发布
   每场赛前：指定 as_of_time → predict → 写 manifest + prediction.json

4. 赛中循环
   赛后 ingest → build_features
   → 可选 calibrate（仅新 cal 文件）→ 更新 serving 指针
   → 对未赛比赛重新 predict → 新 manifest，旧标记 superseded

5. 赛后
   冻结 data_snapshots/wc2026_final
   导出 backtest 与 prediction 台账 → 写报告
```

---

## 6. 风险与常见错误

| 错误 | 后果 | 预防 |
|------|------|------|
| 用赛后 Elo 回算赛前特征 | 回测与实盘虚高 | 所有 join 带 `rating_date <= as_of_time` |
| 覆盖同一 prediction_id | 无法写报告 | 只追加 manifest，用 superseded_by |
| 赛中 full retrain 无验证 | 模型漂移不可控 | 默认只做 L0+L1 |
| 2026 赛程队名不一致 | ingest 失败 | 提前维护 aliases |
| projected 首发无 snapshot_ts | PIT 测试失效 | 每条 projected 必填 snapshot |

---

## 7. 相关文档

- [`constraints.md`](./constraints.md) — 口径、PIT、矩阵唯一输出
- [`data-sources.md`](./data-sources.md) — CSV 列定义与 ingest 路径
- [`system-design.md`](./system-design.md) — 架构、registry、API
- [`data/external/README.md`](../data/external/README.md) — 外部 CSV 放置方式

---

## 8. 当前项目差距（实施前须知）

以下为 playbook 要求、但 MVP 代码**尚未完整自动化**的部分，2026 前可按优先级补齐：

1. **预测台账目录** `artifacts/predictions/` 与 manifest 写入（目前 API 预测未强制落盘）。
2. **独立 calibration 文件**（当前 temperature 写在 checkpoint JSON 内；赛中版本化建议拆文件或复制 checkpoint 副本）。
3. **data_snapshots 冻结脚本**（手动拷贝 curated / feature_mart 亦可）。
4. **2026 赛程 catalog**（可仿 `world_cup_catalog.py` 新增 `WORLD_CUP_2026`）。
5. **赛中 incremental calibrate** 仅针对「已结束 2026 场」验证集的自动化 CLI。

以上不影响本文作为 **数据准备与流程设计** 的基准；实现时可按 MVP 任务单独立排期。

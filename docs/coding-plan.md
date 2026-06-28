# MVP 初步编码计划

本文档在 [MVP 阶段任务清单](./mvp-task-list.md) 基础上，给出更具体的编码阶段划分、目录结构、执行环境与可勾选 Todo。实施前请先阅读 [工程约束与边界文档](./constraints.md)。

## 1. 总体策略

**目标**：在 v0 数据（公开赛果 + Elo + FIFA ranking + 少量手工输入）下，跑通以下闭环：

```text
数据接入 → point-in-time 特征 → baseline 模型 → 统一比分矩阵 → API → Dashboard → 2018/2022 回测
```

**三条硬约束贯穿全程**：

1. 所有衍生概率（1X2 / OU2.5 / BTTS / top-3）必须从 **8×8 比分矩阵聚合**，不得单独训练分类头覆盖矩阵结果。
2. 特征与回测必须绑定 `as_of_time`，禁止赛后信息泄漏。
3. **训练 / 回测 / GPU 任务在实验室 RTX 4090 主机执行**；开发机（Cursor 本地）只做编码、静态检查、不依赖 GPU 的单测。

**MVP 技术栈**：

- Python 3.11+
- PyTorch 2.x
- Hydra（配置管理）
- Parquet + DuckDB（离线数据层）
- FastAPI（推理服务）
- Streamlit（Dashboard 原型）
- MLflow（实验跟踪，可在 Phase 2 训练跑通后再接入）

## 2. 阶段划分（约 4 周 MVP）

```text
Phase 0 脚手架 → Phase 1 数据层 → Phase 2 特征 + Baseline
    → Phase 3 Serving → Phase 4 回测 + 校准 → Phase 5 Dashboard
```

| 阶段 | 周期 | 产出 | 执行环境 |
|------|------|------|----------|
| Phase 0 脚手架 | 1–2 天 | 目录结构、依赖、配置骨架、测试框架 | 本地 |
| Phase 1 数据层 | 3–5 天 | raw/curated Parquet、实体表、首批数据 | 本地写脚本，实验室跑全量 |
| Phase 2 特征 + Baseline | 5–7 天 | PIT 特征、Dixon-Coles、矩阵解码 | 实验室 RTX 4090 |
| Phase 3 Serving | 2–3 天 | FastAPI 5 个核心端点 | 本地 / 实验室均可 |
| Phase 4 回测 + 校准 | 3–4 天 | 2018/2022 回测报告、calibration | 实验室 RTX 4090 |
| Phase 5 Dashboard | 2–3 天 | Streamlit 三页 MVP | 本地（连 API） |

## 3. 目标目录结构

```text
WorldCup/
├─ configs/              # Hydra: data / features / models / training / api
├─ data/
│  ├─ raw/
│  ├─ curated/
│  ├─ feature_mart/
│  └─ external_mappings/
├─ artifacts/
│  ├─ checkpoints/
│  ├─ backtests/
│  └─ reports/
├─ src/
│  ├─ data_ingestion/
│  ├─ entities/          # Pydantic / dataclass schema
│  ├─ features/
│  ├─ models/
│  │  └─ baseline/
│  ├─ training/
│  ├─ inference/
│  ├─ api/
│  ├─ dashboard/
│  └─ utils/
├─ scripts/              # ingest / train / backtest / serve
└─ tests/
   ├─ unit/
   └─ integration/
```

## 4. Phase 0：工程脚手架

**目标**：可安装、可测试、模块边界清晰。

### Todo

- [ ] 初始化 `pyproject.toml`（或 `requirements.txt`）与 `.gitignore`
- [ ] 建立上述目录与空 `__init__.py`
- [ ] 添加 `configs/` 骨架（`data/default.yaml`, `models/baseline.yaml`, `training/default.yaml`）
- [ ] 定义核心实体 schema：`Team`, `Player`, `Match`（含 `source_system`, `ingested_at` 等追溯字段）
- [ ] 编写 `scripts/` 入口：`ingest`, `build_features`, `train`, `backtest`, `serve`
- [ ] 配置 `pytest` + `ruff`（本地可跑的静态检查与单测框架）

**执行环境**：本地

## 5. Phase 1：数据层（v0 数据源）

**目标**：三层数据（raw → curated → feature_mart 目录就绪），能查询历史赛果与 Elo/FIFA。

### 数据源建议（MVP）

| 数据 | 建议来源 | 优先级 |
|------|----------|--------|
| 国际赛 / 世界杯赛果 | 公开 CSV（如 Kaggle / GitHub 国际赛数据集） | P0 |
| Elo | clubelo / eloratings 公开快照 | P0 |
| FIFA ranking | FIFA 官网历史排名 CSV | P0 |
| 手工伤停 / 预计首发 | 本地 CSV + override 字段 | P1（schema 先定，数据可后填） |

### Todo

- [ ] 实现 `data_ingestion/base.py`：统一 `source_system`, `ingested_at`, `source_record_id`
- [ ] 实现 CSV/JSON 导入器 → `data/raw/`
- [ ] 实现 curated 转换：`teams`, `players`, `matches` Parquet
- [ ] 建立 `data/external_mappings/team_aliases.csv` 实体对齐表
- [ ] 实现 Elo / FIFA ranking 快照表（含 `rating_date` / `ranking_date`）
- [ ] 设计手工表 schema：`injuries`, `lineups`（含 override 审计字段）
- [ ] 用 DuckDB 写集成测试：表关联、行数、主键唯一性
- [ ] 输出首份数据 QA 报告（缺失率、世界杯场次覆盖）

### 实验室验证命令（示例）

```bash
conda activate worldcup
cd /path/to/WorldCup
python -m scripts.ingest --config configs/data/default.yaml
python -m scripts.validate_data
```

## 6. Phase 2：特征 + Baseline 模型

**目标**：严格 PIT 特征 + Dixon-Coles baseline，输出 8×8 矩阵及全部衍生概率。

### 6.1 特征层 Todo

- [ ] 实现 `features/point_in_time.py`：`as_of_time` / `kickoff_ts` 截止 join
- [ ] Match context：`stage_type`, `is_knockout`, `is_world_cup`, 主客场
- [ ] Team strength：开赛前最近 Elo、FIFA rank/points
- [ ] Recent form：滚动进失球（3/5/10 场，仅 cutoff 前比赛）
- [ ] Fatigue/rest：距上场间隔天数
- [ ] 生成 `feature_mart/match_features.parquet`（每行 = 一场比赛 + `as_of_time`）
- [ ] **反泄漏单测**：构造「赛后才出现的数据」，断言特征构建器读不到

### 6.2 模型层 Todo

- [ ] 实现 `models/baseline/dixon_coles.py`（λ_home, λ_away, ρ 低比分修正）
- [ ] 实现 `models/score_matrix.py`：Poisson 网格 → 8×8 截断 + `overflow_prob`
- [ ] 实现 `inference/decoder.py`：矩阵 → top-3 / 1X2 / OU2.5 / BTTS / xG / entropy
- [ ] 实现 `training/trainer.py`：时间切分 train/val（禁止 random split）
- [ ] 实现 `models/registry.py`：checkpoint 路径与版本元数据
- [ ] 训练 baseline 并保存 checkpoint 到 `artifacts/checkpoints/`

### 实验室验证命令（示例）

```bash
python -m scripts.build_features --cutoff 2024-12-31
python -m scripts.train --config configs/models/baseline.yaml
python -m scripts.predict --match-id <id>   # smoke test
```

**执行环境**：实验室 RTX 4090

## 7. Phase 3：FastAPI Serving

**目标**：Dashboard 只调 API，不碰训练内部对象。

### Todo

- [ ] 实现 Pydantic 响应模型（对齐 [system-design.md](./system-design.md) §10 JSON 结构）
- [ ] `GET /matches` — 比赛列表 + 筛选
- [ ] `GET /matches/{match_id}` — 比赛详情
- [ ] `POST /predict` — 加载 checkpoint + 特征组装 + 返回完整预测
- [ ] `GET /predictions/{match_id}` — 最近一次预测缓存
- [ ] `GET /score-matrix/{match_id}` — 8×8 矩阵 + overflow
- [ ] 推理路径复用 `inference/decoder.py`（与训练同一套聚合逻辑）
- [ ] API 集成测试（TestClient，mock checkpoint）

### 启动命令（示例）

```bash
uvicorn src.api.main:app --host 0.0.0.0 --port 8000
```

**执行环境**：本地 / 实验室均可

## 8. Phase 4：回测 + 校准

**目标**：2018 / 2022 世界杯独立 out-of-time 测试，产出可对比指标。

### Todo

- [ ] 实现 `backtesting/runner.py`：逐场 `as_of_time = kickoff_ts - ε`
- [ ] 2018 世界杯回测脚本
- [ ] 2022 世界杯回测脚本
- [ ] 指标：score NLL、top-3 hit rate、1X2/OU2.5/BTTS Brier、RPS、ECE
- [ ] 实现 `calibration/`：温度缩放或 isotonic（矩阵边际校准）
- [ ] 输出 `artifacts/backtests/` 报告（JSON + Markdown）
- [ ] 反泄漏验收：回测日志记录每场使用的特征 cutoff

### 实验室验证命令（示例）

```bash
python -m scripts.backtest --test-set world_cup_2018 --model baseline_v1
python -m scripts.backtest --test-set world_cup_2022 --model baseline_v1
python -m scripts.calibrate --model baseline_v1
```

**执行环境**：实验室 RTX 4090

## 9. Phase 5：Streamlit Dashboard

**目标**：MVP 三页，全部通过 API 访问数据。

### Todo

- [ ] Match Selector：调用 `GET /matches`，筛选 + 搜索
- [ ] Prediction Summary：top-3、1X2、OU2.5、BTTS、confidence KPI 卡片
- [ ] Score Matrix Heatmap：`0-7 × 0-7` 热力图 + overflow 提示
- [ ] 展示数据 freshness（各表 `updated_at`）
- [ ] 手工 override 版预测标注（MVP 可先做 POST /predict 按钮）

### 启动命令（示例）

```bash
streamlit run src/dashboard/app.py
```

**执行环境**：本地（连接 API）

## 10. 工程治理（贯穿各阶段）

- [ ] Hydra 配置与 CLI 入口统一
- [ ] MLflow 实验跟踪（Phase 2 训练跑通后再接）
- [ ] 根目录 `README.md`：环境安装、实验室运行步骤、API 启动
- [ ] 每新增数据源 / 特征组更新 `docs/` 对应章节

## 11. 第一 Sprint 建议（优先开工）

按 MVP 建议顺序：**数据 → baseline → API → dashboard → 回测**。

| 顺序 | 任务 | 环境 |
|------|------|------|
| 1 | Phase 0 脚手架 + 实体 schema | 本地 |
| 2 | 导入历史赛果 + Elo + FIFA → curated Parquet | 本地写脚本，实验室跑全量 |
| 3 | PIT 特征 builder + 反泄漏单测 | 本地单测，实验室跑 feature build |
| 4 | Dixon-Coles + score matrix decoder | 实验室训练 |
| 5 | FastAPI `/predict` + `/score-matrix` | 本地 |
| 6 | Streamlit 三页原型 | 本地 |
| 7 | 2018/2022 回测 + calibration | 实验室 |

## 12. MVP 验收 Checklist

- [ ] 指定比赛可生成有效预测
- [ ] 输出统一 8×8 矩阵及全部衍生概率
- [ ] Dashboard 可交互查看，且与 API 数值一致
- [ ] 2018 / 2022 世界杯严格回测可复现
- [ ] 有测试/日志证明未使用赛后信息

## 13. MVP 范围外（刻意不做）

- Mid-level / Advanced 模型（v1+ 再做）
- 实时赔率流、Graph Transformer
- PostgreSQL（先用 Parquet + DuckDB）
- 多卡训练、生产级调度

## 14. 相关文档

- [工程约束与边界文档](./constraints.md) — 预测口径、反泄漏、执行环境
- [系统设计总文档](./system-design.md) — 架构、数据模型、API 契约
- [MVP 阶段任务清单](./mvp-task-list.md) — 高层任务拆解与验收标准

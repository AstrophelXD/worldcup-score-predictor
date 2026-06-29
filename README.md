# ⚽ WorldCup Predictor

> 世界杯指定场次 **90 分钟常规时间** 比分预测系统 — 从数据接入、特征构建、模型训练到 API 与 Dashboard 的完整 MVP 流水线。

![Python](https://img.shields.io/badge/Python-3.11+-3776AB?logo=python&logoColor=white)
![FastAPI](https://img.shields.io/badge/FastAPI-009688?logo=fastapi&logoColor=white)
![PyTorch](https://img.shields.io/badge/PyTorch-EE4C2C?logo=pytorch&logoColor=white)
![Hydra](https://img.shields.io/badge/Hydra-89b8cd?logo=python&logoColor=white)
![Streamlit](https://img.shields.io/badge/Streamlit-FF4B4B?logo=streamlit&logoColor=white)
![Status](https://img.shields.io/badge/Status-MVP-blue)
![Tests](https://img.shields.io/badge/Tests-pytest-0A9EDC?logo=pytest&logoColor=white)
![License](https://img.shields.io/badge/License-MIT-green)

---

## ✨ 特性

| 能力 | 说明 |
|------|------|
| 🎯 **比分联合分布** | 输出 `0–7 × 0–7` 比分概率矩阵，所有衍生指标从同一分布聚合 |
| 📊 **多维度预测** | Top-3 比分、胜平负、O/U 2.5、BTTS、期望进球、不确定性 |
| 🛡️ **严格 PIT 特征** | Point-in-time 特征构造，避免赛后信息泄漏 |
| 📈 **Dixon-Coles Baseline** | MLE 训练的经典泊松比分模型，支持 2018 / 2022 世界杯回测 |
| 🚀 **FastAPI + Streamlit** | REST API Serving 与可视化 Dashboard |
| ⚙️ **Hydra 配置驱动** | 数据、训练、回测均可通过 YAML 灵活覆盖 |

---

## 🧠 模型架构与原理

系统采用 **「统一比分联合分布」** 设计：无论哪条模型线，最终都输出 `0–7 × 0–7` 的比分概率矩阵；胜平负、O/U 2.5、BTTS、Top-3 比分等全部从该矩阵聚合，避免多头模型口径不一致。

特征在 **point-in-time（PIT）** 约束下构造——只使用开球时刻之前可见的数据（Elo、FIFA 排名、近期赛果、预计首发、伤停等），回测与 serving 共用同一套特征视图。

### 三层模型

| 层级 | 配置名 | 原理 | 典型用途 |
|------|--------|------|----------|
| **Baseline** | `baseline_dixon_coles` | 经典 **Dixon-Coles** 泊松比分模型：MLE 学习每队 attack/defense + 主场优势；低比分用 `ρ` 修正；独立泊松生成矩阵后做 DC 调整 | 默认一键启动、快速回测、无 GPU 可跑 |
| **Mid-level** | `midlevel_tabular` | **MLP** 直接编码 tabular 特征 → `64` 维 logits → softmax 为 `8×8` 矩阵；辅以辅助损失对齐 1X2 | 更强 tabular 表达，训练成本低 |
| **ScoreGen** | `scoregen_football` | **多模态 Transformer**：tabular + 近期赛果序列 + 球员图（队内 attention + 对位边）+ 赔率 → Match Context Transformer → **Dixon-Coles 混合双变量分布头**（非 flat 64 类分类） | 当前主力高级模型，融合球员/伤停/赔率 |

### Baseline：Dixon-Coles

对主队期望进球 λ_home、客队 λ_away：

```text
log λ_home = home_advantage + attack_home − defense_away
log λ_away = attack_away − defense_home
```

在 `(home_goals, away_goals)` 上建立独立泊松，并对 `0-0 / 1-0 / 0-1 / 1-1` 施加 Dixon-Coles 低比分修正因子 τ。训练用 **scipy L-BFGS-B** 极大似然（CPU），checkpoint 为 JSON（球队攻防参数字典）。

### ScoreGen-Football：多模态 → 混合分布头

```text
Tabular 特征 ──┐
主/客队近期序列 ──┼──► Match Context Transformer (CLS) ──► 混合 DC 分布头 ──► 8×8 矩阵
球员图 + 对位边 ──┤                                      (K 个分量 × λ_h, λ_a, ρ)
市场赔率 ────────┘
```

- **序列编码器**：各队最近 N 场 embedding，Transformer 聚合为球队 token  
- **球员图编码器**：预计首发/伤停球员特征 → 队内 self-attention → 跨队对位 message passing  
- **分布头**：输出 K 个 Dixon-Coles 分量及混合权重，矩阵由分量解析生成（保持比分结构先验，而非无约束 64-way 分类）  
- **辅助头**：1X2 / 事件类辅助损失仅用于训练，serving 仍以矩阵为唯一概率源  

Serving 默认模型由环境变量 `WORLDCUP_MODEL` 控制（见 `.env.example`）；一键启动脚本缺 checkpoint 时会训练 **Baseline**。

### 校准

`scripts.calibrate` 在验证集上拟合 **temperature / lambda 缩放**，降低概率校准误差；缩放后仍从同一矩阵解码，不改变「矩阵为唯一输出源」的约束。

更完整的设计说明见 [`docs/system-design.md`](docs/system-design.md) 与 [`docs/constraints.md`](docs/constraints.md)。

---

## 🚀 快速开始

### 1️⃣ 安装

```bash
git clone https://github.com/AstrophelXD/worldcup-score-predictor.git
cd worldcup-score-predictor

python -m venv .venv

# Windows
.venv\Scripts\activate
# Linux / macOS
source .venv/bin/activate

pip install -e ".[dev]"
```

### 2️⃣ 一键启动（API + Dashboard）

```bash
# PowerShell / Windows
.\scripts\start_local.ps1

# Linux / macOS
bash scripts/start_local.sh
```

脚本会自动检查 feature mart 与 checkpoint；缺失时依次执行 `ingest` → `build_features` → `train`，然后启动服务。

数据已就绪、仅需重启 UI 时：

```bash
.\scripts\start_local.ps1 -SkipBootstrap
```

| 服务 | 地址 |
|------|------|
| 🌐 API | http://127.0.0.1:8000 |
| 📋 API Docs | http://127.0.0.1:8000/docs |
| 📊 Dashboard | Streamlit 自动打开 |

### 3️⃣ 验证安装

```bash
# 生成本地样例 CSV（public 仓库不包含 data/samples/*.csv）
python -m scripts.export_sample_data

pytest
ruff check src tests scripts
python -m scripts.ingest --help
```

---

## 🔧 流水线命令

完整数据处理与建模可按以下顺序手动执行：

```bash
# 样例数据
python -m scripts.ingest

# 外部真实数据（下载 CSV 放入 data/external/downloads/ 后）
python -m scripts.prepare_data --config-name=config data=external
python -m scripts.ingest --config-name=config data=external

# Point-in-time 特征构建
python -m scripts.build_features

# Baseline 模型训练
python -m scripts.train

# Mid-level tabular 模型（需要 GPU）
python -m scripts.train --config-name=config models=midlevel
python -m scripts.calibrate --config-name=config models=midlevel

# ScoreGen-Football Transformer（Graph + Transformer + 混合双变量比分头）
python -m scripts.train --config-name=config models=scoregen
python -m scripts.calibrate --config-name=config models=scoregen

# 世界杯回测（默认 2018 + 2022）
python -m scripts.backtest
python -m scripts.backtest test_set=world_cup_2018

# 概率校准（lambda 缩放，保持矩阵为唯一输出源）
python -m scripts.calibrate

# 数据 QA 报告
python -m scripts.data_qa

# 单独启动 API / Dashboard
python -m scripts.serve
python -m scripts.dashboard

# CLI 单场预测
python -m scripts.predict --match-id wc2022_arg_fra_final
```

---

## 🌐 API 端点

| 方法 | 路径 | 说明 |
|------|------|------|
| `GET` | `/health` | 健康检查 |
| `GET` | `/matches/{match_id}` | 比赛详情 |
| `GET` | `/features/{match_id}` | 赛前特征摘要 |
| `POST` | `/predict` | 提交 `match_id` 获取预测 |
| `GET` | `/predictions/{match_id}` | 按 ID 查询预测 |
| `GET` | `/score-matrix/{match_id}` | 比分概率矩阵 |
| `GET` | `/backtest/runs` | 回测报告列表 |
| `GET` | `/backtest/{run_id}` | 单场回测详情 |
| `GET` | `/data/freshness` | 数据与模型新鲜度 |

**预测响应示例字段：** `top3_scorelines` · `result_probs` · `ou25_probs` · `btts_probs` · `expected_goals` · `uncertainty` · `overflow_prob`

---

## 📁 目录结构

```text
WorldCup/
├── configs/          # Hydra 配置（data / training / backtest / models）
├── data/
│   ├── raw/              # 原始抓取数据
│   ├── curated/          # 标准化实体表
│   ├── feature_mart/     # PIT 特征视图
│   ├── samples/          # 内置样例 CSV
│   └── external_mappings/
├── src/worldcup/     # 核心 Python 包
├── scripts/          # CLI 入口
├── tests/            # 单元 & 集成测试
├── artifacts/        # checkpoint / 回测报告
└── docs/             # 设计文档
```

---

## 📖 文档

详细设计见 [`docs/README.md`](docs/README.md)。

| 文档 | 内容 |
|------|------|
| [`constraints.md`](docs/constraints.md) | 预测口径、数据边界、反泄漏规则 |
| [`system-design.md`](docs/system-design.md) | 整体架构与模型方案 |
| [`mvp-task-list.md`](docs/mvp-task-list.md) | MVP 范围与验收标准 |
| [`data-sources.md`](docs/data-sources.md) | 数据源 CSV 格式与 external 对接 |
| [`world-cup-2026-playbook.md`](docs/world-cup-2026-playbook.md) | 2026 数据准备、赛中调参、可回滚与报告口径 |

> 💡 建议阅读顺序：`constraints.md` → `system-design.md` → `mvp-task-list.md`；若面向 2026 实战，再加 `world-cup-2026-playbook.md`。

---

## ✅ 已实现（Phase 0–3）

- [x] 项目脚手架与 Hydra 配置
- [x] 数据导入 `raw → curated`（`scripts.ingest`）
- [x] Point-in-time 特征 mart（`scripts.build_features`）
- [x] Dixon-Coles baseline MLE 训练（`scripts.train`）
- [x] 2018 / 2022 世界杯 strict PIT 回测（`scripts.backtest`）
- [x] Streamlit Dashboard（`scripts.dashboard`，通过 API 展示）
- [x] FastAPI 全套 MVP 端点 + `/data/freshness` + `/backtest` + `/features`
- [x] lambda 缩放校准（`scripts.calibrate`，矩阵仍为唯一概率源）
- [x] 数据 QA 报告（`scripts.data_qa`）
- [x] 本机一键启动（Windows `.ps1` / Linux `start_local.sh`）

---

## 🗺️ Roadmap

- [x] 外部 CSV adapter + `scripts.prepare_data`（Kaggle / Elo / FIFA）
- [x] Mid-level tabular 模型（`models=midlevel`，PyTorch MLP → 8×8 矩阵）
- [x] ScoreGen-Football Transformer（`models=scoregen`，Graph + Transformer + 混合双变量分布头）
- [x] 球员级状态 + 对位图（`players` / `lineups` / `player_match_stats` → Graph Encoder）
- [x] 预计首发 + 伤停（`projected lineups` + `injuries`，严格 PIT）
- [ ] Advanced 模型（实时官方首发 / 事件数据 / xG 全量接入）

---

## 🧪 开发

```bash
# 代码检查
ruff check src tests scripts

# 运行测试
pytest

# 数据校验
python -m scripts.validate_data
```

---

## 📄 License

[MIT](LICENSE)

---

<p align="center">
  Made with ⚽ for the beautiful game
</p>

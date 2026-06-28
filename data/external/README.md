# 外部真实数据放置说明

将下载的 CSV 放入 `downloads/`，或在本地执行 bootstrap 自动拉取公开数据集。

## 推荐公开数据源

| 数据 | 推荐来源 | 适配格式 |
|------|----------|----------|
| 国际赛 / 世界杯赛果 | [martj42/international_results](https://github.com/martj42/international_results) | `kaggle_international` / bootstrap 自动 |
| Elo 历史 | 赛果推导 + [eloratings.net](https://www.eloratings.net/) 快照 | bootstrap 自动 |
| FIFA 排名 | [Dato-Futbol/fifa-ranking](https://github.com/Dato-Futbol/fifa-ranking) | `fifa_rankings` / bootstrap 自动 |
| WC roster / 阵容 | `data/external/seeds/`（仓库内种子） | `canonical` |
| 赔率 | `seeds/odds.csv` 或 football-data.co.uk | `canonical` / `football_data_odds` |

## 目录

```text
data/external/
├── downloads/          # bootstrap 输出（gitignore）
├── seeds/              # WC 2018/2022 种子（可提交）
├── examples/           # adapter 单元测试用
└── README.md
```

## 一键 bootstrap + 导入

```bash
# 0. 生成/刷新 WC 种子（仓库内）
python -m scripts.export_external_seeds

# 1. 下载 Kaggle 国际赛 + FIFA 历史 + Elo，并复制 seeds
python -m scripts.bootstrap_external_downloads

# 2. 准备 canonical CSV
python -m scripts.prepare_data --config-name=config data=external

# 3. 导入 raw / curated
python -m scripts.ingest --config-name=config data=external

# 4. 特征 mart（含 P1 球员/赔率摘要）
python -m scripts.build_features --config-name=config data=external
python -m scripts.train --config-name=config data=external models=scoregen_local training=local
```

Windows 一键脚本（含训练）：

```powershell
.\scripts\bootstrap_external_local.ps1
```

合并样例数据（调试 adapter）：

```bash
python -m scripts.prepare_data data=external data.include_samples=true
```

路线图与短板说明：[`docs/gap-analysis-and-roadmap.md`](../docs/gap-analysis-and-roadmap.md)

## 模板

可复制 `data/templates/*.template.csv` 作为手工整理起点。

## 球队别名

下载数据中的队名若与项目不一致，请更新 `data/external_mappings/team_aliases.csv`。

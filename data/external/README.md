# 外部真实数据放置说明

将下载的 CSV 放入 `downloads/`，然后在 `configs/data/external.yaml` 中配置路径与格式。

## 推荐公开数据源

| 数据 | 推荐来源 | 适配格式 |
|------|----------|----------|
| 国际赛 / 世界杯赛果 | [Kaggle International Football Results](https://www.kaggle.com/datasets) | `kaggle_international` |
| Elo 历史 | eloratings.net / 自建 CSV | `elo_history` |
| FIFA 排名 | FIFA 官网历史导出 / 第三方 CSV | `fifa_rankings` |

## 目录

```text
data/external/
├── downloads/          # 你下载的原始 CSV（gitignore）
├── examples/           # 内置示例（用于测试 adapter）
└── README.md
```

## 一键准备 + 导入

```bash
# 1. 准备 canonical CSV（合并 examples/downloads/samples）
python -m scripts.prepare_data --config-name=config data=external

# 2. 导入 raw / curated
python -m scripts.ingest --config-name=config data=external

# 3. 后续流水线
python -m scripts.build_features
python -m scripts.train
```

仅使用 Kaggle 下载文件（不含 samples）：

```bash
python -m scripts.prepare_data data=external data.include_samples=false
```

## 模板

可复制 `data/templates/*.template.csv` 作为手工整理起点。

## 球队别名

下载数据中的队名若与项目不一致，请更新 `data/external_mappings/team_aliases.csv`。

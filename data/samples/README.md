# 本地样例数据（不提交 Git）

本目录下的 `*.csv` 由脚本 **本地生成**，已在 `.gitignore` 中排除，**不会进入 public 仓库**。

## 生成完整 2018/2022 世界杯样例

```bash
python -m scripts.export_sample_data
python -m scripts.ingest
python -m scripts.build_features
```

生成内容约：139 场比赛、128 场 WC、球员/首发/统计等。源定义见：

- `src/worldcup/data_ingestion/sources/world_cup_catalog.py`
- `src/worldcup/data_ingestion/sources/world_cup_squads.py`

## 运行测试前

克隆仓库后需先执行 `export_sample_data`，否则 `data/samples/*.csv` 不存在，`pytest` 会失败。

## 可提交 Git 的相关文件

| 路径 | 说明 |
|------|------|
| `data/templates/` | CSV 列模板 |
| `data/external/examples/` | 小型 external adapter 示例 |
| `data/external_mappings/team_aliases.csv` | 队名映射（export 会更新，可酌情 commit） |
| `scripts/export_sample_data.py` | 生成脚本 |

## 切勿提交

| 路径 | 说明 |
|------|------|
| `data/raw/`、`curated/`、`feature_mart/`、`staging/` | ingest 产物 |
| `data/external/downloads/` | 你下载的真实 CSV |
| `artifacts/checkpoints/` | 模型权重 |
| `artifacts/predictions/` | 预测台账（2026 实战） |

详见 [`docs/world-cup-2026-playbook.md`](../../docs/world-cup-2026-playbook.md)。

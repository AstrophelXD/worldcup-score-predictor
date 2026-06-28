# WC 2018/2022 外部数据种子

本目录存放**可提交到 Git** 的世界杯 roster / 阵容 / 赔率 / 伤停种子，供 `bootstrap_external_downloads` 复制到 `downloads/`。

| 文件 | 内容 | 真实度 |
|------|------|--------|
| `wc_matches.csv` | 128 场 WC 2018/2022 赛果 | 赛果真实 |
| `players.csv` | 41 队 roster | 名单来自公开 catalog |
| `lineups.csv` | 历史首发 + 1 场 projected | 结构真实 |
| `player_match_stats.csv` | 由赛果派生进球归属 | 场次真实，归属简化 |
| `odds.csv` | WC 128 场赔率结构 | 样例化数值 |
| `injuries.csv` | 公开伤停案例 | 部分真实 |

生成/刷新：

```bash
python -m scripts.export_external_seeds
```

完整外部流水线见 [`docs/gap-analysis-and-roadmap.md`](../../docs/gap-analysis-and-roadmap.md)。

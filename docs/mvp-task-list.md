# MVP 阶段任务清单

本文档聚焦第一阶段最小可行版本，不追求一次把高级模型和所有实时能力都做完，而是先建立一条从数据到预测再到展示的完整闭环。

## 目标

在仅依赖公开数据和少量手工输入的前提下，完成一个可在实验室 RTX 4090 主机上训练与回测、可展示的世界杯比赛比分预测系统初版。

详细分阶段编码计划与 Todo 见 [MVP 初步编码计划](./coding-plan.md)。

## 范围

MVP 目标覆盖：

- 历史比赛数据接入
- Elo / FIFA ranking 接入
- 基础 point-in-time 特征
- baseline 模型训练与推理
- 胜平负 / 大小球 / BTTS / top-3 比分输出
- FastAPI 推理接口
- Streamlit dashboard 原型
- 2018 / 2022 世界杯回测

MVP 暂不强制覆盖：

- 完整球员级深度特征
- Graph Transformer / GNN
- 实时赔率流
- 自动化生产级调度

## 任务拆解

### 1. 数据层

- [ ] 建立 `data/raw`
- [ ] 建立 `data/curated`
- [ ] 建立 `data/feature_mart`
- [ ] 定义 `matches`, `teams`, `players` 基础 schema
- [ ] 准备球队和球员实体映射表
- [ ] 导入历史比赛结果
- [ ] 导入 Elo rating
- [ ] 导入 FIFA ranking
- [ ] 设计手工伤停 / 预计首发表

### 2. 特征层

- [ ] 实现 point-in-time join 工具
- [ ] 实现 match context 特征
- [ ] 实现 team strength 特征
- [ ] 实现 recent form 特征
- [ ] 实现 fatigue / rest 特征
- [ ] 编写反泄漏单元测试

### 3. 模型层

- [ ] 实现 baseline Poisson / Dixon-Coles
- [ ] 支持输出 `0-7 × 0-7` 比分矩阵
- [ ] 从矩阵聚合 top-3、1X2、OU2.5、BTTS
- [ ] 产出 uncertainty 初版（先用 entropy）

### 4. 训练与评估

- [ ] 定义时间切分策略
- [ ] 编写训练脚本
- [ ] 编写验证脚本
- [ ] 编写回测脚本
- [ ] 在 2018 世界杯上回测
- [ ] 在 2022 世界杯上回测
- [ ] 增加 calibration 流程

### 5. Serving

- [ ] 实现 `GET /matches`
- [ ] 实现 `GET /matches/{match_id}`
- [ ] 实现 `POST /predict`
- [ ] 实现 `GET /predictions/{match_id}`
- [ ] 实现 `GET /score-matrix/{match_id}`

### 6. Dashboard

- [ ] 实现 Match Selector 页面
- [ ] 实现 Prediction Summary 页面
- [ ] 实现 Score Matrix Heatmap 页面
- [ ] 展示数据更新时间

### 7. 工程治理

- [ ] 建立 `configs/` 结构
- [ ] 引入 Hydra
- [ ] 引入 MLflow
- [ ] 增加基础日志
- [ ] 编写 README 使用说明

## 验收标准

满足以下条件即视为 MVP 完成：

1. 能针对指定比赛生成有效预测。
2. 能输出统一的比分概率矩阵及衍生概率。
3. 能在 dashboard 上交互式查看结果。
4. 能对 2018 / 2022 世界杯执行严格回测。
5. 能证明预测未使用赛后信息。

## 建议执行顺序

1. 先打通 `数据 -> baseline -> API`
2. 再补 `dashboard`
3. 再做 `回测 + 校准`
4. 最后再引入球员级和高级模型

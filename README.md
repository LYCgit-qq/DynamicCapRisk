# 驾驶能力与风险状态评估系统

基于多模态数据的动态驾驶能力评估与风险状态预测系统

## 项目概述

本项目实现了基于多模态数据的动态驾驶能力评估与风险状态预测系统，对应硕士学位论文《基于多模态数据的动态驾驶能力评估与风险状态预测》的完整技术实现。

## 核心功能

### 1. 数据处理（第2章）
- **多模态数据加载**：支持驾驶模拟器(60Hz)、眼动追踪(120Hz)、生理信号(500Hz)数据
- **数据同步对齐**：统一采样率至20Hz，线性插值对齐
- **预处理流程**：异常值检测(3σ+LOF)、滤波(带通+Savitzky-Golay)、标准化
- **特征提取**：从18项候选特征筛选11项（Pearson相关性+VIF检验）

### 2. 动态驾驶能力评估（第3章）
- **基准能力评估**：K-means++聚类分为高(0.88)、中(0.75)、低(0.63)三级
- **能力波动量计算**：AHP-熵权组合赋权，Afl = 0.4×Sfl - 0.2
- **动态能力计算**：Ad = Ab + Afl，实时量化驾驶能力
- **有效性验证**：与异常事件相关性r=-0.78 (p<0.001)

### 3. 风险状态评估（第4章）
- **风险场强量化**：DSF理论，整合道路几何(0.25)、交通设施(0.20)、车辆交互(0.55)
- **TCI模型评估**：R = 0.58×Fs - 0.42×Ãd
- **风险等级划分**：阈值θ=0.10，分为高、中、低三级
- **模型验证**：与主观感知r=0.68-0.74，与异常事件r=0.73

### 4. 联合预测模型（第5章）
- **MT-JP架构**：多模态Transformer，4层编码器×8头注意力
- **双分支预测**：能力预测(R²=0.977) + 风险预测(R²=0.965)
- **交叉注意力**：显式建模能力-风险耦合关系
- **实时推理**：单样本3.7ms，满足实时预警需求

## 项目结构

```
DynamicCapRisk/
├── config/              # 配置文件
├── data/               # 数据目录
├── src/                # 源代码
│   ├── 1_data_processing/        # 数据预处理（第2章）
│   ├── 2_capability_assessment/  # 能力评估（第3章）
│   ├── 3_risk_assessment/        # 风险评估（第4章）
│   ├── 4_prediction/             # 预测模型（第5章）
│   ├── models/                   # 模型定义
│   ├── utils/                    # 工具函数
│   └── visualization/            # 可视化
├── output/            # 输出结果
└── docs/              # 文档
```

## 技术特点

### 数据驱动
- 32名驾驶人，4类场景，25600个时间窗口样本
- 操纵行为、车辆响应、眼动认知、生理状态四模态融合
- 基准能力 + 能力波动量双维度建模

### 理论支撑
- 行车安全场理论（DSF）量化风险场强
- 任务-能力接口模型（TCI）建立评估框架
- 多任务学习 + 一致性约束确保理论逻辑

### 工程实现
- 模块化设计，清晰的代码结构
- 完整的配置文件驱动
- PyTorch实现，支持GPU加速

## 应用场景

### ADAS个性化干预
- 基于能力等级定制干预策略
- 新手：早期预警，低阈值触发
- 专家：降低干预强度，高阈值触发

### 驾驶培训优化
- 识别能力短板，针对性训练
- 量化培训效果，动态调整方案
- 高风险场景专项训练

### 交通安全研究
- 驾驶行为分析
- 风险因素识别
- 事故预防策略

## 性能指标

### 能力预测
- MAE: 0.045
- RMSE: 0.060
- R²: 0.9770

### 风险预测
- 回归MAE: 0.081
- 分类准确率: 87.1%
- 高风险召回率: 89.3%

### 对比基线
优于SVR、LSTM、GRU、CNN-LSTM等模型

## 安装

```bash
pip install -r requirements.txt
```

## 快速开始

### 1. 数据预处理
```bash
python -m src.1_data_processing.main --config config/dataset.yaml
```

### 2. 能力评估
```bash
python -m src.2_capability_assessment.main --config config/baseline_capability.yaml
```

### 3. 风险评估
```bash
python -m src.3_risk_assessment.main --config config/risk_evaluator.yaml
```

### 4. 模型训练
```bash
python -m src.4_prediction.trainer --config config/trainer_dl.yaml
```

### 5. 模型评估
```bash
python -m src.4_prediction.evaluator --config config/evaluator.yaml
```

## 文档

- [项目概览](docs/PROJECT_OVERVIEW.md)
- [代码结构](docs/STRUCTURE.md)
- [模型架构](docs/model_architecture.md)

```python
# 1. 数据预处理
python scripts/preprocess_data.py --config config/experiment_config.yaml

# 2. 能力评估
python scripts/run_capability_assessment.py

# 3. 风险评估
python scripts/run_risk_assessment.py

# 4. 训练预测模型
python scripts/train_model.py --config config/model_config.yaml

# 5. 评估模型
python scripts/evaluate_model.py --model_path checkpoints/best_model.pth
```

## 数据格式

详见 `docs/data_format.md`

## 模型架构

详见 `docs/model_architecture.md`

## 引用

如果使用本代码，请引用相关论文。

## 许可证

MIT License

# 代码库结构说明

## 目录结构

```
driving_capability_risk_assessment/
├── README.md                    # 项目说明
├── requirements.txt             # 依赖包
├── STRUCTURE.md                 # 本文件 - 结构说明
│
├── config/                      # 配置文件目录
│   ├── experiment_config.yaml   # 实验配置（第2章）
│   ├── model_config.yaml        # 模型配置（第5章）
│   └── feature_config.yaml      # 特征配置（第3章）
│
├── data/                        # 数据目录
│   ├── raw/                     # 原始数据
│   │   ├── driving_simulator/   # 驾驶模拟器数据（60Hz）
│   │   ├── eye_tracking/        # 眼动追踪数据（120Hz）
│   │   ├── physiological/       # 生理信号数据（500Hz）
│   │   └── questionnaire.csv    # 问卷数据
│   ├── processed/               # 预处理后数据
│   └── README.md                # 数据说明
│
├── src/                         # 源代码目录
│   ├── data_processing/         # 数据处理模块（第2章）
│   │   ├── __init__.py
│   │   ├── data_loader.py       # 数据加载器
│   │   ├── synchronization.py   # 数据同步
│   │   ├── preprocessing.py     # 数据预处理
│   │   └── feature_extraction.py # 特征提取
│   │
│   ├── capability_assessment/   # 能力评估模块（第3章）
│   │   ├── __init__.py
│   │   ├── questionnaire_processor.py  # 问卷处理
│   │   ├── baseline_capability.py      # 基准能力评估
│   │   ├── capability_fluctuation.py   # 能力波动量
│   │   └── dynamic_capability.py       # 动态能力计算
│   │
│   ├── risk_assessment/         # 风险评估模块（第4章）
│   │   ├── __init__.py
│   │   ├── risk_field.py        # 风险场强计算
│   │   ├── tci_model.py         # TCI模型
│   │   └── risk_evaluator.py    # 风险评估器
│   │
│   ├── prediction/              # 预测模块（第5章）
│   │   ├── __init__.py
│   │   ├── dataset.py           # 数据集定义
│   │   ├── trainer.py           # 模型训练器
│   │   └── evaluator.py         # 模型评估器
│   │
│   ├── models/                  # 模型定义
│   │   ├── __init__.py
│   │   └── mtjp_model.py        # MT-JP模型
│   │
│   ├── utils/                   # 工具函数
│   │   ├── __init__.py
│   │   ├── metrics.py           # 评估指标
│   │   ├── losses.py            # 损失函数
│   │   └── logger.py            # 日志工具
│   │
│   └── visualization/           # 可视化模块
│       ├── __init__.py
│       ├── plot_capability.py   # 能力可视化
│       ├── plot_risk.py         # 风险可视化
│       └── plot_prediction.py   # 预测结果可视化
│
├── notebooks/                   # Jupyter notebooks
│   ├── 01_data_exploration.ipynb
│   ├── 02_capability_assessment.ipynb
│   ├── 03_risk_assessment.ipynb
│   └── 04_prediction_model.ipynb
│
├── tests/                       # 单元测试
│   ├── test_data_processing.py
│   ├── test_capability_assessment.py
│   ├── test_risk_assessment.py
│   └── test_prediction.py
│
├── scripts/                     # 运行脚本
│   ├── preprocess_data.py       # 数据预处理脚本
│   ├── run_capability_assessment.py  # 能力评估脚本
│   ├── run_risk_assessment.py   # 风险评估脚本
│   ├── train_model.py           # 模型训练脚本
│   └── evaluate_model.py        # 模型评估脚本
│
└── docs/                        # 文档目录
    ├── data_format.md           # 数据格式说明
    ├── model_architecture.md    # 模型架构说明
    └── api_reference.md         # API参考文档
```

## 模块对应关系

| 论文章节 | 模块 | 主要功能 |
|---------|------|---------|
| 第2章 | data_processing | 数据采集与预处理 |
| 第3章 | capability_assessment | 动态驾驶能力评估 |
| 第4章 | risk_assessment | 风险状态评估 |
| 第5章 | prediction + models | 联合预测模型 |

## 核心算法实现

### 第3章：动态驾驶能力评估
- **基准能力评估**: K-means++聚类（baseline_capability.py）
- **特征筛选**: Pearson相关性 + VIF检验（feature_extraction.py）
- **权重计算**: AHP-熵权组合赋权（capability_fluctuation.py）
- **能力计算**: Ad = Ab + Afl（dynamic_capability.py）

### 第4章：风险状态评估
- **风险场强**: DSF理论 + AHP权重（risk_field.py）
- **TCI模型**: R = α·Fs - β·Ãd（tci_model.py）
- **参数校准**: Logistic回归（tci_model.py）
- **等级划分**: 阈值敏感性分析（tci_model.py）

### 第5章：联合预测模型
- **MT-JP架构**: Transformer编码器（mtjp_model.py）
- **多模态融合**: 跨模态注意力（mtjp_model.py）
- **损失函数**: 多任务学习 + 一致性约束（losses.py）
- **模型训练**: AdamW + 余弦退火（trainer.py）

## 数据流程

```
原始数据(raw/) 
  ↓ [data_loader.py]
多模态数据
  ↓ [synchronization.py]
同步数据（20Hz）
  ↓ [preprocessing.py]
清洗标准化数据
  ↓ [feature_extraction.py]
11项驾驶特征
  ├─→ [capability_assessment/] → 动态能力 Ad
  ├─→ [risk_assessment/] → 风险度 R, 风险等级
  └─→ [prediction/] → MT-JP模型 → 预测结果
```

## 使用流程

### 1. 数据预处理
```bash
python scripts/preprocess_data.py --config config/experiment_config.yaml
```

### 2. 能力评估
```bash
python scripts/run_capability_assessment.py --data data/processed/
```

### 3. 风险评估
```bash
python scripts/run_risk_assessment.py --data data/processed/
```

### 4. 模型训练
```bash
python scripts/train_model.py --config config/model_config.yaml
```

### 5. 模型评估
```bash
python scripts/evaluate_model.py --model checkpoints/best_model.pth
```

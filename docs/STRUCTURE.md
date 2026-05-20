# 代码库结构说明

## 目录结构

```
DynamicCapRisk/
├── README.md                    # 项目说明
├── requirements.txt             # 依赖包
├── config/                      # 配置文件目录
│   ├── baseline_capability.yaml    # 基准能力评估配置
│   ├── capability_fluctuation.yaml # 能力波动量配置
│   ├── dataset.yaml                # 数据集配置
│   ├── dynamic_capability.yaml     # 动态能力计算配置
│   ├── evaluator_compare.yaml      # 评估器比较配置
│   ├── evaluator.yaml              # 评估器配置
│   ├── risk_evaluator.yaml         # 风险评估器配置
│   ├── risk_field.yaml             # 风险场强配置
│   ├── risk_validator.yaml         # 风险验证器配置
│   ├── trainer_dl.yaml             # 深度学习训练器配置
│   └── trainer_svr_cart.yaml       # SVR和CART训练器配置
├── data/                        # 数据目录
│   ├── README_AHP.md             # AHP方法说明
│   ├── README_raw_data.md        # 原始数据说明
│   ├── README_workzone.md        # 工作区说明
│   ├── dataset/                  # 数据集文件
│   │   ├── dataset_stats_aug-True_onlyTrain-True_gaussStd-0.05_timeWarpSigma-0.2_featDrop-0.1_magWarpSigma-0.1.csv
│   │   └── mtrp_window_aligned.csv
│   ├── processed/                # 预处理后数据
│   │   ├── mtrp_dataset_stats_aug-False.csv
│   │   ├── mtrp_window_aligned.csv
│   │   ├── questionnaire_preprocessed.csv
│   │   ├── questionnaire_standardized.csv
│   │   ├── work_zone_1_continuous.csv
│   │   ├── work_zone_2_continuous.csv
│   │   └── work_zone_3_continuous.csv
│   └── raw/                      # 原始数据
│       ├── act.mat               # 行为数据
│       ├── ahp_afl_judgment_matrix.csv # AHP判断矩阵
│       ├── ahp_risk_field_main.csv    # 风险场强主权重
│       ├── ahp_risk_field_sign.csv    # 风险场强符号权重
│       ├── eye.mat               # 眼动数据
│       ├── phy.mat               # 生理数据
│       ├── questionnaire.csv      # 问卷数据
│       └── 被试-实验ID映射.csv     # 被试映射表
├── docs/                        # 文档目录
│   ├── model_architecture.md    # 模型架构说明
│   ├── PROJECT_OVERVIEW.md      # 项目概览
│   └── STRUCTURE.md             # 本文件 - 结构说明
├── output/                      # 输出结果目录
│   ├── 1_capability_assessment/ # 能力评估结果
│   ├── 2_risk_assessment/       # 风险评估结果
│   └── 3_prediction/            # 预测结果
└── src/                         # 源代码目录
    ├── __init__.py
    ├── __pycache__/             # Python缓存文件
    ├── data_processing/       # 数据处理模块（第2章）
    ├── 1_capability_assessment/ # 能力评估模块（第3章）
    ├── 2_risk_assessment/       # 风险评估模块（第4章）
    ├── 3_prediction/            # 预测模块（第5章）
    ├── models/                  # 模型定义
    ├── trash/                   # 废弃代码目录
    ├── utils/                   # 工具函数
    └── visualization/           # 可视化模块
```

## 模块对应关系

| 论文章节 | 模块 | 主要功能 |
|---------|------|---------|
| 第2章 | data_processing | 数据采集与预处理 |
| 第3章 | 1_capability_assessment | 动态驾驶能力评估 |
| 第4章 | 2_risk_assessment | 风险状态评估 |
| 第5章 | 3_prediction + models | 联合预测模型 |

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
- **MT-RP架构**: Transformer编码器（mtrp_model.py）
- **多模态融合**: 跨模态注意力（mtrp_model.py）
- **损失函数**: 多任务学习 + 一致性约束（losses.py）
- **模型训练**: AdamW + 余弦退火（trainer.py）

## 数据流程

```
原始数据(raw/) 
  ↓ [data_processing/]
多模态数据
  ↓ [数据同步与预处理]
同步数据（20Hz）
  ↓ [特征提取]
11项驾驶特征
  ├─→ [1_capability_assessment/] → 动态能力 Ad
  ├─→ [2_risk_assessment/] → 风险度 R, 风险等级
  └─→ [3_prediction/] → MT-RP模型 → 预测结果
```

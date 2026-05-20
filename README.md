# 驾驶能力与风险状态评估系统

基于多模态数据的动态驾驶能力评估与风险状态预测系统。

## 项目概述

本项目实现了基于多模态信号的驾驶能力评估、施工区风险评估与联合预测模型。
其中：
- `src/1_capability_assessment/` 负责基准驾驶能力、能力波动、动态能力计算与验证；
- `src/2_risk_assessment/` 负责施工区风险场强、风险度计算与风险结果验证；
- `src/3_prediction/` 负责构建预测数据集、训练模型、评估与结果汇总。

## 目录结构

```
DynamicCapRisk/
├── config/                      # YAML 配置文件
├── data/                        # 原始与处理后数据
├── output/                      # 结果输出目录
├── src/                         # 源代码
│   ├── 1_capability_assessment/  # 能力评估模块
│   ├── 2_risk_assessment/        # 风险评估模块
│   ├── 3_prediction/             # 预测模块
│   ├── data_processing/          # 数据预处理与数据集构建
│   ├── models/                   # 模型定义
│   ├── utils/                    # 工具函数
│   └── visualization/            # 可视化工具
├── docs/                        # 文档
└── requirements.txt              # 依赖列表
```

## 模块说明

### 1_capability_assessment
- `baseline_capability.py`：计算基准能力（Ab/Abc），使用聚类评估并生成聚类结果和能力量化结果。
- `capability_fluctuation.py`：计算能力波动指标 Afl，支持 AHP+熵权或 CRITIC+熵权方法。
- `dynamic_capability.py`：结合基准能力和能力波动计算动态驾驶能力 Ad，支持 `Ab` / `Abc` 两种模式。
- `capability_validator.py`：验证动态能力 Ad 的有效性，基于异常事件、车道稳定性等绩效指标进行对比分析。

### 2_risk_assessment
- `risk_field.py`：计算施工区风险场强 F_S，生成场强结果和可视化图表。
- `risk_field_ahp.py`：基于 AHP+熵权进行风险场强权重组合校准。
- `risk_evaluator.py`：计算风险度 R，读取能力评估与风险场强结果并输出风险结果。
- `risk_validator.py`：验证风险度结果与客观绩效指标之间的相关性。

### 3_prediction
- `dataset.py`：构建预测模型数据集，提取历史时序特征并构造能力和风险标签。
- `trainer_dl.py`：训练深度学习模型（如 MT-RP）并自动执行测试评估。
- `trainer_svr_cart.py`：训练传统基线模型 SVR 和 CART。
- `run_grid_search.py`：执行网格搜索与批量实验。
- `evaluator.py`：评估单个模型 checkpoint 的预测结果。
- `evaluator_compare.py`：对比不同模型表现。
- `summarize_results.py`：汇总实验结果与评估指标。


## 依赖

```bash
pip install -r requirements.txt
```

## 运行流程

### 1. 数据预处理
```bash
python src/data_processing/data_loader.py
```

### 2. 能力评估
```bash
python src/1_capability_assessment/baseline_capability.py --config config/baseline_capability.yaml
python src/1_capability_assessment/capability_fluctuation.py --config config/capability_fluctuation.yaml
python src/1_capability_assessment/dynamic_capability.py --config config/dynamic_capability.yaml
python src/1_capability_assessment/capability_validator.py --config config/capability_validator.yaml
```

### 3. 风险评估
```bash
python src/2_risk_assessment/risk_field.py
python src/2_risk_assessment/risk_evaluator.py
python src/2_risk_assessment/risk_validator.py
```

### 4. 预测训练与评估
```bash
python src/3_prediction/dataset.py -c config/dataset.yaml
python src/3_prediction/trainer_dl.py -c config/trainer_dl.yaml
python src/3_prediction/evaluator.py -c config/evaluator.yaml
```

## 推荐配置文件

- `config/baseline_capability.yaml`
- `config/capability_fluctuation.yaml`
- `config/dynamic_capability.yaml`
- `config/capability_validator.yaml`
- `config/risk_field.yaml`
- `config/risk_evaluator.yaml`
- `config/risk_validator.yaml`
- `config/dataset.yaml`
- `config/trainer_dl.yaml`
- `config/trainer_svr_cart.yaml`
- `config/evaluator.yaml`
- `config/evaluator_compare.yaml`
- `config/run_grid_search.yaml`

## 说明

- 所有模块均支持 YAML 配置文件驱动，默认从项目根目录 `config/` 读取配置。
- 预测模块需先构建数据集再进行模型训练与评估。

## 文档

- `docs/PROJECT_OVERVIEW.md`
- `docs/STRUCTURE.md`
- `docs/model_architecture.md`


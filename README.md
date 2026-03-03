# 驾驶能力与风险状态评估系统

基于多模态数据的动态驾驶能力评估与风险状态预测系统

## 项目概述

本项目实现了一个完整的驾驶能力评估和风险预测系统，包括：

1. **动态驾驶能力评估**（第3章）
   - 基准驾驶能力评估（K-means++聚类）
   - 能力波动量计算（AHP-熵权组合赋权）
   - 动态能力值计算

2. **风险状态评估**（第4章）
   - 风险场强量化（基于行车安全场理论）
   - TCI模型风险度计算
   - 风险等级划分

3. **联合预测模型**（第5章）
   - MT-JP多模态Transformer模型
   - 动态驾驶能力预测
   - 风险状态预测

## 项目结构

```
driving_capability_risk_assessment/
├── config/              # 配置文件
├── data/               # 数据目录
├── src/                # 源代码
│   ├── data_processing/        # 数据预处理（第2章）
│   ├── capability_assessment/  # 能力评估（第3章）
│   ├── risk_assessment/        # 风险评估（第4章）
│   ├── prediction/             # 预测模型（第5章）
│   ├── models/                 # 模型定义
│   ├── utils/                  # 工具函数
│   └── visualization/          # 可视化
├── notebooks/          # Jupyter notebooks
├── tests/             # 单元测试
├── scripts/           # 运行脚本
└── docs/              # 文档
```

## 安装

```bash
pip install -r requirements.txt
```

## 快速开始

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

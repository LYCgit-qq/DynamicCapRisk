# 1_capability_assessment 模块介绍

本模块包含驾驶能力评估相关的核心脚本，主要用于计算和验证驾驶员的基准能力、能力波动、动态能力和有效性验证。

## 文件列表及功能

### baseline_capability.py
- **功能**: 计算基准驾驶能力 (Ab/Abc)
- **描述**: 使用 K-means 聚类对标准化问卷或驾驶数据进行聚类，评估候选聚类数的性能指标，并计算基准能力值。生成聚类结果、能力量化结果、个体化基准能力以及可视化报告。
- **输入**: 预处理后数据或指定输入路径
- **输出**: `Ab_quantification.csv`、`Abc_individualized_baseline_ability.csv`、聚类标签、可视化图表

### capability_fluctuation.py
- **功能**: 计算驾驶能力波动 (Afl)
- **描述**: 通过滑动窗口分析时序驾驶数据，进行相关性与 VIF 筛选，使用 AHP+熵权或 CRITIC+熵权组合赋权，计算综合波动指标并输出结果与可视化。
- **输入**: 预处理后的 `act/eye/phy` 数据 PKL
- **输出**: `Afl` 结果 PKL、波动指标 CSV、可视化图表

### dynamic_capability.py
- **功能**: 计算动态驾驶能力 (Ad)
- **描述**: 结合基准能力和能力波动，计算最终动态驾驶能力值。支持两种基准能力模式：`Ab`（聚类均值）和 `Abc`（个体化基准能力）。
- **输入**: 基准能力 CSV 和能力波动 PKL
- **输出**: 动态能力结果 CSV、可视化图表

### capability_validator.py
- **功能**: 验证动态驾驶能力 (Ad)
- **描述**: 使用原始驾驶数据计算异常事件、车道稳定性、方向盘转角标准差等绩效指标，评估 Ad 的有效性并生成验证报告。
- **输入**: 原始驾驶数据 PKL、动态能力结果
- **输出**: 对比分析 CSV、验证报告文本

## 执行顺序
1. 运行 `baseline_capability.py` 生成基准能力
2. 运行 `capability_fluctuation.py` 生成能力波动
3. 运行 `dynamic_capability.py` 生成动态能力
4. 运行 `capability_validator.py` 验证动态能力

## 快速启动
```bash
python src/1_capability_assessment/baseline_capability.py --config config/baseline_capability.yaml
python src/1_capability_assessment/capability_fluctuation.py --config config/capability_fluctuation.yaml
python src/1_capability_assessment/dynamic_capability.py --config config/dynamic_capability.yaml
python src/1_capability_assessment/capability_validator.py --config config/capability_validator.yaml
```

## 注意项
- `dynamic_capability.py` 默认读取 `config/dynamic_capability.yaml`，可通过 `--ab_mode Abc` 切换为个体化基准能力模式。
- `capability_fluctuation.py` 默认读取 `config/capability_fluctuation.yaml`，也可通过 `--data_path` / `--output_path` / `--corr_thresh` / `--vif_thresh` 等命令行参数覆盖。
- `capability_validator.py` 默认读取 `config/capability_validator.yaml`。

## 配置
所有脚本都支持 YAML 配置文件，位于项目根目录的 `config/` 目录下。

## 依赖
- numpy
- pandas
- scipy
- scikit-learn
- matplotlib
- seaborn
- PyYAML
- statsmodels

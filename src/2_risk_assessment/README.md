# 2_risk_assessment

本目录包含施工区风险评估与风险度验证相关脚本。执行顺序基于数据生成和验证流程，先计算风险场强，再评估风险度 R，最后验证模型有效性。

## 目录脚本说明

- `risk_field.py`
  - 计算施工区风险场强 F_S
  - 生成每个场景的 `Fs_<scenario>.csv`
  - 生成场强可视化图表
  - 默认配置文件：`config/risk_field.yaml`

- `risk_field_ahp.py`
  - 基于 AHP + 熵权法对风险场强权重进行组合赋权
  - 计算并保存动态组合权重
  - 生成风险场强可视化图表
  - 应用于需使用专家 AHP 权重与数据驱动熵权联合校准的场景

- `risk_evaluator.py`
  - 计算风险度 R
  - 读取 `output/1_capability_assessment/results/Afl_capability_fluctuation.pkl` 和 `Ad_result.pkl`
  - 读取 `output/2_risk_assessment/results/Fs_*.csv`
  - 保存 `risk_windows_all.csv`、`risk_summary_by_sample.csv` 等结果
  - 默认配置文件：`config/risk_evaluator.yaml`

- `risk_validator.py`
  - 验证风险度 R 与客观绩效指标之间的相关性
  - 读取 `output/1_capability_assessment/validation/performance_metrics.csv`
  - 读取 `output/2_risk_assessment/results/risk_windows_all.csv`
  - 输出验证结果到 `output/2_risk_assessment/validation/`
  - 默认配置文件：`config/risk_validator.yaml`

## 推荐执行顺序

1. `risk_field_ahp.py`（可选）
   - 需要生成或覆盖 AHP 权重时运行
   - 默认使用 `config/risk_field.yaml` 中的输入路径和 AHP 权重文件

2. `risk_field.py`
   - 生成 F_S 场强数据
   - 输出目录：`output/2_risk_assessment/results`

3. `risk_evaluator.py`
   - 计算风险度 R
   - 输出风险结果：`output/2_risk_assessment/results/risk_windows_all.csv`

4. `risk_validator.py`
   - 对风险度结果进行统计验证
   - 输出验证分析结果：`output/2_risk_assessment/validation/`

## 运行示例

```bash
python src/2_risk_assessment/risk_field.py
python src/2_risk_assessment/risk_evaluator.py
python src/2_risk_assessment/risk_validator.py
```

如果使用 AHP 版本：

```bash
python src/2_risk_assessment/risk_field_ahp.py
```

## 关键配置文件

- `config/risk_field.yaml`
- `config/risk_evaluator.yaml`
- `config/risk_validator.yaml`

## 说明

- `risk_field.py` 和 `risk_field_ahp.py` 负责风险场强(F_S)的生成与可视化。
- `risk_evaluator.py` 负责风险度 R 的计算、全局归一化和风险等级划分。
- `risk_validator.py` 负责从客观绩效指标角度验证风控结果的有效性。
- 如果希望使用 `risk_field_ahp.py` 的 AHP 权重计算，请先准备 `feature_name, ahp_weight` 格式的 AHP CSV 文件。

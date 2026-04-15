# MT-JP 联合预测模型 — 数据集输入说明

### 数据来源（3个核心文件）

| 文件 | 路径 | 核心内容 |
|------|------|----------|
| `raw_data.pkl` | `data/processed/raw_data.pkl` | 原始多模态时序信号：<br>- act (60Hz, 10维)：行为信号<br>- eye (120Hz, 3维)：眼动信号<br>- phy (100Hz, 5维)：生理信号 |
| `Afl_capability_fluctuation.pkl` | `output/1_capability_assessment/` | 能力评估结果：<br>- sample_fluctuations：各样本逐窗口能力波动量<br>- sample_field：各样本逐窗口场景标签<br>- sample_window_counts：各样本窗口数 |
| `risk_windows_all.csv` | `output/2_risk_assessment/results/` | 风险评估结果（每窗口）：<br>- R_star：风险度<br>- risk_level：风险等级（低/中/高）<br>- F_S：环境特征<br>- field_label：场景标签 |

---

### 原始信号结构

| 模态 | 键名 | 采样率 | 维度 | 字段含义 |
|------|------|--------|------|---------|
| 行为 | `act` | 60 Hz | 10 维 | 油门、刹车、转向角、速度、纵向加速度、侧向偏移、侧向加速度等 + 场景标签 |
| 眼动 | `eye` | 120 Hz | 3 维 | 眨眼频率、gaze_x、gaze_y |
| 生理 | `phy` | 100 Hz | 5 维 | BVP、ECG、呼吸、HR、HRV |

每个键均为 **List，长度 = 67（驾驶人样本数）**，每个元素为该驾驶人的时序 ndarray。

---

### 模型输入窗口设计（论文 §5.1）
模型采用「历史序列 → 未来单步」的预测范式：
- **时间窗口长度**：T = 5 步（每步对应 3s 能力评估窗口，共 15s 历史序列）
- **预测目标**：第 T+1 步（未来 3s 窗口）的能力值/风险度/风险等级
- **特征维度**：D = 17 维（行为8维 + 眼动3维 + 生理5维 + 环境1维）

#### 模型输入张量 X
```
X : np.ndarray  shape = (N, T=5, D=17)   dtype=float32
```
**D=17 维特征组成：**
```
[0-7]   行为 8 维：steer_angle / steer_vel / brake / throttle /
                   lon_acc / lat_off / lat_acc / speed
[8-10]  眼动 3 维：blink_freq / gaze_x / gaze_y
[11-15] 生理 5 维：bvp / ecg / resp / hr / hrv
[16]    环境 1 维：F_S（来自 risk_csv，已归一化至[0,1]，不参与 Z-score）
```

---

### 预测标签（均对应未来第 T+1 步）

| 字段 | 类型 | 取值范围 | 含义 |
|------|------|----------|------|
| `y_ability` | float32 | [0, 1] | 未来步归一化能力值 Ã_d |
| `y_risk_reg` | float32 | [-1, 1] | 未来步风险度 R |
| `y_risk_cls` | int64 | {0, 1, 2} | 未来步风险等级（0=低/1=中/2=高） |

---

### 数据集划分
按**驾驶人**（sample_idx）分层划分，保证各驾驶人平均能力的高/中/低三组在训练/验证/测试集中比例均衡：
```
train : val : test = 70% : 10% : 20%
```

---

### 输出文件结构
数据集最终保存至 `data/processed/` 目录，包含 2 个核心文件（文件名附带数据增强关键参数）：

#### 1. 主数据集文件：`dataset_xxx.pkl`
文件内为字典结构，包含 `train / val / test` 三个 split，每个 split 对应以下内容：
```python
{
  "X"         : np.ndarray  (N, T=5, D=17)  float32  # 15s历史输入序列
  "y_ability" : np.ndarray  (N,)            float32  # 未来3s能力标签
  "y_risk_reg": np.ndarray  (N,)            float32  # 未来3s风险度回归标签
  "y_risk_cls": np.ndarray  (N,)            int64    # 未来3s风险分类标签
  "meta"      : pd.DataFrame (N,)           # 元信息（可追溯性）
                - sample_idx：驾驶人索引
                - window_idx：目标步（未来3s）的窗口索引
                - field_label：目标步的场景标签
                - augmented（可选）：是否为数据增强样本（True/False）
}
```
此外，文件根层级还包含：
- `norm`：字典，含 `mu`（训练集特征均值）、`sigma`（训练集特征标准差），供推理时 Z-score 标准化使用；
- `feature_names`：列表，对应 17 维特征的名称（与输入张量维度一一对应）。

#### 2. 特征统计文件：`dataset_stats_xxx.csv`
CSV 格式，记录每个特征的训练集统计量（供推理/部署时标准化）：
| 列名 | 内容 |
|------|------|
| `feature` | 17 维特征的名称（如 steer_angle、F_S 等） |
| `train_mean` | 训练集该特征的均值（F_S 均值固定为 0） |
| `train_std` | 训练集该特征的标准差（F_S 标准差固定为 1） |

#### 文件名规则
文件名后缀 `xxx` 为数据增强关键参数，示例：
- 禁用增强：`dataset_aug-False.pkl`
- 启用增强：`dataset_aug-True_onlyTrain-True_gaussStd-0.05.pkl`

---

### 补充说明
1. 数据增强仅作用于训练集，且在 Z-score 标准化前执行；
2. F_S 特征已归一化至 [0,1]，不参与 Z-score 标准化；
3. 所有张量均采用低精度 dtype（float32/int64），平衡精度与存储开销。

### 总结
1. 数据集核心逻辑是「15s 历史多模态序列 → 预测未来 3s 能力/风险」，输入维度固定为 (N,5,17)；
2. 输出文件包含完整的训练/验证/测试集及标准化统计量，文件名附带增强参数便于版本管理；
3. meta 元信息保留样本溯源能力，支持后续分析不同驾驶人/场景的预测效果。
## MT-JP 联合预测模型 — 数据集输入说明

### 数据来源（3个文件）

| 文件 | 路径 | 内容 |
|------|------|------|
| `raw_data.pkl` | `data/processed/raw_data.pkl` | 原始多模态信号（行为/眼动/生理） |
| `Afl_capability_fluctuation.pkl` | `output/1_capability_assessment/` | 能力评估结果 |
| `risk_windows_all.csv` | `output/2_risk_assessment/results/` | 风险评估结果 |

---

### 原始信号结构

| 模态 | 键名 | 采样率 | 维度 | 字段含义 |
|------|------|--------|------|---------|
| 行为 | `act` | 60 Hz | 10 维 | 油门、刹车、转向角、速度、纵向加速度、侧向偏移、侧向加速度等 + 场景标签 |
| 眼动 | `eye` | 120 Hz | 3 维 | 眨眼频率、gaze_x、gaze_y |
| 生理 | `phy` | 100 Hz | 5 维 | BVP、ECG、呼吸、HR、HRV |

每个键均为 **List，长度 = 67（驾驶人样本数）**，每个元素为该驾驶人的时序 ndarray。

---

### 模型输入张量 X

```
X : np.ndarray  shape = (N, T=5, D=17)   dtype=float32
```

**T=5 步历史**（每步对应一个 3s 能力窗口，共 15s）→ 预测第 T+1 步（未来 3s）

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

| 字段 | 类型 | 范围 | 含义 |
|------|------|------|------|
| `y_ability` | float32 | [0, 1] | 归一化能力值 Ã_d |
| `y_risk_reg` | float32 | [-1, 1] | 风险度 R* |
| `y_risk_cls` | int64 | {0, 1, 2} | 风险等级（低/中/高） |

---

### 数据集划分

按**驾驶人**（sample_idx）分层划分，分层依据为各驾驶人平均能力高/中/低三组：

```
train : val : test = 70% : 10% : 20%
```

输出保存至 `output/3_prediction/mtjp_dataset.pkl`，含 `train / val / test` 三个 split 及训练集 Z-score 统计量（`mu` / `sigma`）。
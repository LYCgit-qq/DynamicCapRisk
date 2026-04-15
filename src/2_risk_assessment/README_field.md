# Afl_capability_fluctuation.pkl 样本路段类别字段说明
## 1. 字段基本信息
| 项               | 说明                                                                 |
|------------------|----------------------------------------------------------------------|
| 字段名           | `sample_field`                                                       |
| 所属文件         | `output/1_capability_assessment/results/Afl_capability_fluctuation.pkl`      |
| 核心作用         | 存储67个样本中每个窗口对应的路段类别（自由路段/施工区类型）|
| 整体数据类型     | `list`（长度固定为67，与样本数量一一对应）|
| 列表元素类型     | `numpy.ndarray`（每个数组对应单个样本的所有窗口的路段类别值）|
| 数组元素数据类型 | `int`（仅包含 0/1/2/3 四个取值）|

## 2. 数值-路段类别映射关系

| 数值 | 对应的路段类别 | 场景标识 |
|------|----------------|----------|
| 0    | 自由路段       | test0    |
| 1    | 标准施工区     | work_zone_1   |
| 2    | 占一还一       | work_zone_2   |
| 3    | 占而不还       | work_zone_3   |

- test0 区域的 F_S 统一记为 0
- work_zone_1–work_zone_3 区域的 F_S 计算结果见 `output/2_risk_assessment/Fs_work_zone_1.csv` ~ `Fs_work_zone_3.csv`

## 3. 数据结构与读取示例
### 3.1 读取数据
```python
import pickle

# 读取pkl文件
with open("output/1_capability_assessment/results/Afl_capability_fluctuation.pkl", "rb") as f:
    result = pickle.load(f)

# 获取sample_field字段
sample_field = result["sample_field"]

# 查看基本信息
print(f"样本总数：{len(sample_field)}")  # 输出：67
print(f"第1个样本的窗口数：{len(sample_field[0])}")  # 输出该样本的窗口数量
print(f"第1个样本第1个窗口的路段类别：{sample_field[0][0]}")  # 输出0/1/2/3中的一个
```

### 3.2 结构说明
- `sample_field[0]`：第1个样本的所有窗口路段类别数组
- `sample_field[1]`：第2个样本的所有窗口路段类别数组
- ...
- `sample_field[66]`：第67个样本的所有窗口路段类别数组
- 若某个样本无有效窗口，对应数组为空（`numpy.ndarray([])`）

## 4. 补充说明
1. 每个窗口的路段类别值生成规则：
   - 以3秒为一个非重叠窗口，基于原始act数据最后一列计算；
   - 窗口内全0 → 窗口值为0（自由路段）；
   - 窗口内有非0值 → 取出现次数最多的数（1/2/3）作为窗口值；
   - 最终所有值会被强制限制在0-3范围内，确保无异常值。
2. 该字段与 `sample_window_counts` 字段一一对应：`sample_window_counts[i]` 表示第i个样本的有效窗口数，对应 `sample_field[i]` 的数组长度。
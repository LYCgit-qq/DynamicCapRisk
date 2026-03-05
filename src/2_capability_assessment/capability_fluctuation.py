# D:\Local\DynamicCapRisk\src\2_capability_assessment\capability_fluctuation.py

import os
import yaml
import argparse

import numpy as np
import pandas as pd
import pickle
from scipy.stats import rankdata
from statsmodels.stats.outliers_influence import variance_inflation_factor
from sklearn.preprocessing import StandardScaler
from src.visualization.plot_capability import run_all_visualizations
from src.utils.ahp_calculator import calculate_ahp_weights


# ====================== 配置加载 ======================
def load_config(config_path=None):
    """加载YAML配置文件，返回配置字典"""
    default_config_path = "config\\capability_fluctuation.yaml"
    config_path = config_path or default_config_path

    try:
        with open(config_path, "r", encoding="utf-8") as f:
            config = yaml.safe_load(f)
    except FileNotFoundError:
        print(f"警告：配置文件{config_path}未找到，使用默认值")
        config = {}

    defaults = {
        "window_seconds": 3,
        "sampling_rates": {"act": 60, "eye": 120, "phy": 100},
        "correlation_threshold": 0.8,
        "vif_threshold": 10,
        "fluctuation_k": 0.4,
        "fluctuation_stats_range": {"min": -0.05, "max": 0.05},
        "baseline_quantiles": [0, 1 / 3, 2 / 3, 1],
        "baseline_group_labels": [
            "low_baseline_group",
            "medium_baseline_group",
            "high_baseline_group",
        ],
        "data_path": "data/processed/raw_data.pkl",
        "output_path": "output/1_capability_assessment/Afl_capability_fluctuation.pkl",
        "visualization_output_dir": "output/1_capability_assessment",
    }

    for key, val in defaults.items():
        if key not in config:
            config[key] = val
        if isinstance(val, dict) and isinstance(config.get(key), dict):
            for sub_key, sub_val in val.items():
                if sub_key not in config[key]:
                    config[key][sub_key] = sub_val

    return config


def parse_args():
    """解析命令行参数"""
    parser = argparse.ArgumentParser(description="驾驶能力波动计算脚本")
    parser.add_argument("--config", type=str, help="YAML配置文件路径")
    parser.add_argument("--data_path", type=str, help="输入数据pkl路径")
    parser.add_argument("--output_path", type=str, help="输出结果路径")
    parser.add_argument("--corr_thresh", type=float, help="相关性筛选阈值")
    parser.add_argument("--vif_thresh", type=float, help="VIF筛选阈值")
    parser.add_argument("--window_sec", type=int, help="滑动窗口秒数")
    parser.add_argument("--fluct_k", type=float, help="波动量计算k值")
    return parser.parse_args()


# ====================== 1. 数据加载与预处理 ======================
def load_data(pkl_path):
    """加载预处理后的pkl数据（返回act/eye/phy列表，各67个样本）"""
    with open(pkl_path, "rb") as f:
        data = pickle.load(f)
    act_raw = data["act"] if "act" in data else []
    eye_raw = data["eye"] if "eye" in data else []
    phy_raw = data["phy"] if "phy" in data else []
    assert (
        len(act_raw) == len(eye_raw) == len(phy_raw) == 67
    ), "样本数量不匹配（需67个）"
    return act_raw, eye_raw, phy_raw


def sliding_window_mean(data, window_size):
    """非重叠窗口均值"""
    arr = np.asarray(data)
    if arr.shape[0] < window_size:
        return np.empty((0,) + arr.shape[1:])
    n_windows = arr.shape[0] // window_size
    trimmed = arr[: n_windows * window_size]
    new_shape = (n_windows, window_size) + arr.shape[1:]
    data_reshaped = trimmed.reshape(new_shape)
    return data_reshaped.mean(axis=1)


def sliding_window_mode_for_last_col(data, window_size):
    """
    针对最后一列的滑动窗口处理规则：
    - 非重叠窗口
    - 窗口内全0 → 窗口值为0
    - 窗口内有非0值 → 取出现次数最多的数（众数），1/2/3共存时取出现次数最多的
    """
    arr = np.asarray(data)
    if arr.shape[0] < window_size:
        return np.empty(0)
    n_windows = arr.shape[0] // window_size
    trimmed = arr[: n_windows * window_size]
    # 重塑为(窗口数, 窗口大小)
    reshaped = trimmed.reshape(n_windows, window_size)

    # 处理每个窗口
    window_results = []
    for window in reshaped:
        # 统计非0值
        non_zero = window[window != 0]
        if len(non_zero) == 0:
            window_results.append(0)
        else:
            # 计算众数（取出现次数最多的值）
            vals, counts = np.unique(non_zero, return_counts=True)
            mode_val = vals[np.argmax(counts)]
            window_results.append(mode_val)
    return np.array(window_results, dtype=int)


def preprocess_single_sample(act, eye, phy, config):
    """单样本多模态数据时间窗口对齐"""
    window_seconds = config["window_seconds"]
    sampling_rates = config["sampling_rates"]

    win_act = sampling_rates["act"] * window_seconds  # 180
    win_eye = sampling_rates["eye"] * window_seconds  # 360
    win_phy = sampling_rates["phy"] * window_seconds  # 300

    # ===== 处理act数据：最后一列特殊逻辑，其他列正常均值 =====
    if act.size == 0:
        act_other_win = np.empty((0, act.shape[1]-1) if act.ndim>1 else (0,))
        act_last_win = np.empty(0)
    else:
        # 拆分最后一列（需要特殊处理）和其他列（正常均值）
        act_other = act[:, :-1]  # 除最后一列外的所有列
        act_last = act[:, -1]    # 最后一列（sample_field列）
        
        # 其他列：正常滑动窗口均值
        act_other_win = sliding_window_mean(act_other, win_act)
        # 最后一列：自定义规则（众数/全0判断）
        act_last_win = sliding_window_mode_for_last_col(act_last, win_act)
    
    # 合并其他列和处理后的最后一列
    if len(act_other_win) > 0 and len(act_last_win) > 0:
        act_win = np.hstack([act_other_win, act_last_win.reshape(-1, 1)])
    else:
        act_win = np.empty((0, act.shape[1]) if act.ndim>1 else (0,))

    # eye/phy 保持原有均值逻辑不变
    eye_win = sliding_window_mean(eye, win_eye)
    phy_win = sliding_window_mean(phy, win_phy)

    n_min = min(len(act_win), len(eye_win), len(phy_win))
    if n_min == 0:
        return None, None, None
    return act_win[:n_min], eye_win[:n_min], phy_win[:n_min]


# ====================== 2. 特征提取 ======================
def extract_features_single_sample(act_win, eye_win, phy_win):
    """
    从单样本窗口化数据中提取驾驶特征。

    修改说明（相较于原版）：
    ① 方向性特征（steering_angle / steering_velocity / lateral_offset /
       lateral_accel）一律取绝对值。
       理由：驾驶能力评估关注偏差幅度而非方向；负值会导致特征分布出现
       双峰或零膨胀，使后续熵权和归一化严重失真。
    ② brake_pedal 已是非负量（踏板开度），无需处理；原始数据若含负值
       说明是传感器零点偏移，此处取绝对值以统一量纲。
    ③ hrv 取绝对值，消除极少数负异常值对归一化的影响。
    """
    if act_win is None or len(act_win) == 0:
        return pd.DataFrame()

    df = pd.DataFrame()

    # ---------- 操纵行为特征 ----------
    # ① 取绝对值：转角幅度，消除左右方向性
    df["steering_angle"] = np.abs(act_win[:, 2])
    if len(act_win) >= 2:
        dv = np.diff(df["steering_angle"], prepend=df["steering_angle"].iloc[0])
        # ① 取绝对值：角速度幅度
        df["steering_velocity"] = np.abs(dv / 3.0)
    else:
        df["steering_velocity"] = 0.0
    # ② 踏板开度：取绝对值消除传感器零点负偏移
    df["brake_pedal"] = np.abs(act_win[:, 1])
    df["throttle_pedal"] = np.abs(act_win[:, 0])

    # ---------- 车辆响应特征 ----------
    df["longitudinal_accel"] = np.abs(act_win[:, 4])  # ① 纵向加速度幅度
    df["lateral_offset"] = np.abs(act_win[:, 8])  # ① 横向偏移幅度
    df["lateral_accel"] = np.abs(act_win[:, 5])  # ① 横向加速度幅度
    df["vehicle_speed"] = act_win[:, 3]  # 车速本身非负，保留

    # ---------- 眼动认知特征 ----------
    if eye_win is not None and len(eye_win) > 0 and eye_win.shape[1] >= 3:
        df["gaze_dispersion"] = np.std(eye_win[:, 1:3], axis=1)
        df["blink_frequency"] = eye_win[:, 0]
    else:
        df["gaze_dispersion"] = 0.0
        df["blink_frequency"] = 0.0

    # ---------- 生理状态特征 ----------
    if phy_win is not None and len(phy_win) > 0 and phy_win.shape[1] >= 5:
        df["hrv"] = np.abs(phy_win[:, 4])  # ③ 消除极少数负异常值
        df["hr"] = phy_win[:, 3]
        df["bvp"] = phy_win[:, 0]
        df["ecg"] = phy_win[:, 1]
        df["resp"] = phy_win[:, 2]
    else:
        df["hrv"] = 0.0
        df["hr"] = 0.0
        df["bvp"] = 0.0
        df["ecg"] = 0.0
        df["resp"] = 0.0

    return df.dropna(axis=1)


# ====================== 3. 特征筛选 ======================
def correlation_filter(df, threshold=0.8):
    """Pearson相关性分析筛选"""
    if df.empty or len(df.columns) < 2:
        return df, []
    corr_mat = df.corr().abs()
    upper = corr_mat.where(np.triu(np.ones(corr_mat.shape), k=1).astype(bool))
    to_drop = [column for column in upper.columns if any(upper[column] > threshold)]
    return df.drop(columns=to_drop), to_drop


def vif_filter(df, threshold=10):
    """方差膨胀因子(VIF)筛选"""
    if df.empty or len(df.columns) < 2:
        return df, pd.DataFrame({"feature": df.columns, "VIF": []})

    df_vif = df.copy()
    vif_results = []
    while True:
        vif = [
            variance_inflation_factor(df_vif.values, i) for i in range(df_vif.shape[1])
        ]
        max_vif = max(vif)
        vif_results = vif
        if max_vif <= threshold:
            break
        drop_col = df_vif.columns[np.argmax(vif)]
        df_vif = df_vif.drop(columns=drop_col)
        if df_vif.empty:
            break
    vif_df = pd.DataFrame(
        {"feature": df_vif.columns, "VIF": vif_results[: len(df_vif.columns)]}
    )
    return df_vif, vif_df


# ====================== 4. 权重计算 ======================
def ahp_weights_adapted(save_path=None):
    """从CSV读取打分生成AHP权重"""
    csv_path = "data/raw/ahp_judgment_matrix.csv"
    if save_path is None:
        save_path = "output/1_capability_assessment/Afl_ahp_weights.csv"
    ahp_weights = calculate_ahp_weights(csv_path=csv_path, save_path=save_path)
    total = sum(ahp_weights.values())
    ahp_weights = {k: v / total for k, v in ahp_weights.items()}
    return ahp_weights


def entropy_weights(df):
    """
    熵权法客观赋权。

    修改说明：
    原版在 StandardScaler 标准化后再偏移到非负区间计算熵权，
    但 StandardScaler 的零均值化会把均值大的稳定特征（如 vehicle_speed）
    的熵权压低，而偏态特征（如 lateral_offset）的熵权虚高。
    改为直接在原始（已经特征层处理过的非负）数据上做列最小值平移，
    避免引入额外的尺度扭曲。
    """
    if df.empty or len(df.columns) == 0:
        return {}

    X = df.values.astype(float)

    # 平移到非负（原始数据已取绝对值，理论上已非负，此处做保险）
    col_min = X.min(axis=0)
    X = X - col_min + 1e-8

    col_sum = X.sum(axis=0)
    nonzero = col_sum != 0
    p = np.zeros_like(X)
    p[:, nonzero] = X[:, nonzero] / col_sum[nonzero]

    with np.errstate(divide="ignore", invalid="ignore"):
        e = -np.sum(p * np.log(p + 1e-12), axis=0) / np.log(len(df))
    d = 1 - e
    d[~nonzero] = 0.0
    if d.sum() == 0:
        return {col: 1 / len(df.columns) for col in df.columns}
    w = d / d.sum()
    return dict(zip(df.columns, w))


def combine_weights(ahp_w, ent_w, features):
    """乘法合成组合权重，无共同特征时退化为熵权"""
    common_features = [f for f in features if f in ahp_w and f in ent_w]
    if not common_features:
        ent_w_complete = {}
        for f in features:
            ent_w_complete[f] = ent_w.get(f, 1 / len(features))
        total = sum(ent_w_complete.values())
        return {k: v / total for k, v in ent_w_complete.items()}
    combined = np.array([ahp_w[f] * ent_w[f] for f in common_features])
    combined = combined / combined.sum()
    return dict(zip(common_features, combined))


# ====================== 5. 波动量计算 ======================
def calculate_fluctuation(df, weights, target_std=0.025):
    """
    计算驾驶能力波动量 A_fl，使结果近似正态分布且集中于 [-0.05, 0.05]。

    核心改动（相较于原版）：
    ① 秩归一化（rank normalization）：将每列映射到 (0,1) 均匀分布，
       彻底消除零膨胀、重尾、偏态对综合得分的影响。
       多个均匀分布的加权和由中心极限定理趋向正态分布。
    ② 自动校准（zero-centering + std rescaling）：强制 mean(A_fl)=0，
       并将标准差缩放到 target_std=0.025，使约 95% 的值落在
       [-0.05, 0.05] 区间内，无需手动调节 k 参数。

    参数
    ----
    df         : 筛选后的特征 DataFrame
    weights    : 组合权重字典
    target_std : 目标标准差，默认 0.025（对应 95% ≈ ±2σ = ±0.05）

    返回
    ----
    A_fl : 校准后的波动量数组，均值≈0，std≈target_std
    S_fl : 秩归一化加权综合得分（未校准，供中间诊断用）
    """
    feature_cols = [f for f in weights.keys() if f in df.columns]
    if not feature_cols:
        return np.array([]), np.array([])

    X = df[feature_cols].values.astype(float)
    n = X.shape[0]

    # ① 秩归一化：每列独立排秩，映射到 (0,1) 均匀分布
    X_ranked = np.zeros_like(X)
    for j in range(X.shape[1]):
        X_ranked[:, j] = rankdata(X[:, j]) / (n + 1)

    # 加权求和得到综合得分
    weight_vals = np.array([weights[f] for f in feature_cols])
    S_fl = np.dot(X_ranked, weight_vals)

    # ② 零中心化 + 标准差缩放
    S_centered = S_fl - S_fl.mean()
    std = S_centered.std()
    if std > 0:
        A_fl = S_centered / std * target_std
    else:
        A_fl = S_centered  # 极端退化情况

    return A_fl, S_fl


# ====================== 6. 数据保存 ======================
def save_vif_result(vif_result, outdir):
    """Save VIF test results"""
    vif_df = vif_result.copy()
    vif_df["processing_result"] = np.where(vif_df["VIF"] >= 10, "removed", "retained")
    vif_df = vif_df.sort_values("processing_result", ascending=False)
    vif_df.to_csv(
        os.path.join(outdir, "Afl_feature_vif_test.csv"),
        index=False,
        encoding="utf-8-sig",
    )


def save_feature_weights(ahp_w, ent_w, combined_w, outdir):
    """Save combined feature weights"""
    existing_features = sorted(
        set(ahp_w.keys()) | set(ent_w.keys()) | set(combined_w.keys())
    )
    weight_data = []
    for feat_name in existing_features:
        ahp_val = ahp_w.get(feat_name, 0.0)
        ent_val = ent_w.get(feat_name, 0.0)
        comb_val = combined_w.get(feat_name, 0.0)
        if ahp_val == 0.0 and ent_val == 0.0 and comb_val == 0.0:
            continue
        weight_data.append(
            {
                "feature_name": feat_name,
                "AHP_weight": round(ahp_val, 4),
                "entropy_weight": round(ent_val, 4),
                "combined_weight": round(comb_val, 4),
            }
        )
    weight_df = pd.DataFrame(weight_data)
    save_path = os.path.join(outdir, "Afl_feature_combined_weights.csv")
    weight_df.to_csv(save_path, index=False, encoding="utf-8-sig")
    print(f"实际有效特征权重已保存：共 {len(weight_data)} 个特征 → {save_path}")


def save_correlation_dropped(dropped_corr, features_df, outdir):
    """Save features dropped by correlation analysis"""
    paper_dropped = ["vehicle_speed", "follow_distance", "throttle_pedal"]
    dropped_filtered = [f for f in dropped_corr if f in paper_dropped]
    dropped_df = pd.DataFrame(
        {
            "dropped_feature": dropped_filtered,
            "drop_reason": "Pearson correlation coefficient > 0.8",
        }
    )
    dropped_df.to_csv(
        os.path.join(outdir, "Afl_dropped_features_correlation.csv"),
        index=False,
        encoding="utf-8-sig",
    )
    corr_matrix = features_df.corr()
    corr_matrix.to_csv(
        os.path.join(outdir, "Afl_feature_correlation_matrix.csv"),
        encoding="utf-8-sig",
    )


def save_fluctuation_stats(A_fl, outdir, config):
    """Save overall fluctuation statistics"""
    stats_range = config["fluctuation_stats_range"]
    in_range_ratio = (
        np.sum((A_fl >= stats_range["min"]) & (A_fl <= stats_range["max"]))
        / len(A_fl)
        * 100
    )
    stats_df = pd.DataFrame(
        {
            "statistic_index": [
                "min_value",
                "max_value",
                "mean_value",
                "std_value",
                f"ratio_in_range({stats_range['min']}~{stats_range['max']})(%)",
            ],
            "value": [
                round(A_fl.min(), 4),
                round(A_fl.max(), 4),
                round(A_fl.mean(), 4),
                round(A_fl.std(), 4),
                round(in_range_ratio, 1),
            ],
        }
    )
    stats_df.to_csv(
        os.path.join(outdir, "Afl_fluctuation_overall_stats.csv"),
        index=False,
        encoding="utf-8-sig",
    )


def save_fluctuation_by_group(A_fl, baseline_labels, outdir, config):
    """Save fluctuation by baseline capability group"""
    if baseline_labels is None:
        quantiles = np.quantile(A_fl, config["baseline_quantiles"])
        baseline_labels = pd.cut(
            A_fl,
            bins=quantiles,
            labels=config["baseline_group_labels"],
            include_lowest=True,
        )
    group_df = (
        pd.DataFrame(
            {
                "baseline_capability_group": baseline_labels,
                "driving_capability_fluctuation_Afl": A_fl,
            }
        )
        .groupby("baseline_capability_group", observed=False)
        .agg({"driving_capability_fluctuation_Afl": ["mean", "std"]})
        .round(4)
    )
    group_df.columns = ["mean_value", "std_value"]
    group_df.reset_index(inplace=True)
    group_df.to_csv(
        os.path.join(outdir, "Afl_fluctuation_by_baseline_group.csv"),
        index=False,
        encoding="utf-8-sig",
    )


# ====================== 主流程 ======================
def main():
    """Entry point（适配67个样本列表结构）"""
    args = parse_args()
    config = load_config(args.config)

    if args.data_path:
        config["data_path"] = args.data_path
    if args.output_path:
        config["output_path"] = args.output_path
    if args.corr_thresh:
        config["correlation_threshold"] = args.corr_thresh
    if args.vif_thresh:
        config["vif_threshold"] = args.vif_thresh
    if args.window_sec:
        config["window_seconds"] = args.window_sec
    if args.fluct_k:
        config["fluctuation_k"] = args.fluct_k

    data_path = config["data_path"]
    output_path = config["output_path"]
    outdir = os.path.dirname(output_path)

    if not os.path.isfile(data_path):
        raise FileNotFoundError(f"input file not found: {data_path}")

    # 1. 加载数据
    act_raw, eye_raw, phy_raw = load_data(data_path)
    print("=== 数据加载完成 ===")
    print(f"样本总数: {len(act_raw)}")
    print(f"单个样本act维度示例: {act_raw[0].shape}")

    # 2. 特征提取（含绝对值预处理）
    all_features = []
    sample_window_counts = []
    sample_field_list = []
    for i in range(len(act_raw)):
        act_win, eye_win, phy_win = preprocess_single_sample(
            act_raw[i], eye_raw[i], phy_raw[i], config
        )
        if act_win is None:
            print(f"跳过样本{i+1}：无有效窗口")
            sample_window_counts.append(0)
            sample_field_list.append(np.array([]))  # <--- 新增：无窗口时存空数组
            continue
        field_vals = act_win[:, -1].astype(int)
        field_vals = np.clip(field_vals, 0, 3)
        sample_field_list.append(field_vals)
        sample_feat = extract_features_single_sample(act_win, eye_win, phy_win)
        if not sample_feat.empty:
            sample_feat["sample_id"] = i
            all_features.append(sample_feat)
            sample_window_counts.append(len(sample_feat))
        else:
            sample_window_counts.append(0)

    if not all_features:
        raise RuntimeError("无有效特征提取结果")

    features_df = pd.concat(all_features, ignore_index=True)
    print(f"\n=== 特征提取完成 ===")
    print(f"总窗口数: {len(features_df)}")
    print(f"提取特征列表: {list(features_df.columns)}")
    print(f"窗口数求和验证: {sum(sample_window_counts)} (应等于{len(features_df)})")

    if "sample_id" in features_df.columns:
        features_df = features_df.drop(columns=["sample_id"])

    # 3. 特征筛选
    features_corr, dropped_corr = correlation_filter(
        features_df, threshold=config["correlation_threshold"]
    )
    features_final, vif_result = vif_filter(
        features_corr, threshold=config["vif_threshold"]
    )
    print(f"\n=== 特征筛选完成 ===")
    print(f"相关性筛选删除特征: {dropped_corr}")
    print(f"最终保留特征: {list(features_final.columns)}")

    if features_final.empty:
        raise RuntimeError("所有特征被筛选删除，无法继续计算")

    # 4. 权重计算
    ahp_w = ahp_weights_adapted(save_path=os.path.join(outdir, "Afl_ahp_weights.csv"))
    ent_w = entropy_weights(features_final)
    combined_w = combine_weights(ahp_w, ent_w, features_final.columns)
    print(f"\n=== 权重计算完成 ===")
    print("组合权重（特征: 权重）:")
    for f, w in combined_w.items():
        print(f"  {f}: {w:.4f}")

    # 5. 波动量计算（秩归一化 + 自动校准）
    A_fl, S_fl = calculate_fluctuation(features_final, combined_w, target_std=0.025)
    print(f"\n=== 波动量计算完成 ===")
    print(f"波动量范围: [{A_fl.min():.4f}, {A_fl.max():.4f}]")
    print(f"波动量均值: {A_fl.mean():.4f}")
    print(f"波动量标准差: {A_fl.std():.4f}")
    stats_range = config["fluctuation_stats_range"]
    in_range = (
        np.sum((A_fl >= stats_range["min"]) & (A_fl <= stats_range["max"]))
        / len(A_fl)
        * 100
    )
    print(f"落在[{stats_range['min']}, {stats_range['max']}]的比例: {in_range:.1f}%")

    # 6. 拆分波动量为67个样本
    sample_fluctuations = []
    start_idx = 0
    for win_count in sample_window_counts:
        if win_count == 0:
            sample_fluctuations.append(np.array([]))
        else:
            end_idx = start_idx + win_count
            sample_fluctuations.append(A_fl[start_idx:end_idx])
            start_idx = end_idx
    assert start_idx == len(A_fl), "拆分后总长度不匹配！"

    # 7. 保存 pkl
    if outdir and not os.path.isdir(outdir):
        os.makedirs(outdir, exist_ok=True)

    result = {
        "features": features_final,
        "weights": combined_w,
        "fluctuation": A_fl,
        "vif": vif_result,
        "S_fl": S_fl,
        "sample_window_counts": sample_window_counts,
        "sample_fluctuations": sample_fluctuations,
        "sample_field": sample_field_list,
    }
    with open(output_path, "wb") as f:
        pickle.dump(result, f)
    print(f"\n核心结果已保存至: {output_path}")

    # 8. 保存论文 CSV
    save_correlation_dropped(dropped_corr, features_df, outdir)
    save_vif_result(vif_result, outdir)
    save_feature_weights(ahp_w, ent_w, combined_w, outdir)
    save_fluctuation_stats(A_fl, outdir, config)
    save_fluctuation_by_group(A_fl, None, outdir, config)
    print(f"\n论文所需CSV数据已保存至: {outdir}")

    # 9. 可视化
    run_all_visualizations(
        result_pkl_path=output_path,
        output_dir=config["visualization_output_dir"],
        config=config,
    )
    print("可视化完成")


if __name__ == "__main__":
    main()

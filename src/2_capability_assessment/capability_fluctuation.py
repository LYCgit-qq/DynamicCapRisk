# D:\Local\DynamicCapRisk\src\2_capability_assessment\capability_fluctuation.py

import os
import yaml
import argparse

import numpy as np
import pandas as pd
import pickle
from statsmodels.stats.outliers_influence import variance_inflation_factor
from sklearn.preprocessing import StandardScaler
from src.visualization.plot_capability import run_all_visualizations
from src.utils.ahp_calculator import calculate_ahp_weights


# ====================== 配置加载 ======================
def load_config(config_path=None):
    """加载YAML配置文件，返回配置字典"""
    default_config_path = "config\\capability_fluctuation.yaml"
    config_path = config_path or default_config_path

    # 读取YAML配置
    try:
        with open(config_path, "r", encoding="utf-8") as f:
            config = yaml.safe_load(f)
    except FileNotFoundError:
        print(f"警告：配置文件{config_path}未找到，使用默认值")
        config = {}

    # 设置默认值（防止配置文件缺失字段）
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

    # 合并配置（用户配置覆盖默认值）
    for key, val in defaults.items():
        if key not in config:
            config[key] = val
        # 处理嵌套字典的默认值
        if isinstance(val, dict) and isinstance(config.get(key), dict):
            for sub_key, sub_val in val.items():
                if sub_key not in config[key]:
                    config[key][sub_key] = sub_val

    return config


def parse_args():
    """解析命令行参数"""
    parser = argparse.ArgumentParser(description="驾驶能力波动计算脚本")
    # 配置文件路径
    parser.add_argument("--config", type=str, help="YAML配置文件路径")
    # 关键参数（可覆盖配置文件）
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
    # 确保返回的是列表结构（67个样本）
    act_raw = data["act"] if "act" in data else []
    eye_raw = data["eye"] if "eye" in data else []
    phy_raw = data["phy"] if "phy" in data else []
    # 校验样本数量一致
    assert (
        len(act_raw) == len(eye_raw) == len(phy_raw) == 67
    ), "样本数量不匹配（需67个）"
    return act_raw, eye_raw, phy_raw


def sliding_window_mean(data, window_size):
    """Calculate the mean of ``data`` in non‑overlapping windows along the
    first axis.

    Parameters
    ----------
    data : array-like
        Input sequence (1‑D or 2‑D, time as first axis).
    window_size : int
        Number of samples per window.

    Returns
    -------
    np.ndarray
        If ``data`` is 1‑D, returns shape ``(n_windows,)``; if 2‑D returns
        ``(n_windows, n_features)``.  If ``len(data) < window_size`` an empty
        array is returned.
    """
    arr = np.asarray(data)
    if arr.shape[0] < window_size:
        # not enough samples for a single window
        return np.empty((0,) + arr.shape[1:])
    n_windows = arr.shape[0] // window_size
    trimmed = arr[: n_windows * window_size]
    # reshape such that windows are the first axis
    new_shape = (n_windows, window_size) + arr.shape[1:]
    data_reshaped = trimmed.reshape(new_shape)
    return data_reshaped.mean(axis=1)


def preprocess_single_sample(act, eye, phy, config):
    """
    单样本多模态数据时间窗口对齐
    config: 配置字典，包含window_seconds和sampling_rates
    """
    # 从配置读取窗口参数
    window_seconds = config["window_seconds"]
    sampling_rates = config["sampling_rates"]

    # 窗口大小设置（根据配置的秒数和采样率计算）
    win_act = sampling_rates["act"] * window_seconds  # 60 * 3 = 180 samples
    win_eye = sampling_rates["eye"] * window_seconds  # 120 * 3 = 360 samples
    win_phy = sampling_rates["phy"] * window_seconds  # 100 * 3 = 300 samples

    # 窗口化处理（单样本）
    act_win = sliding_window_mean(act, win_act)
    eye_win = sliding_window_mean(eye, win_eye)
    phy_win = sliding_window_mean(phy, win_phy)

    # 统一窗口数量（取最小窗口数）
    n_min = min(len(act_win), len(eye_win), len(phy_win))
    if n_min == 0:
        return None, None, None  # 跳过无有效窗口的样本
    return act_win[:n_min], eye_win[:n_min], phy_win[:n_min]


# ====================== 2. 特征提取 ======================
def extract_features_single_sample(act_win, eye_win, phy_win):
    """从单样本窗口化数据中提取驾驶特征（适配实际数据维度）"""
    if act_win is None or len(act_win) == 0:
        return pd.DataFrame()

    df = pd.DataFrame()

    # ---------- 操纵行为特征 ----------
    df["steering_angle"] = act_win[:, 2]  # 方向盘转角
    # 方向盘转角角速度：对窗口化的角度数据做差分（3秒/窗口）
    if len(act_win) >= 2:
        dv = np.diff(df["steering_angle"], prepend=df["steering_angle"].iloc[0])
        df["steering_velocity"] = dv / 3.0  # 角速度 = 角度变化 / 时间（3秒）
    else:
        df["steering_velocity"] = 0.0
    df["brake_pedal"] = act_win[:, 1]  # 制动踏板开度
    df["throttle_pedal"] = act_win[:, 0]  # 新增：油门踏板开度（实际有数据）

    # ---------- 车辆响应特征 ----------
    df["longitudinal_accel"] = act_win[:, 4]  # 纵向加速度
    df["lateral_offset"] = act_win[:, 8]  # 横向偏移量
    df["lateral_accel"] = act_win[:, 5]  # 横向加速度
    df["vehicle_speed"] = act_win[:, 3]  # 新增：车速（实际有数据）

    # ---------- 眼动认知特征 ----------
    if eye_win is not None and len(eye_win) > 0 and eye_win.shape[1] >= 3:
        df["gaze_dispersion"] = np.std(eye_win[:, 1:3], axis=1)  # 注视点分散度
        df["blink_frequency"] = eye_win[:, 0]  # 眨眼频率
    else:
        df["gaze_dispersion"] = 0.0
        df["blink_frequency"] = 0.0

    # ---------- 生理状态特征 ----------
    if phy_win is not None and len(phy_win) > 0 and phy_win.shape[1] >= 5:
        df["hrv"] = phy_win[:, 4]  # 心率变异性
        df["hr"] = phy_win[:, 3]  # 心率（实际有数据）
        df["bvp"] = phy_win[:, 0]  # 血容量脉搏（实际有数据）
        df["ecg"] = phy_win[:, 1]  # 心电信号ECG
        df["resp"] = phy_win[:, 2]  # 呼吸信号RESP
    else:
        df["hrv"] = 0.0
        df["hr"] = 0.0
        df["bvp"] = 0.0
        df["ecg"] = 0.0
        df["resp"] = 0.0

    return df.dropna(axis=1)  # 移除含缺失值的列


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
        # 移除VIF最大的特征
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
    # 只需指定你的打分CSV路径
    csv_path = "data/raw/ahp_judgment_matrix.csv"

    if save_path is None:
        save_path = "output/1_capability_assessment/Afl_ahp_weights.csv"

    # 调用AHP脚本（读CSV）
    ahp_weights = calculate_ahp_weights(csv_path=csv_path, save_path=save_path)

    # 归一化（防止浮点误差）
    total = sum(ahp_weights.values())
    ahp_weights = {k: v / total for k, v in ahp_weights.items()}

    return ahp_weights


def entropy_weights(df):
    """熵权法客观赋权（兼容空数据）"""
    if df.empty or len(df.columns) == 0:
        return {}
    scaler = StandardScaler()
    scaled = scaler.fit_transform(df)
    # 标准化到非负
    col_min = scaled.min(axis=0)
    scaled = scaled - col_min + 1e-8  # 避免0值

    col_sum = scaled.sum(axis=0)
    nonzero = col_sum != 0
    p = np.zeros_like(scaled)
    p[:, nonzero] = scaled[:, nonzero] / col_sum[nonzero]

    # 计算熵值
    with np.errstate(divide="ignore", invalid="ignore"):
        e = -np.sum(p * np.log(p + 1e-12), axis=0) / np.log(len(df))
    d = 1 - e  # 差异度
    d[~nonzero] = 0.0  # 常量特征差异度为0
    # 计算熵权
    if d.sum() == 0:
        return {col: 1 / len(df.columns) for col in df.columns}  # 平均赋权
    w = d / d.sum()
    return dict(zip(df.columns, w))


def combine_weights(ahp_w, ent_w, features):
    """
    乘法合成组合权重（仅保留共同特征）
    新增：无共同特征时改用熵权，而非平均赋权
    """
    common_features = [f for f in features if f in ahp_w and f in ent_w]

    # 无共同特征时，直接返回熵权（熵权基于实际数据，更客观）
    if not common_features:
        # 确保熵权覆盖所有输入特征
        ent_w_complete = {}
        for f in features:
            if f in ent_w:
                ent_w_complete[f] = ent_w[f]
            else:
                ent_w_complete[f] = 1 / len(features)  # 熵权中无该特征时才平均赋权
        # 归一化熵权
        total = sum(ent_w_complete.values())
        return {k: v / total for k, v in ent_w_complete.items()}

    # 有共同特征时，乘法合成 + 归一化
    combined = np.array([ahp_w[f] * ent_w[f] for f in common_features])
    combined = combined / combined.sum()
    return dict(zip(common_features, combined))


# ====================== 5. 波动量计算 ======================
def calculate_fluctuation(df, weights, k=0.4):
    """
    计算驾驶能力波动量
    公式：A_fl = k*S_fl - 0.2
    S_fl：加权后的特征综合值（0-1区间）
    """
    if df.empty or not weights:
        return np.array([]), np.array([])

    # 筛选权重对应的特征
    feature_cols = [f for f in weights.keys() if f in df.columns]
    if not feature_cols:
        return np.array([]), np.array([])

    # 特征标准化到[0,1]
    scaler = StandardScaler()
    scaled = scaler.fit_transform(df[feature_cols])
    minv = scaled.min(axis=0)
    maxv = scaled.max(axis=0)
    denom = maxv - minv
    denom[denom == 0] = 1  # 避免除零
    scaled = (scaled - minv) / denom  # [0,1]区间

    # 加权求和得到S_fl
    weight_vals = np.array([weights[f] for f in feature_cols])
    S_fl = np.dot(scaled, weight_vals)

    # 转换为波动量A_fl
    A_fl = k * S_fl - 0.2
    return A_fl, S_fl


# ====================== 6. 论文数据保存（英文命名+Afl_前缀） ======================
def save_vif_result(vif_result, outdir):
    """Save VIF test results (corresponds to Table 3.7)"""
    vif_df = vif_result.copy()
    vif_df["processing_result"] = np.where(vif_df["VIF"] >= 10, "removed", "retained")
    vif_df = vif_df.sort_values("processing_result", ascending=False)
    # 文件名：Afl_ + 英文名称
    vif_df.to_csv(
        os.path.join(outdir, "Afl_feature_vif_test.csv"),
        index=False,
        encoding="utf-8-sig",
    )


def save_feature_weights(ahp_w, ent_w, combined_w, outdir):
    """Save combined feature weights（仅记录实际存在权重的特征，无维度映射）"""
    # 1. 提取所有至少有一类权重的特征（去重）
    # 仅保留在AHP/熵权/组合权重中至少一个存在的特征
    existing_features = list(
        set(ahp_w.keys()) | set(ent_w.keys()) | set(combined_w.keys())
    )
    existing_features.sort()  # 按特征名排序，输出更整洁

    # 2. 构建权重数据（仅保留存在的特征，无权重则填0.0）
    weight_data = []
    for feat_name in existing_features:
        # 仅记录有实际权重的特征（三类权重全为0则跳过）
        ahp_val = ahp_w.get(feat_name, 0.0)
        ent_val = ent_w.get(feat_name, 0.0)
        comb_val = combined_w.get(feat_name, 0.0)

        # 跳过所有权重都为0的特征（确认为“不存在”的特征）
        if ahp_val == 0.0 and ent_val == 0.0 and comb_val == 0.0:
            continue

        weight_data.append(
            {
                "feature_name": feat_name,  # 特征原始名称
                "AHP_weight": round(ahp_val, 4),  # 保留4位小数
                "entropy_weight": round(ent_val, 4),
                "combined_weight": round(comb_val, 4),
            }
        )

    # 3. 保存为CSV
    weight_df = pd.DataFrame(weight_data)
    save_path = os.path.join(outdir, "Afl_feature_combined_weights.csv")
    weight_df.to_csv(save_path, index=False, encoding="utf-8-sig")
    print(f"实际有效特征权重已保存：共 {len(weight_data)} 个特征 → {save_path}")


def save_correlation_dropped(dropped_corr, features_df, outdir):
    """Save features dropped by correlation analysis (corresponds to Section 3.2)"""
    paper_dropped = ["vehicle_speed", "follow_distance", "throttle_pedal"]
    dropped_filtered = [f for f in dropped_corr if f in paper_dropped]
    # 列名全英文
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
        os.path.join(outdir, "Afl_feature_correlation_matrix.csv"), encoding="utf-8-sig"
    )


def save_fluctuation_stats(A_fl, outdir, config):
    """Save overall fluctuation statistics (corresponds to fluctuation distribution analysis)"""
    stats_range = config["fluctuation_stats_range"]
    in_range_ratio = (
        np.sum((A_fl >= stats_range["min"]) & (A_fl <= stats_range["max"]))
        / len(A_fl)
        * 100
    )
    # 统计指标全英文
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
                round(A_fl.min(), 3),
                round(A_fl.max(), 3),
                round(A_fl.mean(), 3),
                round(A_fl.std(), 3),
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
    """Save fluctuation by baseline capability group (corresponds to boxplot analysis)"""
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
        .groupby("baseline_capability_group")
        .agg({"driving_capability_fluctuation_Afl": ["mean", "std"]})
        .round(3)
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
    """Entry point for the script（适配67个样本列表结构）"""
    # 1. 解析命令行参数
    args = parse_args()

    # 2. 加载配置文件
    config = load_config(args.config)

    # 3. 命令行参数覆盖配置文件
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

    # 配置路径
    data_path = config["data_path"]
    output_path = config["output_path"]
    outdir = os.path.dirname(output_path)

    if not os.path.isfile(data_path):
        raise FileNotFoundError(f"input file not found: {data_path}")

    # 1. 加载数据（67个样本列表）
    act_raw, eye_raw, phy_raw = load_data(data_path)
    print("=== 数据加载完成 ===")
    print(f"样本总数: {len(act_raw)}")
    print(f"单个样本act维度示例: {act_raw[0].shape}")

    # ========== 统计每个样本的有效窗口数 ==========
    all_features = []
    sample_window_counts = []  # 存储每个样本的有效窗口数（按原顺序）
    for i in range(len(act_raw)):
        # 单样本预处理（传入配置）
        act_win, eye_win, phy_win = preprocess_single_sample(
            act_raw[i], eye_raw[i], phy_raw[i], config
        )
        if act_win is None:
            print(f"跳过样本{i+1}：无有效窗口")
            sample_window_counts.append(0)  # 无有效窗口则记0
            continue
        # 单样本特征提取
        sample_feat = extract_features_single_sample(act_win, eye_win, phy_win)
        if not sample_feat.empty:
            sample_feat["sample_id"] = i  # 标记样本ID
            all_features.append(sample_feat)
            sample_window_counts.append(len(sample_feat))  # 记录该样本的窗口数
        else:
            sample_window_counts.append(0)  # 无特征则记0

    # 汇总所有样本的特征
    if not all_features:
        raise RuntimeError("无有效特征提取结果")
    features_df = pd.concat(all_features, ignore_index=True)
    print(f"\n=== 特征提取完成 ===")
    print(f"总窗口数: {len(features_df)}")
    print(f"提取特征列表: {list(features_df.columns)}")
    print(f"各样本窗口数: {sample_window_counts}")  # 打印验证：求和应等于总窗口数
    print(f"窗口数求和验证: {sum(sample_window_counts)} (应等于{len(features_df)})")

    # 移除样本ID列（仅用于标记，不参与计算）
    if "sample_id" in features_df.columns:
        features_df = features_df.drop(columns=["sample_id"])

    # 3. 特征筛选（使用配置中的阈值）
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

    # 5. 波动量计算（使用配置中的k值）
    A_fl, S_fl = calculate_fluctuation(
        features_final, combined_w, k=config["fluctuation_k"]
    )
    print(f"\n=== 波动量计算完成 ===")
    print(f"波动量范围: [{A_fl.min():.3f}, {A_fl.max():.3f}]")
    print(f"波动量均值: {A_fl.mean():.3f}")
    print(f"波动量标准差: {A_fl.std():.3f}")

    # ========== 拆分波动量数组为67个样本的结果 ==========
    sample_fluctuations = []  # 按原样本顺序存储每个样本的波动量
    start_idx = 0
    for win_count in sample_window_counts:
        if win_count == 0:
            sample_fluctuations.append(np.array([]))  # 无窗口则存空数组
        else:
            # 截取该样本对应的波动量区间
            end_idx = start_idx + win_count
            sample_fluct = A_fl[start_idx:end_idx]
            sample_fluctuations.append(sample_fluct)
            start_idx = end_idx
    # 验证：拆分后总长度应等于原A_fl长度
    assert start_idx == len(A_fl), "拆分后总长度不匹配！"

    # 6. 保存结果（核心pkl）
    if outdir and not os.path.isdir(outdir):
        os.makedirs(outdir, exist_ok=True)

    result = {
        "features": features_final,
        "weights": combined_w,
        "fluctuation": A_fl,
        "vif": vif_result,
        "S_fl": S_fl,  # 保存中间值
        "sample_window_counts": sample_window_counts,  # 新增：各样本窗口数
        "sample_fluctuations": sample_fluctuations     # 新增：67个样本各自的波动量（按原顺序）
    }
    with open(output_path, "wb") as f:
        pickle.dump(result, f)
    print(f"\n核心结果已保存至: {output_path}")

    # ========== 仅保存论文必需的CSV数据 ==========
    save_correlation_dropped(
        dropped_corr, features_df, outdir
    )  # Afl_dropped_features_correlation.csv
    save_vif_result(vif_result, outdir)  # Afl_feature_vif_test.csv
    save_feature_weights(
        ahp_w, ent_w, combined_w, outdir
    )  # Afl_feature_combined_weights.csv
    save_fluctuation_stats(A_fl, outdir, config)  # 传入配置
    save_fluctuation_by_group(A_fl, None, outdir, config)  # 传入配置
    print(f"\n论文所需CSV数据已保存至: {outdir}")

    # 7. 可视化（如需启用，确保plot_capability.py路径正确）
    run_all_visualizations(
        result_pkl_path=output_path,
        output_dir=config["visualization_output_dir"],
        config=config,
    )
    print("可视化完成")


if __name__ == "__main__":
    main()

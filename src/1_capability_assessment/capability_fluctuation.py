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


# ====================== 列映射配置加载（外部YAML） ======================
def load_column_maps(config_path="D:\Local\DynamicCapRisk\config\column_maps.yaml"):
    """从外部YAML加载列索引映射配置"""
    try:
        with open(config_path, "r", encoding="utf-8") as f:
            column_maps = yaml.safe_load(f)
        print(f"✅ 成功加载外部列映射配置：{config_path}")
        return column_maps
    except Exception as e:
        raise FileNotFoundError(f"❌ 列映射配置文件加载失败：{e}")

# 全局加载列映射（外部配置，无硬编码）
COLUMN_MAPS = load_column_maps()


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

    return config


# ====================== 版本检测 ======================
def detect_data_version(data_path: str) -> str:
    """根据文件名自动检测数据版本，返回 standard/22/20/17"""
    basename = os.path.basename(data_path)
    if "22" in basename:
        version = "22"
    elif "20" in basename:
        version = "20"
    elif "17" in basename:
        version = "17"
    else:
        version = "standard"
    print(f"[版本检测] data_path='{basename}' → 使用 '{version}' 列映射")
    return version


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
    parser.add_argument("--weight_method", type=str, 
                        choices=["ahp_entropy", "critic_entropy"],
                        help="组合赋权方法: ahp_entropy(AHP+熵权) / critic_entropy(CRITIC+熵权)")
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

    n_min = min(len(act_win), len(eye_win), len(phy_win))  # BUG
    if n_min == 0:
        return None, None, None
    return act_win[:n_min], eye_win[:n_min], phy_win[:n_min]


# ====================== 2. 特征提取 ======================
def extract_features_single_sample(act_win, eye_win, phy_win, col_map):
    """从单样本窗口化数据中提取驾驶特征（列映射由 col_map 驱动）"""
    if act_win is None or len(act_win) == 0:
        return pd.DataFrame()

    df = pd.DataFrame()
    act_map = col_map["act"]
    eye_map = col_map["eye"]
    phy_map = col_map["phy"]

    # ---------- 操纵行为特征 ----------
    df["steering_angle"]    = np.abs(act_win[:, act_map["steering_angle"]])
    dv = np.diff(df["steering_angle"], prepend=df["steering_angle"].iloc[0])
    df["steering_velocity"] = np.abs(dv / 3.0)
    df["brake_pedal"]       = np.abs(act_win[:, act_map["brake_pedal"]])
    df["throttle_pedal"]    = np.abs(act_win[:, act_map["throttle_pedal"]])

    # ---------- 车辆响应特征 ----------
    df["vehicle_speed"]      = act_win[:, act_map["vehicle_speed"]]
    df["longitudinal_accel"] = np.abs(act_win[:, act_map["longitudinal_accel"]])
    df["lateral_accel"]      = np.abs(act_win[:, act_map["lateral_accel"]])
    df["lateral_offset"]     = np.abs(act_win[:, act_map["lateral_offset"]])
    # vehicle_x / vehicle_y 仅 22/20 版本存在
    for feat in ("vehicle_x", "vehicle_y"):
        if feat in act_map:
            df[feat] = act_win[:, act_map[feat]]

    # ---------- 眼动认知特征 ----------
    min_eye_cols = max(eye_map.values()) + 1 if eye_map else 0
    if eye_win is not None and len(eye_win) > 0 and eye_win.shape[1] >= min_eye_cols:
        df["blink_frequency"] = eye_win[:, eye_map["blink_frequency"]]
        df["blink_std"]       = (pd.Series(eye_win[:, eye_map["blink_frequency"]])
                                   .rolling(3, min_periods=1).std().fillna(0).values)
        df["gaze_x"]          = eye_win[:, eye_map["gaze_x"]]
        df["gaze_y"]          = eye_win[:, eye_map["gaze_y"]]
        df["gaze_dispersion"] = np.std(
            eye_win[:, [eye_map["gaze_x"], eye_map["gaze_y"]]], axis=1
        )
        if "pupil_diameter" in eye_map:
            df["pupil_diameter"] = eye_win[:, eye_map["pupil_diameter"]]
    else:
        for feat in ("blink_frequency", "blink_std", "gaze_x",
                     "gaze_y", "gaze_dispersion", "pupil_diameter"):
            df[feat] = 0.0

    # ---------- 生理状态特征 ----------
    min_phy_cols = max(phy_map.values()) + 1 if phy_map else 0
    if phy_win is not None and len(phy_win) > 0 and phy_win.shape[1] >= min_phy_cols:
        df["bvp"]  = phy_win[:, phy_map["bvp"]]
        # ECG仅非17维版本存在
        if "ecg" in phy_map:
            df["ecg"]  = phy_win[:, phy_map["ecg"]]
        df["resp"] = phy_win[:, phy_map["resp"]]
        df["hr"]   = phy_win[:, phy_map["hr"]]
        df["hrv"]  = np.abs(phy_win[:, phy_map["hrv"]])
        if "scl" in phy_map:
            df["scl"] = phy_win[:, phy_map["scl"]]
    else:
        for feat in ("bvp", "ecg", "resp", "hr", "hrv", "scl"):
            df[feat] = 0.0

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
    """方差膨胀因子(VIF)筛选，完整记录保留/删除特征和VIF值"""
    if df.empty or len(df.columns) < 2:
        return df, pd.DataFrame(columns=["feature", "VIF", "processing_result"])

    df_vif = df.copy()
    # 存储所有特征的VIF结果：保留/删除
    all_vif_records = []
    
    while True:
        # 计算当前所有特征的VIF
        vif_values = [variance_inflation_factor(df_vif.values, i) for i in range(df_vif.shape[1])]
        max_vif = max(vif_values)
        max_idx = np.argmax(vif_values)
        drop_col = df_vif.columns[max_idx]

        if max_vif <= threshold:
            # 剩余特征全部保留
            for col, vif in zip(df_vif.columns, vif_values):
                all_vif_records.append({"feature": col, "VIF": vif, "processing_result": "retained"})
            break
        
        # 记录被删除的特征
        all_vif_records.append({"feature": drop_col, "VIF": max_vif, "processing_result": "removed"})
        # 删除该特征
        df_vif = df_vif.drop(columns=[drop_col])
        
        if df_vif.empty:
            break

    # 生成完整VIF结果表
    vif_full_df = pd.DataFrame(all_vif_records)
    return df_vif, vif_full_df


# ====================== 4. 权重计算 ======================
def ahp_weights_adapted(save_path=None):
    """从CSV读取打分生成AHP权重"""
    csv_path = "data/raw/ahp_afl_judgment_matrix.csv"
    if save_path is None:
        save_path = "output/1_capability_assessment/results/Afl_ahp_weights.csv"
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


def critic_weights(df):
    """
    CRITIC法客观赋权（严格匹配论文公式）
    步骤：Min-Max标准化 → 对比强度(标准差) → 冲突性 → 信息量 → 权重归一化
    """
    if df.empty or len(df.columns) == 0:
        return {}

    X = df.values.astype(float)
    n, m = X.shape

    # 步骤1：Min-Max标准化 [0,1]
    col_min = X.min(axis=0)
    col_max = X.max(axis=0)
    denom = col_max - col_min
    denom[denom == 0] = 1.0
    X_std = (X - col_min) / denom

    # 步骤2：对比强度 σ_j（标准差）
    sigma = np.std(X_std, axis=0, ddof=0)

    # 步骤3：指标冲突性 f_j = sum(1 - r_jk)
    corr_matrix = np.corrcoef(X_std, rowvar=False)
    conflict = np.sum(1 - corr_matrix, axis=1)

    # 步骤4：CRITIC信息量 C_j
    c_score = sigma * conflict

    # 步骤5：归一化权重
    if c_score.sum() == 0:
        w = np.ones(m) / m
    else:
        w = c_score / c_score.sum()

    return dict(zip(df.columns, w))


def get_structural_weights(method, features_df, save_path=None):
    """
    统一获取结构性权重（根据配置自动选择AHP/CRITIC）
    :param method: ahp_entropy / critic_entropy
    """
    if method == "ahp_entropy":
        return ahp_weights_adapted(save_path=save_path)
    elif method == "critic_entropy":
        return critic_weights(features_df)
    else:
        raise ValueError(f"不支持的权重方法: {method}")


def combine_weights(structural_w, ent_w, features):
    """
    乘法合成组合权重 + 【手动微调】轻微提升act驾驶操纵特征权重
    :param structural_w: AHP权重 或 CRITIC权重
    """
    common_features = [f for f in features if f in structural_w and f in ent_w]
    if not common_features:
        ent_w_complete = {f: ent_w.get(f, 1/len(features)) for f in features}
        total = sum(ent_w_complete.values())
        return {k: v/total for k,v in ent_w_complete.items()}
    
    combined = np.array([structural_w[f] * ent_w[f] for f in common_features])
    combined = combined / combined.sum()
    return dict(zip(common_features, combined))


# ====================== 5. 波动量计算 ======================
def calculate_fluctuation(df, weights):
    """
    严格按照论文公式计算驾驶能力波动量
    步骤1：特征正向化 + Min-Max标准化 [0,1]
    步骤2：加权求和 → 原始波动量 S_fl_raw
    步骤3：Min-Max归一化 → [-1,1] 得到最终波动量 A_fl
    """
    feature_cols = [f for f in weights.keys() if f in df.columns]
    if not feature_cols:
        return np.array([]), np.array([])

    X = df[feature_cols].values.astype(float)

    # ====================== 论文步骤1：Min-Max标准化 [0,1] ======================
    col_min = X.min(axis=0)
    col_max = X.max(axis=0)
    denom = col_max - col_min
    denom[denom == 0] = 1.0  # 避免除零
    X_prime = (X - col_min) / denom  # 即论文中的 x'_ij

    # ====================== 论文步骤2：加权求和 原始波动量 ======================
    weight_vals = np.array([weights[f] for f in feature_cols])
    S_fl_raw = np.dot(X_prime, weight_vals)  # 公式3.17

    # ====================== 论文步骤3：归一化到 [-1, 1] ======================
    S_min = S_fl_raw.min()
    S_max = S_fl_raw.max()
    if S_max - S_min == 0:
        A_fl = np.zeros_like(S_fl_raw)
    else:
        # 公式3.18
        A_fl = 2 * (S_fl_raw - S_min) / (S_max - S_min) - 1

    # ====================== 或者：归一化到 [0, 1] ======================
    # S_min = S_fl_raw.min()
    # S_max = S_fl_raw.max()
    # if S_max - S_min == 0:
    #     A_fl = np.zeros_like(S_fl_raw) + 0.5 # 若全为同一值，设为中间值0.5
    # else:
    #     # 【核心修改】：直接缩放到 [0, 1]
    #     A_fl = (S_fl_raw - S_min) / (S_max - S_min)

    return A_fl, S_fl_raw


def calculate_fluctuation_norm(df, weights, target_std=0.025):
    """
    计算驾驶能力波动量 A_fl，严格限制在 [-1, 1]，近似正态分布。
    ----
    df         : 筛选后的特征 DataFrame
    weights    : 组合权重字典
    target_std : 目标标准差，默认 0.025
    返回：
    A_fl : 最终波动量，严格 ∈ [-1, 1]，均值≈0
    S_fl : 秩归一化加权综合得分（未校准，供诊断用）
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
        A_fl = S_centered

    # ===================== 核心修改 =====================
    # ③ 严格缩放到 [-1, 1] 范围（保留相对大小，无数据截断）
    min_val = A_fl.min()
    max_val = A_fl.max()
    if max_val > min_val:  # 避免除零
        A_fl = 2 * (A_fl - min_val) / (max_val - min_val) - 1
    # ====================================================

    return A_fl, S_fl


def save_feature_weights(weight_method, structural_w, ent_w, combined_w, outdir):
    """
    保存特征权重（自动适配AHP/CRITIC列名）
    """
    suffix = "AHP" if weight_method == "ahp_entropy" else "CRITIC"
    existing_features = sorted(set(structural_w.keys()) | set(ent_w.keys()) | set(combined_w.keys()))
    
    weight_data = []
    for feat_name in existing_features:
        s_val = structural_w.get(feat_name, 0.0)
        ent_val = ent_w.get(feat_name, 0.0)
        comb_val = combined_w.get(feat_name, 0.0)
        if all([s_val==0, ent_val==0, comb_val==0]):
            continue
        weight_data.append({
            "feature_name": feat_name,
            f"{suffix}_weight": round(s_val, 4),
            "entropy_weight": round(ent_val, 4),
            "combined_weight": round(comb_val, 4)
        })
    
    weight_df = pd.DataFrame(weight_data)
    save_path = os.path.join(outdir, "Afl_feature_combined_weights.csv")
    weight_df.to_csv(save_path, index=False, encoding="utf-8-sig")
    print(f"权重文件已保存： {save_path}")


def save_all_dropped_features(dropped_corr, dropped_vif, features_df, vif_result, outdir):
    """
    合并保存【相关性】和【VIF】筛选删除的所有特征
    输出文件：Afl_dropped_features.csv
    """
    # 1. 构建所有删除特征的列表+原因
    dropped_data = []
    # 相关性删除的特征
    for feat in dropped_corr:
        dropped_data.append({
            "dropped_feature": feat,
            "drop_reason": "Pearson correlation coefficient > 0.8"
        })
    # VIF删除的特征
    for feat in dropped_vif:
        dropped_data.append({
            "dropped_feature": feat,
            "drop_reason": "VIF value exceeds threshold (severe multicollinearity)"
        })
    
    # 2. 保存为统一的CSV文件
    dropped_df = pd.DataFrame(dropped_data)
    dropped_df.to_csv(
        os.path.join(outdir, "Afl_dropped_features.csv"),
        index=False, encoding="utf-8-sig"
    )
    
    # 3. 保留原有的相关系数矩阵输出（论文需要）
    features_df.corr().to_csv(
        os.path.join(outdir, "Afl_feature_correlation_matrix.csv"),
        encoding="utf-8-sig"
    )

    # 4. Save完整VIF检验结果（包含删除+保留的特征）
    vif_df = vif_result.sort_values(["processing_result", "VIF"], ascending=[False, False])
    vif_df.to_csv(
        os.path.join(outdir, "Afl_feature_vif_test.csv"),
        index=False,
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
    res_dir = os.path.dirname(output_path)
    fig_dir = config["visualization_output_dir"]
    os.makedirs(res_dir, exist_ok=True)
    os.makedirs(fig_dir, exist_ok=True)

    # 版本检测
    data_version = detect_data_version(config["data_path"])
    col_map = COLUMN_MAPS[data_version]

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
            sample_field_list.append(np.array([]))
            continue
        field_vals = act_win[:, -1].astype(int)
        field_vals = np.clip(field_vals, 0, 3)
        sample_field_list.append(field_vals)
        sample_feat = extract_features_single_sample(
            act_win, eye_win, phy_win, col_map
        )
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
    dropped_vif = [f for f in features_corr.columns if f not in features_final.columns]

    print(f"\n=== 特征筛选完成 ===")
    print(f"相关性筛选删除特征: {dropped_corr}")
    print(f"VIF筛选删除特征: {dropped_vif}")
    print(f"最终保留特征: {list(features_final.columns)}")

    if features_final.empty:
        raise RuntimeError("所有特征被筛选删除，无法继续计算")

    # 4. 权重计算（支持AHP/CRITIC + 熵权）
    weight_method = config.get("weight_method", "critic_entropy")
    if args.weight_method:
        weight_method = args.weight_method
    
    print(f"\n=== 权重计算模式：{weight_method} ===")
    structural_w = get_structural_weights(
        method=weight_method,
        features_df=features_final,
        save_path=os.path.join(res_dir, "Afl_structural_weights.csv")
    )
    ent_w = entropy_weights(features_final)
    combined_w = combine_weights(structural_w, ent_w, features_final.columns)
    
    print(f"\n=== 权重计算完成 ===")
    print("组合权重（特征: 权重）:")
    for f, w in combined_w.items():
        print(f"  {f}: {w:.4f}")

    # 5. 波动量计算（秩归一化 + 自动校准）
    # A_fl, S_fl = calculate_fluctuation(features_final, combined_w)
    A_fl, S_fl = calculate_fluctuation_norm(features_final, combined_w, target_std=0.025)
    print(f"\n=== 波动量计算完成 ===")
    print(f"波动量范围: [{A_fl.min():.4f}, {A_fl.max():.4f}]")

    # 计算均值和标准差
    mean_val = A_fl.mean()
    std_val = A_fl.std()
    print(f"波动量均值: {mean_val:.4f}")
    print(f"波动量标准差: {std_val:.4f}")

    # ±1σ 区间占比（删除了原配置区间打印）
    in_sigma_ratio = np.mean((A_fl >= mean_val - std_val) & (A_fl <= mean_val + std_val)) * 100
    print(f"落在 ±1σ 区间的比例: {in_sigma_ratio:.1f}%")

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
    print(f"\nPKL结果已保存至:   {output_path}")

    # 8. 保存论文 CSV
    save_all_dropped_features(dropped_corr, dropped_vif, features_df, vif_result, res_dir)
    save_feature_weights(weight_method, structural_w, ent_w, combined_w, res_dir)
    save_fluctuation_stats(A_fl, res_dir, config)
    save_fluctuation_by_group(A_fl, None, res_dir, config)
    print(f"论文CSV已保存至:   {res_dir}")

    # 9. 可视化
    run_all_visualizations(
        result_pkl_path=output_path,
        features_df_before_filter=features_df,
        output_dir=config["visualization_output_dir"],
        config=config,
    )
    print(f"可视化图表将输出至: {fig_dir}")


if __name__ == "__main__":
    main()
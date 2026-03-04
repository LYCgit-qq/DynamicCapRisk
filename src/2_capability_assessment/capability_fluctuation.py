# D:\Local\DynamicCapRisk\src\2_capability_assessment\capability_fluctuation.py

import os
import sys

# third-party dependencies are required; fail early with helpful message
try:
    import numpy as np
    import pandas as pd
    import pickle
    from scipy import stats
    from statsmodels.stats.outliers_influence import variance_inflation_factor
    from sklearn.preprocessing import StandardScaler
    # 注意：确保plot_capability.py路径正确，或注释掉可视化调用（如果不需要）
    from src.visualization.plot_capability import run_all_visualizations
except ImportError as exc:
    missing = str(exc).split("'")[1]
    sys.stderr.write(
        f"\nERROR: missing required package '{missing}'.\n" \
        "Please install dependencies listed in requirements.txt (e.g. `pip install -r requirements.txt`).\n"
    )
    raise

# ====================== 1. 数据加载与预处理 ======================
def load_data(pkl_path):
    """加载预处理后的pkl数据（返回act/eye/phy列表，各67个样本）"""
    with open(pkl_path, 'rb') as f:
        data = pickle.load(f)
    # 确保返回的是列表结构（67个样本）
    act_raw = data['act'] if 'act' in data else []
    eye_raw = data['eye'] if 'eye' in data else []
    phy_raw = data['phy'] if 'phy' in data else []
    # 校验样本数量一致
    assert len(act_raw) == len(eye_raw) == len(phy_raw) == 67, "样本数量不匹配（需67个）"
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

def preprocess_single_sample(act, eye, phy):
    """
    单样本多模态数据时间窗口对齐（3s窗口）
    act: 60Hz → 窗口大小180, eye: 120Hz → 360, phy: 100Hz → 300
    """
    # 窗口大小设置（3秒）
    win_act = 60 * 3   # 180 samples
    win_eye = 120 * 3  # 360 samples
    win_phy = 100 * 3  # 300 samples
    
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
    df['steering_angle'] = act_win[:, 2]          # 方向盘转角
    # 方向盘转角角速度：对窗口化的角度数据做差分（3秒/窗口）
    if len(act_win) >= 2:
        dv = np.diff(df['steering_angle'], prepend=df['steering_angle'].iloc[0])
        df['steering_velocity'] = dv / 3.0  # 角速度 = 角度变化 / 时间（3秒）
    else:
        df['steering_velocity'] = 0.0
    df['brake_pedal'] = act_win[:, 1]             # 制动踏板开度
    df['throttle_pedal'] = act_win[:, 0]          # 新增：油门踏板开度（实际有数据）
    
    # ---------- 车辆响应特征 ----------
    df['longitudinal_accel'] = act_win[:, 4]       # 纵向加速度
    df['lateral_offset'] = act_win[:, 8]           # 横向偏移量
    df['lateral_accel'] = act_win[:, 5]            # 横向加速度
    df['vehicle_speed'] = act_win[:, 3]            # 新增：车速（实际有数据）
    
    # ---------- 眼动认知特征 ----------
    if eye_win is not None and len(eye_win) > 0 and eye_win.shape[1] >= 3:
        df['gaze_dispersion'] = np.std(eye_win[:, 1:3], axis=1)  # 注视点分散度
    else:
        df['gaze_dispersion'] = 0.0
    
    # ---------- 生理状态特征 ----------
    if phy_win is not None and len(phy_win) > 0 and phy_win.shape[1] >= 5:
        df['hrv'] = phy_win[:, 4]                      # 心率变异性
        df['hr'] = phy_win[:, 3]                       # 新增：心率（实际有数据）
        df['bvp'] = phy_win[:, 0]                      # 新增：血容量脉搏（实际有数据）
    else:
        df['hrv'] = 0.0
        df['hr'] = 0.0
        df['bvp'] = 0.0
    
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
        return df, pd.DataFrame({'feature': df.columns, 'VIF': []})
    
    df_vif = df.copy()
    vif_results = []
    while True:
        vif = [variance_inflation_factor(df_vif.values, i) for i in range(df_vif.shape[1])]
        max_vif = max(vif)
        vif_results = vif
        if max_vif <= threshold:
            break
        # 移除VIF最大的特征
        drop_col = df_vif.columns[np.argmax(vif)]
        df_vif = df_vif.drop(columns=drop_col)
        if df_vif.empty:
            break
    vif_df = pd.DataFrame({'feature': df_vif.columns, 'VIF': vif_results[:len(df_vif.columns)]})
    return df_vif, vif_df

# ====================== 4. 权重计算 ======================
def ahp_weights_adapted():
    """适配实际提取特征的AHP权重（删除无数据特征，重新归一化）"""
    base_weights = {
        'steering_angle': 0.252,
        'steering_velocity': 0.139,
        'brake_pedal': 0.076,
        'throttle_pedal': 0.060,  # 新增：油门踏板权重（补充）
        'longitudinal_accel': 0.086,
        'lateral_offset': 0.111,
        'lateral_accel': 0.059,
        'vehicle_speed': 0.070,   # 新增：车速权重（补充）
        'gaze_dispersion': 0.050,
        'hrv': 0.050,
        'hr': 0.040,              # 新增：心率权重（补充）
        'bvp': 0.007              # 新增：血容量脉搏权重（补充）
    }
    # 归一化权重（确保总和为1）
    total = sum(base_weights.values())
    return {k: v/total for k, v in base_weights.items()}

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
    with np.errstate(divide='ignore', invalid='ignore'):
        e = -np.sum(p * np.log(p + 1e-12), axis=0) / np.log(len(df))
    d = 1 - e  # 差异度
    d[~nonzero] = 0.0  # 常量特征差异度为0
    # 计算熵权
    if d.sum() == 0:
        return {col: 1/len(df.columns) for col in df.columns}  # 平均赋权
    w = d / d.sum()
    return dict(zip(df.columns, w))

def combine_weights(ahp_w, ent_w, features):
    """乘法合成组合权重（仅保留共同特征）"""
    common_features = [f for f in features if f in ahp_w and f in ent_w]
    if not common_features:
        return {f: 1/len(features) for f in features}  # 无共同特征则平均赋权
    # 乘法合成 + 归一化
    combined = np.array([ahp_w[f] * ent_w[f] for f in common_features])
    combined = combined / combined.sum()
    return dict(zip(common_features, combined))

# ====================== 5. 波动量计算 ======================
def calculate_fluctuation(df, weights, k=0.4):
    """
    计算驾驶能力波动量
    公式：A_fl = 0.4*S_fl - 0.2
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

# ====================== 主流程 ======================
def main(data_path=None, output_path=None):
    """Entry point for the script（适配67个样本列表结构）"""
    # 配置路径
    data_path = data_path or os.path.join('data', 'processed', 'raw_data.pkl')
    output_path = output_path or os.path.join('output', '1_capability_assessment', 'Afl_capability_fluctuation.pkl')

    if not os.path.isfile(data_path):
        raise FileNotFoundError(f"input file not found: {data_path}")

    # 1. 加载数据（67个样本列表）
    act_raw, eye_raw, phy_raw = load_data(data_path)
    print("=== 数据加载完成 ===")
    print(f"样本总数: {len(act_raw)}")
    print(f"单个样本act维度示例: {act_raw[0].shape}")

    # 2. 遍历所有样本，预处理+特征提取
    all_features = []
    for i in range(len(act_raw)):
        # 单样本预处理
        act_win, eye_win, phy_win = preprocess_single_sample(act_raw[i], eye_raw[i], phy_raw[i])
        if act_win is None:
            print(f"跳过样本{i+1}：无有效窗口")
            continue
        # 单样本特征提取
        sample_feat = extract_features_single_sample(act_win, eye_win, phy_win)
        if not sample_feat.empty:
            sample_feat['sample_id'] = i  # 标记样本ID
            all_features.append(sample_feat)
    
    # 汇总所有样本的特征
    if not all_features:
        raise RuntimeError("无有效特征提取结果")
    features_df = pd.concat(all_features, ignore_index=True)
    print(f"\n=== 特征提取完成 ===")
    print(f"总窗口数: {len(features_df)}")
    print(f"提取特征列表: {list(features_df.columns)}")

    # 移除样本ID列（仅用于标记，不参与计算）
    if 'sample_id' in features_df.columns:
        features_df = features_df.drop(columns=['sample_id'])

    # 3. 特征筛选
    features_corr, dropped_corr = correlation_filter(features_df, threshold=0.8)
    features_final, vif_result = vif_filter(features_corr, threshold=10)
    print(f"\n=== 特征筛选完成 ===")
    print(f"相关性筛选删除特征: {dropped_corr}")
    print(f"最终保留特征: {list(features_final.columns)}")

    if features_final.empty:
        raise RuntimeError("所有特征被筛选删除，无法继续计算")

    # 4. 权重计算
    ahp_w = ahp_weights_adapted()
    ent_w = entropy_weights(features_final)
    combined_w = combine_weights(ahp_w, ent_w, features_final.columns)
    print(f"\n=== 权重计算完成 ===")
    print("组合权重（特征: 权重）:")
    for f, w in combined_w.items():
        print(f"  {f}: {w:.4f}")

    # 5. 波动量计算
    A_fl, S_fl = calculate_fluctuation(features_final, combined_w)
    print(f"\n=== 波动量计算完成 ===")
    print(f"波动量范围: [{A_fl.min():.3f}, {A_fl.max():.3f}]")
    print(f"波动量均值: {A_fl.mean():.3f}")
    print(f"波动量标准差: {A_fl.std():.3f}")

    # 6. 保存结果
    outdir = os.path.dirname(output_path)
    if outdir and not os.path.isdir(outdir):
        os.makedirs(outdir, exist_ok=True)
    
    result = {
        'features': features_final,
        'weights': combined_w,
        'fluctuation': A_fl,
        'vif': vif_result,
        'S_fl': S_fl  # 保存中间值
    }
    with open(output_path, 'wb') as f:
        pickle.dump(result, f)
    print(f"\n结果已保存至: {output_path}")

    # 7. 可视化（如需启用，确保plot_capability.py路径正确）
    try:
        run_all_visualizations(
            result_pkl_path=output_path,
            output_dir="output/1_capability_assessment",
        )
        print("可视化完成")
    except Exception as e:
        print(f"可视化失败: {e}")

if __name__ == "__main__":
    main()
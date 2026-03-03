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
    """加载预处理后的pkl数据"""
    with open(pkl_path, 'rb') as f:
        return pickle.load(f)

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

def preprocess_data(act, eye, phy):
    """
    多模态数据时间窗口对齐（3s窗口）
    act: 60Hz, eye: 120Hz, phy: 100Hz
    """
    # 窗口大小设置
    win_act = 60 * 3   # 180 samples
    win_eye = 120 * 3  # 360 samples
    win_phy = 100 * 3  # 300 samples
    
    # 窗口化处理
    act_win = sliding_window_mean(act, win_act)
    eye_win = sliding_window_mean(eye, win_eye)
    phy_win = sliding_window_mean(phy, win_phy)
    
    # 统一窗口数量
    n_min = min(len(act_win), len(eye_win), len(phy_win))
    return act_win[:n_min], eye_win[:n_min], phy_win[:n_min]

# ====================== 2. 特征提取 ======================
def extract_features(act_win, eye_win, phy_win, act_raw):
    """从多模态数据中提取11项驾驶特征（适配数据集调整版）"""
    df = pd.DataFrame()
    
    # ---------- 操纵行为特征 ----------
    df['steering_angle'] = act_win[:, 2]          # 方向盘转角(X1)

    # 方向盘转角角速度(X2)：对窗口化的角度数据做差分，保持长度一致
    if len(act_win) >= 2:
        dv = np.diff(df['steering_angle'], prepend=df['steering_angle'].iloc[0])
        # 每个窗口代表3秒，60Hz-->180 samples, so velocity ~ dv / 3
        df['steering_velocity'] = dv / 3.0
    else:
        df['steering_velocity'] = 0.0

    df['brake_pedal'] = act_win[:, 1]             # 制动踏板开度(X4)
    
    # ---------- 车辆响应特征 ----------
    df['longitudinal_accel'] = act_win[:, 4]       # 纵向加速度(X6)
    df['lateral_offset'] = act_win[:, 8]           # 横向偏移量(X7)
    df['lateral_accel'] = act_win[:, 5]            # 横向加速度(X8)
    
    # 碰撞时间(X10)：需根据实际跟车距离计算，此处用模拟值（需替换）
    # df['time_to_collision'] = act_win[:, 3] / (act_win[:, 4] + 1e-6)  # 示例公式
    
    # ---------- 眼动认知特征 ----------
    # placeholder for missing modalities; keeps column names consistent
    df['pupil_diameter'] = np.ones(len(eye_win))           # 数据集中无，用占位符（需替换）
    if eye_win.shape[1] >= 3:
        df['gaze_dispersion'] = np.std(eye_win[:, 1:3], axis=1)  # 注视点分散度(X15)
    else:
        df['gaze_dispersion'] = 0.0
    
    # ---------- 生理状态特征 ----------
    df['hrv'] = phy_win[:, 4]                      # 心率变异性(X17)
    df['skin_conductance'] = np.ones(len(phy_win))     # 数据集中无，用占位符（需替换）
    
    return df.dropna(axis=1)  # 移除含缺失值的列

# ====================== 3. 特征筛选 ======================
def correlation_filter(df, threshold=0.8):
    """Pearson相关性分析筛选"""
    corr_mat = df.corr().abs()
    upper = corr_mat.where(np.triu(np.ones(corr_mat.shape), k=1).astype(bool))
    to_drop = [column for column in upper.columns if any(upper[column] > threshold)]
    return df.drop(columns=to_drop), to_drop

def vif_filter(df, threshold=10):
    """方差膨胀因子(VIF)筛选"""
    while True:
        vif = [variance_inflation_factor(df.values, i) for i in range(df.shape[1])]
        max_vif = max(vif)
        if max_vif <= threshold:
            break
        df = df.drop(columns=df.columns[np.argmax(vif)])
    return df, pd.DataFrame({'feature': df.columns, 'VIF': vif})

# ====================== 4. 权重计算 ======================
def ahp_weights():
    """论文AHP权重（表3.9）"""
    return {
        'steering_angle': 0.252,
        'steering_velocity': 0.139,
        'brake_pedal': 0.076,
        'longitudinal_accel': 0.086,
        'lateral_offset': 0.111,
        'lateral_accel': 0.059,
        'time_to_collision': 0.044,
        'pupil_diameter': 0.100,
        'gaze_dispersion': 0.050,
        'hrv': 0.050,
        'skin_conductance': 0.033
    }

def entropy_weights(df):
    """熵权法客观赋权

    Ensures each column is scaled to non‑negative values before computing
    entropy.  Handles constant columns by assigning zero weight.
    """
    if df.empty:
        return {}
    scaler = StandardScaler()
    scaled = scaler.fit_transform(df)
    # make columns non-negative individually
    col_min = scaled.min(axis=0)
    scaled = scaled - col_min + 1e-8

    col_sum = scaled.sum(axis=0)
    # avoid division by zero
    nonzero = col_sum != 0
    p = np.zeros_like(scaled)
    p[:, nonzero] = scaled[:, nonzero] / col_sum[nonzero]

    with np.errstate(divide='ignore', invalid='ignore'):
        e = -np.sum(p * np.log(p + 1e-12), axis=0) / np.log(len(df))
    d = 1 - e
    d[~nonzero] = 0.0
    w = d / d.sum() if d.sum() > 0 else d
    return dict(zip(df.columns, w))

def combine_weights(ahp_w, ent_w, features):
    """乘法合成组合权重"""
    common = [f for f in features if f in ahp_w and f in ent_w]
    combined = np.array([ahp_w[f] * ent_w[f] for f in common])
    return dict(zip(common, combined / combined.sum()))

# ====================== 5. 波动量计算 ======================
def calculate_fluctuation(df, weights, k=0.4):
    """
    计算驾驶能力波动量
    公式：A_fl = 0.4*S_fl - 0.2
    """
    # 特征标准化到[0,1]
    scaler = StandardScaler()
    scaled = scaler.fit_transform(df[list(weights.keys())])
    # normalize each column independently to [0,1]
    minv = scaled.min(axis=0)
    maxv = scaled.max(axis=0)
    denom = maxv - minv
    denom[denom == 0] = 1  # prevent divide-by-zero
    scaled = (scaled - minv) / denom
    
    # 加权求和
    S_fl = np.dot(scaled, np.array([weights[f] for f in weights.keys()]))
    
    # 转换为波动量
    A_fl = k * S_fl - 0.2
    return A_fl, S_fl

# ====================== 主流程 ======================
def main(data_path=None, output_path=None):
    """Entry point for the script.

    Parameters
    ----------
    data_path : str, optional
        Path to the raw_data pickle file.  Defaults to ``data/processed/raw_data.pkl``.
    output_path : str, optional
        Where to dump the results pickle.
    """
    data_path = data_path or os.path.join('data', 'processed', 'raw_data.pkl')
    output_path = output_path or os.path.join('output', '1_capability_assessment', 'capability_fluctuation.pkl')

    if not os.path.isfile(data_path):
        raise FileNotFoundError(f"input file not found: {data_path}")

    # 1. load data
    data = load_data(data_path)
    if not data:
        raise ValueError("loaded data is empty")

    print("=== 数据结构诊断 ===")
    print(f"顶层数据类型: {type(data)}")

    # ---------- 核心修复：适配列表结构 ----------
    if isinstance(data, list):
        # 假设 data 是 [act, eye, phy] 的列表
        if len(data) >= 3:
            act_raw = np.array(data[0])
            eye_raw = np.array(data[1])
            phy_raw = np.array(data[2])
        else:
            # 或者 data 是一个样本列表，取第一个样本
            first_sample = data[0]
            if isinstance(first_sample, list) and len(first_sample) >= 3:
                act_raw = np.array(first_sample[0])
                eye_raw = np.array(first_sample[1])
                phy_raw = np.array(first_sample[2])
            else:
                raise ValueError("无法解析列表结构的数据")
    elif isinstance(data, dict):
        # 回退到原来的字典逻辑
        sample_key = next(iter(data))
        samp = data[sample_key]
        print(f"samp 类型: {type(samp)}")
        if isinstance(samp, dict):
            act_raw = samp.get('act')
            eye_raw = samp.get('eye')
            phy_raw = samp.get('phy')
        elif isinstance(samp, list) and len(samp) >= 3:
            act_raw = np.array(samp[0])
            eye_raw = np.array(samp[1])
            phy_raw = np.array(samp[2])
        else:
            raise TypeError(f"不支持的 samp 类型: {type(samp)}")
    else:
        raise TypeError(f"不支持的 data 类型: {type(data)}")

    # 确保数据是 numpy 数组
    act_raw = np.asarray(act_raw)
    eye_raw = np.asarray(eye_raw)
    phy_raw = np.asarray(phy_raw)
    
    print(f"act 形状: {act_raw.shape}")
    print(f"eye 形状: {eye_raw.shape}")
    print(f"phy 形状: {phy_raw.shape}")
    # ---------------------------------------------

    # 2. preprocessing
    act_win, eye_win, phy_win = preprocess_data(act_raw, eye_raw, phy_raw)

    # 3. feature extraction
    features = extract_features(act_win, eye_win, phy_win, act_raw)
    print(f"提取特征数: {len(features.columns)}")

    if features.empty:
        raise RuntimeError("no features could be extracted")

    # 4. feature selection
    features_corr, dropped_corr = correlation_filter(features)
    features_final, vif_result = vif_filter(features_corr)
    print(f"最终保留特征: {list(features_final.columns)}")

    if features_final.empty:
        raise RuntimeError("all features were dropped by filtering")

    # 5. weight calculation
    ahp_w = ahp_weights()
    ent_w = entropy_weights(features_final)
    combined_w = combine_weights(ahp_w, ent_w, features_final.columns)
    print("组合权重:", combined_w)

    # 6. fluctuation
    A_fl, S_fl = calculate_fluctuation(features_final, combined_w)
    print(f"波动量范围: [{A_fl.min():.3f}, {A_fl.max():.3f}]")

    # ensure output directory exists
    outdir = os.path.dirname(output_path)
    if outdir and not os.path.isdir(outdir):
        os.makedirs(outdir, exist_ok=True)

    # 7. save
    result = {
        'features': features_final,
        'weights': combined_w,
        'fluctuation': A_fl,
        'vif': vif_result
    }
    with open(output_path, 'wb') as f:
        pickle.dump(result, f)
    print(f"结果已保存至: {output_path}")

    # 8. visualization
    run_all_visualizations(
        result_pkl_path="output/1_capability_assessment/capability_fluctuation.pkl",
        output_dir="output/1_capability_assessment",
    )

if __name__ == "__main__":
    main()
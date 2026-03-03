import numpy as np
import pandas as pd
import pickle
from scipy import stats
from statsmodels.stats.outliers_influence import variance_inflation_factor
from sklearn.preprocessing import StandardScaler

# ====================== 1. 数据加载与预处理 ======================
def load_data(pkl_path):
    """加载预处理后的pkl数据"""
    with open(pkl_path, 'rb') as f:
        return pickle.load(f)

def sliding_window_mean(data, window_size):
    """滑动窗口切分并计算均值"""
    if len(data) < window_size:
        return np.array([])
    n_windows = len(data) // window_size
    data_trimmed = data[:n_windows * window_size]
    data_reshaped = data_trimmed.reshape(n_windows, window_size, -1)
    return np.mean(data_reshaped, axis=1)

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
    
    # 方向盘转角角速度(X2)：计算原始数据差分
    sa_raw = act_raw[:, 2]
    sa_vel_raw = np.diff(sa_raw, prepend=sa_raw[0]) * 60  # 60Hz采样率转换
    df['steering_velocity'] = sliding_window_mean(sa_vel_raw.reshape(-1,1), 180)[:len(df)].flatten()
    
    df['brake_pedal'] = act_win[:, 1]             # 制动踏板开度(X4)
    
    # ---------- 车辆响应特征 ----------
    df['longitudinal_accel'] = act_win[:, 4]       # 纵向加速度(X6)
    df['lateral_offset'] = act_win[:, 8]           # 横向偏移量(X7)
    df['lateral_accel'] = act_win[:, 5]            # 横向加速度(X8)
    
    # 碰撞时间(X10)：需根据实际跟车距离计算，此处用模拟值（需替换）
    # df['time_to_collision'] = act_win[:, 3] / (act_win[:, 4] + 1e-6)  # 示例公式
    
    # ---------- 眼动认知特征 ----------
    df['pupil_diameter'] = eye_win[:, 1] * 0 + 1  # 数据集中无，用占位符（需替换）
    df['gaze_dispersion'] = np.std(eye_win[:, 1:3], axis=1)  # 注视点分散度(X15)
    
    # ---------- 生理状态特征 ----------
    df['hrv'] = phy_win[:, 4]                      # 心率变异性(X17)
    df['skin_conductance'] = phy_win[:, 0] * 0 + 1 # 数据集中无，用占位符（需替换）
    
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
    """熵权法客观赋权"""
    scaler = StandardScaler()
    scaled = scaler.fit_transform(df)
    scaled = scaled - scaled.min() + 1e-8  # 非负化
    
    p = scaled / scaled.sum(axis=0)
    e = -np.sum(p * np.log(p), axis=0) / np.log(len(df))
    d = 1 - e
    return dict(zip(df.columns, d / d.sum()))

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
    scaled = (scaled - scaled.min()) / (scaled.max() - scaled.min())
    
    # 加权求和
    S_fl = np.dot(scaled, np.array([weights[f] for f in weights.keys()]))
    
    # 转换为波动量
    A_fl = k * S_fl - 0.2
    return A_fl, S_fl

# ====================== 主流程 ======================
def main():
    # 1. 加载数据
    data = load_data('data/processed/raw_data.pkl')
    
    # 示例：处理第一个样本（需根据实际数据结构调整）
    sample_key = list(data.keys())[0]
    act_raw = data[sample_key]['act']
    eye_raw = data[sample_key]['eye']
    phy_raw = data[sample_key]['phy']
    
    # 2. 预处理
    act_win, eye_win, phy_win = preprocess_data(act_raw, eye_raw, phy_raw)
    
    # 3. 特征提取
    features = extract_features(act_win, eye_win, phy_win, act_raw)
    print(f"提取特征数: {len(features.columns)}")
    
    # 4. 特征筛选
    features_corr, dropped_corr = correlation_filter(features)
    features_final, vif_result = vif_filter(features_corr)
    print(f"最终保留特征: {list(features_final.columns)}")
    
    # 5. 权重计算
    ahp_w = ahp_weights()
    ent_w = entropy_weights(features_final)
    combined_w = combine_weights(ahp_w, ent_w, features_final.columns)
    print("组合权重:", combined_w)
    
    # 6. 波动量计算
    A_fl, S_fl = calculate_fluctuation(features_final, combined_w)
    print(f"波动量范围: [{A_fl.min():.3f}, {A_fl.max():.3f}]")
    
    # 7. 保存结果
    result = {
        'features': features_final,
        'weights': combined_w,
        'fluctuation': A_fl,
        'vif': vif_result
    }
    with open('data/processed/driving_ability_result.pkl', 'wb') as f:
        pickle.dump(result, f)
    print("结果已保存至: data/processed/driving_ability_result.pkl")

if __name__ == "__main__":
    main()
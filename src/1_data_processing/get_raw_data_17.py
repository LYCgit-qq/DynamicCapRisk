import pickle
import numpy as np
import os

def resample_signal(signal, original_rate, target_rate, target_length=None):
    """重采样信号到目标频率（线性插值）"""
    T_orig, D = signal.shape
    if target_length is None:
        T_new = int(np.round(T_orig * target_rate / original_rate))
    else:
        T_new = target_length
    
    if T_orig == T_new:
        return signal
    
    t_orig = np.linspace(0, T_orig / original_rate, T_orig)
    t_new = np.linspace(0, T_orig / original_rate, T_new)
    
    resampled = np.zeros((T_new, D))
    for d in range(D):
        resampled[:, d] = np.interp(t_new, t_orig, signal[:, d])
    return resampled

def calculate_gaze_dispersion(gaze_xy, sample_rate=60, window_sec=10):
    """计算注视点分散度（滑动窗口标准差）"""
    window_size = int(sample_rate * window_sec)
    T = len(gaze_xy)
    dispersion = np.zeros(T)
    for t in range(T):
        start = max(0, t - window_size // 2)
        end = min(T, t + window_size // 2)
        window = gaze_xy[start:end]
        if len(window) >= 2:
            dispersion[t] = np.mean(np.std(window, axis=0))
        else:
            dispersion[t] = 0.0
    return dispersion

# ====================== 路径配置 ======================
INPUT_PKL = "data/processed/raw_data.pkl"
OUTPUT_DIR = "data/processed"
OUTPUT_PKL = os.path.join(OUTPUT_DIR, "raw_data_17.pkl")  # 重命名为17维
os.makedirs(OUTPUT_DIR, exist_ok=True)

# ====================== 1. 加载原始数据 ======================
print("正在加载原始数据...")
with open(INPUT_PKL, "rb") as f:
    raw_data = pickle.load(f)
act_list = raw_data["act"]
eye_list = raw_data["eye"]
phy_list = raw_data["phy"]

# ====================== 2. 逐样本生成新的 17维特征 ======================
print("正在生成17维特征并按模态拆分...")
new_act_list = []  # 8维  （删除X/Y坐标）
new_eye_list = []  # 5维  （不变）
new_phy_list = []  # 4维  （删除ECG）

for i in range(len(act_list)):
    act = act_list[i]
    eye = eye_list[i]
    phy = phy_list[i]
    T_act = act.shape[0]
    
    # 重采样对齐
    eye_60 = resample_signal(eye, 120, 60, T_act)
    phy_60 = resample_signal(phy, 100, 60, T_act)
    
    # ====================== 构建新的 act (8维) ======================
    new_act = np.zeros((T_act, 8))  # 10维 → 8维
    # [0] 方向盘转角
    new_act[:, 0] = np.abs(act[:, 2])
    # [1] 方向盘转角角速度
    if T_act >= 2:
        d_angle = np.diff(new_act[:, 0], prepend=new_act[0, 0])
        new_act[:, 1] = np.abs(d_angle * 60)
    # [2] 加速踏板开度
    new_act[:, 2] = np.abs(act[:, 0])
    # [3] 制动踏板开度
    new_act[:, 3] = np.abs(act[:, 1])
    # [4] 车速
    new_act[:, 4] = act[:, 3]
    # [5] 纵向加速度
    new_act[:, 5] = np.abs(act[:, 4])
    # [6] 横向加速度
    new_act[:, 6] = np.abs(act[:, 5])
    # [7] 相对理想路径偏移量
    new_act[:, 7] = np.abs(act[:, 8])
    # 已删除：车辆X坐标、车辆Y坐标
    
    # ====================== 构建新的 eye (5维) ======================
    new_eye = np.zeros((T_act, 5))
    new_eye[:, 0] = eye_60[:, 1]      # 注视X
    new_eye[:, 1] = eye_60[:, 2]      # 注视Y
    new_eye[:, 2] = eye_60[:, 0]      # 眨眼频率
    new_eye[:, 3] = eye_60[:, 0]      # 眨眼标准差
    new_eye[:, 4] = calculate_gaze_dispersion(eye_60[:, 1:3])  # 分散度
    
    # ====================== 构建新的 phy (4维) ======================
    new_phy = np.zeros((T_act, 4))  # 5维 → 4维
    new_phy[:, 0] = phy_60[:, 3]    # 心率均值
    new_phy[:, 1] = np.abs(phy_60[:, 4])  # 心率变异性
    new_phy[:, 2] = phy_60[:, 0]    # 血容量脉搏
    new_phy[:, 3] = phy_60[:, 2]    # 呼吸信号
    # 已删除：心电信号ECG
    
    # 加入列表
    new_act_list.append(new_act)
    new_eye_list.append(new_eye)
    new_phy_list.append(new_phy)
    print(f"  样本 {i+1}/67 处理完成")

# ====================== 3. 保存 ======================
output_data = {
    "act": new_act_list,
    "eye": new_eye_list,
    "phy": new_phy_list
}

with open(OUTPUT_PKL, "wb") as f:
    pickle.dump(output_data, f)

# ====================== 4. 输出验证信息 ======================
print("\n" + "="*60)
print("✅ 17维特征处理完成！(已删除X/Y坐标、ECG)")
print(f"📦 输出文件: {OUTPUT_PKL}")
print(f"🔢 总维度：17维 (act8 + eye5 + phy4)")
print(f"📊 维度详情:")
print(f"   - act: {len(new_act_list)} 个样本，单样本维度 {new_act_list[0].shape}")
print(f"   - eye: {len(new_eye_list)} 个样本，单样本维度 {new_eye_list[0].shape}")
print(f"   - phy: {len(new_phy_list)} 个样本，单样本维度 {new_phy_list[0].shape}")
print("\n📋 最终17维特征索引表：")
print(" 【act 8维】: 0-方向盘转角,1-角速度,2-油门,3-刹车,4-车速,5-纵向加,6-横向加,7-偏移量")
print(" 【eye 5维】: 0-注视X,1-注视Y,2-眨眼频率,3-眨眼标准差,4-分散度")
print(" 【phy 4维】: 0-心率均值,1-HRV,2-BVP,3-呼吸信号")
print("="*60)
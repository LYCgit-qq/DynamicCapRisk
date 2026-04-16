import pickle
import numpy as np
import os

def calculate_gaze_dispersion(gaze_xy, sample_rate=120, window_sec=10):
    """计算注视点分散度（使用眼动原始频率120Hz）"""
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
OUTPUT_PKL = os.path.join(OUTPUT_DIR, "raw_data_17.pkl")
os.makedirs(OUTPUT_DIR, exist_ok=True)

# ====================== 1. 加载原始数据 ======================
print("正在加载原始数据...")
with open(INPUT_PKL, "rb") as f:
    raw_data = pickle.load(f)
act_list = raw_data["act"]
eye_list = raw_data["eye"]
phy_list = raw_data["phy"]

# ====================== 2. 逐样本生成新的 17维特征 ======================
print("正在生成17维特征（保留原始频率，不重采样、不统一长度）...")
new_act_list = []  # 9维  （8维特征 + 最后一列=路段类型）60Hz 原始长度
new_eye_list = []  # 5维  120Hz 原始长度
new_phy_list = []  # 4维  100Hz 原始长度

for i in range(len(act_list)):
    # 直接使用原始数据，不做任何重采样/长度修改
    act = act_list[i]    # 原始 60Hz，原始长度
    eye = eye_list[i]    # 原始 120Hz，原始长度
    phy = phy_list[i]    # 原始 100Hz，原始长度
    T_act = act.shape[0]

    # ====================== 构建新的 act (9维，原始60Hz) ======================
    new_act = np.zeros((T_act, 9))
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
    # [8] 路段类型（场景标签）act最后一列
    new_act[:, 8] = act[:, 9]

    # ====================== 构建新的 eye (5维，原始120Hz) ======================
    T_eye = eye.shape[0]
    new_eye = np.zeros((T_eye, 5))
    new_eye[:, 0] = eye[:, 1]      # 注视X 原始数据
    new_eye[:, 1] = eye[:, 2]      # 注视Y 原始数据
    new_eye[:, 2] = eye[:, 0]      # 眨眼频率 原始数据
    new_eye[:, 3] = eye[:, 0]      # 眨眼标准差 原始数据
    # 眼动分散度使用原始120Hz计算
    new_eye[:, 4] = calculate_gaze_dispersion(eye[:, 1:3], sample_rate=120)

    # ====================== 构建新的 phy (4维，原始100Hz) ======================
    T_phy = phy.shape[0]
    new_phy = np.zeros((T_phy, 4))
    new_phy[:, 0] = phy[:, 3]    # 心率均值 原始数据
    new_phy[:, 1] = np.abs(phy[:, 4])  # 心率变异性 原始数据
    new_phy[:, 2] = phy[:, 0]    # 血容量脉搏 原始数据
    new_phy[:, 3] = phy[:, 2]    # 呼吸信号 原始数据

    # 加入列表（三者长度完全不同，各自保留原始长度）
    new_act_list.append(new_act)
    new_eye_list.append(new_eye)
    new_phy_list.append(new_phy)
    print(f"  样本 {i+1}/67 处理完成 | act长度:{T_act} eye长度:{T_eye} phy长度:{T_phy}")

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
print("✅ 17维特征处理完成！【已关闭重采样，保留原始频率】")
print(f"📦 输出文件: {OUTPUT_PKL}")
print(f"📶 原始频率保留：")
print(f"   - act: 60Hz  (行为数据，原始长度)")
print(f"   - eye: 120Hz (眼动数据，原始长度)")
print(f"   - phy: 100Hz (生理数据，原始长度)")
print(f"✅ act 最后一列 = 路段类型（场景标签）")
print("\n📋 最终 act 索引表：")
print("0-方向盘转角,1-角速度,2-油门,3-刹车,4-车速,5-纵向加,6-横向加,7-偏移量,8-路段类型")
print("="*60)
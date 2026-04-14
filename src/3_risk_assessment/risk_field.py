# D:\Local\DynamicCapRisk\src\3_risk_assessment\risk_field.py

"""
风险场强量化计算模块
基于Driving Safety Field理论，计算施工区场景的综合风险场强
参考论文第4章：风险场强量化方法
"""

import pandas as pd
import numpy as np
import argparse
import os
import sys
import yaml
from typing import Tuple, Dict
from scipy.ndimage import gaussian_filter1d

# 导入可视化函数
from src.visualization.plot_risk import plot_radar_chart, plot_stacked_bar, plot_field_evolution, plot_three_scenarios_evolution

# 全局配置变量
CONFIG: Dict = None

# ============= 配置加载 =============

def load_config(config_path: str) -> Dict:
    """
    加载配置文件（无默认值模式）。
    如果文件不存在或解析失败，直接终止程序。

    Args:
        config_path: 配置文件路径

    Returns:
        配置字典
    """
    if not os.path.exists(config_path):
        print(f"[错误] 配置文件不存在: {config_path}")
        print("        请确保 YAML 配置文件位于正确路径。")
        sys.exit(1)
    
    try:
        with open(config_path, 'r', encoding='utf-8') as f:
            config = yaml.safe_load(f)
        print(f"成功加载配置文件: {config_path}")
        
        # 简单的完整性检查
        required_keys = ['weights', 'geometry', 'paths', 'scenarios']
        for key in required_keys:
            if key not in config:
                raise ValueError(f"配置文件缺少根级键: '{key}'")
                
        return config
    except yaml.YAMLError as e:
        print(f"[错误] YAML 解析失败: {e}")
        sys.exit(1)
    except Exception as e:
        print(f"[错误] 加载配置时发生未知错误: {e}")
        sys.exit(1)

# =============================================================================
# 工具函数 (保持不变)
# =============================================================================

def _get_dx(distances: np.ndarray) -> float:
    if len(distances) < 2:
        return 1.0
    dx = float(np.median(np.diff(distances)))
    return dx if dx > 1e-6 else 1.0

def _gaussian_smooth(values: np.ndarray,
                     distances: np.ndarray,
                     sigma_m: float) -> np.ndarray:
    dx = _get_dx(distances)
    sigma_samples = max(sigma_m / dx, 0.5)
    return gaussian_filter1d(values.astype(float),
                             sigma=sigma_samples,
                             mode='nearest')

def _smooth_mask(binary_mask: np.ndarray,
                 distances: np.ndarray,
                 sigma_m: float) -> np.ndarray:
    return _gaussian_smooth(binary_mask.astype(float), distances, sigma_m)

# =============================================================================
# CRITIC-熵权组合赋权法 (保持不变)
# =============================================================================

def calculate_critic_entropy_weights(data: pd.DataFrame) -> pd.Series:
    """
    基于 CRITIC 法和熵权法的组合客观赋权。

    逻辑：
      1. Min-Max 标准化数据
      2. 分别计算 CRITIC 权重 (w_critic) 和熵权 (w_entropy)
      3. 乘法合成：w = (w_critic * w_entropy) / sum(w_critic * w_entropy)

    Args:
        data: 指标数据 DataFrame，每一列是一个待赋权的指标

    Returns:
        组合权重 Series，index 对应 data 的列名
    """
    # -------------------------- 1. 数据预处理 (Min-Max 标准化) --------------------------
    # 避免修改原数据
    df = data.copy()
    
    # Min-Max 归一化到 [0, 1]
    for col in df.columns:
        min_val = df[col].min()
        max_val = df[col].max()
        if max_val - min_val < 1e-6:
            df[col] = 0.0
        else:
            df[col] = (df[col] - min_val) / (max_val - min_val)
    
    # 处理标准化后全为 0 或 NaN 的列
    df = df.fillna(0.0)

    # -------------------------- 2. 计算 CRITIC 权重 --------------------------
    n_samples, n_metrics = df.shape
    
    # 2.1 对比强度 (标准差 - 总体标准差 ddof=0)
    sigma = df.std(axis=0, ddof=0)
    
    # 2.2 冲突性 (基于 Pearson 相关系数)
    # 如果数据没有变异，相关系数矩阵会报错，这里加个保护
    if n_metrics > 1 and (sigma > 1e-6).any():
        corr_matrix = df.corr(method='pearson')
        # f_j = sum(1 - r_jk)，对角线 r_jj=1，所以不影响
        conflict = (1 - corr_matrix).sum(axis=0)
    else:
        # 如果无法计算相关系数（例如只有1列数据），假设冲突性为1
        conflict = pd.Series([1.0] * n_metrics, index=df.columns)
    
    # 2.3 CRITIC 信息量 & 归一化
    C = sigma * conflict
    # 防止全零
    if C.sum() < 1e-6:
        w_critic = pd.Series([1.0 / n_metrics] * n_metrics, index=df.columns)
    else:
        w_critic = C / C.sum()

    # -------------------------- 3. 计算熵权 --------------------------
    # 3.1 计算比重 p_ij
    col_sum = df.sum(axis=0)
    # 防止除以0，如果某列全为0，则比重均匀分布
    col_sum = col_sum.where(col_sum > 1e-6, 1.0)
    
    # 使用 numpy 广播计算比重
    p = df.values / col_sum.values[np.newaxis, :]
    
    # 3.2 计算信息熵 e_j
    # 约定 p=0 时 p*ln(p)=0
    log_p = np.log(p, where=(p > 1e-6), out=np.zeros_like(p))
    
    # 防止 ln(0) 当 n_samples=1 时的情况
    if n_samples > 1:
        e = -np.sum(p * log_p, axis=0) / np.log(n_samples)
    else:
        e = pd.Series([1.0] * n_metrics, index=df.columns)
    
    # 3.3 计算熵权
    d = 1 - e
    if d.sum() < 1e-6:
        w_entropy = pd.Series([1.0 / n_metrics] * n_metrics, index=df.columns)
    else:
        w_entropy = d / d.sum()

    # -------------------------- 4. 组合权重 (乘法合成) --------------------------
    w_combined = w_critic * w_entropy
    if w_combined.sum() < 1e-6:
        w_combined = pd.Series([1.0 / n_metrics] * n_metrics, index=df.columns)
    else:
        w_combined = w_combined / w_combined.sum()

    return w_combined

def _collect_and_calculate_weights(scenario_list: list, input_dir: str) -> Dict[str, float]:
    """
    内部辅助函数：遍历所有待处理场景，加载数据，提取指标，计算权重。
    
    Returns:
        包含所有权重的扁平字典，可直接用于更新 CONFIG['weights']
    """
    all_data_frames = []
    
    # 1. 加载所有场景的数据
    print(f"\n[CRITIC] 正在加载数据用于客观赋权...")
    for scenario in scenario_list:
        input_file = os.path.join(input_dir, f"{scenario}_continuous.csv")
        if os.path.exists(input_file):
            try:
                df = pd.read_csv(input_file, encoding='utf-8-sig')
                all_data_frames.append(df)
                print(f"[CRITIC]   - 加载成功: {scenario}")
            except Exception as e:
                print(f"[CRITIC]   - 加载失败: {scenario} ({e})")
    
    if not all_data_frames:
        print("[CRITIC] 警告: 未找到任何有效数据文件，无法计算客观权重。")
        return {}

    # 合并所有数据
    full_df = pd.concat(all_data_frames, axis=0, ignore_index=True)
    print(f"[CRITIC] 数据合并完成，总样本量: {len(full_df)}")

    new_weights = {}

    # =========================================================================
    #  1. 计算 s_sign 的权重 (lambda_1, lambda_2)
    # =========================================================================
    print("\n[CRITIC] 正在计算道路设施场强 (s_sign) 权重...")
    try:
        indicators_sign = pd.DataFrame()
        
        # 指标 1: 标志存在/密度 (简单的 0/1 统计)
        if '标识牌类型' in full_df.columns:
            indicators_sign['sign_density'] = (full_df['标识牌类型'] != '-').astype(float)
        else:
            print("[CRITIC]   跳过: 未找到 '标识牌类型' 列")
            raise KeyError("缺少列")

        # 指标 2: 施工区状态
        if '施工区状态' in full_df.columns:
            indicators_sign['work_zone'] = (full_df['施工区状态'] == '是').astype(float)
        else:
            print("[CRITIC]   跳过: 未找到 '施工区状态' 列")
            raise KeyError("缺少列")

        # 计算权重
        if len(indicators_sign.columns) == 2:
            w_sign = calculate_critic_entropy_weights(indicators_sign)
            new_weights['lambda_1'] = float(w_sign.iloc[0])
            new_weights['lambda_2'] = float(w_sign.iloc[1])
            print(f"[CRITIC]   成功: lambda_1={new_weights['lambda_1']:.4f}, lambda_2={new_weights['lambda_2']:.4f}")
        else:
            print("[CRITIC]   跳过: 指标数量不足。")
            
    except Exception as e:
        print(f"[CRITIC]   计算 s_sign 权重时发生错误: {str(e)}")

    # =========================================================================
    #  2. 计算 s_veh 的权重 (mu_1, mu_2, mu_3) - 可选
    #  注：由于当前 s_veh 是启发式模型，且无实测数据，这一块保持跳过即可
    # =========================================================================
    print("\n[CRITIC] 正在计算车辆交互场强 (s_veh) 权重 (可选)...")
    print("[CRITIC]   跳过: 无实测车辆交互数据 (TH/侧向密度/速度差)，使用 YAML 默认值。")

    # =========================================================================
    #  3. 计算 s_geo 的权重 (nu_1, nu_2, nu_3)
    #  【修改】：不再依赖 CSV 中的具体数值列，而是根据现有分类数据 + 映射规则生成指标
    # =========================================================================
    print("\n[CRITIC] 正在计算道路几何场强 (s_geo) 权重...")
    try:
        # 确保 CONFIG 已加载（因为这里需要用到 geometry 映射）
        # 注意：此函数在 main 中被调用时 CONFIG 应该已经是全局变量了
        if CONFIG is None:
            raise ValueError("CONFIG 未初始化")
            
        geo_cfg = CONFIG['geometry']
        
        # 基于现有数据列，反向构建 c_hat, w_hat, l_hat 作为 CRITIC 的输入指标
        indicators_geo = pd.DataFrame()
        
        # 指标 1: 曲率映射值 (c_hat)
        if '道路几何类型' in full_df.columns:
            indicators_geo['curvature_hat'] = full_df['道路几何类型'].map(geo_cfg['curvature_map']).fillna(0.0)
        else:
            print("[CRITIC]   跳过: 未找到 '道路几何类型' 列")
            raise KeyError("缺少列")
            
        # 指标 2: 车道宽度映射值 (w_hat)
        if '车道数' in full_df.columns:
            indicators_geo['lane_width_hat'] = full_df['车道数'].map(geo_cfg['lane_width_map']).fillna(0.0)
        else:
            print("[CRITIC]   跳过: 未找到 '车道数' 列")
            raise KeyError("缺少列")
            
        # 指标 3: 车道数映射值 (l_hat)
        if '车道数' in full_df.columns:
            indicators_geo['lane_count_hat'] = full_df['车道数'].map(geo_cfg['lane_map']).fillna(0.0)
        
        # 检查是否有足够的变异 (如果全是直道，标准差为0，计算会有问题)
        if len(indicators_geo.columns) >= 2:
            # 只有当数据存在变异时才计算，否则回退默认值
            if indicators_geo.std().sum() > 1e-3:
                w_geo = calculate_critic_entropy_weights(indicators_geo)
                new_weights['nu_1'] = float(w_geo.iloc[0])
                new_weights['nu_2'] = float(w_geo.iloc[1])
                new_weights['nu_3'] = float(w_geo.iloc[2])
                print(f"[CRITIC]   成功: nu_1={new_weights['nu_1']:.4f}, nu_2={new_weights['nu_2']:.4f}, nu_3={new_weights['nu_3']:.4f}")
            else:
                print("[CRITIC]   跳过: 道路几何数据无变异 (全为直道)，使用默认平均权重。")
        else:
            print("[CRITIC]   跳过: 指标数量不足。")
            
    except Exception as e:
        print(f"[CRITIC]   计算 s_geo 权重时跳过: {str(e)}")

    return new_weights

# =============================================================================
# 场强分量计算 (保持不变，但需确保引用 CONFIG 的地方正确)
# =============================================================================

def calculate_geo_field(df: pd.DataFrame) -> pd.Series:
    geo_cfg = CONFIG['geometry']
    weights = CONFIG['weights']  # 新增：读取nu权重
    distances = df['距离 (m)'].values.astype(float)

    c_hat = df['道路几何类型'].map(geo_cfg['curvature_map']).fillna(0.0).values
    w_hat = df['车道数'].map(geo_cfg['lane_width_map']).fillna(0.0).values
    l_hat = df['车道数'].map(geo_cfg['lane_map']).fillna(0.0).values

    # 修改：加权融合替代简单平均
    raw_geo = (
        weights.get('nu_1', 1/3) * c_hat +
        weights.get('nu_2', 1/3) * w_hat +
        weights.get('nu_3', 1/3) * l_hat
    )
    
    sigma_m = geo_cfg.get('smooth_sigma', 30.0)
    smoothed = _gaussian_smooth(raw_geo, distances, sigma_m)
    return pd.Series(smoothed, index=df.index)

def calculate_sign_field(df: pd.DataFrame) -> pd.Series:
    """
    计算道路设施场强。
    注意：这里直接从 CONFIG['weights'] 读取 lambda_1 和 lambda_2，
    如果启用了 CRITIC，CONFIG 在 main 中已被更新。
    """
    sign_cfg  = CONFIG.get('sign_field', {})
    sigma_front    = sign_cfg.get('sigma_front', 120.0)
    sigma_back     = sign_cfg.get('sigma_back',   40.0)
    wz_sigma       = sign_cfg.get('wz_smooth_sigma', 20.0)

    weights   = CONFIG['weights']
    distances = df['距离 (m)'].values.astype(float)

    # ---------- 标志牌非对称指数衰减影响 ----------
    has_sign       = (df['标识牌类型'] != '-').values
    sign_positions = distances[has_sign]
    sign_influence = np.zeros(len(distances))

    for sign_pos in sign_positions:
        d_vec  = distances - sign_pos
        kernel = np.where(
            d_vec <= 0,
            np.exp(d_vec  / sigma_front),
            np.exp(-d_vec / sigma_back)
        )
        sign_influence = np.maximum(sign_influence, kernel)

    max_inf = sign_influence.max()
    if max_inf > 1e-6:
        sign_influence /= max_inf

    # ---------- 施工区存在项 ----------
    delta_wz_raw    = (df['施工区状态'] == '是').values.astype(float)
    delta_wz_smooth = _smooth_mask(delta_wz_raw, distances, wz_sigma)

    return pd.Series(
        weights['lambda_1'] * sign_influence + weights['lambda_2'] * delta_wz_smooth,
        index=df.index
    )

def calculate_vehicle_field_static(df: pd.DataFrame) -> pd.Series:
    """
    计算车辆交互场强（平滑阶跃 + 施工区入口高斯峰）

    基础层：
      将施工区状态和复杂几何的二值掩码分别用高斯核软化，
      再通过加权叠加得到连续变化的基础场强，彻底消除阶跃。

    入口峰叠加：
      在每个施工区入口（0→1 转变）处叠加高斯峰，
      模拟汇道区车辆冲突骤升后的平滑衰减。

    Args:
        df: 包含 '施工区状态'、'道路几何类型'、'距离 (m)' 的 DataFrame

    Returns:
        车辆交互场强序列（估计值）
    """
    veh_cfg   = CONFIG['vehicle_interaction']
    distances = df['距离 (m)'].values.astype(float)

    baseline   = veh_cfg['baseline']
    wz_level   = veh_cfg['work_zone']
    cx_level   = veh_cfg['complex_geometry']
    bnd_sigma  = veh_cfg.get('boundary_smooth_sigma', 25.0)

    wz_raw = (df['施工区状态'] == '是').values.astype(float)
    cx_raw = df['道路几何类型'].isin(['bend', 'cross']).values.astype(float)

    # 软化两个二值掩码
    wz_smooth = _smooth_mask(wz_raw, distances, bnd_sigma)
    cx_smooth = _smooth_mask(cx_raw, distances, bnd_sigma)

    # 基础场强：三层叠加
    #   baseline          —— 始终存在的底部
    #   (wz_level - baseline) * wz_smooth   —— 施工区贡献
    #   (cx_level - wz_level) * cx_smooth * wz_smooth  —— 复杂几何额外贡献（仅在施工区内）
    s_veh = (
        baseline
        + (wz_level - baseline) * wz_smooth
        + (cx_level - wz_level) * cx_smooth * wz_smooth
    )

    # ---------- 施工区入口高斯峰（汇道区冲突峰值） ----------
    entry_peak   = veh_cfg.get('entry_peak',   0.98)
    entry_sigma  = veh_cfg.get('entry_sigma',  25.0)
    entry_offset = veh_cfg.get('entry_offset', -15.0)

    wz_int = wz_raw.astype(int)
    entry_indices = np.where(np.diff(wz_int) == 1)[0] + 1

    for idx in entry_indices:
        center   = distances[idx] + entry_offset
        gaussian = entry_peak * np.exp(
            -0.5 * ((distances - center) / entry_sigma) ** 2
        )
        # 取最大值叠加，保证峰值不被压低
        s_veh = np.maximum(s_veh, gaussian)

    return pd.Series(s_veh, index=df.index)

# =============================================================================
# 归一化 / 等级判定 (保持不变)
# =============================================================================

def normalize_field(field: pd.Series) -> pd.Series:
    min_val = field.min()
    max_val = field.max()
    if max_val - min_val < 1e-6:
        return pd.Series(np.zeros(len(field)), index=field.index)
    return (field - min_val) / (max_val - min_val)

def get_field_level(mean_field: float) -> str:
    levels = CONFIG['field_levels']
    if mean_field < levels['medium'][0]:
        return '低'
    elif mean_field < levels['medium_high'][0]:
        return '中'
    elif mean_field < levels['high'][0]:
        return '中高'
    else:
        return '高'

# =============================================================================
# 综合场强 (保持不变)
# =============================================================================

def calculate_comprehensive_field(df: pd.DataFrame) -> pd.DataFrame:
    weights    = CONFIG['weights']
    levels     = CONFIG['field_levels']
    final_sigma = CONFIG['calculation'].get('final_smooth_sigma', 8.0)

    result    = df[['距离 (m)']].copy()
    distances = df['距离 (m)'].values.astype(float)

    # 计算各分量 (注意：请确保 calculate_vehicle_field_static 已正确实现)
    s_geo  = calculate_geo_field(df)
    s_sign = calculate_sign_field(df)
    s_veh  = calculate_vehicle_field_static(df) # 确保此函数已在上方补全

    # 归一化
    s_geo_norm  = normalize_field(s_geo)
    s_sign_norm = normalize_field(s_sign)
    s_veh_norm  = normalize_field(s_veh)

    # 加权合成
    F_S_raw = (weights['w_veh']  * s_veh_norm +
               weights['w_geo']  * s_geo_norm  +
               weights['w_sign'] * s_sign_norm)

    # 最终平滑
    F_S = pd.Series(
        _gaussian_smooth(F_S_raw.values, distances, final_sigma),
        index=df.index
    )
    F_S = F_S.clip(0.0, 1.0)

    # 保存结果
    result['s_geo']       = s_geo
    result['s_geo_norm']  = s_geo_norm
    result['s_sign']      = s_sign
    result['s_sign_norm'] = s_sign_norm
    result['s_veh']       = s_veh
    result['s_veh_norm']  = s_veh_norm
    result['F_S']         = F_S

    # 场强等级
    result['field_level'] = pd.cut(
        F_S,
        bins=[levels['low'][0],  levels['medium'][0],
              levels['medium_high'][0], levels['high'][0], levels['high'][1]],
        labels=['低', '中', '中高', '高'],
        include_lowest=True
    )

    return result

# =============================================================================
# 统计 / 场景处理 (保持不变)
# =============================================================================

def get_scenario_statistics(result_df: pd.DataFrame, scenario_name: str) -> Dict:
    return {
        'scenario':    scenario_name,
        's_geo_mean':  result_df['s_geo_norm'].mean(),
        's_sign_mean': result_df['s_sign_norm'].mean(),
        's_veh_mean':  result_df['s_veh_norm'].mean(),
        'F_S_mean':    result_df['F_S'].mean(),
        'F_S_max':     result_df['F_S'].max(),
        'F_S_min':     result_df['F_S'].min(),
        'F_S_std':     result_df['F_S'].std(),
        'field_level': get_field_level(result_df['F_S'].mean())
    }

def process_scenario(
    input_path:    str,
    output_path:   str,
    scenario_name: str
) -> Tuple[pd.DataFrame, Dict]:
    df = pd.read_csv(input_path, encoding='utf-8-sig')

    print(f"\n{'='*60}")
    print(f"处理场景: {scenario_name}")
    print(f"输入文件: {input_path}")
    print(f"数据行数: {len(df)}")
    print(f"{'='*60}")

    result_df = calculate_comprehensive_field(df)
    stats     = get_scenario_statistics(result_df, scenario_name)

    result_df.to_csv(output_path, index=False, encoding='utf-8-sig')
    print(f"\n结果已保存: {output_path}")
    return result_df, stats

def save_weights_record(weights_dict: Dict, output_dir: str, calculated_weights: Dict):
    """
    将权重保存为 CSV 记录文件，并精确标记来源
    
    Args:
        weights_dict: 所有使用的权值字典
        output_dir: 输出目录
        calculated_weights: 仅包含由 CRITIC 计算出的权值字典
    """
    # 构建记录 DataFrame
    record = []
    for key, val in weights_dict.items():
        # 精确判断来源：只有当该键名存在于本次计算出的字典中时，才标记为 CRITIC
        source = 'CRITIC-Entropy' if key in calculated_weights else 'YAML_Fixed'
        
        record.append({
            'weight_name': key,
            'value': val,
            'source': source
        })
    
    df_weights = pd.DataFrame(record)
    
    # 保存路径
    weights_path = os.path.join(output_dir, 'used_weights.csv')
    df_weights.to_csv(weights_path, index=False, encoding='utf-8-sig')
    print(f"\n权重记录已保存至: {weights_path}")
    
    # 控制台打印提示
    if calculated_weights:
        print(f"  数据驱动更新的参数: {list(calculated_weights.keys())}")
    
    return df_weights

# =============================================================================
# 主函数
# =============================================================================

def main():
    global CONFIG

    parser = argparse.ArgumentParser(
        description='计算施工区场景风险场强',
        formatter_class=argparse.RawDescriptionHelpFormatter
    )

    parser.add_argument('-c', '--config',     type=str, default='config/risk_field.yaml', help='配置文件路径')
    parser.add_argument('-i', '--input_dir',  type=str, default=None)
    parser.add_argument('-o', '--output_dir', type=str, default=None)
    parser.add_argument('-v', '--vis_dir',    type=str, default=None)
    parser.add_argument('-s', '--scenarios',  type=str, nargs='+', default=None)
    
    # 客观赋权开关
    critic_group = parser.add_mutually_exclusive_group()
    critic_group.add_argument('--use-critic', action='store_true', dest='use_critic', help='强制启用客观赋权')
    critic_group.add_argument('--no-use-critic', action='store_false', dest='use_critic', help='强制禁用客观赋权')
    parser.set_defaults(use_critic=None)

    args = parser.parse_args()
    
    # 1. 加载配置 (无默认值)
    CONFIG = load_config(args.config)

    # 2. 命令行覆盖配置
    if args.use_critic is not None:
        CONFIG['use_critic_entropy'] = args.use_critic

    input_dir  = args.input_dir  or CONFIG['paths']['input_dir']
    output_dir = args.output_dir or CONFIG['paths']['output_dir']
    vis_dir    = args.vis_dir    or CONFIG['paths']['vis_dir']
    scenarios  = args.scenarios  or CONFIG['scenarios']['default']

    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(vis_dir,    exist_ok=True)

    # 3. 打印启动信息
    print("=" * 60)
    print("风险场强计算与可视化")
    print("=" * 60)
    print(f"配置文件: {args.config}")
    print(f"输入目录: {input_dir}")
    print(f"输出目录: {output_dir}")
    print(f"处理场景: {', '.join(scenarios)}")
    
    # 4. 客观赋权逻辑 (已取消注释并激活)
    calculated_weights = {}
    if CONFIG.get('use_critic_entropy', False):
        print("\n" + "=" * 60)
        print("模式: 启用 CRITIC-熵权客观赋权")
        print("=" * 60)
        
        # >>> 激活调用 <<<
        calculated_weights = _collect_and_calculate_weights(scenarios, input_dir)
        if calculated_weights:
            CONFIG['weights'].update(calculated_weights)
            print("\n[CRITIC] 已更新当前运行时权重配置。")
    else:
        print("\n模式: 使用 YAML 固定权重")
    
    # 5. 详细打印权重
    w = CONFIG['weights']
    print("\n" + "-" * 40)
    print("本次运行最终权重配置:")
    print("-" * 40)
    
    # 分类打印，更清晰
    print(f"[顶层综合权重]")
    print(f"  w_veh  (车辆交互):   {w.get('w_veh', 'N/A'):.4f}")
    print(f"  w_geo  (道路几何):   {w.get('w_geo', 'N/A'):.4f}")
    print(f"  w_sign (道路设施):   {w.get('w_sign', 'N/A'):.4f}")
    
    print(f"\n[道路设施场强 (s_sign)]")
    print(f"  lambda_1 (标志密度): {w.get('lambda_1', 'N/A'):.4f}")
    print(f"  lambda_2 (施工区):   {w.get('lambda_2', 'N/A'):.4f}")
    
    if 'mu_1' in w:
        print(f"\n[车辆交互场强 (s_veh)]")
        print(f"  mu_1 (跟车时距):    {w.get('mu_1', 'N/A'):.4f}")
        print(f"  mu_2 (侧向密度):    {w.get('mu_2', 'N/A'):.4f}")
        print(f"  mu_3 (速度差):      {w.get('mu_3', 'N/A'):.4f}")
    print("-" * 40)
    
    # 6. 保存权重到 CSV
    save_weights_record(w, output_dir, calculated_weights)

    print("=" * 60)

    # 7. 处理循环
    all_stats   = []
    all_results = {}

    for scenario in scenarios:
        input_file  = os.path.join(input_dir,  f"{scenario}_continuous.csv")
        output_file = os.path.join(output_dir, f"{scenario}_risk_field.csv")

        if not os.path.exists(input_file):
            print(f"警告: 文件不存在，跳过 - {input_file}")
            continue

        try:
            result_df, stats = process_scenario(input_file, output_file, scenario)
            all_stats.append(stats)
            all_results[scenario] = result_df
        except Exception as e:
            print(f"错误: 处理 {scenario} 时出错 - {str(e)}")
            continue

    # 8. 收尾：保存场景汇总统计
    if all_stats:
        summary_df   = pd.DataFrame(all_stats)
        summary_path = os.path.join(output_dir, 'scenario_summary.csv')
        summary_df.to_csv(summary_path, index=False, encoding='utf-8-sig')
        print(f"\n场景汇总统计已保存: {summary_path}")
        print(f"\n可视化已保存: {vis_dir}")
        print("\n场景对比:")
        print(summary_df.to_string(index=False))

    # 9. 可视化分析（论文图4.3/4.4）
    if all_stats:
        print("\n" + "=" * 60)
        print("生成可视化图表...")
        print("=" * 60)

        # 准备可视化数据字典
        vis_data = {row['scenario']: row for row in all_stats}

        # 四维雷达图
        plot_radar_chart(
            scenarios=scenarios,
            data=vis_data,
            output_dir=vis_dir
        )

        # 堆叠柱状图（传入论文顶层权重）
        plot_stacked_bar(
            scenarios=scenarios,
            data=vis_data,
            output_dir=vis_dir,
            w_geo=CONFIG['weights']['w_geo'],
            w_sign=CONFIG['weights']['w_sign'],
            w_veh=CONFIG['weights']['w_veh']
        )

        # 各场景场强演化曲线
        for scenario, result_df in all_results.items():
            plot_field_evolution(
                df=result_df,
                scenario_name=scenario,
                output_dir=vis_dir
            )
            plot_three_scenarios_evolution(
                results_dict=all_results,
                output_dir=vis_dir
            )

if __name__ == '__main__':
    main()
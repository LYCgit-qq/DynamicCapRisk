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
import yaml
from typing import Tuple, Dict
from scipy.ndimage import gaussian_filter1d

# 导入可视化函数
from src.visualization.plot_risk import plot_radar_chart, plot_stacked_bar, plot_field_evolution


# ============= 全局配置变量 =============

def load_config(config_path: str = 'config/risk_field.yaml') -> Dict:
    """
    加载配置文件

    Args:
        config_path: 配置文件路径

    Returns:
        配置字典
    """
    try:
        with open(config_path, 'r', encoding='utf-8') as f:
            config = yaml.safe_load(f)
        print(f"成功加载配置文件: {config_path}")
        return config
    except FileNotFoundError:
        print(f"警告: 配置文件不存在 {config_path}，使用默认参数")
        return get_default_config()
    except Exception as e:
        print(f"警告: 配置文件加载失败 {e}，使用默认参数")
        return get_default_config()


def get_default_config() -> Dict:
    """
    获取默认配置（与YAML文件保持一致）

    Returns:
        默认配置字典
    """
    return {
        'weights': {
            'w_veh': 0.50,
            'w_geo': 0.22,
            'w_sign': 0.28,
            'lambda_1': 0.65,
            'lambda_2': 0.35
        },
        'geometry': {
            'curvature_map': {'straight': 0.0, 'cross': 0.5, 'bend': 1.0},
            'lane_map': {1: 1.0, 2: 0.5, 4: 0.0},
            'lane_width_map': {4: 0.0, 2: 0.5, 1: 1.0},
            'smooth_sigma': 30.0
        },
        'vehicle_interaction': {
            'baseline': 0.30,
            'work_zone': 0.65,
            'complex_geometry': 0.85,
            'boundary_smooth_sigma': 25.0,
            'entry_peak': 0.98,
            'entry_sigma': 25.0,
            'entry_offset': -15.0
        },
        'sign_field': {
            'sigma_front': 120.0,
            'sigma_back': 40.0,
            'wz_smooth_sigma': 20.0
        },
        'field_levels': {
            'low': [0.0, 0.3],
            'medium': [0.3, 0.5],
            'medium_high': [0.5, 0.7],
            'high': [0.7, 1.0]
        },
        'calculation': {
            'window_size': 100,
            'final_smooth_sigma': 8.0
        },
        'paths': {
            'input_dir': 'data/processed',
            'output_dir': 'data/risk_field',
            'vis_dir': 'visualizations'
        },
        'scenarios': {
            'default': ['test00', 'test01', 'test02']
        }
    }


# =============================================================================
# 工具函数
# =============================================================================

def _get_dx(distances: np.ndarray) -> float:
    """
    推断距离序列的采样间隔（m/sample）。
    用中位数而非均值，对偶发的缺失点更鲁棒。
    """
    if len(distances) < 2:
        return 1.0
    dx = float(np.median(np.diff(distances)))
    return dx if dx > 1e-6 else 1.0


def _gaussian_smooth(values: np.ndarray,
                     distances: np.ndarray,
                     sigma_m: float) -> np.ndarray:
    """
    对值序列进行高斯平滑（sigma 以米为单位）。

    将物理单位的 sigma_m 换算为样本数，再调用
    scipy.ndimage.gaussian_filter1d，边界使用 'nearest' 模式
    避免两端产生额外衰减。

    Args:
        values:    待平滑的 1-D 数组
        distances: 对应的沿程距离坐标（m），用于换算 sigma
        sigma_m:   高斯核标准差（m）

    Returns:
        平滑后的数组，形状与 values 相同
    """
    dx = _get_dx(distances)
    sigma_samples = max(sigma_m / dx, 0.5)   # 至少半个样本，避免无效调用
    return gaussian_filter1d(values.astype(float),
                             sigma=sigma_samples,
                             mode='nearest')


def _smooth_mask(binary_mask: np.ndarray,
                 distances: np.ndarray,
                 sigma_m: float) -> np.ndarray:
    """
    将二值掩码（0/1）软化为连续过渡的 [0,1] 曲线。

    原理：对二值序列做高斯卷积，边界处自然形成 S 形过渡，
    过渡区宽度约为 4·sigma_m。

    Args:
        binary_mask: 0/1 的 numpy 数组
        distances:   对应的沿程距离坐标（m）
        sigma_m:     控制过渡区宽度的高斯标准差（m）

    Returns:
        连续过渡掩码，值域 [0, 1]
    """
    return _gaussian_smooth(binary_mask.astype(float), distances, sigma_m)


# =============================================================================
# 场强分量计算
# =============================================================================

def calculate_geo_field(df: pd.DataFrame) -> pd.Series:
    """
    计算道路几何特征场强
    s_geo = 1/3 · (ĉ + ŵ + l̂)

    离散映射后对整条曲线做高斯平滑，消除几何类型切换处的阶跃。
    平滑 sigma 由 geometry.smooth_sigma（默认 30 m）控制。

    Args:
        df: 包含 '道路几何类型'、'车道数'、'距离 (m)' 的 DataFrame

    Returns:
        平滑后的道路几何场强序列（未归一化）
    """
    geo_cfg = CONFIG['geometry']
    distances = df['距离 (m)'].values.astype(float)

    # 离散映射
    c_hat = df['道路几何类型'].map(geo_cfg['curvature_map']).fillna(0.0).values
    w_hat = df['车道数'].map(geo_cfg['lane_width_map']).fillna(0.0).values
    l_hat = df['车道数'].map(geo_cfg['lane_map']).fillna(0.0).values

    raw_geo = (c_hat + w_hat + l_hat) / 3.0

    # 高斯平滑：消除几何类型突变带来的阶跃
    sigma_m = geo_cfg.get('smooth_sigma', 30.0)
    smoothed = _gaussian_smooth(raw_geo, distances, sigma_m)

    return pd.Series(smoothed, index=df.index)


def calculate_sign_field(df: pd.DataFrame) -> pd.Series:
    """
    计算道路设施场强（非对称指数衰减核 + 平滑施工区项）

    标志牌影响：
      - 车辆尚未到达（前方）：慢衰减，sigma_front（默认 120 m）
      - 车辆已通过（后方）：快衰减，sigma_back（默认 40 m）
      取所有标志牌影响的逐点最大值

    施工区存在项 δ_wz：
      将二值掩码用高斯核软化（sigma = wz_smooth_sigma，默认 20 m），
      消除进出施工区时的硬跳变。

    s_sign = λ1 · influence_max(x) + λ2 · δ_wz_smooth(x)

    Args:
        df: 包含 '标识牌类型'、'施工区状态'、'距离 (m)' 的 DataFrame

    Returns:
        道路设施场强序列（未归一化）
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
        d_vec  = distances - sign_pos          # 负：未到达；正：已通过
        kernel = np.where(
            d_vec <= 0,
            np.exp(d_vec  / sigma_front),      # 前方慢衰减
            np.exp(-d_vec / sigma_back)         # 后方快衰减
        )
        sign_influence = np.maximum(sign_influence, kernel)

    max_inf = sign_influence.max()
    if max_inf > 1e-6:
        sign_influence /= max_inf

    # ---------- 施工区存在项（软化二值边界） ----------
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
# 归一化 / 等级判定
# =============================================================================

def normalize_field(field: pd.Series) -> pd.Series:
    """
    将场强归一化到 [0, 1] 区间

    Args:
        field: 原始场强

    Returns:
        归一化后的场强
    """
    min_val = field.min()
    max_val = field.max()

    if max_val - min_val < 1e-6:
        return pd.Series(np.zeros(len(field)), index=field.index)

    return (field - min_val) / (max_val - min_val)


def get_field_level(mean_field: float) -> str:
    """
    根据平均场强判断总体等级

    Args:
        mean_field: 平均场强值

    Returns:
        场强等级（低/中/中高/高）
    """
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
# 综合场强
# =============================================================================

def calculate_comprehensive_field(df: pd.DataFrame) -> pd.DataFrame:
    """
    计算综合风险场强
    F_S = w_veh·s̃_veh + w_geo·s̃_geo + w_sign·s̃_sign

    最终对 F_S 做一次轻量高斯平滑（sigma = final_smooth_sigma，默认 8 m），
    消除各分量归一化后可能残余的微小锯齿。

    Args:
        df: 场景数据

    Returns:
        包含各分量场强和综合场强的 DataFrame
    """
    weights    = CONFIG['weights']
    levels     = CONFIG['field_levels']
    final_sigma = CONFIG['calculation'].get('final_smooth_sigma', 8.0)

    result    = df[['距离 (m)']].copy()
    distances = df['距离 (m)'].values.astype(float)

    # 计算各分量
    s_geo  = calculate_geo_field(df)
    s_sign = calculate_sign_field(df)
    s_veh  = calculate_vehicle_field_static(df)

    # 归一化
    s_geo_norm  = normalize_field(s_geo)
    s_sign_norm = normalize_field(s_sign)
    s_veh_norm  = normalize_field(s_veh)

    # 加权合成
    F_S_raw = (weights['w_veh']  * s_veh_norm +
               weights['w_geo']  * s_geo_norm  +
               weights['w_sign'] * s_sign_norm)

    # 最终轻量平滑：消除残余锯齿，保留宏观形态
    F_S = pd.Series(
        _gaussian_smooth(F_S_raw.values, distances, final_sigma),
        index=df.index
    )
    # 确保仍在 [0,1]（高斯平滑是线性操作，不会越界，但防御性裁剪）
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
# 统计 / 场景处理 / 主函数
# =============================================================================

def get_scenario_statistics(result_df: pd.DataFrame, scenario_name: str) -> Dict:
    """
    获取场景统计信息

    Args:
        result_df:     计算结果 DataFrame
        scenario_name: 场景名称

    Returns:
        统计信息字典
    """
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
    """
    处理单个场景

    Args:
        input_path:    输入 CSV 路径
        output_path:   输出 CSV 路径
        scenario_name: 场景名称

    Returns:
        (结果 DataFrame, 统计信息字典)
    """
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

    print(f"\n场景统计:")
    print(f"  道路几何场强 (归一化): {stats['s_geo_mean']:.3f}")
    print(f"  道路设施场强 (归一化): {stats['s_sign_mean']:.3f}")
    print(f"  车辆交互场强 (归一化): {stats['s_veh_mean']:.3f}")
    print(f"  综合场强 F_S: {stats['F_S_mean']:.3f} "
          f"(范围: {stats['F_S_min']:.3f} - {stats['F_S_max']:.3f})")
    print(f"  场强等级: {stats['field_level']}")

    return result_df, stats


def main():
    """主函数"""
    global CONFIG

    parser = argparse.ArgumentParser(
        description='计算施工区场景风险场强并生成可视化',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例用法:
  python risk_field.py
  python risk_field.py -c config/custom.yaml
  python risk_field.py -i data/processed -o data/risk_field -v visualizations
  python risk_field.py --scenarios test00 test01
        """
    )

    parser.add_argument('-c', '--config',     type=str, default='config/risk_field.yaml')
    parser.add_argument('-i', '--input_dir',  type=str, default=None)
    parser.add_argument('-o', '--output_dir', type=str, default=None)
    parser.add_argument('-v', '--vis_dir',    type=str, default=None)
    parser.add_argument('-s', '--scenarios',  type=str, nargs='+', default=None)

    args   = parser.parse_args()
    CONFIG = load_config(args.config)

    input_dir  = args.input_dir  or CONFIG['paths']['input_dir']
    output_dir = args.output_dir or CONFIG['paths']['output_dir']
    vis_dir    = args.vis_dir    or CONFIG['paths']['vis_dir']
    scenarios  = args.scenarios  or CONFIG['scenarios']['default']

    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(vis_dir,    exist_ok=True)

    print("=" * 60)
    print("风险场强计算与可视化")
    print("=" * 60)
    print(f"输入目录: {input_dir}")
    print(f"输出目录: {output_dir}")
    print(f"可视化目录: {vis_dir}")
    print(f"处理场景: {', '.join(scenarios)}")
    w = CONFIG['weights']
    print(f"权重配置: 车辆={w['w_veh']}, 几何={w['w_geo']}, 设施={w['w_sign']}")
    print("=" * 60)

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

    if all_stats:
        summary_df   = pd.DataFrame(all_stats)
        summary_path = os.path.join(output_dir, 'scenario_summary.csv')
        summary_df.to_csv(summary_path, index=False, encoding='utf-8-sig')

        print(f"\n{'='*60}")
        print(f"所有场景处理完成！")
        print(f"汇总统计已保存: {summary_path}")
        print(f"{'='*60}")
        print(f"\n场景对比:")
        print(summary_df.to_string(index=False))

    if all_results:
        print(f"\n{'='*60}")
        print("生成可视化...")
        print(f"{'='*60}")

        stats_dict    = {s['scenario']: s for s in all_stats}
        scenarios_list = list(all_results.keys())
        w             = CONFIG['weights']

        print("\n生成雷达图...")
        plot_radar_chart(scenarios_list, stats_dict, vis_dir)

        print("生成堆叠柱状图...")
        plot_stacked_bar(scenarios_list, stats_dict, vis_dir,
                         w['w_geo'], w['w_sign'], w['w_veh'])

        print("\n生成场景演化曲线...")
        for scenario, df in all_results.items():
            plot_field_evolution(df, scenario, vis_dir)

        print(f"\n{'='*60}")
        print("可视化完成！")
        print(f"输出目录: {vis_dir}")
        print(f"{'='*60}")


if __name__ == '__main__':
    main()
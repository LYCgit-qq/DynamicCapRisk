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
            'w_veh': 0.55,
            'w_geo': 0.25,
            'w_sign': 0.20,
            'lambda_1': 0.6,
            'lambda_2': 0.4
        },
        'geometry': {
            'curvature_map': {'straight': 0.0, 'cross': 0.5, 'bend': 1.0},
            'lane_map': {1: 1.0, 2: 0.5, 4: 0.0},
            'lane_width_map': {4: 0.0, 2: 0.5, 1: 1.0}
        },
        'vehicle_interaction': {
            'baseline': 0.3,
            'work_zone': 0.65,
            'complex_geometry': 0.85
        },
        'field_levels': {
            'low': [0.0, 0.3],
            'medium': [0.3, 0.5],
            'medium_high': [0.5, 0.7],
            'high': [0.7, 1.0]
        },
        'calculation': {
            'window_size': 100
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


def calculate_geo_field(df: pd.DataFrame) -> pd.Series:
    """
    计算道路几何特征场强
    s_geo = 1/3(ĉ + ŵ + l̂)
    
    Args:
        df: 包含道路几何信息的DataFrame
        
    Returns:
        道路几何场强序列
    """
    geo_config = CONFIG['geometry']
    
    c_hat = df['道路几何类型'].map(geo_config['curvature_map']).fillna(0.0)
    w_hat = df['车道数'].map(geo_config['lane_width_map']).fillna(0.0)
    l_hat = df['车道数'].map(geo_config['lane_map']).fillna(0.0)
    
    return (c_hat + w_hat + l_hat) / 3.0


def calculate_sign_field(df: pd.DataFrame, window_size: int = None) -> pd.Series:
    """
    计算道路设施场强
    s_sign = λ1·(n_sign/n_sign_max) + λ2·δ_wz
    
    Args:
        df: 包含道路设施信息的DataFrame
        window_size: 滑动窗口大小（米），用于计算标志密度
        
    Returns:
        道路设施场强序列
    """
    if window_size is None:
        window_size = CONFIG['calculation']['window_size']
    
    weights = CONFIG['weights']
    
    delta_wz = (df['施工区状态'] == '是').astype(int)
    has_sign = (df['标识牌类型'] != '-').astype(int)
    
    n_sign = has_sign.rolling(window=window_size, min_periods=1, center=True).sum()
    n_sign_max = n_sign.max() if n_sign.max() > 0 else 1.0
    
    return weights['lambda_1'] * (n_sign / n_sign_max) + weights['lambda_2'] * delta_wz


def calculate_vehicle_field_static(df: pd.DataFrame) -> pd.Series:
    """
    计算车辆交互场强（静态估计版本）
    基于场景类型给出估计值，参考论文表4.3的典型值
    
    Args:
        df: 场景数据
        
    Returns:
        车辆交互场强序列（估计值）
    """
    veh_config = CONFIG['vehicle_interaction']
    
    s_veh = np.full(len(df), veh_config['baseline'])  # 基线值
    
    work_zone_mask = df['施工区状态'] == '是'
    s_veh[work_zone_mask] = veh_config['work_zone']
    
    complex_geom_mask = df['道路几何类型'].isin(['bend', 'cross'])
    s_veh[work_zone_mask & complex_geom_mask] = veh_config['complex_geometry']
    
    return pd.Series(s_veh, index=df.index)


def normalize_field(field: pd.Series) -> pd.Series:
    """
    将场强归一化到[0,1]区间
    
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


def calculate_comprehensive_field(df: pd.DataFrame) -> pd.DataFrame:
    """
    计算综合风险场强
    F_S = w1·s̃_veh + w2·s̃_geo + w3·s̃_sign
    
    Args:
        df: 场景数据
        
    Returns:
        包含各分量场强和综合场强的DataFrame
    """
    weights = CONFIG['weights']
    levels = CONFIG['field_levels']
    
    result = df[['距离 (m)']].copy()
    
    # 计算各分量场强
    s_geo = calculate_geo_field(df)
    s_sign = calculate_sign_field(df)
    s_veh = calculate_vehicle_field_static(df)
    
    # 归一化
    s_geo_norm = normalize_field(s_geo)
    s_sign_norm = normalize_field(s_sign)
    s_veh_norm = normalize_field(s_veh)
    
    # 计算综合场强
    F_S = (weights['w_veh'] * s_veh_norm + 
           weights['w_geo'] * s_geo_norm + 
           weights['w_sign'] * s_sign_norm)
    
    # 保存结果
    result['s_geo'] = s_geo
    result['s_geo_norm'] = s_geo_norm
    result['s_sign'] = s_sign
    result['s_sign_norm'] = s_sign_norm
    result['s_veh'] = s_veh
    result['s_veh_norm'] = s_veh_norm
    result['F_S'] = F_S
    
    # 添加场强等级
    result['field_level'] = pd.cut(
        F_S,
        bins=[levels['low'][0], levels['medium'][0], 
              levels['medium_high'][0], levels['high'][0], levels['high'][1]],
        labels=['低', '中', '中高', '高'],
        include_lowest=True
    )
    
    return result


def get_scenario_statistics(result_df: pd.DataFrame, scenario_name: str) -> Dict:
    """
    获取场景统计信息
    
    Args:
        result_df: 计算结果DataFrame
        scenario_name: 场景名称
        
    Returns:
        统计信息字典
    """
    return {
        'scenario': scenario_name,
        's_geo_mean': result_df['s_geo_norm'].mean(),
        's_sign_mean': result_df['s_sign_norm'].mean(),
        's_veh_mean': result_df['s_veh_norm'].mean(),
        'F_S_mean': result_df['F_S'].mean(),
        'F_S_max': result_df['F_S'].max(),
        'F_S_min': result_df['F_S'].min(),
        'F_S_std': result_df['F_S'].std(),
        'field_level': get_field_level(result_df['F_S'].mean())
    }


def process_scenario(
    input_path: str,
    output_path: str,
    scenario_name: str
) -> Tuple[pd.DataFrame, Dict]:
    """
    处理单个场景
    
    Args:
        input_path: 输入CSV路径
        output_path: 输出CSV路径
        scenario_name: 场景名称
        
    Returns:
        (结果DataFrame, 统计信息字典)
    """
    df = pd.read_csv(input_path, encoding='utf-8-sig')
    
    print(f"\n{'='*60}")
    print(f"处理场景: {scenario_name}")
    print(f"输入文件: {input_path}")
    print(f"数据行数: {len(df)}")
    print(f"{'='*60}")
    
    result_df = calculate_comprehensive_field(df)
    stats = get_scenario_statistics(result_df, scenario_name)
    
    result_df.to_csv(output_path, index=False, encoding='utf-8-sig')
    print(f"\n结果已保存: {output_path}")
    
    print(f"\n场景统计:")
    print(f"  道路几何场强 (归一化): {stats['s_geo_mean']:.3f}")
    print(f"  道路设施场强 (归一化): {stats['s_sign_mean']:.3f}")
    print(f"  车辆交互场强 (归一化): {stats['s_veh_mean']:.3f}")
    print(f"  综合场强 F_S: {stats['F_S_mean']:.3f} (范围: {stats['F_S_min']:.3f} - {stats['F_S_max']:.3f})")
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
    
    parser.add_argument(
        '-c', '--config',
        type=str,
        default='config/risk_field.yaml',
        help='配置文件路径（默认: config/risk_field.yaml）'
    )
    
    parser.add_argument(
        '-i', '--input_dir',
        type=str,
        default=None,
        help='输入目录（包含*_continuous.csv文件），覆盖配置文件'
    )
    
    parser.add_argument(
        '-o', '--output_dir',
        type=str,
        default=None,
        help='输出目录（保存风险场强CSV），覆盖配置文件'
    )
    
    parser.add_argument(
        '-v', '--vis_dir',
        type=str,
        default=None,
        help='可视化输出目录，覆盖配置文件'
    )
    
    parser.add_argument(
        '-s', '--scenarios',
        type=str,
        nargs='+',
        default=None,
        help='要处理的场景列表，覆盖配置文件'
    )
    
    args = parser.parse_args()
    
    # 加载配置文件
    CONFIG = load_config(args.config)
    
    # 命令行参数覆盖配置文件
    input_dir = args.input_dir if args.input_dir else CONFIG['paths']['input_dir']
    output_dir = args.output_dir if args.output_dir else CONFIG['paths']['output_dir']
    vis_dir = args.vis_dir if args.vis_dir else CONFIG['paths']['vis_dir']
    scenarios = args.scenarios if args.scenarios else CONFIG['scenarios']['default']
    
    # 创建输出目录
    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(vis_dir, exist_ok=True)
    
    print("=" * 60)
    print("风险场强计算与可视化")
    print("=" * 60)
    print(f"输入目录: {input_dir}")
    print(f"输出目录: {output_dir}")
    print(f"可视化目录: {vis_dir}")
    print(f"处理场景: {', '.join(scenarios)}")
    print(f"权重配置: 车辆={CONFIG['weights']['w_veh']}, "
          f"几何={CONFIG['weights']['w_geo']}, "
          f"设施={CONFIG['weights']['w_sign']}")
    print("=" * 60)
    
    # 处理每个场景
    all_stats = []
    all_results = {}
    
    for scenario in scenarios:
        input_file = os.path.join(input_dir, f"{scenario}_continuous.csv")
        output_file = os.path.join(output_dir, f"{scenario}_risk_field.csv")
        
        if not os.path.exists(input_file):
            print(f"警告: 文件不存在，跳过 - {input_file}")
            continue
        
        try:
            result_df, stats = process_scenario(
                input_path=input_file,
                output_path=output_file,
                scenario_name=scenario
            )
            all_stats.append(stats)
            all_results[scenario] = result_df
        except Exception as e:
            print(f"错误: 处理 {scenario} 时出错 - {str(e)}")
            continue
    
    # 保存汇总统计
    if all_stats:
        summary_df = pd.DataFrame(all_stats)
        summary_path = os.path.join(output_dir, 'scenario_summary.csv')
        summary_df.to_csv(summary_path, index=False, encoding='utf-8-sig')
        
        print(f"\n{'='*60}")
        print(f"所有场景处理完成！")
        print(f"汇总统计已保存: {summary_path}")
        print(f"{'='*60}")
        
        # 打印对比表格
        print(f"\n场景对比:")
        print(summary_df.to_string(index=False))
    
    # 生成可视化
    if all_results:
        print(f"\n{'='*60}")
        print("生成可视化...")
        print(f"{'='*60}")
        
        # 构建统计数据字典
        stats_dict = {stat['scenario']: stat for stat in all_stats}
        scenarios_list = list(all_results.keys())
        
        weights = CONFIG['weights']
        
        # 1. 绘制雷达图
        print("\n生成雷达图...")
        plot_radar_chart(scenarios_list, stats_dict, vis_dir)
        
        # 2. 绘制堆叠柱状图
        print("生成堆叠柱状图...")
        plot_stacked_bar(scenarios_list, stats_dict, vis_dir, 
                        weights['w_geo'], weights['w_sign'], weights['w_veh'])
        
        # 3. 绘制每个场景的演化曲线
        print("\n生成场景演化曲线...")
        for scenario, df in all_results.items():
            plot_field_evolution(df, scenario, vis_dir)
        
        print(f"\n{'='*60}")
        print("可视化完成！")
        print(f"输出目录: {vis_dir}")
        print(f"{'='*60}")


if __name__ == '__main__':
    main()
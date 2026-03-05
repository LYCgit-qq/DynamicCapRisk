"""
风险场强可视化函数模块
提供雷达图、堆叠柱状图、演化曲线等可视化功能
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
import os
from typing import List, Dict

# 设置中文字体支持
matplotlib.rcParams['font.sans-serif'] = ['SimHei', 'Arial Unicode MS', 'DejaVu Sans']
matplotlib.rcParams['axes.unicode_minus'] = False


def plot_radar_chart(
    scenarios: List[str], 
    data: Dict[str, Dict], 
    output_dir: str,
    save_name: str = "figure_4_3_radar_chart.png"
):
    """
    绘制雷达图（论文图4.3）
    
    Args:
        scenarios: 场景名称列表
        data: 数据字典 {场景名: {指标名: 值}}
        output_dir: 输出目录
        save_name: 保存文件名
    """
    categories = ['道路几何', '道路设施', '车辆交互', '综合场强']
    N = len(categories)
    
    angles = [n / float(N) * 2 * np.pi for n in range(N)]
    angles += angles[:1]
    
    fig, ax = plt.subplots(figsize=(10, 8), subplot_kw=dict(projection='polar'))
    
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728']
    
    for idx, scenario in enumerate(scenarios):
        values = [
            data[scenario].get('s_geo_mean', 0),
            data[scenario].get('s_sign_mean', 0),
            data[scenario].get('s_veh_mean', 0),
            data[scenario].get('F_S_mean', 0)
        ]
        values += values[:1]
        
        ax.plot(angles, values, 'o-', linewidth=2, 
               label=scenario, color=colors[idx % len(colors)])
        ax.fill(angles, values, alpha=0.15, color=colors[idx % len(colors)])
    
    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(categories, fontsize=11)
    ax.set_ylim(0, 1)
    ax.set_yticks([0.2, 0.4, 0.6, 0.8, 1.0])
    ax.set_yticklabels(['0.2', '0.4', '0.6', '0.8', '1.0'], fontsize=9)
    ax.grid(True, linestyle='--', alpha=0.6)
    
    plt.legend(loc='upper right', bbox_to_anchor=(1.3, 1.1), fontsize=10)
    plt.title("风险场强四维雷达图对比", fontsize=14, pad=20, weight='bold')
    
    save_path = os.path.join(output_dir, save_name)
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"雷达图已保存: {save_path}")
    plt.close()


def plot_stacked_bar(
    scenarios: List[str], 
    data: Dict[str, Dict], 
    output_dir: str,
    w_geo: float = 0.25,
    w_sign: float = 0.20,
    w_veh: float = 0.55,
    save_name: str = "figure_4_4_stacked_bar.png"
):
    """
    绘制堆叠柱状图（论文图4.4）
    
    Args:
        scenarios: 场景名称列表
        data: 数据字典 {场景名: {指标名: 值}}
        output_dir: 输出目录
        w_geo: 道路几何权重
        w_sign: 道路设施权重
        w_veh: 车辆交互权重
        save_name: 保存文件名
    """
    geo_values = [data[s]['s_geo_mean'] * w_geo for s in scenarios]
    sign_values = [data[s]['s_sign_mean'] * w_sign for s in scenarios]
    veh_values = [data[s]['s_veh_mean'] * w_veh for s in scenarios]
    
    fig, ax = plt.subplots(figsize=(10, 6))
    
    x = np.arange(len(scenarios))
    width = 0.6
    
    p1 = ax.bar(x, geo_values, width, label=f'道路几何 (w={w_geo})', 
               color='#1f77b4', edgecolor='white', linewidth=1.5)
    p2 = ax.bar(x, sign_values, width, bottom=geo_values,
               label=f'道路设施 (w={w_sign})', 
               color='#ff7f0e', edgecolor='white', linewidth=1.5)
    p3 = ax.bar(x, veh_values, width,
               bottom=[i+j for i,j in zip(geo_values, sign_values)],
               label=f'车辆交互 (w={w_veh})', 
               color='#2ca02c', edgecolor='white', linewidth=1.5)
    
    for idx, scenario in enumerate(scenarios):
        total = data[scenario]['F_S_mean']
        ax.text(idx, total + 0.03, f'{total:.2f}', 
               ha='center', va='bottom', fontsize=10, weight='bold')
    
    ax.set_xlabel('场景', fontsize=12, weight='bold')
    ax.set_ylabel('场强贡献值', fontsize=12, weight='bold')
    ax.set_title("风险场强子项贡献堆叠柱状图", fontsize=14, pad=15, weight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(scenarios, fontsize=11)
    ax.set_ylim(0, 1.1)
    ax.grid(axis='y', linestyle='--', alpha=0.3)
    ax.legend(loc='upper left', fontsize=10)
    
    save_path = os.path.join(output_dir, save_name)
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"堆叠柱状图已保存: {save_path}")
    plt.close()


def plot_field_evolution(
    df: pd.DataFrame, 
    scenario_name: str, 
    output_dir: str,
    save_name: str = None
):
    """
    绘制场强沿距离的演化曲线
    
    Args:
        df: 风险场强结果DataFrame
        scenario_name: 场景名称
        output_dir: 输出目录
        save_name: 保存文件名（默认为 {scenario_name}_evolution.png）
    """
    if save_name is None:
        save_name = f"{scenario_name}_evolution.png"
    
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8), 
                                   sharex=True, height_ratios=[2, 1])
    
    # 上图：各分量场强
    ax1.plot(df['距离 (m)'], df['s_geo_norm'], 
            label='道路几何', linewidth=2, color='#1f77b4')
    ax1.plot(df['距离 (m)'], df['s_sign_norm'], 
            label='道路设施', linewidth=2, color='#ff7f0e')
    ax1.plot(df['距离 (m)'], df['s_veh_norm'], 
            label='车辆交互', linewidth=2, color='#2ca02c')
    
    ax1.set_ylabel('归一化场强', fontsize=11, weight='bold')
    ax1.set_ylim(0, 1.05)
    ax1.grid(True, linestyle='--', alpha=0.3)
    ax1.legend(loc='upper right', fontsize=10)
    ax1.set_title(f"{scenario_name} 风险场强沿距离演化", fontsize=13, pad=10, weight='bold')
    
    # 下图：综合场强
    ax2.fill_between(df['距离 (m)'], df['F_S'], 
                    color='#d62728', alpha=0.3, label='综合场强')
    ax2.plot(df['距离 (m)'], df['F_S'], 
            linewidth=2.5, color='#d62728')
    
    # 添加场强等级区域
    ax2.axhspan(0, 0.3, alpha=0.1, color='green', label='低')
    ax2.axhspan(0.3, 0.5, alpha=0.1, color='yellow', label='中')
    ax2.axhspan(0.5, 0.7, alpha=0.1, color='orange', label='中高')
    ax2.axhspan(0.7, 1.0, alpha=0.1, color='red', label='高')
    
    ax2.set_xlabel('距离 (m)', fontsize=11, weight='bold')
    ax2.set_ylabel('综合场强 F_S', fontsize=11, weight='bold')
    ax2.set_ylim(0, 1.05)
    ax2.grid(True, linestyle='--', alpha=0.3)
    ax2.legend(loc='upper right', fontsize=9, ncol=5)
    
    save_path = os.path.join(output_dir, save_name)
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"演化曲线已保存: {save_path}")
    plt.close()
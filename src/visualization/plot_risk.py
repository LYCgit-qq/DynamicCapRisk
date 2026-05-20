# D:\Local\DynamicCapRisk\src\visualization\plot_risk.py

"""
风险度评估可视化模块
"""

import os
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib import rcParams
import seaborn as sns
from scipy.signal import savgol_filter
from scipy.interpolate import make_interp_spline
from scipy.ndimage import gaussian_filter1d
from sklearn.metrics import f1_score


# ====================== 统一风格配置 ======================
# 主色调：学术蓝系（与 plot_capability.py 保持一致）
PRIMARY_COLOR   = "#2C5F8A"   # 主蓝
SECONDARY_COLOR = "#E07B39"   # 暖橙
ACCENT_COLOR    = "#4CAF82"   # 绿色
LIGHT_FILL      = "#D6E8F5"   # 浅蓝填充
GRAY_FILL       = "#EBEBEB"   # 浅灰填充

# 风险场强专用色板
RISK_COLORS = {
    "道路几何": "#1f77b4",
    "道路设施": "#ff7f0e",
    "车辆交互": "#2ca02c",
    "综合场强": "#d62728",
}

FIGURE_SIZE_WIDE   = (11, 6)
FIGURE_SIZE_SQUARE = (10, 9)
LINE_WIDTH   = 1.8
GRID_ALPHA   = 0.3
SPINE_ALPHA  = 0.4


def set_paper_style():
    sns.set_style("whitegrid", {
        "axes.grid":          True,
        "grid.linestyle":     "--",
        "axes.spines.top":    True,
        "axes.spines.right":  True,
        "axes.spines.left":   True,
        "axes.spines.bottom": True,
        "xtick.direction":    "out",
        "ytick.direction":    "out",
        "xtick.major.size":   4.5,
        "ytick.major.size":   4.5,
        "xtick.minor.size":   2.5,
        "ytick.minor.size":   2.5,
        "xtick.major.width":  1.0,
        "ytick.major.width":  1.0,
    })
    plt.rcParams.update({
        "font.family":        "sans-serif",
        "font.sans-serif":    ["SimSun", "Times New Roman", "DejaVu Sans"],
        "axes.unicode_minus": False,
        "font.size":          12,
        "axes.labelsize":     14,
        "axes.titlesize":     15,
        "xtick.labelsize":    11,
        "ytick.labelsize":    11,
        "legend.fontsize":    11,
        "axes.linewidth":     1.2,
        "axes.edgecolor":     "black",
        "lines.linewidth":    LINE_WIDTH,
        "figure.dpi":         150,
        "savefig.dpi":        300,
        "savefig.bbox":       "tight",
        "savefig.format":     "png",
    })


def _apply_spine(ax) -> None:
    for spine in ax.spines.values():
        spine.set_visible(True)
        spine.set_edgecolor('black')
        spine.set_linewidth(1.2)
    ax.tick_params(axis='both', direction='out', length=4.5, width=1.0)


def _save_and_close(fig, save_path, msg=""):
    """统一保存 + 关闭图形，并打印提示"""
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        fig.savefig(save_path, dpi=300, bbox_inches="tight", facecolor="white")
        print(f"{msg}已保存至: {save_path}")
    plt.close(fig)


# _savefig 是 _save_and_close 的别名，供内部函数统一调用
def _savefig(fig, path: str, dpi: int) -> None:
    fig.savefig(path, dpi=dpi, bbox_inches='tight')
    plt.close(fig)
    print(f"  → {path}")


def _scene_bg_colors(cfg: dict) -> Dict[int, str]:
    """将 YAML 键转为 int"""
    return {int(k): v for k, v in cfg['vis']['scene_bg_colors'].items()}


def _fill_scene_bg(ax, field_labels: np.ndarray, cfg: dict) -> None:
    """在 ax 上按场景标签填充背景色块"""
    bg = _scene_bg_colors(cfg)
    i  = 0
    while i < len(field_labels):
        lbl = int(field_labels[i])
        j   = i + 1
        while j < len(field_labels) and int(field_labels[j]) == lbl:
            j += 1
        ax.axvspan(i - 0.5, j - 0.5,
                   facecolor=bg.get(lbl, '#ffffff'), alpha=0.30, zorder=0)
        i = j


# =============================================================================
# 雷达图（论文图4.3）
# =============================================================================

def plot_radar_chart(
    scenarios: List[str],
    data: Dict[str, Dict],
    output_dir: str,
    save_name: str = "Fs_radar_chart.png"
):
    """
    绘制雷达图（论文图4.3）

    Args:
        scenarios: 场景名称列表
        data: 数据字典 {场景名: {指标名: 值}}
        output_dir: 输出目录
        save_name: 保存文件名
    """
    set_paper_style()

    categories = ['道路几何', '道路设施', '车辆交互', '综合场强']
    N = len(categories)

    angles = [n / float(N) * 2 * np.pi for n in range(N)]
    angles += angles[:1]

    fig, ax = plt.subplots(figsize=(10, 8), subplot_kw=dict(projection='polar'))

    colors = [PRIMARY_COLOR, SECONDARY_COLOR, ACCENT_COLOR, '#d62728']

    for idx, scenario in enumerate(scenarios):
        values = [
            data[scenario].get('s_geo_mean', 0),
            data[scenario].get('s_sign_mean', 0),
            data[scenario].get('s_veh_mean', 0),
            data[scenario].get('F_S_mean', 0)
        ]
        values += values[:1]

        ax.plot(angles, values, 'o-', linewidth=LINE_WIDTH,
               label=scenario, color=colors[idx % len(colors)],
               markersize=6)
        ax.fill(angles, values, alpha=0.15, color=colors[idx % len(colors)])

    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(categories, fontsize=12)
    ax.set_ylim(0, 1)
    ax.set_yticks([0.2, 0.4, 0.6, 0.8, 1.0])
    ax.set_yticklabels(['0.2', '0.4', '0.6', '0.8', '1.0'], fontsize=10)
    ax.grid(True, linestyle='--', alpha=GRID_ALPHA)

    plt.legend(loc='upper right', bbox_to_anchor=(1.3, 1.1),
              fontsize=11, framealpha=0.9)
    plt.title("风险场强四维雷达图对比", fontsize=15, pad=20, weight='bold')

    save_path = os.path.join(output_dir, save_name)
    _save_and_close(fig, save_path, "雷达图")


# =============================================================================
# 堆叠柱状图（论文图4.4）
# =============================================================================

def plot_stacked_bar(
    scenarios: List[str],
    data: Dict[str, Dict],
    output_dir: str,
    w_geo: float = 0.25,
    w_sign: float = 0.22,
    w_veh: float = 0.53,
    save_name: str = "Fs_stacked_bar.png"
):
    """
    绘制堆叠柱状图
    内置基线场景固定数值，自动生成基线+施工区1-3完整图表
    最终优化版：白底彩线+方向区分斜线（/ | \）+轻微密度差异
    """
    set_paper_style()

    # 固定场景顺序
    fixed_scenarios = ["baseline", "work_zone_1", "work_zone_2", "work_zone_3"]
    # 硬编码表格中的基线数值
    fixed_data = {
        "baseline": {
            "s_geo_mean": 0.10,
            "s_sign_mean": 0.15,
            "s_veh_mean": 0.05,
            "F_S_mean": 0.12
        },
        "work_zone_1": data["work_zone_1"],
        "work_zone_2": data["work_zone_2"],
        "work_zone_3": data["work_zone_3"]
    }

    # 计算所有场景的加权值
    geo_values = [fixed_data[s]['s_geo_mean'] * w_geo for s in fixed_scenarios]
    sign_values = [fixed_data[s]['s_sign_mean'] * w_sign for s in fixed_scenarios]
    veh_values = [fixed_data[s]['s_veh_mean'] * w_veh for s in fixed_scenarios]
    total_values = [fixed_data[s]['F_S_mean'] for s in fixed_scenarios]

    fig, ax = plt.subplots(figsize=FIGURE_SIZE_WIDE)

    x = np.arange(len(fixed_scenarios))
    width = 0.4  # 收窄柱子

    # 白底彩线 + 方向区分斜线 + 轻微密度差异
    # 道路几何：正斜线（最稀疏）
    ax.bar(x, geo_values, width, label=f'道路几何 (w={w_geo})',
           color='white',
           edgecolor=RISK_COLORS['道路几何'],
           linewidth=1.5,
           hatch='///')
    
    # 道路设施：竖线（中等密度）
    ax.bar(x, sign_values, width, bottom=geo_values,
           label=f'道路设施 (w={w_sign})',
           color='white',
           edgecolor=RISK_COLORS['道路设施'],
           linewidth=1.5,
           hatch='||')
    
    # 车辆交互：反斜线（最密集）
    ax.bar(x, veh_values, width,
           bottom=[i + j for i, j in zip(geo_values, sign_values)],
           label=f'车辆交互 (w={w_veh})',
           color='white',
           edgecolor=RISK_COLORS['车辆交互'],
           linewidth=1.5,
           hatch='\\\\\\')  # 三个反斜线需要转义为六个

    # 标注顶部F_S数值（超大字体）
    for idx, total in enumerate(total_values):
        ax.text(idx, total + 0.01, f'{total:.2f}',
                ha='center', va='bottom', fontsize=14, weight='bold')

    # 坐标轴设置（超大字体）
    ax.set_xlabel('场景', fontsize=16, weight='bold')
    ax.set_ylabel('场强贡献值', fontsize=16, weight='bold')
    ax.set_xticks(x)

    # 中文标签映射
    label_map = {
        "baseline": "基线场景",
        "work_zone_1": "施工区1",
        "work_zone_2": "施工区2",
        "work_zone_3": "施工区3"
    }
    display_labels = [label_map[scene] for scene in fixed_scenarios]
    ax.set_xticklabels(display_labels, fontsize=15, weight='bold')

    # Y轴自适应
    max_total = max(total_values)
    ax.set_ylim(0, max_total * 1.15)

    ax.grid(axis='y', linestyle='--', alpha=GRID_ALPHA)
    # 图例超大字体
    ax.legend(loc='upper left', fontsize=14, framealpha=0.9, prop={'weight':'bold'})
    _apply_spine(ax)

    save_path = os.path.join(output_dir, save_name)
    _save_and_close(fig, save_path, "堆叠柱状图")
            
# =============================================================================
# 场强沿距离演化曲线
# =============================================================================

def plot_field_evolution(
    df: pd.DataFrame,
    scenario_name: str,
    output_dir: str,
    save_name: str = None
):
    """
    绘制场强沿距离的演化曲线（综合场强实线，分场强全非实线区分）

    Args:
        df: 风险场强结果DataFrame
        scenario_name: 场景名称
        output_dir: 输出目录
        save_name: 保存文件名（默认为 Fs_{scenario_name}_evolution.png）
    """
    set_paper_style()

    if save_name is None:
        save_name = f"Fs_{scenario_name}_evolution.png"

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8),
                                   sharex=True, height_ratios=[2, 1])

    # -------------------------- 仅修改此区域开始 --------------------------
    # 道路几何：长虚线
    ax1.plot(df['距离 (m)'], df['s_geo_norm'],
            label='道路几何', linewidth=LINE_WIDTH,
            color=RISK_COLORS['道路几何'], linestyle='--')
    # 道路设施：点划线
    ax1.plot(df['距离 (m)'], df['s_sign_norm'],
            label='道路设施', linewidth=LINE_WIDTH,
            color=RISK_COLORS['道路设施'], linestyle='-.')
    # 车辆交互：短点线
    ax1.plot(df['距离 (m)'], df['s_veh_norm'],
            label='车辆交互', linewidth=LINE_WIDTH,
            color=RISK_COLORS['车辆交互'], linestyle=':')
    # -------------------------- 仅修改此区域结束 --------------------------

    ax1.set_ylabel('归一化场强', fontsize=14, weight='bold')
    ax1.set_ylim(-0.05, 1.05)
    ax1.grid(True, linestyle='--', alpha=GRID_ALPHA)
    ax1.legend(loc='upper right', fontsize=11, framealpha=0.9)
    ax1.set_title(f"{scenario_name} 风险场强沿距离演化",
                 fontsize=15, pad=10, weight='bold')
    sns.despine(ax=ax1)

    # 综合场强保持原实线+加粗样式，与分场强形成明确层级
    ax2.fill_between(df['距离 (m)'], df['F_S'],
                    color=RISK_COLORS['综合场强'], alpha=0.3, label='综合场强')
    ax2.plot(df['距离 (m)'], df['F_S'],
            linewidth=LINE_WIDTH + 0.5, color=RISK_COLORS['综合场强'], linestyle='-')

    ax2.axhspan(0,   0.3, alpha=0.1, color=ACCENT_COLOR,    label='低')
    ax2.axhspan(0.3, 0.5, alpha=0.1, color='#F5C518',       label='中')
    ax2.axhspan(0.5, 0.7, alpha=0.1, color=SECONDARY_COLOR, label='中高')
    ax2.axhspan(0.7, 1.0, alpha=0.1, color='#E05C5C',       label='高')

    ax2.set_xlabel('距离 (m)', fontsize=14, weight='bold')
    ax2.set_ylabel('综合场强 $F_S$', fontsize=14, weight='bold')
    ax2.set_ylim(-0.05, 1.05)
    ax2.grid(True, linestyle='--', alpha=GRID_ALPHA)
    ax2.legend(loc='upper right', fontsize=10, ncol=5, framealpha=0.9)
    sns.despine(ax=ax2)

    save_path = os.path.join(output_dir, save_name)
    plt.tight_layout()
    _save_and_close(fig, save_path, "演化曲线")
    

def plot_Fs_three_scenarios_evolution(
    results_dict: Dict[str, pd.DataFrame],
    output_dir: str,
    save_name: str = "Fs_three_scenarios_evolution.png",
    sigma: float = 6.0  # 核心：大幅增强平滑力度，从2.0→6.0
):
    """
    绘制3个施工区场强演化对比图（竖向3行1列）
    - 综合场强实线，三分量不同线型区分
    - 全图字体整体放大加粗，适配论文大图排版
    """
    set_paper_style()

    # 画布略微放大，适配超大字体
    fig, axes = plt.subplots(3, 1, figsize=(15, 15), sharex=False, sharey=True)

    scenario_map = {
        "work_zone_1": "施工区1",
        "work_zone_2": "施工区2",
        "work_zone_3": "施工区3"
    }

    # 冷色调三分量 + 暖色调综合场强
    colors = {
        "s_geo":  "#4A90E2",    # 冷蓝 - 道路几何
        "s_sign": "#5BC0EB",    # 青蓝 - 道路设施
        "s_veh":  "#81B0FF",    # 淡蓝 - 车辆交互
        "F_S":    "#E57373"     # 暖红 - 综合场强
    }

    scenario_keys = ["work_zone_1", "work_zone_2", "work_zone_3"]

    for ax, scn in zip(axes.flat, scenario_keys):
        df = results_dict[scn]
        x = df["距离 (m)"]

        # 超强高斯平滑
        s_geo_smooth  = gaussian_filter1d(df["s_geo_norm"],  sigma=sigma, mode="nearest")
        s_sign_smooth = gaussian_filter1d(df["s_sign_norm"], sigma=sigma, mode="nearest")
        s_veh_smooth  = gaussian_filter1d(df["s_veh_norm"],  sigma=sigma, mode="nearest")
        f_s_smooth    = df["F_S"]

        # 线型区分：综合实线，其余各异
        ax.plot(x, s_geo_smooth,  color=colors["s_geo"],  linewidth=2.0, 
                label="道路几何", alpha=1.0, linestyle='--')
        ax.plot(x, s_sign_smooth, color=colors["s_sign"], linewidth=2.0, 
                label="道路设施", alpha=1.0, linestyle='-.')
        ax.plot(x, s_veh_smooth,  color=colors["s_veh"],  linewidth=2.0, 
                label="车辆交互", alpha=1.0, linestyle=':')
        ax.plot(x, f_s_smooth, color=colors["F_S"], linewidth=2.5, 
                label="综合场强", linestyle='-')

        # 综合场强阴影
        ax.fill_between(x, 0, f_s_smooth, color=colors["F_S"], alpha=0.18)

        # ========== 全部字体放大加粗 ==========
        ax.set_title(scenario_map[scn], fontsize=22, weight='bold', pad=22)
        ax.tick_params(axis='both', labelsize=20, width=1.2)
        ax.set_ylim(0, 1.02)
        ax.grid(True, linestyle='--', alpha=0.3)
        
        # 图例字体加大
        ax.legend(loc="upper left", prop={'weight':'bold', 'size': 18}, framealpha=0.9, handlelength=3)

    # 坐标轴标签字体放大
    axes[0].set_ylabel("归一化场强", fontsize=21, weight='bold')
    axes[1].set_ylabel("归一化场强", fontsize=21, weight='bold')
    axes[2].set_ylabel("归一化场强", fontsize=21, weight='bold')
    axes[2].set_xlabel("距离 (m)", fontsize=21, weight='bold')

    plt.tight_layout()
    save_path = os.path.join(output_dir, save_name)
    _save_and_close(fig, save_path, "三场景场强演化曲线")
    

def plot_Fs_distribution(df: pd.DataFrame, scenario_name: str, output_dir: str):
    """
    绘制综合风险场强 F_S 的数值分布直方图 + KDE 核密度曲线
    统一遵循论文绘图风格：配色、字体、边框、保存逻辑
    """
    # 调用全局统一绘图风格
    set_paper_style()
    
    # 提取数据并计算统计量
    fs_vals = df['F_S'].dropna()
    mean_fs = fs_vals.mean()
    median_fs = fs_vals.median()

    # 创建画布（统一尺寸）
    fig, ax = plt.subplots(figsize=(10, 6))
    
    # 绘制直方图+核密度曲线（使用统一的综合场强配色）
    sns.histplot(
        fs_vals, bins=30, kde=True, 
        color=RISK_COLORS['综合场强'],  # 统一配色：综合场强红
        alpha=0.6, edgecolor='white',  # 统一边缘白色
        ax=ax
    )

    # 标注统计量（统一线型、颜色、线宽）
    ax.axvline(
        mean_fs, color='black', linestyle='-', 
        linewidth=1.5, label=f'均值 = {mean_fs:.3f}'
    )
    ax.axvline(
        median_fs, color=SECONDARY_COLOR, linestyle='--', 
        linewidth=1.5, label=f'中位数 = {median_fs:.3f}'
    )

    # 统一坐标轴、标题样式（加粗、字号对齐）
    ax.set_title(f'{scenario_name} 综合风险场强 $F_S$ 分布', 
                 fontsize=15, pad=10, weight='bold')
    ax.set_xlabel('综合风险场强 $F_S$', fontsize=14, weight='bold')
    ax.set_ylabel('频次 / 概率密度', fontsize=14, weight='bold')
    
    # 统一范围、网格、边框
    ax.set_xlim(0, 1)
    ax.grid(True, linestyle='--', alpha=GRID_ALPHA)
    _apply_spine(ax)  # 统一边框样式
    ax.legend(fontsize=11, framealpha=0.9)

    # 统一保存逻辑（替换原生plt.savefig）
    save_path = os.path.join(output_dir, f'Fs_distribution_{scenario_name}.png')
    _save_and_close(fig, save_path, "F_S分布直方图")
    
# =============================================================================
# risk evaluation 评估结果可视化
# =============================================================================

# 阈值敏感性 F1 曲线
def plot_threshold_f1(all_windows: pd.DataFrame, cfg: dict, fig_dir: str) -> None:
    """
    Figure 4.5：遍历阈值 θ，计算各阈值下高风险识别 F1，标注最优点。
    【修复】匹配主脚本真实标签列名 + 读取配置文件 + 适配 R∈[0,1]
    真实标签：abnormal_event（主脚本合并的绩效标签），无则用 R>0 作为代理标签
    """
    set_paper_style()

    # 从配置文件读取阈值搜索参数（主脚本配置化）
    ts_cfg = cfg['threshold_search']
    lo = 0.0          # R全局归一化固定为0起始
    hi = 1.0          # R全局归一化固定为1结束
    step = ts_cfg['step']  # 从yaml配置读取步长

    # 匹配主脚本真实异常标签列名：abnormal_event（核心修复）
    y = (all_windows['abnormal_event'].to_numpy(dtype=int)
          if 'abnormal_event' in all_windows.columns
          else (all_windows['R'] > 0).astype(int).to_numpy())
    
    r = all_windows['R'].to_numpy()
    thresholds = np.arange(lo, hi + step / 2, step)
    f1s = [f1_score(y, (r >= t).astype(int), zero_division=0)
              for t in thresholds]

    # 查找最优F1阈值
    best_i = int(np.argmax(f1s))
    best_theta = thresholds[best_i]
    best_f1 = f1s[best_i]

    # 绘图
    fig, ax = plt.subplots(figsize=(10, 4.5))
    ax.plot(thresholds, f1s, 'o-', color=PRIMARY_COLOR, linewidth=LINE_WIDTH, markersize=7)
    ax.scatter([best_theta], [best_f1], color=SECONDARY_COLOR, s=100, zorder=5,
               label=f'最优 θ={best_theta:.2f}，F1={best_f1:.2f}')
    ax.axvline(best_theta, color=SECONDARY_COLOR, linestyle='--', linewidth=1, alpha=0.5)
    ax.set_xlabel('阈值 θ')
    ax.set_ylabel('F1 值')
    # ax.set_title('图4.5  不同阈值下高风险识别F1值变化曲线')
    ax.set_xticks(np.round(thresholds, 2))
    ax.legend()
    ax.grid(True, linestyle='--', alpha=GRID_ALPHA)
    _apply_spine(ax)
    plt.tight_layout()
    _savefig(fig, os.path.join(fig_dir, 'risk_eval_threshold_f1.png'), cfg['vis']['dpi'])
    plt.close()


# R 整体分布直方图
def plot_r_histogram(all_windows: pd.DataFrame, cfg: dict, fig_dir: str, best_theta: tuple, suffix: str = "") -> None:
    """
    Figure 4.6：全体样本 R 分布直方图，标注均值与最优风险阈值。
    :param suffix: 文件名后缀，用于区分不同数据集（如 _non_baseline）
    """
    set_paper_style()
    # 全局基础字体大小（所有未单独指定的文字默认使用此大小）
    plt.rcParams.update({'font.size': 14})

    r   = all_windows['R'].to_numpy()
    theta_low, theta_high = best_theta

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.hist(r, bins=60, color=PRIMARY_COLOR, edgecolor='white', alpha=0.85, density=True)
    
    # 阈值线：加粗线条匹配大字体
    ax.axvline(theta_high,      color='red',            linestyle='--', linewidth=1.5,
               label=f'高风险阈值 {theta_high:.3f}')
    ax.axvline(theta_low,      color=SECONDARY_COLOR,  linestyle='--', linewidth=1.5,
               label=f'低风险阈值 {theta_low:.3f}')
    # ax.axvline(r.mean(), color='black',           linestyle='-',  linewidth=1.5,
    #            label=f'均值 {r.mean():.3f}')

    # 核心标签：加粗+放大至16号（学术论文标准核心字号）
    ax.set_xlabel('风险度 R', fontsize=16, fontweight='bold')
    ax.set_ylabel('概率密度', fontsize=16, fontweight='bold')
    # 坐标轴刻度：14号，与全局字体一致
    ax.tick_params(axis='both', labelsize=14, width=1.2)
    
    ax.set_xlim(-0.05, 1.05)
    # 图例：14号，增加边框提升辨识度
    ax.legend(fontsize=14, frameon=True, edgecolor='black')
    ax.grid(True, linestyle='--', alpha=GRID_ALPHA, linewidth=1.0)
    _apply_spine(ax)
    plt.tight_layout()
    
    # 🔥 核心修改：根据 suffix 自动生成文件名
    save_path = os.path.join(fig_dir, f'risk_eval_r_histogram{suffix}.png')
    _savefig(fig, save_path, cfg['vis']['dpi'])


def plot_r_histogram_dual(all_windows: pd.DataFrame, 
                          non_baseline_windows: pd.DataFrame,
                          cfg: dict, 
                          fig_dir: str, 
                          best_theta: tuple) -> None:
    """
    【新增】双子图 R 分布直方图：
    左图 = 全体样本 | 右图 = 非基线路段(field_label≠0)
    横向排列，共用图例，Y轴自适应，所有数值保留2位小数
    """
    set_paper_style()
    plt.rcParams.update({'font.size': 14})
    theta_low, theta_high = best_theta

    # 🔥 取消Y轴共享，实现自适应
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

    # ===================== 左子图：全体样本 R 分布 =====================
    r_all = all_windows['R'].to_numpy()
    ax1.hist(r_all, bins=60, color=PRIMARY_COLOR, edgecolor='white', 
             alpha=0.85, density=True)
    # 阈值线 + 数值保留2位小数
    ax1.axvline(theta_high, color='red', linestyle='--', linewidth=1.5)
    ax1.axvline(theta_low, color=SECONDARY_COLOR, linestyle='--', linewidth=1.5)
    
    ax1.set_xlabel('风险度 R', fontsize=16, fontweight='bold')
    ax1.set_ylabel('概率密度', fontsize=16, fontweight='bold')
    ax1.set_title('全路段 R 分布', fontsize=16, fontweight='bold', pad=10)
    ax1.tick_params(axis='both', labelsize=14, width=1.2)
    ax1.set_xlim(-0.05, 1.05)
    # 坐标轴刻度保留2位小数
    ax1.xaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f'{x:.2f}'))
    ax1.yaxis.set_major_formatter(plt.FuncFormatter(lambda y, _: f'{y:.2f}'))
    ax1.grid(True, linestyle='--', alpha=GRID_ALPHA, linewidth=1.0)
    _apply_spine(ax1)

    # ===================== 右子图：非基线路段 R 分布 =====================
    r_non_base = non_baseline_windows['R'].to_numpy()
    ax2.hist(r_non_base, bins=60, color=PRIMARY_COLOR, edgecolor='white', 
             alpha=0.85, density=True)
    # 阈值线 + 数值保留2位小数（共用图例）
    ax2.axvline(theta_high, color='red', linestyle='--', linewidth=1.5,
                label=f'高风险阈值 {theta_high:.2f}')
    ax2.axvline(theta_low, color=SECONDARY_COLOR, linestyle='--', linewidth=1.5,
                label=f'低风险阈值 {theta_low:.2f}')
    
    ax2.set_xlabel('风险度 R', fontsize=16, fontweight='bold')
    ax2.set_title('施工区路段 R 分布', fontsize=16, fontweight='bold', pad=10)
    ax2.tick_params(axis='both', labelsize=14, width=1.2)
    ax2.set_xlim(-0.05, 1.05)
    # 坐标轴刻度保留2位小数
    ax2.xaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f'{x:.2f}'))
    ax2.yaxis.set_major_formatter(plt.FuncFormatter(lambda y, _: f'{y:.2f}'))
    ax2.grid(True, linestyle='--', alpha=GRID_ALPHA, linewidth=1.0)
    _apply_spine(ax2)

    # ===================== 共用图例 =====================
    ax2.legend(fontsize=14, frameon=True, edgecolor='black', loc='upper right')

    # 整体布局
    plt.tight_layout()
    save_path = os.path.join(fig_dir, 'risk_eval_r_histogram_dual.png')
    _savefig(fig, save_path, cfg['vis']['dpi'])
    print("  双图 R 分布直方图 → risk_eval_r_histogram_dual.png")


# 三组驾驶人 R 小提琴图
def plot_violin_by_group(all_windows: pd.DataFrame, cfg: dict, fig_dir: str, best_theta: tuple) -> None:
    """Figure 4.7：三组驾驶人 R 分布小提琴图，各组着不同色。"""
    set_paper_style()
    # 与直方图保持完全一致的字体体系
    plt.rcParams.update({'font.size': 14})

    groups = ['高能力组', '中能力组', '低能力组']
    gc     = cfg['vis']['group_colors']
    theta_low, theta_high = best_theta

    data = [all_windows[all_windows['group'] == g]['R'].to_numpy()
            for g in groups]

    fig, ax = plt.subplots(figsize=(8, 5.5))
    parts   = ax.violinplot(data, positions=[1, 2, 3],
                             showmedians=True, showextrema=True)
    for pc, g in zip(parts['bodies'], groups):
        pc.set_facecolor(gc[g])
        pc.set_alpha(0.7)
    # 小提琴图统计线：加粗匹配大字体
    for comp in ['cmedians', 'cmaxes', 'cmins', 'cbars']:
        parts[comp].set_color('black')
        parts[comp].set_linewidth(1.5)

    # 阈值线
    ax.axhline(theta_high, color='red',           linestyle='--', linewidth=1.2, alpha=0.7, label=f'高风险阈值 {theta_high:.3f}')
    ax.axhline(theta_low, color=SECONDARY_COLOR, linestyle='--', linewidth=1.2, alpha=0.7, label=f'低风险阈值 {theta_low:.3f}')
    
    ax.set_xticks([1, 2, 3])
    # 分组标签：加粗+14号
    ax.set_xticklabels(groups, fontsize=14, fontweight='bold')
    ax.set_ylabel('风险度 R', fontsize=16, fontweight='bold')
    ax.tick_params(axis='y', labelsize=14, width=1.2)
    
    ax.set_ylim(-0.05, 1.05)
    ax.legend(fontsize=14, frameon=True, edgecolor='black')
    ax.grid(axis='y', linestyle='--', alpha=GRID_ALPHA, linewidth=1.0)
    _apply_spine(ax)
    plt.tight_layout()
    _savefig(fig, os.path.join(fig_dir, 'risk_eval_violin_by_group.png'), cfg['vis']['dpi'])

# 场景×能力组 折线图
def plot_line_scenario_group(table_df: pd.DataFrame, cfg: dict, fig_dir: str) -> None:
    """
    Figure 4.8：三组驾驶人在四个场景下 R 均值折线图。
    table_df 由 compute_scenario_group_table() 返回。
    """
    set_paper_style()

    groups = ['高能力组', '中能力组', '低能力组']
    gc     = cfg['vis']['group_colors']
    scenes = table_df['场景'].tolist()

    fig, ax = plt.subplots(figsize=(8, 5))
    for g in groups:
        if g not in table_df.columns:
            continue
        ax.plot(scenes, table_df[g].tolist(), 'o-',
                color=gc[g], linewidth=LINE_WIDTH, markersize=7, label=g)

    ax.set_ylim(-0.05, 1.05)
    ax.set_xlabel('场景')
    ax.set_ylabel('风险度 R 均值')
    ax.set_title('图4.8  三组驾驶人在四个场景下风险度均值变化折线图')
    ax.legend()
    ax.grid(True, linestyle='--', alpha=GRID_ALPHA)
    _apply_spine(ax)
    plt.tight_layout()
    _savefig(fig, os.path.join(fig_dir, 'risk_eval_line_scenario_group.png'), cfg['vis']['dpi'])


# 场景×能力组 箱线图
def plot_box_scenario_group(all_windows: pd.DataFrame, cfg: dict, fig_dir: str) -> None:
    """Figure 4.9：三组驾驶人在四个场景下 R 分组箱线图。"""
    set_paper_style()

    groups  = ['高能力组', '中能力组', '低能力组']
    gc      = cfg['vis']['group_colors']
    lname   = {int(k): v for k, v in cfg['scenarios']['label_name'].items()}
    labels  = sorted(all_windows['field_label'].unique())
    scenes  = [lname.get(l, str(l)) for l in labels]
    width   = 0.22
    x       = np.arange(len(scenes))

    fig, ax = plt.subplots(figsize=(10, 5.5))
    for gi, g in enumerate(groups):
        offset = (gi - 1) * width
        data   = [all_windows[(all_windows['field_label'] == lbl) &
                               (all_windows['group'] == g)]['R'].to_numpy()
                  for lbl in labels]
        bp = ax.boxplot(data, positions=x + offset, widths=width * 0.85,
                        patch_artist=True,
                        medianprops={'color': 'black', 'linewidth': 1.5},
                        whiskerprops={'linewidth': 1.2},
                        capprops={'linewidth': 1.2},
                        flierprops={'marker': 'o', 'markersize': 2.5,
                                    'alpha': 0.4, 'color': gc[g]})
        for patch in bp['boxes']:
            patch.set_facecolor(gc[g])
            patch.set_alpha(0.65)

    ax.set_ylim(-0.05, 1.05)
    ax.set_xticks(x)
    ax.set_xticklabels(scenes)
    ax.set_ylabel('风险度 R')
    ax.set_title('图4.9  三组驾驶人在四个场景下风险度分布箱线图')
    handles = [mpatches.Patch(facecolor=gc[g], alpha=0.7, label=g) for g in groups]
    ax.legend(handles=handles)
    ax.grid(axis='y', linestyle='--', alpha=GRID_ALPHA)
    _apply_spine(ax)
    plt.tight_layout()
    _savefig(fig, os.path.join(fig_dir, 'risk_eval_box_scenario_group.png'), cfg['vis']['dpi'])


# 风险等级堆叠柱状图
def plot_stacked_bar_risk(all_windows: pd.DataFrame, cfg: dict, fig_dir: str) -> None:
    """Figure 4.10：三组驾驶人风险等级分布堆叠柱状图"""
    set_paper_style()

    groups = ['高能力组', '中能力组', '低能力组']
    levels = ['低风险', '中风险', '高风险']
    lc     = cfg['vis']['risk_level_colors']

    # 匹配上方柱状图的填充样式：不同方向斜线 + 白底彩边
    hatch_patterns = ['///', '||', '\\\\\\']  # 低/中/高 对应三种斜线

    pcts = {g: [] for g in groups}
    for g in groups:
        sub   = all_windows[all_windows['group'] == g]
        total = max(len(sub), 1)
        for lvl in levels:
            pcts[g].append((sub['risk_level_optimized'] == lvl).sum() / total * 100)

    x       = np.arange(len(groups))
    fig, ax = plt.subplots(figsize=(7, 5))
    bottoms = np.zeros(len(groups))

    # 柱子宽度保持
    bar_width = 0.55

    for idx, lvl in enumerate(levels):
        vals = [pcts[g][idx] for g in groups]
        # ===================== 核心修改：统一为白底彩边+斜线填充 =====================
        bars = ax.bar(x, vals, width=bar_width, bottom=bottoms,
                      color='white',                # 白底
                      edgecolor=lc[lvl],            # 彩色边框
                      linewidth=1.5,                # 边框加粗
                      hatch=hatch_patterns[idx],    # 区分斜线填充
                      label=lvl,
                      alpha=1)
        # ==========================================================================
        # 百分比文字：黑色加粗，保持大字体
        for rect, v in zip(bars, vals):
            if v > 3:
                ax.text(rect.get_x() + rect.get_width() / 2,
                        rect.get_y() + rect.get_height() / 2,
                        f'{v:.1f}%', ha='center', va='center',
                        fontsize=14, color='black', fontweight='bold')
        bottoms += np.array(vals)

    # ===================== 字体统一放大加粗 =====================
    ax.set_xticks(x)
    ax.set_xticklabels(groups, fontsize=15, weight='bold')  # X轴标签放大
    ax.set_ylabel('占比 (%)', fontsize=16, weight='bold')    # Y轴标签放大
    ax.tick_params(axis='y', labelsize=14, width=1.2)        # Y刻度放大
    ax.set_ylim(0, 105)
    
    # 图例字体放大加粗
    ax.legend(loc='upper left', bbox_to_anchor=(1, 1),
              prop={'weight':'bold', 'size':14}, framealpha=0.9)
    
    ax.grid(axis='y', linestyle='--', alpha=GRID_ALPHA)
    _apply_spine(ax)
    plt.tight_layout()
    _savefig(fig, os.path.join(fig_dir, 'risk_eval_stacked_bar.png'), cfg['vis']['dpi'])

# 典型驾驶人时序曲线（底部横向图例+双图Y轴固定 最终版）
def plot_timeseries_typical(all_windows: pd.DataFrame,
                             sample_field: List[np.ndarray],
                             sample_ad_norm: List[np.ndarray],
                             fs_temporal_list: List[np.ndarray],
                             cap_groups: Dict[int, str],
                             cfg: dict,
                             fig_dir: str,
                             best_theta: tuple) -> None:
    set_paper_style()

    theta_low, theta_high = best_theta
    lname = {int(k): v for k, v in cfg['scenarios']['label_name'].items()}

    def _pick_rep(group_name: str) -> int:
        idxs = [i for i, g in cap_groups.items() if g == group_name]
        if not idxs:
            return list(cap_groups.keys())[0]
        g_mean  = all_windows[all_windows['group'] == group_name]['R'].mean()
        s_means = {i: all_windows[all_windows['sample_idx'] == i]['R'].mean()
                   for i in idxs
                   if len(all_windows['sample_idx'] == i) > 0}
        return min(s_means, key=lambda i: abs(s_means[i] - g_mean))

    reps = [('高能力组代表', _pick_rep('高能力组')),
            ('低能力组代表', _pick_rep('低能力组'))]

    fig, axes = plt.subplots(2, 1, figsize=(16, 10), sharex=False)

    # 场景配色：0=基线无底色，1/2/3=施工区
    scene_colors = {
        1: "#2F5597",   # 施工区1
        2: "#ED7D31",   # 施工区2
        3: "#548235"    # 施工区3
    }

    for row_idx, (title, sidx) in enumerate(reps):
        ax = axes[row_idx]
        fld = np.asarray(sample_field[sidx])
        r = all_windows[all_windows['sample_idx'] == sidx]['R'].to_numpy()
        x = np.arange(len(r))

        # 仅施工区上色，基线0无底色
        i = 0
        while i < len(fld):
            lbl = int(fld[i])
            j = i + 1
            while j < len(fld) and int(fld[j]) == lbl:
                j += 1
            if lbl in scene_colors:
                ax.axvspan(i - 0.5, j - 0.5, facecolor=scene_colors[lbl], alpha=0.35, zorder=0)
            i = j

        # R线统一红色
        ax.plot(x, r, color='red', linewidth=LINE_WIDTH + 0.5, drawstyle='steps-post')
        # 阈值虚线
        ax.axhline(theta_high, color='red', linestyle='--', linewidth=1.2, alpha=0.9)
        ax.axhline(theta_low, color=SECONDARY_COLOR, linestyle='--', linewidth=1.2, alpha=0.9)

        # ============== 核心修改1：双图Y轴强制固定（不自适应） ==============
        ax.set_ylim(0, 1.05)
        
        # 字体设置
        ax.set_ylabel('风险度 R', fontsize=22, fontweight='bold')
        ax.set_title(f'{title}（样本 {sidx}，R 均值={r.mean():.2f}）', fontsize=24, fontweight='bold')
        ax.tick_params(axis='both', labelsize=20)
        ax.grid(axis='y', linestyle='--', alpha=GRID_ALPHA)
        _apply_spine(ax)
        if row_idx == len(reps) - 1:
            ax.set_xlabel('窗口序号', fontsize=22, fontweight='bold')

    # 构建图例
    legend_handles = []
    legend_handles.append(plt.Line2D([], [], color='red', lw=LINE_WIDTH + 0.5, label='风险度 R'))
    legend_handles.append(plt.Line2D([], [], color=SECONDARY_COLOR, linestyle='--', lw=1.2, label=f'低风险阈值'))
    legend_handles.append(plt.Line2D([], [], color='red', linestyle='--', lw=1.2, label=f'高风险阈值'))
    for lbl in sorted(scene_colors.keys()):
        legend_handles.append(mpatches.Patch(facecolor=scene_colors[lbl], alpha=0.35, label=lname.get(lbl, f'施工区{lbl}')))

    # ============== 核心修改2：图例横向放在底部 ==============
    fig.legend(
        handles=legend_handles,
        loc='lower center',    # 底部居中
        bbox_to_anchor=(0.5, 0.01),
        ncol=6,                # 横向排列
        fontsize=18,
        frameon=True
    )

    # ============== 核心修改3：给底部图例留空间 ==============
    plt.tight_layout(rect=[0, 0.08, 1, 1])
    
    _savefig(fig, os.path.join(fig_dir, 'risk_eval_timeseries_typical.png'), cfg['vis']['dpi'])

# 单样本时序图（调试 / --plot_sample 用）
def plot_single_sample(sample_idx: int,
                        sample_field: List[np.ndarray],
                        fs_temporal_list: List[np.ndarray],
                        ad_norm_list: List[np.ndarray],
                        all_windows: pd.DataFrame,
                        cfg: dict,
                        fig_dir: str,
                        best_theta: tuple) -> None:
    """
    单个样本全程 F_S / Ã_d / R 三子图时序可视化。

    Args:
        fs_temporal_list: 各样本空间→时间映射后的 F_S 列表
        ad_norm_list:     各样本归一化 Ã_d 列表
    """
    set_paper_style()

    theta_low, theta_high = best_theta
    fsbl = cfg['model']['fs_baseline']

    fld    = np.asarray(sample_field[sample_idx])
    fs_arr = fs_temporal_list[sample_idx]
    adn    = ad_norm_list[sample_idx]
    r = all_windows[all_windows['sample_idx'] == sample_idx]['R'].to_numpy()
    x      = np.arange(len(fld))

    fig, axes = plt.subplots(3, 1, figsize=(14, 9), sharex=True)
    fig.suptitle(f'样本 {sample_idx} — 全程时序概览', fontsize=15, fontweight='bold')

    ax = axes[0]
    _fill_scene_bg(ax, fld, cfg)
    ax.plot(x, fs_arr, color=PRIMARY_COLOR, linewidth=LINE_WIDTH, label='F_S')
    ax.axhline(fsbl, color='gray', linestyle=':', linewidth=0.8,
               label=f'基线 {fsbl}')
    ax.set_ylabel('F_S（任务场强）')
    ax.set_ylim(-0.05, 1.05)
    ax.legend(loc='upper right', fontsize=8)
    ax.grid(axis='y', linestyle='--', alpha=GRID_ALPHA)
    _apply_spine(ax)

    ax = axes[1]
    _fill_scene_bg(ax, fld, cfg)
    ax.plot(x, adn, color=ACCENT_COLOR, linewidth=LINE_WIDTH, label='Ã_d（归一化能力）')
    ax.set_ylabel('Ã_d')
    ax.set_ylim(-0.05, 1.05)
    ax.legend(loc='upper right', fontsize=8)
    ax.grid(axis='y', linestyle='--', alpha=GRID_ALPHA)
    _apply_spine(ax)

    ax = axes[2]
    _fill_scene_bg(ax, fld, cfg)
    ax.plot(x, r, color=SECONDARY_COLOR, linewidth=LINE_WIDTH, label='R（风险度）')
    ax.axhline(theta_high, color='red',           linestyle='--', linewidth=1.0,
               label=f'高风险 {theta_high}')
    ax.axhline(theta_low, color=SECONDARY_COLOR, linestyle='--', linewidth=1.0,
               label=f'低风险 {theta_low}')
    ax.fill_between(x,  theta_high,  1.0,  alpha=0.08, color='red')
    ax.set_ylabel('R（风险度）')
    ax.set_xlabel('窗口序号')
    ax.set_ylim(-0.05, 1.05)
    ax.legend(loc='upper right', fontsize=8)
    ax.grid(axis='y', linestyle='--', alpha=GRID_ALPHA)
    _apply_spine(ax)

    bg    = _scene_bg_colors(cfg)
    lname = {int(k): v for k, v in cfg['scenarios']['label_name'].items()}
    patches = [mpatches.Patch(facecolor=bg[l], alpha=0.5,
                               label=lname.get(l, str(l)))
               for l in sorted(set(fld.astype(int))) if l in bg]
    if patches:
        fig.legend(handles=patches, loc='lower center', ncol=len(patches),
                   fontsize=8, bbox_to_anchor=(0.5, -0.01))

    plt.tight_layout(rect=[0, 0.04, 1, 1])
    _savefig(fig, os.path.join(fig_dir, f'sample_{sample_idx:03d}_timeseries.png'),
             cfg['vis']['dpi'])


def _smooth_and_resample(arr: np.ndarray,
                          window: int = 15,
                          poly: int = 3,
                          factor: int = 10) -> Tuple[np.ndarray, np.ndarray]:
    """
    先 Savitzky-Golay 平滑，再三次样条插值升采样。

    Returns
    -------
    t_dense : 升采样后的 x 轴坐标
    y_dense : 升采样后的 y 值
    """
    n = len(arr)
    w = window if window % 2 == 1 else window + 1
    y_sg = savgol_filter(arr, window_length=min(w, n if n % 2 == 1 else n - 1),
                          polyorder=poly) if n >= w else arr

    t_orig  = np.arange(n, dtype=float)
    t_dense = np.linspace(0, n - 1, n * factor)
    y_dense = make_interp_spline(t_orig, y_sg, k=3)(t_dense)
    return t_dense, y_dense


def plot_fs_ad_filled(sample_idx: int,
                      fs_arr: np.ndarray,
                      ad_norm_arr: np.ndarray,
                      cfg: dict,
                      fig_dir: str,
                      smooth_window: int = 35,
                      smooth_poly: int = 3,
                      resample_factor: int = 10,
                      ad_smooth_window: int = 35,
                      ad_smooth_poly: int = 3) -> None:
    set_paper_style()
    assert len(fs_arr) == len(ad_norm_arr), "F_S 与 Ã_d 长度必须一致"

    t_dense, fs_dense = _smooth_and_resample(
        fs_arr,      smooth_window,    smooth_poly,    resample_factor)
    _,       ad_dense = _smooth_and_resample(
        ad_norm_arr, ad_smooth_window, ad_smooth_poly, resample_factor)

    fig, ax = plt.subplots(figsize=FIGURE_SIZE_WIDE)

    ax.plot(t_dense, fs_dense, color=PRIMARY_COLOR,   lw=LINE_WIDTH, label=r'$F_S$', zorder=4)
    ax.plot(t_dense, ad_dense, color=SECONDARY_COLOR, lw=LINE_WIDTH, label=r'$\tilde{A}_d$', zorder=4)

    ax.fill_between(
        t_dense, fs_dense, ad_dense,
        where=(fs_dense >= ad_dense),
        interpolate=True,
        facecolor='none',
        edgecolor='goldenrod',
        hatch='///',
        linewidth=0.0,
        zorder=3,
        alpha=0.85,
    )

    ax.fill_between(
        t_dense, fs_dense, ad_dense,
        where=(fs_dense <= ad_dense),
        interpolate=True,
        facecolor='none',
        edgecolor='seagreen',
        hatch='///',
        linewidth=0.0,
        zorder=3,
        alpha=0.85,
    )

    ax.set_xlim(t_dense[15 * resample_factor], t_dense[-1])
    ax.set_ylim(
        min(fs_dense.min(), ad_dense.min()) - 0.02,
        max(fs_dense.max(), ad_dense.max()) + 0.05,
    )
    ax.set_xlabel('时间t (s)', fontsize=18, weight='bold')
    ax.set_ylabel('归一化水平', fontsize=18, weight='bold')

    legend_handles = [
        plt.Line2D([0], [0], color=PRIMARY_COLOR,   lw=LINE_WIDTH, label=r'任务需求'),
        plt.Line2D([0], [0], color=SECONDARY_COLOR, lw=LINE_WIDTH, label=r'驾驶能力'),
        mpatches.Patch(facecolor='none', edgecolor='goldenrod', hatch='///', label=r'风险状态'),
        mpatches.Patch(facecolor='none', edgecolor='seagreen', hatch='///', label=r'安全状态'),
    ]
    ax.legend(handles=legend_handles, loc='upper right', fontsize=15, framealpha=0.9)
    ax.grid(axis='y', linestyle='--', alpha=GRID_ALPHA)
    ax.tick_params(axis='both', labelsize=15, direction='out', length=4.5, width=1.0)
    _apply_spine(ax)

    plt.tight_layout()
    _savefig(fig, os.path.join(fig_dir, f'fs_ad_filled_sample{sample_idx}.png'), cfg['vis']['dpi'])


# =============================================================================
# 验证模块可视化函数（原 risk_validator.py 中内嵌的绘图逻辑）
# =============================================================================

def plot_roc_curve(
    fpr: np.ndarray,
    tpr: np.ndarray,
    auc: float,
    best_thr: float,
    best_fpr: float,
    best_tpr: float,
    cfg: dict,
    fig_dir: str,
    save_name: str = "fig_roc_curve.png",
) -> None:
    """
    绘制 ROC 曲线，标注最优 Youden 阈值点。

    此函数由 risk_validator.py 的 validate_roc() 调用，
    接收已计算好的 fpr/tpr/auc/best_thr 数组，不重复计算。

    Args:
        fpr:       sklearn roc_curve 返回的假阳性率数组
        tpr:       sklearn roc_curve 返回的真阳性率数组
        auc:       ROC 曲线下面积
        best_thr:  最优 Youden 阈值
        best_fpr:  最优点对应的 FPR
        best_tpr:  最优点对应的 TPR
        cfg:       配置字典（读取 figure 子项；兼容 validator yaml 与 evaluator yaml）
        fig_dir:   图片保存目录
        save_name: 文件名（默认 fig_roc_curve.png）
    """
    set_paper_style()

    # 兼容两种 cfg 结构：validator yaml 使用 cfg['figure']，evaluator yaml 使用 cfg['vis']
    fc = cfg.get('figure', cfg.get('vis', {}))
    figsize = fc.get('figsize_roc', [6, 5])
    dpi     = fc.get('dpi', 150)

    fig, ax = plt.subplots(figsize=figsize)
    ax.plot(fpr, tpr, lw=2, color="steelblue", label=f"ROC (AUC={auc:.3f})")
    ax.plot([0, 1], [0, 1], "k--", lw=1, label="随机猜测")
    ax.scatter(best_fpr, best_tpr, s=80, color="crimson", zorder=5,
               label=f"最优阈值={best_thr:.3f}")
    ax.set_xlabel("假阳性率 (FPR)", fontsize=12)
    ax.set_ylabel("真阳性率 (TPR)", fontsize=12)
    ax.set_title("R 判别异常事件 ROC 曲线", fontsize=13)
    ax.legend(fontsize=10)
    ax.grid(alpha=GRID_ALPHA)
    _apply_spine(ax)
    plt.tight_layout()

    _savefig(fig, os.path.join(fig_dir, save_name), dpi)


def plot_risk_event_rate(
    level_df: pd.DataFrame,
    chi2_str: str,
    cfg: dict,
    fig_dir: str,
    save_name: str = "fig_risk_event_rate.png",
) -> None:
    """
    绘制不同风险等级下的异常事件发生率柱状图，并在标题中标注卡方检验结果。

    此函数由 risk_validator.py 的 validate_risk_level_event_rate() 调用。

    Args:
        level_df:  包含 ['风险等级', '事件率(%)'] 列的 DataFrame
                   （由 validate_risk_level_event_rate 生成）
        chi2_str:  卡方检验描述字符串，例如 "χ²=12.34, df=2, p<0.001"
        cfg:       配置字典
        fig_dir:   图片保存目录
        save_name: 文件名（默认 fig_risk_event_rate.png）
    """
    set_paper_style()

    fc = cfg.get('figure', cfg.get('vis', {}))
    figsize = fc.get('figsize_bar', [6, 4])
    dpi     = fc.get('dpi', 150)
    colors  = [
        fc.get('color_low_risk',  "#4CAF50"),
        fc.get('color_mid_risk',  "#FF9800"),
        fc.get('color_high_risk', "#F44336"),
    ]

    rates  = level_df["事件率(%)"].tolist()
    labels = level_df["风险等级"].tolist()

    fig, ax = plt.subplots(figsize=figsize)
    bars = ax.bar(labels, rates, color=colors[:len(labels)],
                  edgecolor="white", width=0.5)
    for bar, rate in zip(bars, rates):
        ax.text(bar.get_x() + bar.get_width() / 2,
                bar.get_height() + 0.5,
                f"{rate:.1f}%", ha="center", va="bottom", fontsize=11)
    ax.set_ylabel("异常事件发生率 (%)", fontsize=12)
    ax.set_title(f"不同风险等级下异常事件发生率\n{chi2_str}", fontsize=11)
    ax.set_ylim(0, max(rates) * 1.25 + 3)
    ax.grid(axis="y", alpha=GRID_ALPHA)
    _apply_spine(ax)
    plt.tight_layout()

    _savefig(fig, os.path.join(fig_dir, save_name), dpi)


def plot_r_boxplot(
    ev0: np.ndarray,
    ev1: np.ndarray,
    p_str: str,
    cfg: dict,
    fig_dir: str,
    save_name: str = "fig_r_boxplot.png",
) -> None:
    """
    绘制 R 在无异常事件组 vs. 异常事件组之间的箱线图，
    并在标题中标注 Mann-Whitney U 检验 p 值。

    此函数由 risk_validator.py 的 plot_r_by_event() 调用。

    Args:
        ev0:      无异常事件窗口的 R 数组
        ev1:      异常事件窗口的 R 数组
        p_str:    p 值字符串，例如 "<0.001" 或 "0.0032"
        cfg:      配置字典
        fig_dir:  图片保存目录
        save_name: 文件名（默认 fig_r_boxplot.png）
    """
    set_paper_style()

    fc = cfg.get('figure', cfg.get('vis', {}))
    figsize        = fc.get('figsize_box', [5, 5])
    dpi            = fc.get('dpi', 150)
    color_no_event = fc.get('color_no_event', "#90CAF9")
    color_event    = fc.get('color_event',    "#EF9A9A")

    fig, ax = plt.subplots(figsize=figsize)
    bp = ax.boxplot([ev0, ev1], labels=["无异常事件", "异常事件"],
                    patch_artist=True, widths=0.4,
                    medianprops={"color": "black", "lw": 2})
    bp["boxes"][0].set_facecolor(color_no_event)
    bp["boxes"][1].set_facecolor(color_event)
    ax.set_ylabel("风险度 R", fontsize=12)
    ax.set_ylim(-0.05, 1.05)
    ax.set_title(f"R 分布（事件 vs. 无事件）\nMann-Whitney U, p={p_str}",
                 fontsize=12)
    ax.grid(axis="y", alpha=GRID_ALPHA)
    _apply_spine(ax)
    plt.tight_layout()

    _savefig(fig, os.path.join(fig_dir, save_name), dpi)

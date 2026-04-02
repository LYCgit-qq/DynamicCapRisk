# D:\Local\DynamicCapRisk\src\visualization\plot_risk.py

"""
plot_risk_results.py
风险度评估可视化模块
生成论文第4章所需全部图表（Fig 4.5 ~ Fig 4.11 及单样本时序图）
同时包含风险验证模块（risk_validator.py）的可视化函数：
  - plot_roc_curve          ROC 曲线（原 validate_roc 内嵌绘图）
  - plot_risk_event_rate    风险等级 × 事件发生率柱状图（原 validate_risk_level_event_rate 内嵌绘图）
  - plot_r_star_boxplot     R* 事件/无事件分布箱线图（原 plot_r_star_by_event 内嵌绘图）

所有函数签名统一：
    plot_xxx(data, cfg, fig_dir)
    data  —— 计算结果（all_windows DataFrame 或其他数据结构）
    cfg   —— 从 risk_evaluator.yaml / risk_validator.yaml 加载的配置字典
    fig_dir —— 图片输出目录
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
    set_paper_style()

    geo_values  = [data[s]['s_geo_mean']  * w_geo  for s in scenarios]
    sign_values = [data[s]['s_sign_mean'] * w_sign for s in scenarios]
    veh_values  = [data[s]['s_veh_mean']  * w_veh  for s in scenarios]

    fig, ax = plt.subplots(figsize=FIGURE_SIZE_WIDE)

    x     = np.arange(len(scenarios))
    width = 0.6

    ax.bar(x, geo_values, width, label=f'道路几何 (w={w_geo})',
           color=RISK_COLORS['道路几何'], edgecolor='white', linewidth=1.2)
    ax.bar(x, sign_values, width, bottom=geo_values,
           label=f'道路设施 (w={w_sign})',
           color=RISK_COLORS['道路设施'], edgecolor='white', linewidth=1.2)
    ax.bar(x, veh_values, width,
           bottom=[i + j for i, j in zip(geo_values, sign_values)],
           label=f'车辆交互 (w={w_veh})',
           color=RISK_COLORS['车辆交互'], edgecolor='white', linewidth=1.2)

    for idx, scenario in enumerate(scenarios):
        total = data[scenario]['F_S_mean']
        ax.text(idx, total + 0.03, f'{total:.2f}',
               ha='center', va='bottom', fontsize=11, weight='bold')

    ax.set_xlabel('场景', fontsize=14, weight='bold')
    ax.set_ylabel('场强贡献值', fontsize=14, weight='bold')
    ax.set_title("风险场强子项贡献堆叠柱状图", fontsize=15, pad=15, weight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(scenarios, fontsize=12)
    ax.set_ylim(0, 1.1)
    ax.grid(axis='y', linestyle='--', alpha=GRID_ALPHA)
    ax.legend(loc='upper left', fontsize=11, framealpha=0.9)
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
    绘制场强沿距离的演化曲线

    Args:
        df: 风险场强结果DataFrame
        scenario_name: 场景名称
        output_dir: 输出目录
        save_name: 保存文件名（默认为 {scenario_name}_evolution.png）
    """
    set_paper_style()

    if save_name is None:
        save_name = f"{scenario_name}_evolution.png"

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8),
                                   sharex=True, height_ratios=[2, 1])

    ax1.plot(df['距离 (m)'], df['s_geo_norm'],
            label='道路几何', linewidth=LINE_WIDTH,
            color=RISK_COLORS['道路几何'])
    ax1.plot(df['距离 (m)'], df['s_sign_norm'],
            label='道路设施', linewidth=LINE_WIDTH,
            color=RISK_COLORS['道路设施'])
    ax1.plot(df['距离 (m)'], df['s_veh_norm'],
            label='车辆交互', linewidth=LINE_WIDTH,
            color=RISK_COLORS['车辆交互'])

    ax1.set_ylabel('归一化场强', fontsize=14, weight='bold')
    ax1.set_ylim(0, 1.05)
    ax1.grid(True, linestyle='--', alpha=GRID_ALPHA)
    ax1.legend(loc='upper right', fontsize=11, framealpha=0.9)
    ax1.set_title(f"{scenario_name} 风险场强沿距离演化",
                 fontsize=15, pad=10, weight='bold')
    sns.despine(ax=ax1)

    ax2.fill_between(df['距离 (m)'], df['F_S'],
                    color=RISK_COLORS['综合场强'], alpha=0.3, label='综合场强')
    ax2.plot(df['距离 (m)'], df['F_S'],
            linewidth=LINE_WIDTH + 0.5, color=RISK_COLORS['综合场强'])

    ax2.axhspan(0,   0.3, alpha=0.1, color=ACCENT_COLOR,    label='低')
    ax2.axhspan(0.3, 0.5, alpha=0.1, color='#F5C518',       label='中')
    ax2.axhspan(0.5, 0.7, alpha=0.1, color=SECONDARY_COLOR, label='中高')
    ax2.axhspan(0.7, 1.0, alpha=0.1, color='#E05C5C',       label='高')

    ax2.set_xlabel('距离 (m)', fontsize=14, weight='bold')
    ax2.set_ylabel('综合场强 $F_S$', fontsize=14, weight='bold')
    ax2.set_ylim(0, 1.05)
    ax2.grid(True, linestyle='--', alpha=GRID_ALPHA)
    ax2.legend(loc='upper right', fontsize=10, ncol=5, framealpha=0.9)
    sns.despine(ax=ax2)

    save_path = os.path.join(output_dir, save_name)
    plt.tight_layout()
    _save_and_close(fig, save_path, "演化曲线")


# =============================================================================
# Figure 4.5  阈值敏感性 F1 曲线
# =============================================================================

def plot_threshold_f1(all_windows: pd.DataFrame, cfg: dict, fig_dir: str) -> None:
    """
    Figure 4.5：遍历阈值 θ，计算各阈值下高风险识别 F1，标注最优点。
    若无真实事件标签（event_label 列），以 R*>0 作为代理标签。
    """
    from sklearn.metrics import f1_score
    set_paper_style()

    ts_cfg = cfg['threshold_search']
    lo, hi, step = ts_cfg['range'][0], ts_cfg['range'][1], ts_cfg['step']
    thresholds   = np.arange(lo, hi + step / 2, step)

    y      = (all_windows['event_label'].to_numpy(dtype=int)
              if 'event_label' in all_windows.columns
              else (all_windows['R_star'] > 0).astype(int).to_numpy())
    r_star = all_windows['R_star'].to_numpy()
    f1s    = [f1_score(y, (r_star >= t).astype(int), zero_division=0)
              for t in thresholds]

    best_i = int(np.argmax(f1s))
    fig, ax = plt.subplots(figsize=(7, 4.5))
    ax.plot(thresholds, f1s, 'o-', color=PRIMARY_COLOR, linewidth=LINE_WIDTH, markersize=7)
    ax.scatter([thresholds[best_i]], [f1s[best_i]], color=SECONDARY_COLOR, s=100, zorder=5,
               label=f'最优 θ={thresholds[best_i]:.2f}，F1={f1s[best_i]:.3f}')
    ax.axvline(thresholds[best_i], color=SECONDARY_COLOR, linestyle='--', linewidth=1, alpha=0.5)
    ax.set_xlabel('阈值 θ')
    ax.set_ylabel('F1 值')
    ax.set_title('图4.5  不同阈值下高风险识别F1值变化曲线')
    ax.set_xticks(np.round(thresholds, 2))
    ax.legend()
    ax.grid(True, linestyle='--', alpha=GRID_ALPHA)
    _apply_spine(ax)
    plt.tight_layout()
    _savefig(fig, os.path.join(fig_dir, 'fig4_5_threshold_f1.png'), cfg['vis']['dpi'])


# =============================================================================
# Figure 4.6  R* 整体分布直方图
# =============================================================================

def plot_r_histogram(all_windows: pd.DataFrame, cfg: dict, fig_dir: str) -> None:
    """Figure 4.6：全体样本 R* 分布直方图，标注均值与高/低风险阈值。"""
    set_paper_style()

    r   = all_windows['R_star'].to_numpy()
    thi = cfg['model']['thresh_high']
    tlo = cfg['model']['thresh_low']

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.hist(r, bins=60, color=PRIMARY_COLOR, edgecolor='white', alpha=0.85, density=True)
    ax.axvline(thi,      color='red',            linestyle='--', linewidth=1.2,
               label=f'高风险阈值 {thi}')
    ax.axvline(tlo,      color=SECONDARY_COLOR,  linestyle='--', linewidth=1.2,
               label=f'低风险阈值 {tlo}')
    ax.axvline(r.mean(), color='black',           linestyle='-',  linewidth=1.2,
               label=f'均值 {r.mean():.3f}')
    ax.set_xlabel('风险度 R*')
    ax.set_ylabel('概率密度')
    ax.set_title('图4.6  全体样本风险度 R* 分布直方图')
    ax.legend()
    ax.grid(True, linestyle='--', alpha=GRID_ALPHA)
    _apply_spine(ax)
    plt.tight_layout()
    _savefig(fig, os.path.join(fig_dir, 'fig4_6_r_histogram.png'), cfg['vis']['dpi'])


# =============================================================================
# Figure 4.7  三组驾驶人 R* 小提琴图
# =============================================================================

def plot_violin_by_group(all_windows: pd.DataFrame, cfg: dict, fig_dir: str) -> None:
    """Figure 4.7：三组驾驶人 R* 小提琴图，各组着不同色。"""
    set_paper_style()

    groups = ['高能力组', '中能力组', '低能力组']
    gc     = cfg['vis']['group_colors']
    thi    = cfg['model']['thresh_high']
    tlo    = cfg['model']['thresh_low']

    data = [all_windows[all_windows['group'] == g]['R_star'].to_numpy()
            for g in groups]

    fig, ax = plt.subplots(figsize=(8, 5.5))
    parts   = ax.violinplot(data, positions=[1, 2, 3],
                             showmedians=True, showextrema=True)
    for pc, g in zip(parts['bodies'], groups):
        pc.set_facecolor(gc[g])
        pc.set_alpha(0.7)
    for comp in ['cmedians', 'cmaxes', 'cmins', 'cbars']:
        parts[comp].set_color('black')
        parts[comp].set_linewidth(1.2)

    ax.axhline(thi, color='red',           linestyle='--', linewidth=1, alpha=0.7)
    ax.axhline(tlo, color=SECONDARY_COLOR, linestyle='--', linewidth=1, alpha=0.7)
    ax.set_xticks([1, 2, 3])
    ax.set_xticklabels(groups)
    ax.set_ylabel('风险度 R*')
    ax.set_title('图4.7  三组驾驶人风险度分布小提琴图')
    ax.grid(axis='y', linestyle='--', alpha=GRID_ALPHA)
    _apply_spine(ax)
    plt.tight_layout()
    _savefig(fig, os.path.join(fig_dir, 'fig4_7_violin_by_group.png'), cfg['vis']['dpi'])


# =============================================================================
# Figure 4.8  场景×能力组 折线图
# =============================================================================

def plot_line_scenario_group(table_df: pd.DataFrame, cfg: dict, fig_dir: str) -> None:
    """
    Figure 4.8：三组驾驶人在四个场景下 R* 均值折线图。
    table_df 由 compute_scenario_group_table() 返回。
    """
    set_paper_style()

    groups = ['高能力组', '中能力组', '低能力组']
    gc     = cfg['vis']['group_colors']
    thi    = cfg['model']['thresh_high']
    tlo    = cfg['model']['thresh_low']
    scenes = table_df['场景'].tolist()

    fig, ax = plt.subplots(figsize=(8, 5))
    for g in groups:
        if g not in table_df.columns:
            continue
        ax.plot(scenes, table_df[g].tolist(), 'o-',
                color=gc[g], linewidth=LINE_WIDTH, markersize=7, label=g)

    ax.axhline(thi, color='red',           linestyle='--', linewidth=1, alpha=0.6)
    ax.axhline(tlo, color=SECONDARY_COLOR, linestyle='--', linewidth=1, alpha=0.6)
    ax.axhline(0,   color='gray',          linestyle=':',  linewidth=0.8)
    ax.set_xlabel('场景')
    ax.set_ylabel('风险度 R* 均值')
    ax.set_title('图4.8  三组驾驶人在四个场景下风险度均值变化折线图')
    ax.legend()
    ax.grid(True, linestyle='--', alpha=GRID_ALPHA)
    _apply_spine(ax)
    plt.tight_layout()
    _savefig(fig, os.path.join(fig_dir, 'fig4_8_line_scenario_group.png'), cfg['vis']['dpi'])


# =============================================================================
# Figure 4.9  场景×能力组 箱线图
# =============================================================================

def plot_box_scenario_group(all_windows: pd.DataFrame, cfg: dict, fig_dir: str) -> None:
    """Figure 4.9：三组驾驶人在四个场景下 R* 分组箱线图。"""
    set_paper_style()

    groups  = ['高能力组', '中能力组', '低能力组']
    gc      = cfg['vis']['group_colors']
    thi     = cfg['model']['thresh_high']
    tlo     = cfg['model']['thresh_low']
    lname   = {int(k): v for k, v in cfg['scenarios']['label_name'].items()}
    labels  = sorted(all_windows['field_label'].unique())
    scenes  = [lname.get(l, str(l)) for l in labels]
    width   = 0.22
    x       = np.arange(len(scenes))

    fig, ax = plt.subplots(figsize=(10, 5.5))
    for gi, g in enumerate(groups):
        offset = (gi - 1) * width
        data   = [all_windows[(all_windows['field_label'] == lbl) &
                               (all_windows['group'] == g)]['R_star'].to_numpy()
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

    ax.axhline(thi, color='red',           linestyle='--', linewidth=1, alpha=0.6)
    ax.axhline(tlo, color=SECONDARY_COLOR, linestyle='--', linewidth=1, alpha=0.6)
    ax.set_xticks(x)
    ax.set_xticklabels(scenes)
    ax.set_ylabel('风险度 R*')
    ax.set_title('图4.9  三组驾驶人在四个场景下风险度分布箱线图')
    handles = [mpatches.Patch(facecolor=gc[g], alpha=0.7, label=g) for g in groups]
    ax.legend(handles=handles)
    ax.grid(axis='y', linestyle='--', alpha=GRID_ALPHA)
    _apply_spine(ax)
    plt.tight_layout()
    _savefig(fig, os.path.join(fig_dir, 'fig4_9_box_scenario_group.png'), cfg['vis']['dpi'])


# =============================================================================
# Figure 4.10  风险等级堆叠柱状图
# =============================================================================

def plot_stacked_bar_risk(all_windows: pd.DataFrame, cfg: dict, fig_dir: str) -> None:
    """Figure 4.10：三组驾驶人风险等级分布堆叠柱状图。"""
    set_paper_style()

    groups = ['高能力组', '中能力组', '低能力组']
    levels = ['低风险', '中风险', '高风险']
    lc     = cfg['vis']['risk_level_colors']

    pcts = {g: [] for g in groups}
    for g in groups:
        sub   = all_windows[all_windows['group'] == g]
        total = max(len(sub), 1)
        for lvl in levels:
            pcts[g].append((sub['risk_level'] == lvl).sum() / total * 100)

    x       = np.arange(len(groups))
    fig, ax = plt.subplots(figsize=(7, 5))
    bottoms = np.zeros(len(groups))

    for lvl in levels:
        vals = [pcts[g][levels.index(lvl)] for g in groups]
        bars = ax.bar(x, vals, bottom=bottoms, color=lc[lvl], label=lvl,
                      alpha=0.85, edgecolor='white', linewidth=0.5)
        for rect, v in zip(bars, vals):
            if v > 3:
                ax.text(rect.get_x() + rect.get_width() / 2,
                        rect.get_y() + rect.get_height() / 2,
                        f'{v:.1f}%', ha='center', va='center',
                        fontsize=9, color='white', fontweight='bold')
        bottoms += np.array(vals)

    ax.set_xticks(x)
    ax.set_xticklabels(groups)
    ax.set_ylabel('占比 (%)')
    ax.set_ylim(0, 105)
    ax.set_title('图4.10  三组驾驶人风险等级分布堆叠柱状图')
    ax.legend(loc='upper right')
    ax.grid(axis='y', linestyle='--', alpha=GRID_ALPHA)
    _apply_spine(ax)
    plt.tight_layout()
    _savefig(fig, os.path.join(fig_dir, 'fig4_10_stacked_bar.png'), cfg['vis']['dpi'])


# =============================================================================
# Figure 4.11  典型驾驶人时序曲线
# =============================================================================

def plot_timeseries_typical(all_windows: pd.DataFrame,
                             sample_field: List[np.ndarray],
                             sample_ad_norm: List[np.ndarray],
                             fs_temporal_list: List[np.ndarray],
                             cap_groups: Dict[int, str],
                             cfg: dict,
                             fig_dir: str) -> None:
    """
    Figure 4.11：高/低能力组各取一名代表（R* 均值最接近组均值的样本），
    双行子图展示全程 R* 时序 + 场景背景色块 + 阈值线。

    Args:
        sample_ad_norm:    各样本归一化后的 Ã_d 数组列表
        fs_temporal_list:  各样本已完成空间→时间映射的 F_S 数组列表
    """
    set_paper_style()

    gc    = cfg['vis']['group_colors']
    thi   = cfg['model']['thresh_high']
    tlo   = cfg['model']['thresh_low']
    lname = {int(k): v for k, v in cfg['scenarios']['label_name'].items()}

    def _pick_rep(group_name: str) -> int:
        idxs = [i for i, g in cap_groups.items() if g == group_name]
        if not idxs:
            return list(cap_groups.keys())[0]
        g_mean  = all_windows[all_windows['group'] == group_name]['R_star'].mean()
        s_means = {i: all_windows[all_windows['sample_idx'] == i]['R_star'].mean()
                   for i in idxs
                   if len(all_windows[all_windows['sample_idx'] == i]) > 0}
        return min(s_means, key=lambda i: abs(s_means[i] - g_mean))

    reps = [('高能力组代表', _pick_rep('高能力组')),
            ('低能力组代表', _pick_rep('低能力组'))]

    fig, axes = plt.subplots(2, 1, figsize=(14, 10), sharex=False)
    fig.suptitle('图4.11  典型驾驶人风险度时序演化曲线', fontsize=15, fontweight='bold')

    for row_idx, (title, sidx) in enumerate(reps):
        ax     = axes[row_idx]
        fld    = np.asarray(sample_field[sidx])
        r_star = all_windows[all_windows['sample_idx'] == sidx]['R_star'].to_numpy()
        x      = np.arange(len(r_star))

        _fill_scene_bg(ax, fld[:len(r_star)], cfg)
        ax.plot(x, r_star,
                color=gc.get(cap_groups.get(sidx, '中能力组'), '#666666'),
                linewidth=LINE_WIDTH, label='R*')
        ax.axhline(thi, color='red',           linestyle='--', linewidth=1.0, alpha=0.8,
                   label=f'高风险 {thi}')
        ax.axhline(tlo, color=SECONDARY_COLOR, linestyle='--', linewidth=1.0, alpha=0.8,
                   label=f'低风险 {tlo}')
        ax.fill_between(x,  thi,  1.0, alpha=0.06, color='red')
        ax.fill_between(x, -1.0,  tlo, alpha=0.06, color=ACCENT_COLOR)
        ax.set_ylim(-1.1, 1.1)
        ax.set_ylabel('R*')
        ax.set_title(f'{title}（样本 {sidx}，R* 均值={r_star.mean():.3f}）')
        ax.legend(loc='upper right', fontsize=8)
        ax.grid(axis='y', linestyle='--', alpha=GRID_ALPHA)
        _apply_spine(ax)
        if row_idx == len(reps) - 1:
            ax.set_xlabel('窗口序号')

    all_lbls = set(np.concatenate([np.asarray(sample_field[r]).astype(int)
                                    for _, r in reps]))
    bg = _scene_bg_colors(cfg)
    patches = [mpatches.Patch(facecolor=bg[l], alpha=0.5,
                               label=lname.get(l, str(l)))
               for l in sorted(all_lbls) if l in bg]
    if patches:
        fig.legend(handles=patches, loc='lower center', ncol=len(patches),
                   fontsize=9, bbox_to_anchor=(0.5, -0.02))

    plt.tight_layout(rect=[0, 0.05, 1, 1])
    _savefig(fig, os.path.join(fig_dir, 'fig4_11_timeseries_typical.png'),
             cfg['vis']['dpi'])


# =============================================================================
# 单样本时序图（调试 / --plot_sample 用）
# =============================================================================

def plot_single_sample(sample_idx: int,
                        sample_field: List[np.ndarray],
                        fs_temporal_list: List[np.ndarray],
                        ad_norm_list: List[np.ndarray],
                        all_windows: pd.DataFrame,
                        cfg: dict,
                        fig_dir: str) -> None:
    """
    单个样本全程 F_S / Ã_d / R* 三子图时序可视化。

    Args:
        fs_temporal_list: 各样本空间→时间映射后的 F_S 列表
        ad_norm_list:     各样本归一化 Ã_d 列表
    """
    set_paper_style()

    thi  = cfg['model']['thresh_high']
    tlo  = cfg['model']['thresh_low']
    fsbl = cfg['model']['fs_baseline']

    fld    = np.asarray(sample_field[sample_idx])
    fs_arr = fs_temporal_list[sample_idx]
    adn    = ad_norm_list[sample_idx]
    r_star = all_windows[all_windows['sample_idx'] == sample_idx]['R_star'].to_numpy()
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
    ax.plot(x, r_star, color=SECONDARY_COLOR, linewidth=LINE_WIDTH, label='R*（风险度）')
    ax.axhline(thi, color='red',           linestyle='--', linewidth=1.0,
               label=f'高风险 {thi}')
    ax.axhline(tlo, color=SECONDARY_COLOR, linestyle='--', linewidth=1.0,
               label=f'低风险 {tlo}')
    ax.fill_between(x,  thi,  1.0,  alpha=0.08, color='red')
    ax.fill_between(x, -1.0,  tlo,  alpha=0.08, color=ACCENT_COLOR)
    ax.set_ylabel('R*（风险度）')
    ax.set_xlabel('窗口序号')
    ax.set_ylim(-1.1, 1.1)
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
    ax.set_title("R* 判别异常事件 ROC 曲线", fontsize=13)
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


def plot_r_star_boxplot(
    ev0: np.ndarray,
    ev1: np.ndarray,
    p_str: str,
    cfg: dict,
    fig_dir: str,
    save_name: str = "fig_r_star_boxplot.png",
) -> None:
    """
    绘制 R* 在无异常事件组 vs. 异常事件组之间的箱线图，
    并在标题中标注 Mann-Whitney U 检验 p 值。

    此函数由 risk_validator.py 的 plot_r_star_by_event() 调用。

    Args:
        ev0:      无异常事件窗口的 R* 数组
        ev1:      异常事件窗口的 R* 数组
        p_str:    p 值字符串，例如 "<0.001" 或 "0.0032"
        cfg:      配置字典
        fig_dir:  图片保存目录
        save_name: 文件名（默认 fig_r_star_boxplot.png）
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
    ax.set_ylabel("风险度 R*", fontsize=12)
    ax.set_title(f"R* 分布（事件 vs. 无事件）\nMann-Whitney U, p={p_str}",
                 fontsize=12)
    ax.grid(axis="y", alpha=GRID_ALPHA)
    _apply_spine(ax)
    plt.tight_layout()

    _savefig(fig, os.path.join(fig_dir, save_name), dpi)


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
    """
    绘制单个样本的 F_S 与 Ã_d 平滑折线图，并用栅格线填充两者之间的封闭区域：
      · F_S > Ã_d  →  黄色斜线栅格填充（高风险区）
      · F_S < Ã_d  →  绿色斜线栅格填充（低风险区）

    Parameters
    ----------
    sample_idx      : 样本编号（用于标题/文件名）
    fs_arr          : shape (T,) 的 F_S 时序数组
    ad_norm_arr     : shape (T,) 的 Ã_d 归一化时序数组
    cfg             : 配置字典（读取 vis.dpi）
    fig_dir         : 图片保存目录
    smooth_window   : F_S 的 SG 滤波窗口长度（默认 15）
    smooth_poly     : F_S 的 SG 多项式阶数（默认 3）
    resample_factor : 样条插值升采样倍数（默认 10）
    ad_smooth_window: Ã_d 的 SG 滤波窗口长度（默认 51，更平滑）
    ad_smooth_poly  : Ã_d 的 SG 多项式阶数（默认 3）
    """
    set_paper_style()
    assert len(fs_arr) == len(ad_norm_arr), "F_S 与 Ã_d 长度必须一致"

    # ── 平滑 + 升采样（F_S 与 Ã_d 分别使用独立平滑参数）────────
    t_dense, fs_dense = _smooth_and_resample(
        fs_arr,      smooth_window,    smooth_poly,    resample_factor)
    # ad_norm_arr=ad_norm_arr*3-1.7
    _,       ad_dense = _smooth_and_resample(
        ad_norm_arr, ad_smooth_window, ad_smooth_poly, resample_factor)
    ad_dense=ad_dense*1.7-0.5

    fig, ax = plt.subplots(figsize=FIGURE_SIZE_WIDE)

    # ── 平滑折线 ──────────────────────────────────────────────────
    ax.plot(t_dense, fs_dense, color=PRIMARY_COLOR,   lw=LINE_WIDTH,
            label=r'$F_S$',         zorder=4)
    ax.plot(t_dense, ad_dense, color=SECONDARY_COLOR, lw=LINE_WIDTH,
            label=r'$\tilde{A}_d$', zorder=4)

    # ── 黄色栅格：F_S > Ã_d ──────────────────────────────────────
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

    # ── 绿色栅格：F_S < Ã_d ──────────────────────────────────────
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

    # ── 坐标轴装饰 ────────────────────────────────────────────────
    # ax.set_xlim(t_dense[0], t_dense[-1])
    ax.set_xlim(t_dense[15 * resample_factor], t_dense[-1])
    ax.set_ylim(
        min(fs_dense.min(), ad_dense.min()) + 0.02,
        max(fs_dense.max(), ad_dense.max()) + 0.05,
    )
    ax.set_xlabel('时间t (s)', fontsize=18, weight='bold')
    ax.set_ylabel('归一化水平', fontsize=18, weight='bold')
    # ax.set_title(f'样本 {sample_idx}：任务需求与驾驶能力时序对比',
    #              fontsize=15, pad=10, weight='bold')

    legend_handles = [
        plt.Line2D([0], [0], color=PRIMARY_COLOR,   lw=LINE_WIDTH,
                   label=r'任务需求'),
        plt.Line2D([0], [0], color=SECONDARY_COLOR, lw=LINE_WIDTH,
                   label=r'驾驶能力'),
        mpatches.Patch(facecolor='none', edgecolor='goldenrod',
                       hatch='///',     label=r'风险状态'),
        mpatches.Patch(facecolor='none', edgecolor='seagreen',
                       hatch='///', label=r'安全状态'),
    ]
    ax.legend(handles=legend_handles, loc='upper right',
              fontsize=15, framealpha=0.9)
    ax.grid(axis='y', linestyle='--', alpha=GRID_ALPHA)
    ax.tick_params(axis='both', labelsize=15, direction='out',
               length=4.5, width=1.0)
    _apply_spine(ax)

    plt.tight_layout()
    _savefig(fig,
             os.path.join(fig_dir, f'fs_ad_filled_sample{sample_idx}.png'),
             cfg['vis']['dpi'])
# D:\Local\DynamicCapRisk\src\visualization\plot_prediction.py
import os
os.environ["OMP_NUM_THREADS"] = "1"

import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from pathlib import Path
# 新增：导入小数刻度格式化工具
from matplotlib.ticker import ScalarFormatter

RANDOM_STATE = 42

# ====================== 完全对齐你的统一风格配置 ======================
# 主色调：学术蓝系（和plot_capability.py完全一致）
PRIMARY_COLOR   = "#2C5F8A"   # 主蓝（直方图柱体、主折线）
SECONDARY_COLOR = "#E07B39"   # 暖橙（均值线、标注三角）
ACCENT_COLOR    = "#4CAF82"   # 绿色（辅助标注）
LIGHT_FILL      = "#D6E8F5"   # 浅蓝填充（区间着色）
GRAY_FILL       = "#EBEBEB"   # 浅灰填充

# 图表尺寸
FIGURE_SIZE_WIDE   = (11, 6)    # 宽幅图（损失曲线）
FIGURE_SIZE_SQUARE = (10, 9)    # 方形图（混淆矩阵）
LINE_WIDTH   = 1.8
MARKER_SIZE  = 7
GRID_ALPHA   = 0.3
SPINE_ALPHA  = 0.4

# ====================== 统一样式函数（修复负号显示，中文仍为宋体） ======================
def set_paper_style():
    """设置论文级绘图风格，解决中文宋体+负号显示问题"""
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
    # 核心修复：中文优先用宋体，符号/英文用微软雅黑+新罗马（完美支持负号）
    plt.rcParams.update({
        "font.family":        "sans-serif",
        # 👇 仅修改这一行：添加 Microsoft YaHei，解决负号 U+2212 报错
        "font.sans-serif":    ["SimSun", "Microsoft YaHei", "Times New Roman", "DejaVu Sans"],
        "axes.unicode_minus": False,  # 强制负号正常显示
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
    """统一坐标轴样式"""
    for spine in ax.spines.values():
        spine.set_visible(True)
        spine.set_edgecolor('black')
        spine.set_linewidth(1.2)
    ax.tick_params(axis='both', direction='out', length=4.5, width=1.0)

def _save_and_close(fig, save_path, msg=""):
    """统一保存 + 关闭图形，白色背景高清导出"""
    if save_path:
        save_path = Path(save_path)
        os.makedirs(save_path.parent, exist_ok=True)
        fig.savefig(save_path, dpi=300, bbox_inches="tight", facecolor="white")
        print(f"{msg}已保存至: {save_path}")
    plt.close(fig)

# =============================================================================
# 模型训练损失曲线（已修改：y轴显示普通小数，无科学计数法）
# =============================================================================
def plot_training_loss(
    train_loss: np.ndarray,
    val_loss: np.ndarray,
    best_epoch: int,
    save_path: str,
    epochs: int = 150
):
    set_paper_style()
    fig, ax = plt.subplots(figsize=FIGURE_SIZE_WIDE)
    x = np.arange(len(train_loss))

    # 主折线：训练/验证损失
    ax.plot(x, train_loss, label='训练集损失', color=PRIMARY_COLOR, linewidth=LINE_WIDTH)
    ax.plot(x, val_loss, label='验证集损失', color=SECONDARY_COLOR, linewidth=LINE_WIDTH)

    # 最优轮次标注
    ax.scatter(best_epoch, val_loss[best_epoch],
                color=SECONDARY_COLOR, s=120, zorder=5, marker="*",
                label=f'最优轮次(第{best_epoch}轮)')
    ax.axvline(x=best_epoch, color=SECONDARY_COLOR, linestyle='--', alpha=0.7)

    # 标签与样式
    ax.set_xlabel('训练轮数 (Epochs)')
    ax.set_ylabel('损失值')
    
    # ====================== 核心修改 ======================
    # 1. 移除原有的对数刻度 ax.set_yscale('log')
    # 2. 强制y轴显示普通小数，禁止科学计数法
    ax.yaxis.set_major_formatter(ScalarFormatter())
    ax.ticklabel_format(axis='y', style='plain', scilimits=(0, 0))
    # ======================================================
    
    ax.set_xlim(0, epochs)
    # ax.set_title('模型训练损失曲线', fontweight='bold')
    ax.legend(framealpha=0.9)
    ax.grid(alpha=GRID_ALPHA, linestyle="--")
    
    _apply_spine(ax)
    _save_and_close(fig, save_path, msg="训练损失曲线")

# =============================================================================
# 6组风险度回归预测（完全对齐你的绘图风格）
# =============================================================================
def plot_risk_regression_6groups(
    group_data: list,
    save_path: str,
    low_thresh: float = -0.1,
    high_thresh: float = 0.1
):
    set_paper_style()
    fig, axes = plt.subplots(3, 2, figsize=(18, 10))
    axes = axes.flatten()

    PRED_COLOR = PRIMARY_COLOR  # 统一主色调

    for i, (ax, data) in enumerate(zip(axes, group_data)):
        x = np.arange(len(data['true']))
        # 真实/预测曲线
        ax.plot(x, data['true'], label='真实风险度', color='black', linewidth=LINE_WIDTH)
        ax.plot(x, data['pred'], label='预测风险度', color=PRED_COLOR, linewidth=LINE_WIDTH, alpha=0.8)

        # 风险阈值线
        ax.axhline(y=high_thresh, color=SECONDARY_COLOR, linestyle='--', 
                   label=f'高风险阈值({high_thresh})', alpha=0.8)
        ax.axhline(y=low_thresh, color=PRIMARY_COLOR, linestyle='--', 
                   label=f'低风险阈值({low_thresh})', alpha=0.8)

        ax.set_title(f'第{i+1}组测试数据', fontweight='bold')
        ax.grid(alpha=GRID_ALPHA, linestyle="--")
        _apply_spine(ax)

    # 统一图例（和你的脚本格式一致）
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(
        handles, labels,
        loc='lower center',
        bbox_to_anchor=(0.5, 0.02),
        ncol=4,
        fontsize=12,
        frameon=True,
        fancybox=True,
        shadow=True,
        framealpha=0.9
    )

    fig.suptitle('6组测试数据风险度回归预测对比', fontsize=16, fontweight='bold')
    plt.tight_layout(rect=[0, 0.08, 1, 0.96])
    _save_and_close(fig, save_path, msg="6组风险度预测图")
            
# =============================================================================
# 风险等级混淆矩阵（完全对齐你的绘图风格）
# =============================================================================
def plot_confusion_matrix(
    cm: np.ndarray,
    class_names: list,
    save_path: str
):
    set_paper_style()
    fig, ax = plt.subplots(figsize=FIGURE_SIZE_SQUARE)
    
    # 热力图样式统一
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=class_names,
                yticklabels=class_names,
                annot_kws={'size': 14},
                ax=ax,
                linewidths=0.8)

    ax.set_xlabel('预测类别')
    ax.set_ylabel('真实类别')
    ax.set_title('风险等级分类混淆矩阵', fontweight='bold')
    _apply_spine(ax)
    _save_and_close(fig, save_path, msg="混淆矩阵热力图")
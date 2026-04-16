import os
os.environ["OMP_NUM_THREADS"] = "1"

from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import matplotlib.patches as mpatches

RANDOM_STATE = 42

# ====================== 完全对齐参考脚本的统一风格配置 ======================
# 主色调：学术蓝系
PRIMARY_COLOR   = "#2C5F8A"   # 主蓝（直方图柱体、主折线）
SECONDARY_COLOR = "#E07B39"   # 暖橙（均值线、标注三角、早停点）
ACCENT_COLOR    = "#4CAF82"   # 绿色（辅助标注）
LIGHT_FILL      = "#D6E8F5"   # 浅蓝填充（区间着色）
GRAY_FILL       = "#EBEBEB"   # 浅灰填充

# 分组色板
GROUP_PALETTE = {
    "高能力组":  "#2C5F8A",
    "中能力组":  "#4CAF82",
    "低能力组":  "#E07B39",
}
# 聚类/分组配色（6组风险预测专用）
GROUP_COLORS = ["#2C5F8A", "#E07B39", "#4CAF82", "#A259C4", "#E05C5C", "#F5C518"]

# 图形尺寸
FIGURE_SIZE_WIDE   = (11, 6)    # 宽幅图（损失曲线）
FIGURE_SIZE_SQUARE = (10, 9)    # 方形图（混淆矩阵）
FIGURE_SIZE_GRID   = (18, 10)   # 子图网格（6组预测）
LINE_WIDTH   = 1.8
MARKER_SIZE  = 100
GRID_ALPHA   = 0.3
SPINE_ALPHA  = 0.4

# ====================== 统一学术样式设置 ======================
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

# ====================== 统一坐标轴脊柱样式 ======================
def _apply_spine(ax) -> None:
    for spine in ax.spines.values():
        spine.set_visible(True)
        spine.set_edgecolor('black')
        spine.set_linewidth(1.2)
    ax.tick_params(axis='both', direction='out', length=4.5, width=1.0)

# ====================== 统一保存与关闭 ======================
def _save_and_close(fig, save_path, msg=""):
    if save_path:
        save_path = Path(save_path)
        os.makedirs(save_path.parent, exist_ok=True)
        fig.savefig(save_path, dpi=300, bbox_inches="tight", facecolor="white")
        print(f"{msg}已保存至: {save_path}")
    plt.close(fig)

# =============================================================================
# 论文图表1：模型训练损失曲线（图5.6）
# =============================================================================
def plot_training_loss(
    train_loss: np.ndarray,
    val_loss: np.ndarray,
    early_stop_epoch: int,
    save_path: str,
    epochs: int = 150
):
    set_paper_style()
    fig, ax = plt.subplots(figsize=FIGURE_SIZE_WIDE)
    x = np.arange(len(train_loss))

    # 统一配色绘制损失曲线
    ax.plot(x, train_loss, label='训练集损失', color=PRIMARY_COLOR, linewidth=LINE_WIDTH)
    ax.plot(x, val_loss, label='验证集损失', color=SECONDARY_COLOR, linewidth=LINE_WIDTH)

    # 早停点标注（统一风格）
    ax.scatter(early_stop_epoch, val_loss[early_stop_epoch],
                color=SECONDARY_COLOR, s=MARKER_SIZE, zorder=5,
                label=f'早停点(第{early_stop_epoch}轮)')
    ax.axvline(x=early_stop_epoch, color=SECONDARY_COLOR, linestyle='--', alpha=0.7)

    # 样式配置
    ax.set_xlabel('训练轮数 (Epochs)')
    ax.set_ylabel('损失值')
    ax.set_yscale('log')
    ax.set_xlim(0, epochs)
    ax.set_title('模型训练损失曲线', fontweight='bold')
    ax.legend()
    ax.grid(alpha=GRID_ALPHA)
    
    # 应用坐标轴样式
    _apply_spine(ax)
    # 保存
    _save_and_close(fig, save_path, msg="训练损失曲线")

# =============================================================================
# 论文图表2：6组风险度回归预测（图5.8）
# =============================================================================
def plot_risk_regression_6groups(
    group_data: list,
    save_path: str,          # 必选参数移到前面
    low_thresh: float = -0.1,
    high_thresh: float = 0.1
):
    set_paper_style()
    fig, axes = plt.subplots(2, 3, figsize=FIGURE_SIZE_GRID)
    axes = axes.flatten()

    for i, (ax, data, color) in enumerate(zip(axes, group_data, GROUP_COLORS)):
        x = np.arange(len(data['true']))
        # 真实值（黑色）+ 预测值（分组配色）
        ax.plot(x, data['true'], label='真实风险度', color='black', linewidth=LINE_WIDTH)
        ax.plot(x, data['pred'], label='预测风险度', color=color, linewidth=LINE_WIDTH, alpha=0.8)

        # 阈值线（统一配色）
        ax.axhline(y=high_thresh, color=SECONDARY_COLOR, linestyle='--', label='高风险阈值(0.1)', alpha=0.8)
        ax.axhline(y=low_thresh, color=PRIMARY_COLOR, linestyle='--', label='低风险阈值(-0.1)', alpha=0.8)

        ax.set_title(f'第{i+1}组测试数据', fontweight='bold')
        ax.set_ylim(-1.0, 1.0)
        ax.legend(fontsize=10)
        ax.grid(alpha=GRID_ALPHA)
        _apply_spine(ax)

    fig.suptitle('6组测试数据风险度回归预测对比', fontsize=16, fontweight='bold')
    plt.tight_layout()
    plt.subplots_adjust(top=0.92)
    _save_and_close(fig, save_path, msg="6组风险度预测图")
    
# =============================================================================
# 论文图表3：风险等级混淆矩阵（图5.6）
# =============================================================================
def plot_confusion_matrix(
    cm: np.ndarray,
    class_names: list,
    save_path: str
):
    set_paper_style()
    fig, ax = plt.subplots(figsize=FIGURE_SIZE_SQUARE)
    
    # 统一学术蓝热力图
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=class_names,
                yticklabels=class_names,
                annot_kws={'size': 14},
                ax=ax)

    ax.set_xlabel('预测类别')
    ax.set_ylabel('真实类别')
    ax.set_title('风险等级分类混淆矩阵', fontweight='bold')
    _apply_spine(ax)
    _save_and_close(fig, save_path, msg="混淆矩阵热力图")
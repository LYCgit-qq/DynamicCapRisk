import os
os.environ["OMP_NUM_THREADS"] = "1"

# ====================== 修复缓存报错 + 中文显示 ======================
import matplotlib
# 手动创建缓存目录，避免文件不存在报错
cache_dir = matplotlib.get_cachedir()
os.makedirs(cache_dir, exist_ok=True)
# 清理缓存文件，不删除文件夹
for f in os.listdir(cache_dir):
    try:
        os.remove(os.path.join(cache_dir, f))
    except:
        pass
# ==================================================================

from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import matplotlib.patches as mpatches

RANDOM_STATE = 42

# ====================== 统一风格配置 ======================
PRIMARY_COLOR   = "#2C5F8A"
SECONDARY_COLOR = "#E07B39"
ACCENT_COLOR    = "#4CAF82"
LIGHT_FILL      = "#D6E8F5"
GRAY_FILL       = "#EBEBEB"

GROUP_PALETTE = {
    "高能力组":  "#2C5F8A",
    "中能力组":  "#4CAF82",
    "低能力组":  "#E07B39",
}
GROUP_COLORS = ["#2C5F8A", "#E07B39", "#4CAF82", "#A259C4", "#E05C5C", "#F5C518"]

FIGURE_SIZE_WIDE   = (11, 6)
FIGURE_SIZE_SQUARE = (10, 9)
FIGURE_SIZE_GRID   = (18, 10)
LINE_WIDTH   = 1.8
MARKER_SIZE  = 100
GRID_ALPHA   = 0.3
SPINE_ALPHA  = 0.4

# ====================== 中文字体配置（AutoDL 专用） ======================
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
        "font.family":        "WenQuanYi Zen Hei",
        "font.sans-serif":    ["WenQuanYi Zen Hei", "Arial", "Times New Roman"],
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

# ====================== 工具函数 ======================
def _apply_spine(ax) -> None:
    for spine in ax.spines.values():
        spine.set_visible(True)
        spine.set_edgecolor('black')
        spine.set_linewidth(1.2)
    ax.tick_params(axis='both', direction='out', length=4.5, width=1.0)

def _save_and_close(fig, save_path, msg=""):
    if save_path:
        save_path = Path(save_path)
        os.makedirs(save_path.parent, exist_ok=True)
        fig.savefig(save_path, dpi=300, bbox_inches="tight", facecolor="white")
        print(f"{msg}已保存至: {save_path}")
    plt.close(fig)

# =============================================================================
# 模型训练损失曲线
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

    ax.plot(x, train_loss, label='训练集损失', color=PRIMARY_COLOR, linewidth=LINE_WIDTH)
    ax.plot(x, val_loss, label='验证集损失', color=SECONDARY_COLOR, linewidth=LINE_WIDTH)

    ax.scatter(early_stop_epoch, val_loss[early_stop_epoch],
                color=SECONDARY_COLOR, s=MARKER_SIZE, zorder=5,
                label=f'早停点(第{early_stop_epoch}轮)')
    ax.axvline(x=early_stop_epoch, color=SECONDARY_COLOR, linestyle='--', alpha=0.7)

    ax.set_xlabel('训练轮数 (Epochs)')
    ax.set_ylabel('损失值')
    ax.set_yscale('log')
    ax.set_xlim(0, epochs)
    ax.set_title('模型训练损失曲线', fontweight='bold')
    ax.legend()
    ax.grid(alpha=GRID_ALPHA)
    
    _apply_spine(ax)
    _save_and_close(fig, save_path, msg="训练损失曲线")

# =============================================================================
# 6组风险度回归预测
# =============================================================================
def plot_risk_regression_6groups(
    group_data: list,
    save_path: str,
    low_thresh: float = -0.1,
    high_thresh: float = 0.1
):
    set_paper_style()
    # 3行2列布局，宽高比拉宽压低（适配底部图例）
    fig, axes = plt.subplots(3, 2, figsize=(18, 10))
    axes = axes.flatten()

    # 🔥 统一配置：预测风险度固定颜色
    PRED_COLOR = "#1f77b4"  # 标准蓝色（可自行修改）

    for i, (ax, data) in enumerate(zip(axes, group_data)):
        x = np.arange(len(data['true']))
        # 真实值：黑色固定
        ax.plot(x, data['true'], label='真实风险度', color='black', linewidth=LINE_WIDTH)
        # 预测值：统一颜色，不再分组变色
        ax.plot(x, data['pred'], label='预测风险度', color=PRED_COLOR, linewidth=LINE_WIDTH, alpha=0.8)

        # 动态阈值线 + 动态标签
        ax.axhline(y=high_thresh, color=SECONDARY_COLOR, linestyle='--', 
                   label=f'高风险阈值({high_thresh})', alpha=0.8)
        ax.axhline(y=low_thresh, color=PRIMARY_COLOR, linestyle='--', 
                   label=f'低风险阈值({low_thresh})', alpha=0.8)

        ax.set_title(f'第{i+1}组测试数据', fontweight='bold')
        # Y轴自适应（删除固定范围）
        ax.grid(alpha=GRID_ALPHA)
        _apply_spine(ax)
        # 🔥 关键：关闭子图单独图例
        # ax.legend() 

    # 🔥 核心：全图统一图例 + 底部横排
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(
        handles, labels,
        loc='lower center',    # 底部居中
        bbox_to_anchor=(0.5, 0.02),  # 精确定位
        ncol=4,                # 横向4列排列
        fontsize=12,
        frameon=True,
        fancybox=True,
        shadow=True
    )

    fig.suptitle('6组测试数据风险度回归预测对比', fontsize=16, fontweight='bold')
    # 🔥 调整底部间距，给图例留空间
    plt.tight_layout(rect=[0, 0.08, 1, 0.96])
    _save_and_close(fig, save_path, msg="6组风险度预测图")
            
# =============================================================================
# 风险等级混淆矩阵
# =============================================================================
def plot_confusion_matrix(
    cm: np.ndarray,
    class_names: list,
    save_path: str
):
    set_paper_style()
    fig, ax = plt.subplots(figsize=FIGURE_SIZE_SQUARE)
    
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
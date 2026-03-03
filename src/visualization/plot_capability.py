import logging
from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats

RANDOM_STATE = 42  # 随机种子（保证可复现）


def plot_pca_visualization(X: pd.DataFrame, labels: pd.Series, output_dir: Path):
    """PCA降维可视化（图3.3）。"""
    # 第一步：全局字体配置（必须放在最前面）
    plt.rcParams["font.sans-serif"] = [
        "SimSun",
        "Times New Roman",
    ]  # 中文：宋体，英文：Times New Roman
    plt.rcParams["axes.unicode_minus"] = False  # 解决负号显示异常
    plt.rcParams["font.family"] = "sans-serif"

    pca = PCA(n_components=2, random_state=RANDOM_STATE)
    X_pca = pca.fit_transform(X)

    plt.figure(figsize=(10, 8))
    colors = {"高能力组": "darkgreen", "中能力组": "darkorange", "低能力组": "darkred"}
    markers = {"高能力组": "o", "中能力组": "s", "低能力组": "^"}

    for group in ["高能力组", "中能力组", "低能力组"]:
        idx = labels == group
        plt.scatter(
            X_pca[idx, 0],
            X_pca[idx, 1],
            c=colors[group],
            marker=markers[group],
            label=group,
            s=100,
            alpha=0.8,
        )

    # 第二步：标签/标题配置（关键：混合文本不指定fontfamily，纯中文指定）
    plt.xlabel(f"PC1 (方差解释率: {pca.explained_variance_ratio_[0]:.2%})")
    plt.ylabel(f"PC2 (方差解释率: {pca.explained_variance_ratio_[1]:.2%})")
    plt.title("基准驾驶能力聚类结果可视化(PCA降维)", fontfamily="SimSun", fontsize=14)
    plt.legend(prop={"family": "SimSun"})  # 图例纯中文，指定宋体
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(output_dir / "cluster_pca_visualization.png", dpi=300)
    plt.close()
    logging.info(
        "PCA可视化图已保存至: %s", output_dir / "cluster_pca_visualization.png"
    )


def plot_pca_3d_visualization(X: pd.DataFrame, labels: pd.Series, output_dir: Path):
    """PCA降维至3维并可视化（新增3D版本）"""
    # 1. 全局字体配置（解决中文显示）
    plt.rcParams["font.sans-serif"] = [
        "SimSun",
        "Times New Roman",
    ]  # 宋体(中文)+Times New Roman(英文)
    plt.rcParams["axes.unicode_minus"] = False  # 解决负号显示异常
    plt.rcParams["font.family"] = "sans-serif"

    # 2. PCA降维到3维
    pca = PCA(n_components=3, random_state=RANDOM_STATE)
    X_pca = pca.fit_transform(X)

    # 3. 创建3D画布
    fig = plt.figure(figsize=(12, 10))
    ax = fig.add_subplot(111, projection="3d")  # 创建3D坐标轴

    # 4. 定义颜色/标记（与二维图完全一致，保证视觉统一）
    colors = {"高能力组": "darkgreen", "中能力组": "darkorange", "低能力组": "darkred"}
    markers = {"高能力组": "o", "中能力组": "s", "低能力组": "^"}

    # 5. 绘制3D散点图
    for group in ["高能力组", "中能力组", "低能力组"]:
        idx = labels == group
        ax.scatter(
            X_pca[idx, 0],
            X_pca[idx, 1],
            X_pca[idx, 2],
            c=colors[group],
            marker=markers[group],
            label=group,
            s=100,
            alpha=0.8,
            edgecolors="black",
            linewidth=0.5,  # 加黑边增强辨识度
        )

    # 6. 设置坐标轴标签（含3维方差解释率，自动匹配中英字体）
    ax.set_xlabel(
        f"PC1 (方差解释率: {pca.explained_variance_ratio_[0]:.2%})", fontsize=11
    )
    ax.set_ylabel(
        f"PC2 (方差解释率: {pca.explained_variance_ratio_[1]:.2%})", fontsize=11
    )
    ax.set_zlabel(
        f"PC3 (方差解释率: {pca.explained_variance_ratio_[2]:.2%})", fontsize=11
    )

    # 7. 设置标题和图例
    ax.set_title(
        "基准驾驶能力聚类结果可视化(PCA 3D降维)",
        fontfamily="SimSun",
        fontsize=14,
        pad=20,
    )
    ax.legend(prop={"family": "SimSun"}, fontsize=10, loc="upper right")

    # 8. 优化3D视角（常用美观视角）
    ax.view_init(elev=15, azim=45)  # 调整仰角和方位角，可根据需求修改
    ax.grid(alpha=0.3)

    # 9. 保存图片（高分辨率）
    plt.tight_layout()
    plt.savefig(
        output_dir / "cluster_pca_3d_visualization.png", dpi=300, bbox_inches="tight"
    )
    plt.close()
    # plt.show()
    logging.info(
        "3D PCA可视化图已保存至: %s", output_dir / "cluster_pca_3d_visualization.png"
    )


# ====================== 学术绘图风格配置 ======================
def set_paper_style():
    """设置论文级绘图风格（清晰、高DPI、无衬线字体）"""
    sns.set_style("whitegrid")
    plt.rcParams.update(
        {
            "font.family": "Arial",  # 无衬线字体，适合论文
            "font.size": 12,
            "axes.labelsize": 14,
            "axes.titlesize": 16,
            "xtick.labelsize": 10,
            "ytick.labelsize": 10,
            "legend.fontsize": 11,
            "figure.dpi": 300,  # 高分辨率
            "savefig.dpi": 300,
            "savefig.bbox": "tight",  # 自动裁剪空白
            "savefig.format": "png",  # 默认保存PNG（位图）
        }
    )


# ====================== 1. 特征相关性热力图 ======================
def plot_correlation_heatmap(features_df, save_path=None):
    """
    绘制最终特征的Pearson相关性热力图

    Args:
        features_df: 提取的特征DataFrame
        save_path: 保存路径（如 'output/figs/corr_heatmap.pdf'）
    """
    set_paper_style()
    corr_mat = features_df.corr()

    fig, ax = plt.subplots(figsize=(10, 8))
    # 绘制热力图，显示相关系数，保留两位小数
    sns.heatmap(
        corr_mat,
        annot=True,
        fmt=".2f",
        cmap="coolwarm",
        vmin=-1,
        vmax=1,
        square=True,
        cbar_kws={"shrink": 0.8, "label": "Pearson Correlation"},
        ax=ax,
    )
    ax.set_title("Correlation Matrix of Final Driving Features", pad=20)

    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path)
        print(f"相关性热力图已保存至: {save_path}")
    # plt.show()
    plt.close()


# ====================== 2. 驾驶能力波动量分布图 ======================
def plot_fluctuation_distribution(fluctuation_arr, save_path=None):
    """
    绘制驾驶能力波动量的直方图+核密度估计图

    Args:
        fluctuation_arr: 驾驶能力波动量数组 (A_fl)
        save_path: 保存路径
    """
    set_paper_style()

    fig, ax = plt.subplots(figsize=(10, 6))

    # 绘制直方图+KDE
    sns.histplot(
        fluctuation_arr,
        bins=50,
        kde=True,
        color="steelblue",
        edgecolor="white",
        linewidth=0.5,
        ax=ax,
    )

    # 标注关键统计信息
    mean_val = np.mean(fluctuation_arr)
    std_val = np.std(fluctuation_arr)
    ax.axvline(
        mean_val,
        color="crimson",
        linestyle="--",
        linewidth=2,
        label=f"Mean: {mean_val:.3f}",
    )

    # 标注-0.05~0.05区间（论文中提到的集中区间）
    ax.axvspan(
        -0.05, 0.05, color="lightgray", alpha=0.3, label="Stable Range [-0.05, 0.05]"
    )

    # 计算并显示集中区间占比
    in_range = np.sum((fluctuation_arr >= -0.05) & (fluctuation_arr <= 0.05)) / len(
        fluctuation_arr
    )
    ax.text(
        0.95,
        0.9,
        f"In Range: {in_range:.1%}",
        transform=ax.transAxes,
        ha="right",
        va="top",
        bbox=dict(facecolor="white", alpha=0.8),
    )

    ax.set_xlabel("Driving Ability Fluctuation ($A_{fl}$)")
    ax.set_ylabel("Frequency")
    ax.set_title("Distribution of Driving Ability Fluctuation", pad=20)
    ax.legend(loc="upper left")
    sns.despine(ax=ax)

    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path)
        print(f"波动量分布图已保存至: {save_path}")
    # plt.show()
    plt.close()


# ====================== 3. 不同基准能力组波动量箱线图 ======================
def plot_grouped_boxplot(fluctuation_arr, n_groups=3, save_path=None):
    """
    绘制不同基准能力组的驾驶能力波动量箱线图
    (根据波动量分位数模拟基准能力分组：高/中/低)

    Args:
        fluctuation_arr: 驾驶能力波动量数组
        n_groups: 分组数量（默认3组）
        save_path: 保存路径
    """
    set_paper_style()

    # 1. 根据波动量分位数分组（模拟基准能力：波动越小，基准能力越高）
    # 注意：此处假设波动量绝对值越小代表基准能力越稳定，可根据实际逻辑调整
    abs_fluct = np.abs(fluctuation_arr)
    quantiles = np.quantile(abs_fluct, np.linspace(0, 1, n_groups + 1))

    # 分组标签
    group_labels = ["High Baseline", "Medium Baseline", "Low Baseline"]
    groups = pd.cut(abs_fluct, bins=quantiles, labels=group_labels, include_lowest=True)

    # 构建DataFrame用于绘图
    plot_df = pd.DataFrame(
        {"Fluctuation": fluctuation_arr, "Baseline Ability Group": groups}
    )

    # 2. 绘制箱线图
    fig, ax = plt.subplots(figsize=(10, 6))
    sns.boxplot(
        x="Baseline Ability Group",
        y="Fluctuation",
        data=plot_df,
        palette="viridis",
        showfliers=False,  # 隐藏异常值点使图更整洁
        width=0.6,
        ax=ax,
    )

    # 3. 添加统计显著性标注（单因素ANOVA）
    # 提取各组数据
    group_data = [
        plot_df[plot_df["Baseline Ability Group"] == lbl]["Fluctuation"].values
        for lbl in group_labels
    ]
    f_stat, p_val = stats.f_oneway(*group_data)

    # 在图上标注ANOVA结果
    p_text = f"ANOVA: F={f_stat:.2f}, " + (
        f"p<0.001" if p_val < 0.001 else f"p={p_val:.3f}"
    )
    ax.text(
        0.5,
        0.95,
        p_text,
        transform=ax.transAxes,
        ha="center",
        va="top",
        bbox=dict(facecolor="white", alpha=0.8, edgecolor="none"),
    )

    # 4. 标注各组均值
    means = plot_df.groupby("Baseline Ability Group")["Fluctuation"].mean()
    for i, lbl in enumerate(group_labels):
        ax.text(
            i,
            means[lbl],
            f"{means[lbl]:.3f}",
            ha="center",
            va="bottom",
            fontweight="bold",
            color="crimson",
        )

    ax.set_ylabel("Driving Ability Fluctuation ($A_{fl}$)")
    ax.set_title("Driving Ability Fluctuation by Baseline Ability Group", pad=20)
    sns.despine(ax=ax)

    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path)
        print(f"分组箱线图已保存至: {save_path}")
    # plt.show()
    plt.close()


# ====================== 主调用函数 ======================
def run_all_visualizations(result_pkl_path, output_dir="output/figs"):
    """
    一键运行所有可视化

    Args:
        result_pkl_path: 之前保存的 driving_ability_result.pkl 路径
        output_dir: 图片输出目录
    """
    import pickle

    # 加载结果数据
    with open(result_pkl_path, "rb") as f:
        result = pickle.load(f)

    features_df = result["features"]
    fluctuation_arr = result["fluctuation"]

    # 依次绘图
    plot_correlation_heatmap(
        features_df, save_path=os.path.join(output_dir, "corr_heatmap.png")
    )
    plot_fluctuation_distribution(
        fluctuation_arr, save_path=os.path.join(output_dir, "fluctuation_dist.png")
    )
    plot_grouped_boxplot(
        fluctuation_arr, save_path=os.path.join(output_dir, "grouped_boxplot.png")
    )


if __name__ == "__main__":
    # 示例调用（请根据实际路径修改）
    run_all_visualizations(
        result_pkl_path="output/1_capability_assessment/driving_ability_result.pkl",
        output_dir="output/1_capability_assessment",
    )

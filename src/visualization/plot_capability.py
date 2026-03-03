import logging
from pathlib import Path

import pandas as pd
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA

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

# D:\Local\DynamicCapRisk\src\visualization\plot_capability.py

import os

os.environ["OMP_NUM_THREADS"] = "1"

import logging
from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
import numpy as np
import seaborn as sns
from scipy import stats

RANDOM_STATE = 42  # 随机种子（保证可复现）


# ====================== 学术绘图风格配置（优化版） ======================
def set_paper_style():
    """设置论文级绘图风格（清晰、高DPI、兼容中英文）"""
    sns.set_style("whitegrid")
    plt.rcParams.update(
        {
            "font.family": [
                "SimSun",
                "Times New Roman",
            ],  # 中文：宋体，英文：Times New Roman
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
            "axes.unicode_minus": False,  # 解决负号显示异常
        }
    )


# %% 基准能力评估模块的可视化函数


def plot_pca_visualization(X: pd.DataFrame, labels: pd.Series, output_dir: Path):
    """PCA降维可视化（适配任意BEST_K）。"""
    set_paper_style()
    # 第一步：全局字体配置（必须放在最前面）
    plt.rcParams["font.sans-serif"] = [
        "SimSun",
        "Times New Roman",
    ]  # 中文：宋体，英文：Times New Roman
    plt.rcParams["axes.unicode_minus"] = False  # 解决负号显示异常
    plt.rcParams["font.family"] = "sans-serif"

    pca = PCA(n_components=2, random_state=RANDOM_STATE)
    X_pca = pca.fit_transform(X)

    # -------------------------- 动态适配任意分组数 --------------------------
    # 1. 动态获取唯一分组名称（替代硬编码的["高能力组", "中能力组", "低能力组"]）
    unique_groups = labels.unique().tolist()
    # 2. 动态生成颜色（使用matplotlib调色板，适配任意数量分组）
    color_palette = plt.cm.Set3(np.linspace(0, 1, len(unique_groups)))
    # 3. 动态生成标记（优先常用标记，超出则循环）
    marker_list = ["o", "s", "^", "D", "v", "p", "*", "h", "+"]
    markers = {
        group: marker_list[i % len(marker_list)]
        for i, group in enumerate(unique_groups)
    }
    colors = {group: color_palette[i] for i, group in enumerate(unique_groups)}
    # -----------------------------------------------------------------------

    # 循环绘制动态获取的分组（替代硬编码循环）
    for group in unique_groups:
        idx = labels == group
        plt.scatter(
            X_pca[idx, 0],
            X_pca[idx, 1],
            c=[colors[group]],  # 修复单颜色传入格式问题
            marker=markers[group],
            label=group,
            s=100,
            alpha=0.8,
        )

    # 第二步：标签/标题配置
    plt.xlabel(f"PC1 (方差解释率: {pca.explained_variance_ratio_[0]:.2%})")
    plt.ylabel(f"PC2 (方差解释率: {pca.explained_variance_ratio_[1]:.2%})")
    plt.title("基准驾驶能力聚类结果可视化(PCA降维)", fontfamily="SimSun", fontsize=14)
    plt.legend(prop={"family": "SimSun"})  # 图例纯中文，指定宋体
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(output_dir / "Ab_cluster_pca_visualization.png", dpi=300)
    plt.close()
    logging.info(
        "PCA可视化图已保存至: %s", output_dir / "Ab_cluster_pca_visualization.png"
    )


def plot_pca_3d_visualization(X: pd.DataFrame, labels: pd.Series, output_dir: Path):
    """PCA降维至3维并可视化（新增3D版本，适配任意BEST_K）"""
    set_paper_style()
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

    # -------------------------- 动态适配任意分组数 --------------------------
    unique_groups = labels.unique().tolist()
    color_palette = plt.cm.Set3(np.linspace(0, 1, len(unique_groups)))
    marker_list = ["o", "s", "^", "D", "v", "p", "*", "h", "+"]
    markers = {
        group: marker_list[i % len(marker_list)]
        for i, group in enumerate(unique_groups)
    }
    colors = {group: color_palette[i] for i, group in enumerate(unique_groups)}
    # -----------------------------------------------------------------------

    # 5. 绘制3D散点图（动态分组）
    for group in unique_groups:
        idx = labels == group
        ax.scatter(
            X_pca[idx, 0],
            X_pca[idx, 1],
            X_pca[idx, 2],
            c=[colors[group]],  # 修复单颜色传入格式问题
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
        output_dir / "Ab_cluster_pca_3d_visualization.png", dpi=300, bbox_inches="tight"
    )
    plt.close()
    logging.info(
        "3D PCA可视化图已保存至: %s", output_dir / "Ab_cluster_pca_3d_visualization.png"
    )


# ====================== 聚类评价指标可视化（SC/CH/DBI） ======================
def plot_cluster_metrics(eval_df: pd.DataFrame, output_dir: Path, best_k: int = None):
    """
    绘制聚类评价指标可视化图（一行三列子图：SC/CH/DBI）

    Args:
        eval_df: 聚类评价结果DataFrame（索引=k，列=轮廓系数SC/CH指数/DBI指数）
        output_dir: 图片保存目录
        best_k: 最优k值（可选，标注在图上）
    """
    set_paper_style()

    # 创建1行3列的子图
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    fig.suptitle("聚类评价指标对比 (k=2-9)", fontsize=16, y=0.95)

    # 提取数据
    k_vals = eval_df.index.values
    sc_vals = eval_df["轮廓系数SC"].values
    ch_vals = eval_df["CH指数"].values
    dbi_vals = eval_df["DBI指数"].values

    # 子图1：轮廓系数SC（越大越好）
    ax1 = axes[0]
    ax1.plot(k_vals, sc_vals, marker="o", linewidth=2, markersize=8, color="#2E86AB")
    ax1.set_xlabel("聚类数目 k")
    ax1.set_ylabel("轮廓系数 SC")
    ax1.set_title("轮廓系数", fontsize=14)
    ax1.grid(alpha=0.3)
    # 标注最优k（如果指定）
    if best_k is not None and best_k in k_vals:
        ax1.scatter(
            best_k,
            sc_vals[list(k_vals).index(best_k)],
            color="red",
            s=150,
            zorder=5,
            label=f"最优 k={best_k}",
        )
        ax1.legend()

    # 子图2：CH指数（越大越好）
    ax2 = axes[1]
    ax2.plot(k_vals, ch_vals, marker="s", linewidth=2, markersize=8, color="#A23B72")
    ax2.set_xlabel("聚类数目 k")
    ax2.set_ylabel("CH指数")
    ax2.set_title("CH指数", fontsize=14)
    ax2.grid(alpha=0.3)
    if best_k is not None and best_k in k_vals:
        ax2.scatter(
            best_k,
            ch_vals[list(k_vals).index(best_k)],
            color="red",
            s=150,
            zorder=5,
            label=f"Best k={best_k}",
        )
        ax2.legend()

    # 子图3：DBI指数（越小越好）
    ax3 = axes[2]
    ax3.plot(k_vals, dbi_vals, marker="^", linewidth=2, markersize=8, color="#F18F01")
    ax3.set_xlabel("聚类数目 k")
    ax3.set_ylabel("DBI指数")
    ax3.set_title("DBI指数", fontsize=14)
    ax3.grid(alpha=0.3)
    if best_k is not None and best_k in k_vals:
        ax3.scatter(
            best_k,
            dbi_vals[list(k_vals).index(best_k)],
            color="red",
            s=150,
            zorder=5,
            label=f"Best k={best_k}",
        )
        ax3.legend()

    # 调整子图间距
    plt.tight_layout()
    # 保存图片
    save_path = output_dir / "Ab_cluster_metrics_visualization.png"
    plt.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close()
    logging.info(f"聚类评价指标可视化图已保存至: {save_path}")


# %% 波动量计算


# ====================== 1. 特征相关性热力图（优化版） ======================
def plot_correlation_heatmap(features_df, save_path=None):
    """
    绘制最终特征的Pearson相关性热力图

    Args:
        features_df: 提取的特征DataFrame
        save_path: 保存路径（如 'output/figs/corr_heatmap.pdf'）
    """
    # 鲁棒性检查
    if features_df.empty or len(features_df.columns) < 2:
        print("特征数据为空或特征数<2，跳过相关性热力图绘制")
        return

    feat_name_map = {
        "steering_angle": "方向盘转角",
        "steering_velocity": "方向盘角速度",
        "brake_pedal": "制动踏板开度",
        "throttle_pedal": "油门踏板开度",
        "longitudinal_accel": "纵向加速度",
        "lateral_offset": "横向偏移量",
        "lateral_accel": "横向加速度",
        "vehicle_speed": "车速",
        "gaze_dispersion": "注视点分散度",
        "blink_frequency": "眨眼频率",
        "hrv": "心率变异性",
        "bvp": "血容量脉搏",
        "ecg": "心电信号",
        "resp": "呼吸信号",
    }

    set_paper_style()
    corr_mat = features_df.corr()

    # 替换特征名为中文（兼容未匹配的特征名）
    corr_mat.index = [feat_name_map.get(name, name) for name in corr_mat.index]
    corr_mat.columns = [feat_name_map.get(name, name) for name in corr_mat.columns]

    # 动态调整画布尺寸（按特征数适配）
    n_feat = len(features_df.columns)
    fig_size = (max(10, n_feat * 0.8), max(8, n_feat * 0.7))
    fig, ax = plt.subplots(figsize=fig_size)

    # 绘制热力图，优化标注字体大小
    sns.heatmap(
        corr_mat,
        annot=True,
        fmt=".2f",
        cmap="coolwarm",
        vmin=-1,
        vmax=1,
        square=True,
        cbar_kws={"shrink": 0.8, "label": "Pearson相关系数"},  # 色标标签也改中文
        ax=ax,
        annot_kws={"size": 9},  # 缩小标注字体，避免重叠
    )
    ax.set_title("驾驶特征Pearson相关性热力图", fontsize=14, pad=20)  # 标题改中文
    ax.tick_params(axis="both", labelsize=10)  # 调整坐标轴标签字体大小

    if save_path:
        save_path = Path(save_path)  # 统一路径处理
        os.makedirs(save_path.parent, exist_ok=True)
        plt.savefig(save_path, bbox_inches="tight")  # 加bbox_inches避免中文标签被截断
        print(f"相关性热力图已保存至: {save_path}")
    plt.close()


# ====================== 2. 驾驶能力波动量分布图（优化版） ======================
def plot_fluctuation_distribution(fluctuation_arr, save_path=None):
    """
    绘制驾驶能力波动量的直方图+核密度估计图（适配实际数据分布）

    Args:
        fluctuation_arr: 驾驶能力波动量数组 (A_fl)
        save_path: 保存路径
    """
    # 鲁棒性检查
    if len(fluctuation_arr) == 0:
        print("波动量数组为空，跳过波动量分布图绘制")
        return

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
        label=f"均值: {mean_val:.3f}",
    )

    # 优化：动态计算实际集中区间（1个标准差范围），保留原论文区间标注但补充说明
    std_range = [mean_val - std_val, mean_val + std_val]
    ax.axvspan(
        std_range[0],
        std_range[1],
        color="lightblue",
        alpha=0.3,
        label=f"±1σ 区间 [{std_range[0]:.3f}, {std_range[1]:.3f}]",
    )
    # 保留论文区间，标注实际占比（当前数据中该区间占比极低）
    paper_range = [-0.05, 0.05]
    in_paper_range = np.sum(
        (fluctuation_arr >= paper_range[0]) & (fluctuation_arr <= paper_range[1])
    ) / len(fluctuation_arr)
    ax.axvspan(
        paper_range[0],
        paper_range[1],
        color="lightgray",
        alpha=0.3,
        label=f"论文区间 [-0.05, 0.05] (In: {in_paper_range:.1%})",
    )

    # 调整文本位置，避免超出画布
    ax.text(
        0.95,
        0.85,
        f"均值: {mean_val:.3f}\n标准差: {std_val:.3f}\n论文区间 In: {in_paper_range:.1%}",
        transform=ax.transAxes,
        ha="right",
        va="top",
        bbox=dict(facecolor="white", alpha=0.8),
        fontsize=10,
    )

    ax.set_xlabel("驾驶能力波动量 Afl")
    ax.set_ylabel("频次")
    ax.set_title("驾驶能力波动量分布", pad=20)
    ax.legend(loc="upper left", fontsize=9)
    sns.despine(ax=ax)

    if save_path:
        save_path = Path(save_path)
        os.makedirs(save_path.parent, exist_ok=True)
        plt.savefig(save_path)
        print(f"波动量分布图已保存至: {save_path}")
    plt.close()


# ====================== 3. 不同基准能力组波动量箱线图 ======================
def plot_grouped_boxplot(fluctuation_sample, config, save_path=None):
    """
    绘制真实能力等级箱线图（消除所有FutureWarning）
    """
    # ====================== 1. 基础检查 ======================
    if len(fluctuation_sample) != 67:
        raise ValueError(
            f"fluctuation_sample长度必须为67，当前为{len(fluctuation_sample)}"
        )

    required_config_keys = ["subject_exp_map_csv", "ability_label_csv"]
    if not all(k in config for k in required_config_keys):
        raise KeyError(f"config必须包含：{required_config_keys}")

    for csv_key in required_config_keys:
        csv_path = config[csv_key]
        if not os.path.exists(csv_path):
            raise FileNotFoundError(f"CSV文件不存在：{csv_path}")

    # ====================== 2. 加载映射关系 ======================
    map_df = pd.read_csv(config["subject_exp_map_csv"])
    map_df["实验ID"] = map_df["实验ID"].astype(int)

    label_df = pd.read_csv(config["ability_label_csv"])
    label_df.rename(columns={label_df.columns[0]: "被试ID"}, inplace=True)
    label_df["被试ID"] = label_df["被试ID"].astype(int)
    label_df["能力等级"] = label_df["能力等级"].astype(str)  # 强制字符串

    # 合并映射（仅保留高/中/低）
    merge_df = pd.merge(map_df, label_df, on="被试ID", how="left")
    merge_df = merge_df[merge_df["能力等级"].isin(["高能力组", "中能力组", "低能力组"])]
    merge_df["能力等级"] = merge_df["能力等级"].fillna("低能力组")
    expid_to_grade = dict(zip(merge_df["实验ID"], merge_df["能力等级"]))

    # ====================== 3. 收集波动量数据 ======================
    grade_flucts = {"高能力组": [], "中能力组": [], "低能力组": []}

    for exp_id in range(67):
        if exp_id not in expid_to_grade:
            continue
        grade = expid_to_grade[exp_id]

        fluct_arr = fluctuation_sample[exp_id]
        if not isinstance(fluct_arr, (np.ndarray, list)) or len(fluct_arr) == 0:
            continue

        try:
            fluct_arr = np.array(fluct_arr, dtype=np.float64)
            fluct_arr = fluct_arr[np.isfinite(fluct_arr)]
        except:
            continue

        if len(fluct_arr) > 0:
            grade_flucts[grade].extend(fluct_arr.tolist())

    # 过滤空分组
    grade_flucts = {k: v for k, v in grade_flucts.items() if len(v) > 0}
    if len(grade_flucts) == 0:
        print("无有效波动量数据可绘制")
        return

    print("最终有效分组数据量：")
    for grade, vals in grade_flucts.items():
        print(f"  {grade}: {len(vals)}个值")

    # ====================== 4. 绘制箱线图（修复seaborn警告） ======================
    set_paper_style()

    # 准备绘图数据
    plot_data = []
    for grade, fluct_vals in grade_flucts.items():
        for val in fluct_vals:
            plot_data.append({"驾驶能力波动量 Afl": val, "真实能力等级": grade})
    plot_df = pd.DataFrame(plot_data)

    # 动态有序分类
    ordered_grades = []
    for g in ["高能力组", "中能力组", "低能力组"]:
        if g in grade_flucts and len(grade_flucts[g]) > 0:
            ordered_grades.append(g)

    plot_df["真实能力等级"] = pd.Categorical(
        plot_df["真实能力等级"], categories=ordered_grades, ordered=True
    )

    # 创建画布
    fig, ax = plt.subplots(figsize=(10, 6))

    sns.boxplot(
        x="真实能力等级",
        y="驾驶能力波动量 Afl",
        hue="真实能力等级",  # 解决palette警告
        data=plot_df,
        palette="viridis",
        showfliers=False,
        width=0.6,
        ax=ax,
        legend=False       # 解决palette警告
    )

    # ====================== 5. 统计标注 ======================
    if len(ordered_grades) >= 2:
        try:
            group_data = [grade_flucts[g] for g in ordered_grades]
            f_stat, p_val = stats.f_oneway(*group_data)
            p_text = (
                f"ANOVA: F={f_stat:.2f}, p={p_val:.3f}"
                if p_val >= 0.001
                else f"ANOVA: F={f_stat:.2f}, p<0.001"
            )
            ax.text(
                0.5,
                0.95,
                p_text,
                transform=ax.transAxes,
                ha="center",
                va="top",
                bbox=dict(facecolor="white", alpha=0.8),
                fontsize=10,
            )
        except Exception as e:
            print(f"ANOVA统计失败: {e}")

    # 手动计算均值（消除pandas groupby警告）
    try:
        grade_means = {}
        for grade in ordered_grades:
            vals = grade_flucts[grade]
            if len(vals) > 0:
                grade_means[grade] = np.mean(vals)

        for i, grade in enumerate(ordered_grades):
            if grade in grade_means and np.isfinite(grade_means[grade]):
                ax.scatter(
                    i, grade_means[grade], color="crimson", marker="^", s=40, zorder=10
                )
    except Exception as e:
        print(f"均值标注失败: {e}（跳过）")

    # ====================== 6. 美化与保存 ======================
    ax.set_xlabel("")
    ax.set_ylabel("驾驶能力波动量 Afl", fontsize=14)
    ax.set_title("不同真实能力等级的驾驶能力波动量", pad=20, fontsize=16)
    ax.tick_params(axis="x", labelsize=14)
    sns.despine(ax=ax)

    if save_path:
        save_path = Path(save_path)
        os.makedirs(save_path.parent, exist_ok=True)
        plt.ioff()
        plt.savefig(save_path, dpi=300, bbox_inches="tight", facecolor="white")
        plt.close(fig)
        print(f"箱线图已保存至: {save_path}")
    else:
        plt.close(fig)


def plot_grouped_boxplot_abs(fluctuation_arr, save_path=None):
    """
    绘制不同基准能力组的驾驶能力波动量箱线图（修复警告+增强鲁棒性）
    (根据波动量绝对值分位数模拟基准能力等级：高/中/低)

    Args:
        fluctuation_arr: 驾驶能力波动量数组
        n_groups: 分组数量（默认3组）
        save_path: 保存路径
    """
    # 鲁棒性检查
    if len(fluctuation_arr) == 0:
        print("波动量数组为空，跳过分组箱线图绘制")
        return
    if n_groups < 2 or n_groups > 5:
        print(f"分组数{n_groups}不合理，默认改为3组")
        n_groups = 3

    set_paper_style()

    # 1. 根据波动量绝对值分位数分组（模拟基准能力：波动越小，基准能力越高）
    abs_fluct = np.abs(fluctuation_arr)
    # 优化：使用np.linspace生成分位数，避免重复边界
    quantiles = np.unique(np.quantile(abs_fluct, np.linspace(0, 1, n_groups + 1)))
    # 处理分位数重复（数据分布过于集中）
    if len(quantiles) < n_groups + 1:
        quantiles = np.linspace(abs_fluct.min(), abs_fluct.max(), n_groups + 1)

    # 分组标签（适配任意n_groups，全中文）
    if n_groups == 2:
        group_labels = ["高基准能力组", "低基准能力组"]
    elif n_groups == 3:
        group_labels = ["高基准能力组", "中基准能力组", "低基准能力组"]
    elif n_groups == 4:
        group_labels = [
            "极高基准能力组",
            "高基准能力组",
            "低基准能力组",
            "极低基准能力组",
        ]
    else:
        group_labels = [f"第{i+1}组" for i in range(n_groups)]

    # 避免标签数与分位数不匹配
    if len(quantiles) - 1 != len(group_labels):
        group_labels = group_labels[: len(quantiles) - 1]

    groups = pd.cut(
        abs_fluct,
        bins=quantiles,
        labels=group_labels,
        include_lowest=True,
        duplicates="drop",  # 处理重复分位数
    )

    # 构建DataFrame用于绘图
    plot_df = pd.DataFrame({"Fluctuation": fluctuation_arr, "基准能力等级": groups})

    # 2. 绘制箱线图（修复palette警告）
    fig, ax = plt.subplots(figsize=(10, 6))
    sns.boxplot(
        x="基准能力等级",
        y="Fluctuation",
        hue="基准能力等级",  # 添加hue参数，匹配palette
        data=plot_df,
        palette="viridis",
        showfliers=False,  # 隐藏异常值点使图更整洁
        width=0.6,
        ax=ax,
        legend=False,  # 关闭图例（避免重复）
    )

    # 3. 添加统计显著性标注（单因素ANOVA，增加异常处理）
    try:
        # 提取各组数据（过滤空组）
        group_data = []
        valid_labels = []
        for lbl in group_labels:
            group_vals = plot_df[plot_df["基准能力等级"] == lbl]["Fluctuation"].values
            if len(group_vals) > 0:
                group_data.append(group_vals)
                valid_labels.append(lbl)

        if len(group_data) >= 2:
            f_stat, p_val = stats.f_oneway(*group_data)
            # 优化p值显示格式
            if p_val < 0.001:
                p_text = f"ANOVA: F={f_stat:.2f}, p<0.001"
            elif p_val < 0.01:
                p_text = f"ANOVA: F={f_stat:.2f}, p<0.01"
            else:
                p_text = f"ANOVA: F={f_stat:.2f}, p={p_val:.3f}"
            ax.text(
                0.5,
                0.95,
                p_text,
                transform=ax.transAxes,
                ha="center",
                va="top",
                bbox=dict(facecolor="white", alpha=0.8, edgecolor="none"),
                fontsize=10,
            )
    except Exception as e:
        print(f"ANOVA统计计算失败: {e}")

    # 4. 标注各组均值（替换为红色小三角，去掉数字）
    try:
        means = plot_df.groupby("基准能力等级", observed=False)["Fluctuation"].mean()
        for i, lbl in enumerate(means.index):
            # 绘制红色小三角标记均值位置
            ax.scatter(
                i,  # X轴位置（对应分组）
                means[lbl],  # Y轴位置（均值）
                color="crimson",  # 红色
                marker="^",  # 三角形状
                s=40,  # 小尺寸（可根据需要调整）
                zorder=10,  # 置于顶层，避免被遮挡
                edgecolors=None,
                linewidth=0,
            )
    except Exception as e:
        print(f"均值标注失败: {e}")

    # 核心修改1：去掉X轴标题，调整X轴刻度文字大小（和Y轴标签/刻度一致）
    ax.set_xlabel("")
    # 调整X轴刻度文字大小为14（和Y轴标签大小一致，set_paper_style中axes.labelsize=14）
    # 若想和Y轴刻度大小一致，改为labelsize=10（set_paper_style中ytick.labelsize=10）
    ax.tick_params(axis="x", labelsize=14)
    # 核心修改2：Y轴保留中文，标题改中文
    ax.set_ylabel("驾驶能力波动量 Afl")
    ax.set_title("不同基准能力组的驾驶能力波动量", pad=20)
    sns.despine(ax=ax)

    if save_path:
        save_path = Path(save_path)
        os.makedirs(save_path.parent, exist_ok=True)
        plt.savefig(save_path)
        print(f"分组箱线图已保存至: {save_path}")
    plt.close()


# ====================== 主调用函数（优化路径处理） ======================
def run_all_visualizations(
    result_pkl_path,
    output_dir="output/figs",
    config="config/capability_fluctuation.yaml",
):
    """
    一键运行所有可视化（增强鲁棒性+统一路径处理）

    Args:
        result_pkl_path: 之前保存的 driving_ability_result.pkl 路径
        output_dir: 图片输出目录
    """
    # 鲁棒性检查
    if not os.path.exists(result_pkl_path):
        print(f"结果文件不存在: {result_pkl_path}")
        return

    import pickle

    # 加载结果数据
    try:
        with open(result_pkl_path, "rb") as f:
            result = pickle.load(f)
    except Exception as e:
        print(f"加载结果文件失败: {e}")
        return

    # 检查必要的键是否存在
    if "features" not in result or "fluctuation" not in result:
        print("结果文件缺少必要的键（features/fluctuation）")
        return

    features_df = result["features"]
    fluctuation_arr = result["fluctuation"]
    fluctuation_sample = result["sample_fluctuations"]

    # 统一输出目录为Path对象
    output_dir = Path(output_dir)
    os.makedirs(output_dir, exist_ok=True)

    # 依次绘图
    plot_correlation_heatmap(features_df, save_path=output_dir / "Afl_corr_heatmap.png")
    plot_fluctuation_distribution(
        fluctuation_arr, save_path=output_dir / "Afl_fluctuation_dist.png"
    )
    plot_grouped_boxplot(
        fluctuation_sample, config, save_path=output_dir / "Afl_grouped_boxplot.png"
    )

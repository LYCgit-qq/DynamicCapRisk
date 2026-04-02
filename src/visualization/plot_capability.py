# D:\Local\DynamicCapRisk\src\visualization\plot_capability.py

import os

os.environ["OMP_NUM_THREADS"] = "1"

import logging
from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib as mpl
from sklearn.decomposition import PCA
import numpy as np
import seaborn as sns
from scipy import stats

RANDOM_STATE = 42

# ====================== 统一风格配置 ======================
# 主色调：学术蓝系
PRIMARY_COLOR   = "#2C5F8A"   # 主蓝（直方图柱体、主折线）
SECONDARY_COLOR = "#E07B39"   # 暖橙（均值线、标注三角）
ACCENT_COLOR    = "#4CAF82"   # 绿色（辅助标注）
LIGHT_FILL      = "#D6E8F5"   # 浅蓝填充（区间着色）
GRAY_FILL       = "#EBEBEB"   # 浅灰填充

# 分组色板（高/中/低 固定色，可扩展）
GROUP_PALETTE = {
    "高能力组":  "#2C5F8A",
    "中能力组":  "#4CAF82",
    "低能力组":  "#E07B39",
    "高基准能力组": "#2C5F8A",
    "中基准能力组": "#4CAF82",
    "低基准能力组": "#E07B39",
}
# 聚类散点图色板（最多9组）
CLUSTER_COLORS  = ["#2C5F8A", "#E07B39", "#4CAF82", "#A259C4",
                   "#E05C5C", "#F5C518", "#00B4D8", "#6D6875", "#B5E48C"]
CLUSTER_MARKERS = ["o", "s", "^", "D", "v", "p", "*", "h", "+"]

FIGURE_SIZE_WIDE   = (11, 6)    # 宽幅图（分布/箱线/折线）
FIGURE_SIZE_SQUARE = (10, 9)    # 方形图（热力图）
FIGURE_SIZE_3D     = (12, 10)   # 三维图
LINE_WIDTH   = 1.8
MARKER_SIZE  = 7
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
    """统一保存 + 关闭图形，并打印提示。"""
    if save_path:
        save_path = Path(save_path)
        os.makedirs(save_path.parent, exist_ok=True)
        fig.savefig(save_path, dpi=300, bbox_inches="tight", facecolor="white")
        print(f"{msg}已保存至: {save_path}")
    plt.close(fig)


# ====================== 基准能力评估模块 ======================

def plot_pca_visualization(X: pd.DataFrame, labels: pd.Series, output_dir: Path):
    """PCA 2D 降维可视化（统一风格）"""
    set_paper_style()
    pca    = PCA(n_components=2, random_state=RANDOM_STATE)
    X_pca  = pca.fit_transform(X)
    groups = labels.unique().tolist()

    fig, ax = plt.subplots(figsize=FIGURE_SIZE_WIDE)
    for i, group in enumerate(groups):
        idx = labels == group
        ax.scatter(
            X_pca[idx, 0], X_pca[idx, 1],
            color=CLUSTER_COLORS[i % len(CLUSTER_COLORS)],
            marker=CLUSTER_MARKERS[i % len(CLUSTER_MARKERS)],
            label=group, s=80, alpha=0.8, edgecolors="white", linewidths=0.4,
        )

    ax.set_xlabel(f"PC1（方差解释率: {pca.explained_variance_ratio_[0]:.2%}）")
    ax.set_ylabel(f"PC2（方差解释率: {pca.explained_variance_ratio_[1]:.2%}）")
    ax.set_title("基准驾驶能力聚类结果可视化（PCA 降维）")
    ax.legend(framealpha=0.9)
    ax.grid(alpha=GRID_ALPHA, linestyle="--")
    _apply_spine(ax)

    save_path = output_dir / "Ab_cluster_pca_visualization.png"
    _save_and_close(fig, save_path, "PCA 可视化图")
    logging.info("PCA可视化图已保存至: %s", save_path)


def plot_pca_3d_visualization(X: pd.DataFrame, labels: pd.Series, output_dir: Path):
    """PCA 3D 降维可视化（统一风格）"""
    set_paper_style()
    pca    = PCA(n_components=3, random_state=RANDOM_STATE)
    X_pca  = pca.fit_transform(X)
    groups = labels.unique().tolist()

    fig = plt.figure(figsize=FIGURE_SIZE_3D)
    ax  = fig.add_subplot(111, projection="3d")

    for i, group in enumerate(groups):
        idx = labels == group
        ax.scatter(
            X_pca[idx, 0], X_pca[idx, 1], X_pca[idx, 2],
            color=CLUSTER_COLORS[i % len(CLUSTER_COLORS)],
            marker=CLUSTER_MARKERS[i % len(CLUSTER_MARKERS)],
            label=group, s=80, alpha=0.8, edgecolors="white", linewidth=0.4,
        )

    ax.set_xlabel(f"PC1（{pca.explained_variance_ratio_[0]:.2%}）", fontsize=11)
    ax.set_ylabel(f"PC2（{pca.explained_variance_ratio_[1]:.2%}）", fontsize=11)
    ax.set_zlabel(f"PC3（{pca.explained_variance_ratio_[2]:.2%}）", fontsize=11)
    ax.set_title("基准驾驶能力聚类结果可视化（PCA 3D 降维）", pad=20)
    ax.legend(fontsize=10, loc="upper right", framealpha=0.9)
    ax.view_init(elev=15, azim=45)
    ax.grid(alpha=GRID_ALPHA)

    save_path = output_dir / "Ab_cluster_pca_3d_visualization.png"
    _save_and_close(fig, save_path, "3D PCA 可视化图")
    logging.info("3D PCA可视化图已保存至: %s", save_path)


def plot_abc_distribution(abc_df: pd.DataFrame, output_dir: Path):
    """
    Abc 分布可视化，生成三张图：

    Abc_dist_histogram.png  — 直方图 + KDE + 正态性检验结果
    Abc_dist_by_group.png   — 按能力等级分组的小提琴图 + 散点
    Abc_dist_scatter.png    — 按被试 ID 排列的散点图（能力等级着色）
    """
    from scipy.stats import shapiro  # 修复：缺失正态性检验导入
    set_paper_style()
    os.makedirs(output_dir, exist_ok=True)
    id_col  = "被试ID" if "被试ID" in abc_df.columns else "实验ID"
    abc_arr = abc_df["Abc"].values
    mean_v  = abc_arr.mean()
    std_v   = abc_arr.std()

    # ── 图1：直方图 + KDE ──────────────────────────────────────────
    fig, ax = plt.subplots(figsize=(10, 6))
    sns.histplot(abc_arr, bins=12, kde=True, color=PRIMARY_COLOR,
                 edgecolor="white", linewidth=0.4, alpha=0.75, ax=ax,
                 line_kws={"linewidth": 1.8, "color": PRIMARY_COLOR})
    ax.axvline(mean_v, color=SECONDARY_COLOR, linestyle="--", linewidth=1.8,
               label=f"均值 {mean_v:.4f}")
    ax.axvspan(mean_v - std_v, mean_v + std_v, color=LIGHT_FILL, alpha=0.45,  # 修复：LIGHT_FILL
               label=f"±1σ  [{mean_v - std_v:.4f}, {mean_v + std_v:.4f}]")

    # Shapiro-Wilk 正态性检验
    if len(abc_arr) >= 3:
        stat_w, p_sw = shapiro(abc_arr)
        sw_text = f"Shapiro-Wilk: W={stat_w:.4f}, p={p_sw:.4f}"
        normal_hint = "（近似正态 ✓）" if p_sw > 0.05 else "（拒绝正态）"
        ax.text(0.97, 0.95, f"{sw_text}\n{normal_hint}",
                transform=ax.transAxes, ha="right", va="top", fontsize=10,
                bbox=dict(facecolor="white", edgecolor="#CCCCCC",
                          alpha=0.85, boxstyle="round,pad=0.4"))

    ax.set_xlabel("个体化基准能力值 $A_{bc}$")
    ax.set_ylabel("频次")
    ax.set_title("个体化基准驾驶能力值 $A_{bc}$ 分布")
    ax.legend(framealpha=0.9, fontsize=10)
    sns.despine(ax=ax)
    path1 = output_dir / "Abc_dist_histogram.png"
    fig.savefig(path1, dpi=300, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    logging.info("Abc 直方图已保存：%s", path1)

    # ── 图2：分组小提琴图 + 散点 ───────────────────────────────────
    ordered = [g for g in ["高能力组", "中能力组", "低能力组"]
               if g in abc_df["能力等级"].values]
    if ordered:
        abc_df["能力等级"] = pd.Categorical(abc_df["能力等级"],
                                            categories=ordered, ordered=True)
        palette = {g: GROUP_PALETTE.get(g, PRIMARY_COLOR) for g in ordered}  # 修复：GROUP_PALETTE

        fig, ax = plt.subplots(figsize=(9, 6))
        sns.violinplot(x="能力等级", y="Abc", data=abc_df,
                       palette=palette, inner=None, width=0.7,
                       linewidth=1.2, ax=ax, hue="能力等级", legend=False)
        sns.stripplot(x="能力等级", y="Abc", data=abc_df,
                      color="white", edgecolor="gray", linewidth=0.6,
                      size=6, jitter=True, ax=ax, zorder=3)
        # 均值标注
        for i, grp in enumerate(ordered):
            mv = abc_df[abc_df["能力等级"] == grp]["Abc"].mean()
            ax.scatter(i, mv, color=SECONDARY_COLOR, marker="^",
                       s=80, zorder=5, linewidths=0)

        ax.set_xlabel("")
        ax.set_ylabel("个体化基准能力值 $A_{bc}$")
        ax.set_title("不同能力等级的 $A_{bc}$ 分布（小提琴图）")
        ax.tick_params(axis="x", labelsize=13)
        ax.grid(axis="y", alpha=0.3, linestyle="--")
        sns.despine(ax=ax)
        path2 = output_dir / "Abc_dist_by_group.png"
        fig.savefig(path2, dpi=300, bbox_inches="tight", facecolor="white")
        plt.close(fig)
        logging.info("Abc 分组小提琴图已保存：%s", path2)

    # ── 图3：被试级散点图 ──────────────────────────────────────────
    plot_df = abc_df.sort_values(id_col).copy()
    fig, ax = plt.subplots(figsize=(11, 5))
    for grp in ordered:
        sub = plot_df[plot_df["能力等级"] == grp]
        ax.scatter(sub[id_col], sub["Abc"],
                   color=GROUP_PALETTE.get(grp, PRIMARY_COLOR), label=grp,  # 修复
                   s=70, alpha=0.85, edgecolors="white", linewidths=0.4, zorder=3)
    ax.axhline(mean_v, color=SECONDARY_COLOR, linestyle="--",
               linewidth=1.4, label=f"总均值 {mean_v:.4f}")
    ax.set_xlabel(f"{'被试' if id_col == '被试ID' else '实验'} ID")
    ax.set_ylabel("个体化基准能力值 $A_{bc}$")
    ax.set_title("各被试个体化基准驾驶能力值 $A_{bc}$")
    ax.legend(framealpha=0.9, fontsize=10)
    ax.grid(alpha=0.3, linestyle="--")
    sns.despine(ax=ax)
    path3 = output_dir / "Abc_dist_scatter.png"
    fig.savefig(path3, dpi=300, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    logging.info("Abc 散点图已保存：%s", path3)


def plot_cluster_metrics(eval_df: pd.DataFrame, output_dir: Path, best_k: int = None, plot_ch: bool = False):
    """聚类评价指标可视化（SC / CH / DBI 三联图，统一风格）"""
    set_paper_style()

    k_vals   = eval_df.index.values
    sc_vals  = eval_df["轮廓系数SC"].values
    ch_vals  = eval_df["CH指数"].values
    dbi_vals = eval_df["DBI指数"].values

    # 动态配置指标：控制是否绘制CH
    metric_cfg = [("轮廓系数 SC", sc_vals, "o")]
    if plot_ch:
        metric_cfg.append(("CH 指数", ch_vals, "o"))
    metric_cfg.append(("DBI 指数", dbi_vals, "o"))

    # 动态设置子图数量
    n_cols = len(metric_cfg)
    fig, axes = plt.subplots(1, n_cols, figsize=(18, 5) if plot_ch else (12, 5))

    for ax, (ylabel, vals, marker) in zip(axes, metric_cfg):
        ax.plot(k_vals, vals, marker=marker, linewidth=LINE_WIDTH,
                markersize=MARKER_SIZE, color=PRIMARY_COLOR)
        if best_k is not None and best_k in k_vals:
            idx = list(k_vals).index(best_k)
            ax.scatter(best_k, vals[idx], color=SECONDARY_COLOR,
                       s=120, zorder=5, label=f"最优 k={best_k}", marker="*")
            ax.legend(framealpha=0.9, fontsize=14)
        
        ax.set_xlabel("聚类数目 k", fontsize=20)
        ax.set_ylabel(ylabel, fontsize=20)
        ax.tick_params(axis='both', labelsize=14)
        
        ax.grid(alpha=GRID_ALPHA, linestyle="--")
        _apply_spine(ax)

    plt.tight_layout()
    save_path = output_dir / "Ab_cluster_metrics_visualization.png"
    _save_and_close(fig, save_path, "聚类评价指标图")
    logging.info("聚类评价指标图已保存至: %s", save_path)


# ====================== 波动量计算模块 ======================

def plot_correlation_heatmap(features_df, save_path=None):
    """特征 Pearson 相关性热力图（统一风格）"""
    if features_df.empty or len(features_df.columns) < 2:
        print("特征数据为空或特征数 < 2，跳过相关性热力图绘制")
        return

    feat_name_map = {
        "steering_angle":    "方向盘转角",
        "steering_velocity": "方向盘角速度",
        "brake_pedal":       "制动踏板开度",
        "throttle_pedal":    "油门踏板开度",
        "longitudinal_accel":"纵向加速度",
        "lateral_offset":    "横向偏移量",
        "lateral_accel":     "横向加速度",
        "vehicle_speed":     "车速",
        "gaze_dispersion":   "注视点分散度",
        "blink_frequency":   "眨眼频率",
        "hrv":               "心率变异性",
        "bvp":               "血容量脉搏",
        "ecg":               "心电信号",
        "resp":              "呼吸信号",
    }

    set_paper_style()
    corr_mat = features_df.corr()
    corr_mat.index   = [feat_name_map.get(n, n) for n in corr_mat.index]
    corr_mat.columns = [feat_name_map.get(n, n) for n in corr_mat.columns]

    n_feat   = len(features_df.columns)
    fig_size = (max(10, n_feat * 0.85), max(8, n_feat * 0.75))
    fig, ax  = plt.subplots(figsize=fig_size)

    # 使用与主色调一致的冷暖色系
    cmap = sns.diverging_palette(220, 20, as_cmap=True)
    sns.heatmap(
        corr_mat, annot=True, fmt=".2f", cmap=cmap,
        vmin=-1, vmax=1, square=True,
        cbar_kws={"shrink": 0.8, "label": "Pearson 相关系数"},
        annot_kws={"size": 9}, linewidths=0.3, linecolor="#CCCCCC",
        ax=ax,
    )
    ax.set_title("驾驶特征 Pearson 相关性热力图", pad=16)
    ax.tick_params(axis="both", labelsize=10)

    _save_and_close(fig, save_path, "相关性热力图")


def plot_fluctuation_distribution(fluctuation_arr, save_path=None):
    """
    驾驶能力波动量分布图：直方图 + KDE + 统计标注（统一风格）
    """
    if len(fluctuation_arr) == 0:
        print("波动量数组为空，跳过分布图绘制")
        return

    set_paper_style()
    mean_val = np.mean(fluctuation_arr)
    std_val  = np.std(fluctuation_arr)

    fig, ax = plt.subplots(figsize=FIGURE_SIZE_WIDE)

    # 直方图 + KDE
    sns.histplot(
        fluctuation_arr, bins=50, kde=True,
        color=PRIMARY_COLOR, edgecolor="white", linewidth=0.4,
        alpha=0.75, ax=ax,
        line_kws={"linewidth": LINE_WIDTH, "color": PRIMARY_COLOR},
    )

    # ±1σ 区间
    ax.axvspan(mean_val - std_val, mean_val + std_val,
               color=LIGHT_FILL, alpha=0.5,
               label=f"±1σ  [{mean_val - std_val:.3f}, {mean_val + std_val:.3f}]")

    # 论文目标区间 [-0.05, 0.05]
    paper_min, paper_max = -0.05, 0.05
    in_ratio = np.mean((fluctuation_arr >= paper_min) & (fluctuation_arr <= paper_max))
    ax.axvspan(paper_min, paper_max, color=GRAY_FILL, alpha=0.5,
               label=f"目标区间 [{paper_min}, {paper_max}]（占比 {in_ratio:.1%}）")

    # 均值竖线
    ax.axvline(mean_val, color=SECONDARY_COLOR, linestyle="--",
               linewidth=LINE_WIDTH, label=f"均值 {mean_val:.4f}")

    # 文字统计框
    stats_text = (
        f"均值: {mean_val:.4f}\n"
        f"标准差: {std_val:.4f}\n"
        f"目标区间占比: {in_ratio:.1%}"
    )
    ax.text(0.97, 0.95, stats_text, transform=ax.transAxes,
            ha="right", va="top", fontsize=10,
            bbox=dict(facecolor="white", edgecolor="#CCCCCC", alpha=0.85, boxstyle="round,pad=0.4"))

    ax.set_xlabel("驾驶能力波动量 $A_{fl}$")
    ax.set_ylabel("频次")
    ax.set_title("驾驶能力波动量分布")
    ax.legend(loc="upper left", framealpha=0.9, fontsize=10)
    _apply_spine(ax)

    _save_and_close(fig, save_path, "波动量分布图")


def plot_grouped_boxplot(fluctuation_sample, config, save_path=None):
    """
    按真实能力等级绘制波动量箱线图（统一风格）
    """
    if len(fluctuation_sample) != 67:
        raise ValueError(f"fluctuation_sample 长度必须为 67，当前为 {len(fluctuation_sample)}")

    for k in ["subject_exp_map_csv", "ability_label_csv"]:
        if k not in config:
            raise KeyError(f"config 缺少键：{k}")
        if not os.path.exists(config[k]):
            raise FileNotFoundError(f"CSV 文件不存在：{config[k]}")

    # 加载映射
    map_df   = pd.read_csv(config["subject_exp_map_csv"])
    map_df["实验ID"] = map_df["实验ID"].astype(int)

    label_df = pd.read_csv(config["ability_label_csv"])
    label_df.rename(columns={label_df.columns[0]: "被试ID"}, inplace=True)
    label_df["被试ID"]  = label_df["被试ID"].astype(int)
    label_df["能力等级"] = label_df["能力等级"].astype(str)

    merge_df = pd.merge(map_df, label_df, on="被试ID", how="left")
    merge_df = merge_df[merge_df["能力等级"].isin(["高能力组", "中能力组", "低能力组"])]
    expid_to_grade = dict(zip(merge_df["实验ID"], merge_df["能力等级"]))

    grade_flucts = {"高能力组": [], "中能力组": [], "低能力组": []}
    for exp_id in range(67):
        if exp_id not in expid_to_grade:
            continue
        grade    = expid_to_grade[exp_id]
        fluct_arr = fluctuation_sample[exp_id]
        if not isinstance(fluct_arr, (np.ndarray, list)) or len(fluct_arr) == 0:
            continue
        try:
            fluct_arr = np.array(fluct_arr, dtype=np.float64)
            fluct_arr = fluct_arr[np.isfinite(fluct_arr)]
        except Exception:
            continue
        if len(fluct_arr) > 0:
            grade_flucts[grade].extend(fluct_arr.tolist())

    grade_flucts = {k: v for k, v in grade_flucts.items() if len(v) > 0}
    if not grade_flucts:
        print("无有效波动量数据可绘制")
        return

    print("最终有效分组数据量：")
    for grade, vals in grade_flucts.items():
        print(f"  {grade}: {len(vals)} 个值")

    ordered_grades = [g for g in ["高能力组", "中能力组", "低能力组"] if g in grade_flucts]

    # 构建绘图 DataFrame
    rows = [{"驾驶能力波动量 $A_{fl}$": v, "真实能力等级": g}
            for g, vals in grade_flucts.items() for v in vals]
    plot_df = pd.DataFrame(rows)
    plot_df["真实能力等级"] = pd.Categorical(
        plot_df["真实能力等级"], categories=ordered_grades, ordered=True
    )
    palette = {g: GROUP_PALETTE.get(g, PRIMARY_COLOR) for g in ordered_grades}

    set_paper_style()
    fig, ax = plt.subplots(figsize=FIGURE_SIZE_WIDE)

    sns.boxplot(
        x="真实能力等级", y="驾驶能力波动量 $A_{fl}$",
        hue="真实能力等级", data=plot_df,
        palette=palette, showfliers=False, width=0.55,
        linewidth=1.2, ax=ax, legend=False,
    )

    # ANOVA 标注
    if len(ordered_grades) >= 2:
        try:
            f_stat, p_val = stats.f_oneway(*[grade_flucts[g] for g in ordered_grades])
            p_text = (f"ANOVA: F = {f_stat:.2f}, p < 0.001" if p_val < 0.001
                      else f"ANOVA: F = {f_stat:.2f}, p = {p_val:.3f}")
            ax.text(0.5, 0.97, p_text, transform=ax.transAxes,
                    ha="center", va="top", fontsize=10,
                    bbox=dict(facecolor="white", edgecolor="#CCCCCC", alpha=0.85, boxstyle="round,pad=0.4"))
        except Exception as e:
            print(f"ANOVA 计算失败: {e}")

    # 均值三角标注
    for i, grade in enumerate(ordered_grades):
        mean_v = np.mean(grade_flucts[grade])
        ax.scatter(i, mean_v, color=SECONDARY_COLOR, marker="^",
                   s=60, zorder=10, linewidths=0)

    ax.set_xlabel("")
    ax.set_ylabel("驾驶能力波动量 $A_{fl}$")
    ax.set_title("不同真实能力等级的驾驶能力波动量")
    ax.tick_params(axis="x", labelsize=13)
    ax.grid(axis="y", alpha=GRID_ALPHA, linestyle="--")
    _apply_spine(ax)

    _save_and_close(fig, save_path, "箱线图")


def plot_grouped_boxplot_abs(fluctuation_arr, n_groups=3, save_path=None):
    """
    按波动量绝对值分位数分组的箱线图（统一风格）
    """
    if len(fluctuation_arr) == 0:
        print("波动量数组为空，跳过分组箱线图绘制")
        return
    if not (2 <= n_groups <= 5):
        n_groups = 3

    abs_fluct = np.abs(fluctuation_arr)
    quantiles = np.unique(np.quantile(abs_fluct, np.linspace(0, 1, n_groups + 1)))
    if len(quantiles) < n_groups + 1:
        quantiles = np.linspace(abs_fluct.min(), abs_fluct.max(), n_groups + 1)

    label_presets = {
        2: ["高基准能力组", "低基准能力组"],
        3: ["高基准能力组", "中基准能力组", "低基准能力组"],
        4: ["极高基准能力组", "高基准能力组", "低基准能力组", "极低基准能力组"],
    }
    group_labels = label_presets.get(n_groups, [f"第{i+1}组" for i in range(n_groups)])
    group_labels = group_labels[: len(quantiles) - 1]

    groups = pd.cut(abs_fluct, bins=quantiles, labels=group_labels,
                    include_lowest=True, duplicates="drop")
    plot_df = pd.DataFrame({"Fluctuation": fluctuation_arr, "基准能力等级": groups})
    palette  = {g: GROUP_PALETTE.get(g, PRIMARY_COLOR) for g in group_labels}

    set_paper_style()
    fig, ax = plt.subplots(figsize=FIGURE_SIZE_WIDE)

    sns.boxplot(
        x="基准能力等级", y="Fluctuation",
        hue="基准能力等级", data=plot_df,
        palette=palette, showfliers=False, width=0.55,
        linewidth=1.2, ax=ax, legend=False,
    )

    # ANOVA
    try:
        group_data = [plot_df[plot_df["基准能力等级"] == lbl]["Fluctuation"].values
                      for lbl in group_labels if len(plot_df[plot_df["基准能力等级"] == lbl]) > 0]
        if len(group_data) >= 2:
            f_stat, p_val = stats.f_oneway(*group_data)
            p_text = (f"ANOVA: F = {f_stat:.2f}, p < 0.001" if p_val < 0.001
                      else f"ANOVA: F = {f_stat:.2f}, p = {p_val:.3f}")
            ax.text(0.5, 0.97, p_text, transform=ax.transAxes,
                    ha="center", va="top", fontsize=10,
                    bbox=dict(facecolor="white", edgecolor="#CCCCCC", alpha=0.85, boxstyle="round,pad=0.4"))
    except Exception as e:
        print(f"ANOVA 计算失败: {e}")

    # 均值三角
    try:
        means = plot_df.groupby("基准能力等级", observed=False)["Fluctuation"].mean()
        for i, lbl in enumerate(means.index):
            if np.isfinite(means[lbl]):
                ax.scatter(i, means[lbl], color=SECONDARY_COLOR, marker="^",
                           s=60, zorder=10, linewidths=0)
    except Exception as e:
        print(f"均值标注失败: {e}")

    ax.set_xlabel("")
    ax.set_ylabel("驾驶能力波动量 $A_{fl}$")
    ax.set_title("不同基准能力组的驾驶能力波动量")
    ax.tick_params(axis="x", labelsize=13)
    ax.grid(axis="y", alpha=GRID_ALPHA, linestyle="--")
    _apply_spine(ax)

    _save_and_close(fig, save_path, "分组箱线图")


# ====================== 主调用函数 ======================

def run_all_visualizations(
    result_pkl_path,
    output_dir="output/figs",
    config="config/capability_fluctuation.yaml",
):
    """一键运行所有波动量相关可视化"""
    if not os.path.exists(result_pkl_path):
        print(f"结果文件不存在: {result_pkl_path}")
        return

    import pickle
    try:
        with open(result_pkl_path, "rb") as f:
            result = pickle.load(f)
    except Exception as e:
        print(f"加载结果文件失败: {e}")
        return

    if "features" not in result or "fluctuation" not in result:
        print("结果文件缺少必要的键（features / fluctuation）")
        return

    features_df        = result["features"]
    fluctuation_arr    = result["fluctuation"]
    fluctuation_sample = result["sample_fluctuations"]

    output_dir = Path(output_dir)
    os.makedirs(output_dir, exist_ok=True)

    plot_correlation_heatmap(
        features_df, save_path=output_dir / "Afl_corr_heatmap.png"
    )
    plot_fluctuation_distribution(
        fluctuation_arr, save_path=output_dir / "Afl_fluctuation_dist.png"
    )
    plot_grouped_boxplot(
        fluctuation_sample, config, save_path=output_dir / "Afl_grouped_boxplot.png"
    )


# ====================== 动态驾驶能力模块 ======================

def visualize_Ad_results(all_dynamic_cap, dynamic_cap_sample, exp_group_df, config):
    """
    动态驾驶能力可视化：
      图1 — 全局分布直方图
      图2 — 32 名驾驶人均值散点图
    （统一风格）
    """
    set_paper_style()

    full_paths  = config["full_paths"]
    plot_params = config["plot"]
    out_dir     = full_paths["output_dir"]

    ad_mean = np.mean(all_dynamic_cap)

    # ---- 图1：全局分布 ----
    fig, ax = plt.subplots(figsize=FIGURE_SIZE_WIDE)
    sns.histplot(all_dynamic_cap, bins=50, kde=True,
                 color=PRIMARY_COLOR, edgecolor="white", linewidth=0.4,
                 alpha=0.75, ax=ax,
                 line_kws={"linewidth": LINE_WIDTH, "color": PRIMARY_COLOR})
    ax.axvline(ad_mean, color=SECONDARY_COLOR, linestyle="--",
               linewidth=LINE_WIDTH, label=f"均值 = {ad_mean:.2f}")
    for thresh in [0.55, 0.90]:
        ax.axvline(thresh, color=ACCENT_COLOR, linestyle=":",
                   linewidth=1.2, label=f"阈值 {thresh}")
    ax.set_xlabel("动态驾驶能力量化值 $A_d$")
    ax.set_ylabel("频次")
    ax.set_title("动态驾驶能力量化值整体分布")
    ax.legend(framealpha=0.9)
    ax.grid(axis="y", alpha=GRID_ALPHA, linestyle="--")
    _apply_spine(ax)
    _save_and_close(fig, os.path.join(out_dir, "Ad_global_distribution.png"), "Ad 分布图")

    # ---- 图2：32 名驾驶人均值散点图 ----
    subject_ad_list = []
    for exp_id in range(len(dynamic_cap_sample)):
        exp_row = exp_group_df[exp_group_df["实验ID"] == exp_id]
        if exp_row.empty:
            continue
        subject_id    = int(exp_row["被试ID"].iloc[0])
        ability_group = exp_row["能力等级"].iloc[0]
        for ad in dynamic_cap_sample[exp_id]:
            subject_ad_list.append({"被试ID": subject_id, "能力等级": ability_group, "Ad": ad})

    subject_ad_df = pd.DataFrame(subject_ad_list)
    subject_stats = (
        subject_ad_df.groupby("被试ID")
        .agg(Ad_mean=("Ad", "mean"), 能力等级=("能力等级", lambda x: x.mode().iloc[0]))
        .reset_index()
        .sort_values("被试ID")
    )

    fig, ax = plt.subplots(figsize=(11, 5))
    group_order = ["高能力组", "中能力组", "低能力组"]
    for group in group_order:
        sub = subject_stats[subject_stats["能力等级"] == group]
        if not sub.empty:
            ax.scatter(sub["被试ID"], sub["Ad_mean"],
                       color=GROUP_PALETTE.get(group, PRIMARY_COLOR),
                       label=group, s=75, alpha=0.85, edgecolors="white", linewidths=0.4)

    ax.set_xlabel("驾驶人编号（被试 ID）")
    ax.set_ylabel("动态驾驶能力均值 $A_d$")
    ax.set_title("32 名驾驶人动态驾驶能力均值分布")
    ax.legend(title="基准能力等级", framealpha=0.9)
    ax.grid(alpha=GRID_ALPHA, linestyle="--")
    ax.set_xticks(range(1, 33))
    _apply_spine(ax)
    _save_and_close(fig, os.path.join(out_dir, "Ad_32_subjects_mean_distribution.png"), "Ad 均值散点图")

    print("\n📸 已生成：\n  1. 动态能力全局分布直方图\n  2. 32 名驾驶人动态能力均值分布图")
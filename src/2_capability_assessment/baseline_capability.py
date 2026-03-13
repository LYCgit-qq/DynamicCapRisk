# D:\Local\DynamicCapRisk\src\2_capability_assessment\baseline_capability.py

import os
os.environ["OMP_NUM_THREADS"] = "1"
import argparse
import logging
import yaml
from pathlib import Path
from typing import Tuple, Dict, Any

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import rankdata, norm as scipy_norm, shapiro
from sklearn.cluster import KMeans
from sklearn.metrics import (
    silhouette_score,
    calinski_harabasz_score,
    davies_bouldin_score,
)
from src.visualization.plot_capability import (
    plot_pca_visualization,
    plot_pca_3d_visualization,
    plot_cluster_metrics,
)

BASE_DIR = Path(__file__).resolve().parents[2]


# ---------------------------------------------------------------------------
# 配置读取工具函数
# ---------------------------------------------------------------------------
def load_yaml_config(config_path) -> Dict[str, Any]:
    config_path = Path(config_path)
    if not config_path.exists():
        raise FileNotFoundError(f"配置文件不存在: {config_path.resolve()}")
    with open(config_path, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)
    if "cluster_range" in config and isinstance(config["cluster_range"], list):
        s, e = config["cluster_range"]
        config["cluster_range"] = range(s, e)
    for path_key in ["input_path", "output_path"]:
        if path_key in config:
            config[path_key] = BASE_DIR / config[path_key]
    # 可选路径：被试-实验映射（用于 Abc 计算）
    if "subject_exp_map_csv" in config:
        config["subject_exp_map_csv"] = BASE_DIR / config["subject_exp_map_csv"]
    return config


def merge_configs(default_config: Dict[str, Any], cli_args: argparse.Namespace) -> Dict[str, Any]:
    merged = default_config.copy()
    if cli_args.input  is not None: merged["input_path"]    = Path(cli_args.input)
    if cli_args.output is not None: merged["output_path"]   = Path(cli_args.output)
    if cli_args.best_k        is not None: merged["best_k"]        = cli_args.best_k
    if cli_args.max_iter      is not None: merged["max_iter"]      = cli_args.max_iter
    if cli_args.random_state  is not None: merged["random_state"]  = cli_args.random_state
    return merged


# ---------------------------------------------------------------------------
# 核心功能函数
# ---------------------------------------------------------------------------
def evaluate_clustering(X: pd.DataFrame, k: int, config: Dict[str, Any]) -> Tuple[float, float, float]:
    kmeans = KMeans(
        n_clusters=k, init="k-means++",
        max_iter=config["max_iter"], tol=config["tol"],
        random_state=config["random_state"],
    )
    labels = kmeans.fit_predict(X)
    return (silhouette_score(X, labels),
            calinski_harabasz_score(X, labels),
            davies_bouldin_score(X, labels))


def find_best_k(X: pd.DataFrame, output_dir: Path, config: Dict[str, Any]) -> Tuple[int, pd.DataFrame]:
    results = []
    for k in config["cluster_range"]:
        sc, ch, dbi = evaluate_clustering(X, k, config)
        results.append({"k": k, "轮廓系数SC": sc, "CH指数": ch, "DBI指数": dbi})
    eval_df = pd.DataFrame(results).set_index("k")
    eval_df.to_csv(output_dir / "Ab_cluster_evaluation.csv", encoding="utf-8-sig")
    logging.info("=" * 60)
    logging.info("不同聚类数目下的评价指标：\n%s", eval_df.round(2).to_string())
    logging.info("=" * 60)
    return config["best_k"], eval_df


def cluster_analysis(X: pd.DataFrame, k: int, config: Dict[str, Any]):
    kmeans = KMeans(
        n_clusters=k, init="k-means++",
        max_iter=config["max_iter"], tol=config["tol"],
        random_state=config["random_state"],
    )
    labels = kmeans.fit_predict(X)
    centers = pd.DataFrame(kmeans.cluster_centers_, columns=X.columns)

    cluster_means = X.groupby(labels).mean()
    sort_score    = cluster_means[config["key_indicators"]].mean(axis=1)
    cluster_order = sort_score.sort_values(ascending=False).index.tolist()

    def _get_names(k):
        if k == 2: return ["高能力组", "低能力组"]
        if k == 3: return ["高能力组", "中能力组", "低能力组"]
        if k == 4: return ["高能力组", "中高能力组", "中低能力组", "低能力组"]
        names = [f"能力组_{i+1}" for i in range(k)]
        names[0] += "（高）"; names[-1] += "（低）"
        return names

    cluster_names  = _get_names(k)
    label_mapping  = {old: new for old, new in zip(cluster_order, cluster_names)}
    labels_renamed = pd.Series(labels, index=X.index).map(label_mapping)
    centers        = centers.reindex(cluster_order)
    centers.index  = cluster_names
    centers["综合得分"] = centers[config["key_indicators"]].mean(axis=1)

    return centers, labels_renamed, kmeans


def quantify_benchmark_ability(centers: pd.DataFrame, config: Dict[str, Any]) -> pd.DataFrame:
    """聚类中心级 Ab（仅 3 个值，原始逻辑不变）"""
    cn = (centers - centers.min().min()) / (centers.max().max() - centers.min().min())
    composite_score   = cn.mean(axis=1)
    benchmark_ability = 0.55 + 0.4 * composite_score
    return pd.DataFrame({"综合得分S^c": composite_score, "基准能力值A_b": benchmark_ability})


# ---------------------------------------------------------------------------
# ★ 新增：个体化基准能力 Abc 计算
# ---------------------------------------------------------------------------
def compute_individualized_abc(
    X: pd.DataFrame,
    labels_renamed: pd.Series,
    centers: pd.DataFrame,
    config: Dict[str, Any],
) -> pd.DataFrame:
    """
    计算每名被试的个体化基准驾驶能力值 Abc。

    原始 Ab 每个聚类只有一个值，无法区分同组内的个体差异。
    Abc 在保留聚类结构（高/中/低组的宏观排序）的前提下，引入
    个体特征信息，使 32 名被试各自拥有细微差别的量化值，
    且整体分布尽可能服从正态分布。

    计算流程
    --------
    1. 逐实验计算综合能力得分 S_raw
       对 key_indicators 特征做跨全局的 Min-Max 归一化，
       然后等权求和，得到每条实验记录的原始综合得分。

    2. 按被试聚合（67 条实验 → 32 名被试）
       若配置中提供 subject_exp_map_csv，则按映射聚合；
       否则将每行视为独立被试（实验级 Abc）。

    3. 组内排名保序 + 全局秩归一化
       先在每个聚类组内按 S_raw 排名，保证高能力组的
       Abc > 中能力组 > 低能力组；再对全局 32 个值做
       秩归一化 → 均匀分布 → 正态分布（Box-Rank 变换）。

    4. 线性缩放到 [0.55, 0.95]
       与 Ab 量程一致，方便后续联合使用。

    参数
    ----
    X              : 标准化特征 DataFrame，形状 (67, n_features)
    labels_renamed : 每条实验记录的能力组标签，Series(67,)
    centers        : 聚类中心 DataFrame，含 "综合得分" 列
    config         : 配置字典

    返回
    ----
    abc_df : DataFrame，列为 [被试ID / 实验ID, 能力等级, S_raw, Abc]，
             按 Abc 降序排列
    """
    key_cols = config["key_indicators"]

    # ── Step 1：每条实验记录的原始综合得分 ──────────────────────────
    X_sub  = X[key_cols].copy()
    minv   = X_sub.min()
    maxv   = X_sub.max()
    denom  = (maxv - minv).replace(0, 1)
    X_norm = (X_sub - minv) / denom          # 全局 Min-Max，形状 (67, n_key)
    S_exp  = X_norm.mean(axis=1)             # 等权综合得分，形状 (67,)

    # ── Step 2：聚合到被试级 ─────────────────────────────────────────
    map_path = config.get("subject_exp_map_csv")
    if map_path and Path(map_path).exists():
        map_df   = pd.read_csv(map_path)
        map_df["实验ID"] = map_df["实验ID"].astype(int)
        map_df["被试ID"] = map_df["被试ID"].astype(int)

        # 构建实验→被试的映射
        exp_to_subj = dict(zip(map_df["实验ID"], map_df["被试ID"]))
        rows = []
        for exp_id, s_val in S_exp.items():
            subj_id = exp_to_subj.get(exp_id)
            if subj_id is None:
                continue
            rows.append({
                "被试ID": subj_id,
                "实验ID": exp_id,
                "能力等级": labels_renamed.loc[exp_id],
                "S_raw": s_val,
            })
        exp_df = pd.DataFrame(rows)

        # 每名被试取均值（多次实验取平均，代表稳定能力水平）
        subj_df = (
            exp_df.groupby("被试ID")
            .agg(S_raw=("S_raw", "mean"),
                 能力等级=("能力等级", lambda x: x.mode().iloc[0]))
            .reset_index()
        )
        id_col = "被试ID"
    else:
        # 无映射文件：实验级 Abc（67 条）
        logging.warning("未找到 subject_exp_map_csv，将生成实验级 Abc（67 条）")
        subj_df = pd.DataFrame({
            "实验ID": S_exp.index,
            "S_raw":  S_exp.values,
            "能力等级": labels_renamed.values,
        })
        id_col = "实验ID"

    n = len(subj_df)

    # ── Step 3：组内排名保序 + 全局正态化 ───────────────────────────
    # 3a. 获取各组的 Ab 中心值，用于锚定组间位置
    # centers 的索引是能力组名，"综合得分" 列已归一化
    group_ab = {}
    for grp in subj_df["能力等级"].unique():
        if grp in centers.index:
            # 与 quantify_benchmark_ability 相同的映射：0.55 + 0.4 * 归一化得分
            raw_score = centers.loc[grp, "综合得分"]
            cs        = centers["综合得分"]
            norm_s    = (raw_score - cs.min()) / (cs.max() - cs.min() + 1e-12)
            group_ab[grp] = 0.55 + 0.4 * norm_s
        else:
            group_ab[grp] = 0.75   # fallback

    # 3b. 组内对 S_raw 排秩，加上组间偏移量形成保序的全局得分
    #     偏移量 = 各组 Ab 作为基准，使高能力组的最低分 > 中能力组的最高分
    group_order = sorted(subj_df["能力等级"].unique(),
                         key=lambda g: group_ab.get(g, 0.75), reverse=True)
    n_groups    = len(group_order)

    subj_df = subj_df.copy()
    subj_df["S_ordered"] = 0.0

    for rank_g, grp in enumerate(group_order):
        mask      = subj_df["能力等级"] == grp
        n_g       = mask.sum()
        if n_g == 0:
            continue
        # 组内归一化排名：[0, 1]
        intra_rank = rankdata(subj_df.loc[mask, "S_raw"]) / (n_g + 1)
        # 分配到全局区间段（每组占 1/n_groups，从高到低）
        seg_high   = 1.0 - rank_g       / n_groups
        seg_low    = 1.0 - (rank_g + 1) / n_groups
        subj_df.loc[mask, "S_ordered"] = seg_low + intra_rank * (seg_high - seg_low)

    # 3c. Box-Rank 变换：均匀 → 正态（截断避免 ±∞）
    uniform_vals = subj_df["S_ordered"].values.clip(1e-6, 1 - 1e-6)
    normal_vals  = scipy_norm.ppf(uniform_vals)           # 标准正态

    # 3d. 缩放到 [0.55, 0.95]（与 Ab 量程一致）
    #     先将正态值映射到 [0, 1]，再线性拉伸
    z_min, z_max = normal_vals.min(), normal_vals.max()
    if z_max > z_min:
        abc_01 = (normal_vals - z_min) / (z_max - z_min)
    else:
        abc_01 = np.full_like(normal_vals, 0.5)
    subj_df["Abc"] = 0.55 + 0.40 * abc_01

    # ── Step 4：整理输出 ─────────────────────────────────────────────
    abc_df = subj_df[[id_col, "能力等级", "S_raw", "Abc"]].copy()
    abc_df["Abc"]   = abc_df["Abc"].round(4)
    abc_df["S_raw"] = abc_df["S_raw"].round(4)
    abc_df = abc_df.sort_values("Abc", ascending=False).reset_index(drop=True)

    return abc_df



# ---------------------------------------------------------------------------
# Abc 可视化
# ---------------------------------------------------------------------------
# 与 plot_capability.py 共享的风格常量（局部定义，避免循环导入）
_PRIMARY   = "#2C5F8A"
_SECONDARY = "#E07B39"
_LIGHT     = "#D6E8F5"
_GRAY      = "#EBEBEB"
_GP        = {"高能力组": "#2C5F8A", "中能力组": "#4CAF82", "低能力组": "#E07B39"}


def _set_style():
    sns.set_style("whitegrid", {"axes.grid": True, "grid.linestyle": "--"})
    plt.rcParams.update({
        "font.family": "sans-serif",
        "font.sans-serif": ["SimSun", "Times New Roman", "DejaVu Sans"],
        "axes.unicode_minus": False,
        "font.size": 12, "axes.labelsize": 14, "axes.titlesize": 15,
        "xtick.labelsize": 11, "ytick.labelsize": 11, "legend.fontsize": 11,
        "lines.linewidth": 1.8, "savefig.dpi": 300,
        "savefig.bbox": "tight", "savefig.format": "png",
    })


def plot_abc_distribution(abc_df: pd.DataFrame, output_dir: Path):
    """
    Abc 分布可视化，生成三张图：

    Abc_dist_histogram.png  — 直方图 + KDE + 正态性检验结果
    Abc_dist_by_group.png   — 按能力等级分组的小提琴图 + 散点
    Abc_dist_scatter.png    — 按被试 ID 排列的散点图（能力等级着色）
    """
    _set_style()
    os.makedirs(output_dir, exist_ok=True)
    id_col  = "被试ID" if "被试ID" in abc_df.columns else "实验ID"
    abc_arr = abc_df["Abc"].values
    mean_v  = abc_arr.mean()
    std_v   = abc_arr.std()

    # ── 图1：直方图 + KDE ──────────────────────────────────────────
    fig, ax = plt.subplots(figsize=(10, 6))
    sns.histplot(abc_arr, bins=12, kde=True, color=_PRIMARY,
                 edgecolor="white", linewidth=0.4, alpha=0.75, ax=ax,
                 line_kws={"linewidth": 1.8, "color": _PRIMARY})
    ax.axvline(mean_v, color=_SECONDARY, linestyle="--", linewidth=1.8,
               label=f"均值 {mean_v:.4f}")
    ax.axvspan(mean_v - std_v, mean_v + std_v, color=_LIGHT, alpha=0.45,
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
        palette = {g: _GP.get(g, _PRIMARY) for g in ordered}

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
            ax.scatter(i, mv, color=_SECONDARY, marker="^",
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
                   color=_GP.get(grp, _PRIMARY), label=grp,
                   s=70, alpha=0.85, edgecolors="white", linewidths=0.4, zorder=3)
    ax.axhline(mean_v, color=_SECONDARY, linestyle="--",
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



def evaluate_benchmark_driving_ability(config: Dict[str, Any]):
    out     = config["output_path"]
    fig_dir = out / "figures"
    res_dir = out / "results"
    fig_dir.mkdir(parents=True, exist_ok=True)
    res_dir.mkdir(parents=True, exist_ok=True)

    # 1. 加载标准化数据
    logging.info("正在加载标准化数据...")
    X = pd.read_pickle(config["input_path"])
    logging.info("数据加载完成，形状: %s", X.shape)

    # 2. 确定最佳聚类数目
    best_k, eval_df = find_best_k(X, res_dir, config)
    logging.info("最佳聚类数目确定为: k=%d", best_k)

    # 3. 执行 K-means++ 聚类
    logging.info("正在执行 K-means++ 聚类...")
    centers, labels, kmeans = cluster_analysis(X, best_k, config)
    logging.info("聚类完成，迭代次数: %d", kmeans.n_iter_)

    # 4. 聚类结果统计
    sorted_cluster_names = centers.index.tolist()
    cluster_stats = pd.DataFrame({
        "样本数量": labels.value_counts(),
        "占比":    labels.value_counts(normalize=True).round(3) * 100,
    }).reindex(sorted_cluster_names)
    cluster_stats["占比"] = cluster_stats["占比"].astype(str) + "%"

    # 5. 核心指标聚类中心
    key_centers = centers[config["key_indicators"] + ["综合得分"]].round(2)
    key_centers = pd.concat([key_centers, cluster_stats], axis=1)

    # 6. 聚类中心级基准能力量化（Ab，原始逻辑，3 个值）
    benchmark_result = quantify_benchmark_ability(centers.drop("综合得分", axis=1), config)
    benchmark_result = pd.concat([cluster_stats, benchmark_result.round(4)], axis=1)

    # 7. ★ 个体化基准能力量化（Abc，每名被试一个值）
    abc_df = compute_individualized_abc(X, labels, centers, config)

    # 8. 保存结果
    labels.to_csv(res_dir / "Ab_cluster_labels.csv", encoding="utf-8-sig", header=["能力等级"])
    centers.to_csv(res_dir / "Ab_cluster_centers_all_indicators.csv", encoding="utf-8-sig")
    key_centers.to_csv(res_dir / "Ab_cluster_key_centers.csv", encoding="utf-8-sig")
    benchmark_result.to_csv(res_dir / "Ab_quantification.csv", encoding="utf-8-sig")
    abc_df.to_csv(res_dir / "Abc_individualized_baseline_ability.csv",
                  index=False, encoding="utf-8-sig")

    # 打印 Abc 描述性统计
    logging.info("=" * 60)
    logging.info("个体化基准能力 Abc 描述性统计：")
    logging.info("  均值:   %.4f", abc_df["Abc"].mean())
    logging.info("  标准差: %.4f", abc_df["Abc"].std())
    logging.info("  最小值: %.4f", abc_df["Abc"].min())
    logging.info("  最大值: %.4f", abc_df["Abc"].max())
    for grp, sub in abc_df.groupby("能力等级", observed=True):
        logging.info("  %s: %.4f ± %.4f", grp, sub["Abc"].mean(), sub["Abc"].std())
    logging.info("=" * 60)

    # 9. 可视化
    plot_cluster_metrics(eval_df, fig_dir, best_k=config["best_k"])
    plot_pca_visualization(X, labels, fig_dir)
    plot_pca_3d_visualization(X, labels, fig_dir)
    plot_abc_distribution(abc_df, fig_dir)

    # 10. 打印结果摘要
    logging.info("=" * 60)
    logging.info("三类驾驶人核心特征聚类中心值：\n%s", key_centers.to_string())
    logging.info("\n基准驾驶能力等级量化结果（Ab）：\n%s", benchmark_result.to_string())
    logging.info("=" * 60)
    logging.info("所有结果已保存至: %s", out.resolve())

    return labels, centers, benchmark_result, abc_df


# ---------------------------------------------------------------------------
# 命令行参数解析 & 主函数
# ---------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(
        description="基准驾驶能力评估",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("-c", "--config",
        default=BASE_DIR / "config" / "baseline_capability.yaml",
        help="YAML配置文件路径")
    parser.add_argument("-i", "--input",  help="标准化问卷数据路径（PKL格式）")
    parser.add_argument("-o", "--output", help="输出目录")
    parser.add_argument("--best_k",       type=int, help="最佳聚类数")
    parser.add_argument("--max_iter",     type=int, help="KMeans最大迭代次数")
    parser.add_argument("--random_state", type=int, help="随机种子")

    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s - %(levelname)s - %(message)s")
    try:
        default_config = load_yaml_config(args.config)
        final_config   = merge_configs(default_config, args)
        logging.info("配置加载完成：聚类范围=%s，最佳k=%d，输入=%s，输出=%s",
                     final_config["cluster_range"], final_config["best_k"],
                     final_config["input_path"], final_config["output_path"])
        evaluate_benchmark_driving_ability(final_config)
    except Exception as exc:
        logging.error("评估失败：%s", exc, exc_info=True)


if __name__ == "__main__":
    main()
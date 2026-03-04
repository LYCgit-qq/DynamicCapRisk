# D:\Local\DynamicCapRisk\src\2_capability_assessment\baseline_capability.py

import argparse
import logging
from pathlib import Path
from typing import Tuple

import pandas as pd
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

# 项目根目录
BASE_DIR = Path(__file__).resolve().parents[2]

# ---------------------------------------------------------------------------
# 核心配置（与论文一致）
# ---------------------------------------------------------------------------
CLUSTER_RANGE = range(2, 8)  # 聚类数目k从2到7
BEST_K = 4  # 论文确定的最佳聚类数
MAX_ITER = 100  # 最大迭代次数
TOL = 0.01  # 收敛阈值
RANDOM_STATE = 42  # 随机种子（保证可复现）

# 论文中提取的6项核心指标（用于结果展示）
KEY_INDICATORS = [
    "驾龄",
    "每周开车频率",  # 驾驶经验
    "限速遵守度",
    "换道观察充分性",  # 行为规范
    "情绪稳定性",
    "施工区安全感",  # 心理状态
]


# ---------------------------------------------------------------------------
# 辅助函数
# ---------------------------------------------------------------------------
def evaluate_clustering(X: pd.DataFrame, k: int) -> Tuple[float, float, float]:
    """计算单个k值下的三个聚类评价指标。"""
    kmeans = KMeans(
        n_clusters=k,
        init="k-means++",
        max_iter=MAX_ITER,
        tol=TOL,
        random_state=RANDOM_STATE,
    )
    labels = kmeans.fit_predict(X)
    sc = silhouette_score(X, labels)
    ch = calinski_harabasz_score(X, labels)
    dbi = davies_bouldin_score(X, labels)
    return sc, ch, dbi


def find_best_k(X: pd.DataFrame, output_dir: Path) -> int:
    """遍历k=2-6，计算评价指标并确定最佳k（与论文表3.3一致）。"""
    results = []
    for k in CLUSTER_RANGE:
        sc, ch, dbi = evaluate_clustering(X, k)
        results.append({"k": k, "轮廓系数SC": sc, "CH指数": ch, "DBI指数": dbi})

    # 保存并打印评价指标表
    eval_df = pd.DataFrame(results).set_index("k")
    eval_df.to_csv(output_dir / "Ab_cluster_evaluation.csv", encoding="utf-8-sig")

    logging.info("=" * 60)
    logging.info("不同聚类数目下的评价指标（表3.3）：")
    logging.info(eval_df.round(2).to_string())
    logging.info("=" * 60)

    # 按论文逻辑选择最佳k：SC最大、CH最大、DBI最小
    # 这里直接返回论文确定的3，也可根据实际数据自动选择
    return BEST_K, eval_df


def cluster_analysis(
    X: pd.DataFrame, k: int
) -> Tuple[pd.DataFrame, pd.Series, pd.DataFrame]:
    """执行K-means++聚类并返回核心结果（修正命名颠倒问题）。"""
    kmeans = KMeans(
        n_clusters=k,
        init="k-means++",
        max_iter=MAX_ITER,
        tol=TOL,
        random_state=RANDOM_STATE,
    )
    labels = kmeans.fit_predict(X)
    centers = pd.DataFrame(kmeans.cluster_centers_, columns=X.columns)

    # 按6项核心指标综合得分排序
    cluster_means = X.groupby(labels).mean()
    sort_score = cluster_means[KEY_INDICATORS].mean(axis=1)  # 6项核心指标综合得分
    cluster_order = sort_score.sort_values(
        ascending=False
    ).index.tolist()  # 0:高, 1:中, 2:低

    # -------------------------- 动态标签生成（适配任意k） --------------------------
    def _get_dynamic_cluster_names(k: int) -> list:
        """根据k值动态生成聚类名称（优先使用语义化名称，超过4类则用通用名）"""
        if k == 2:
            return ["高能力组", "低能力组"]
        elif k == 3:
            return ["高能力组", "中能力组", "低能力组"]
        elif k == 4:
            return ["高能力组", "中高能力组", "中低能力组", "低能力组"]
        else:
            # k > 4 时使用通用命名
            names = [f"能力组_{i+1}" for i in range(k)]
            names[0] += "（高）"
            names[-1] += "（低）"
            return names

    cluster_names = _get_dynamic_cluster_names(k)

    # 重命名标签
    label_mapping = {
        old_id: new_name for old_id, new_name in zip(cluster_order, cluster_names)
    }
    labels_renamed = pd.Series(labels, index=X.index).map(label_mapping)

    # 按得分排序并更新聚类中心索引
    centers = centers.reindex(cluster_order)
    centers.index = cluster_names
    # -----------------------------------------------------------------------------

    # 【新增】添加6项核心指标综合得分（便于验证结果）
    centers["综合得分"] = centers[KEY_INDICATORS].mean(axis=1)

    return centers, labels_renamed, kmeans


def quantify_benchmark_ability(centers: pd.DataFrame) -> pd.DataFrame:
    """根据论文步骤计算基准能力量化值（表3.5）。"""
    # 步骤1：对35项指标的聚类中心进行Min-Max归一化到[0,1]
    centers_normalized = (centers - centers.min().min()) / (
        centers.max().max() - centers.min().min()
    )

    # 步骤2：等权求和计算综合得分S^c
    composite_score = centers_normalized.mean(axis=1)  # 35项指标等权平均

    # 步骤3：线性映射到[0.55, 0.95]
    benchmark_ability = 0.55 + 0.4 * composite_score

    # 整理结果表
    result_df = pd.DataFrame(
        {"综合得分S^c": composite_score, "基准能力值A_b": benchmark_ability}
    )
    return result_df


# ---------------------------------------------------------------------------
# 主流程
# ---------------------------------------------------------------------------
def evaluate_benchmark_driving_ability(
    standardized_data_path: Path | str, output_dir: Path | str
):
    """执行完整的基准驾驶能力评估流程。"""
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)

    # 1. 加载标准化数据
    logging.info("正在加载标准化数据...")
    X = pd.read_pickle(standardized_data_path)
    logging.info("数据加载完成，形状: %s", X.shape)

    # 2. 确定最佳聚类数目
    best_k, eval_df = find_best_k(X, out)
    logging.info("最佳聚类数目确定为: k=%d", best_k)

    # 3. 执行K-means++聚类
    logging.info("正在执行K-means++聚类...")
    centers, labels, kmeans = cluster_analysis(X, best_k)
    logging.info("聚类完成，迭代次数: %d", kmeans.n_iter_)

    # 4. 聚类结果统计
    # 动态获取排序后的聚类名称（直接从聚类中心索引获取，保证顺序一致）
    sorted_cluster_names = centers.index.tolist()
    cluster_stats = pd.DataFrame(
        {
            "样本数量": labels.value_counts(),
            "占比": labels.value_counts(normalize=True).round(3) * 100,
        }
    ).reindex(sorted_cluster_names)
    cluster_stats["占比"] = cluster_stats["占比"].astype(str) + "%"

    # 5. 核心指标聚类中心（表3.4）
    key_centers = centers[KEY_INDICATORS + ["综合得分"]].round(2)  # 新增综合得分列
    key_centers = pd.concat([key_centers, cluster_stats], axis=1)

    # 6. 基准能力量化（表3.5）
    benchmark_result = quantify_benchmark_ability(
        centers.drop("综合得分", axis=1)
    )  # 排除新增的综合得分列
    benchmark_result = pd.concat([cluster_stats, benchmark_result.round(2)], axis=1)

    # 7. 保存结果
    labels.to_csv(out / "Ab_cluster_labels.csv", encoding="utf-8-sig", header=["能力等级"])
    centers.to_csv(out / "Ab_cluster_centers_all_indicators.csv", encoding="utf-8-sig")
    key_centers.to_csv(out / "Ab_cluster_key_centers.csv", encoding="utf-8-sig")
    benchmark_result.to_csv(
        out / "Ab_quantification.csv", encoding="utf-8-sig"
    )

    # 8. PCA可视化
    plot_cluster_metrics(eval_df, out, best_k=BEST_K)
    plot_pca_visualization(X, labels, out)
    plot_pca_3d_visualization(X, labels, out)

    # 9. 打印结果摘要
    logging.info("=" * 60)
    logging.info("三类驾驶人核心特征聚类中心值（表3.4）：")
    logging.info(key_centers.to_string())
    logging.info("\n基准驾驶能力等级量化结果（表3.5）：")
    logging.info(benchmark_result.to_string())
    logging.info("=" * 60)
    logging.info("所有结果已保存至: %s", out.resolve())

    return labels, centers, benchmark_result


def main():
    parser = argparse.ArgumentParser(
        description="基准驾驶能力评估",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "-i",
        "--input",
        default=BASE_DIR / "data" / "processed" / "questionnaire_standardized.pkl",
        help="标准化问卷数据路径（PKL格式）",
    )
    parser.add_argument(
        "-o",
        "--output",
        default=BASE_DIR / "output" / "1_capability_assessment",
        help="输出目录",
    )
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
    )

    try:
        evaluate_benchmark_driving_ability(args.input, args.output)
    except Exception as exc:
        logging.error("评估失败：%s", exc, exc_info=True)
        return


if __name__ == "__main__":
    main()

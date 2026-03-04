# D:\Local\DynamicCapRisk\src\2_capability_assessment\baseline_capability.py

import os
os.environ["OMP_NUM_THREADS"] = "1"
import argparse
import logging
import yaml
from pathlib import Path
from typing import Tuple, Dict, Any

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
# 配置读取工具函数
# ---------------------------------------------------------------------------
def load_yaml_config(config_path: str | Path) -> Dict[str, Any]:
    """加载YAML配置文件"""
    config_path = Path(config_path)
    if not config_path.exists():
        raise FileNotFoundError(f"配置文件不存在: {config_path.resolve()}")
    
    with open(config_path, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)
    
    # 处理聚类范围（将列表转为range对象）
    if "cluster_range" in config and isinstance(config["cluster_range"], list):
        cluster_start, cluster_end = config["cluster_range"]
        config["cluster_range"] = range(cluster_start, cluster_end)
    
    # 路径转为绝对路径（基于项目根目录）
    for path_key in ["input_path", "output_path"]:
        if path_key in config:
            config[path_key] = BASE_DIR / config[path_key]
    
    return config

def merge_configs(default_config: Dict[str, Any], cli_args: argparse.Namespace) -> Dict[str, Any]:
    """合并默认配置（YAML）和命令行参数，命令行参数优先级更高"""
    merged = default_config.copy()
    
    # 覆盖路径参数
    if cli_args.input is not None:
        merged["input_path"] = Path(cli_args.input)
    if cli_args.output is not None:
        merged["output_path"] = Path(cli_args.output)
    
    # 覆盖聚类核心参数（按需添加）
    if cli_args.best_k is not None:
        merged["best_k"] = cli_args.best_k
    if cli_args.max_iter is not None:
        merged["max_iter"] = cli_args.max_iter
    if cli_args.random_state is not None:
        merged["random_state"] = cli_args.random_state
    
    return merged

# ---------------------------------------------------------------------------
# 核心功能函数（仅参数来源修改，逻辑不变）
# ---------------------------------------------------------------------------
def evaluate_clustering(X: pd.DataFrame, k: int, config: Dict[str, Any]) -> Tuple[float, float, float]:
    """计算单个k值下的三个聚类评价指标。"""
    kmeans = KMeans(
        n_clusters=k,
        init="k-means++",
        max_iter=config["max_iter"],
        tol=config["tol"],
        random_state=config["random_state"],
    )
    labels = kmeans.fit_predict(X)
    sc = silhouette_score(X, labels)
    ch = calinski_harabasz_score(X, labels)
    dbi = davies_bouldin_score(X, labels)
    return sc, ch, dbi

def find_best_k(X: pd.DataFrame, output_dir: Path, config: Dict[str, Any]) -> Tuple[int, pd.DataFrame]:
    """遍历k范围，计算评价指标并确定最佳k"""
    results = []
    for k in config["cluster_range"]:
        sc, ch, dbi = evaluate_clustering(X, k, config)
        results.append({"k": k, "轮廓系数SC": sc, "CH指数": ch, "DBI指数": dbi})

    # 保存并打印评价指标表
    eval_df = pd.DataFrame(results).set_index("k")
    eval_df.to_csv(output_dir / "Ab_cluster_evaluation.csv", encoding="utf-8-sig")

    logging.info("=" * 60)
    logging.info("不同聚类数目下的评价指标（表3.3）：")
    logging.info(eval_df.round(2).to_string())
    logging.info("=" * 60)

    # 返回论文指定的最佳k（或可改为自动选择）
    return config["best_k"], eval_df

def cluster_analysis(
    X: pd.DataFrame, k: int, config: Dict[str, Any]
) -> Tuple[pd.DataFrame, pd.Series, pd.DataFrame]:
    """执行K-means++聚类并返回核心结果"""
    kmeans = KMeans(
        n_clusters=k,
        init="k-means++",
        max_iter=config["max_iter"],
        tol=config["tol"],
        random_state=config["random_state"],
    )
    labels = kmeans.fit_predict(X)
    centers = pd.DataFrame(kmeans.cluster_centers_, columns=X.columns)

    # 按核心指标综合得分排序
    cluster_means = X.groupby(labels).mean()
    sort_score = cluster_means[config["key_indicators"]].mean(axis=1)
    cluster_order = sort_score.sort_values(ascending=False).index.tolist()

    # 动态标签生成
    def _get_dynamic_cluster_names(k: int) -> list:
        if k == 2:
            return ["高能力组", "低能力组"]
        elif k == 3:
            return ["高能力组", "中能力组", "低能力组"]
        elif k == 4:
            return ["高能力组", "中高能力组", "中低能力组", "低能力组"]
        else:
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
    centers["综合得分"] = centers[config["key_indicators"]].mean(axis=1)

    return centers, labels_renamed, kmeans

def quantify_benchmark_ability(centers: pd.DataFrame, config: Dict[str, Any]) -> pd.DataFrame:
    """根据论文步骤计算基准能力量化值"""
    # 步骤1：Min-Max归一化
    centers_normalized = (centers - centers.min().min()) / (centers.max().max() - centers.min().min())

    # 步骤2：等权求和计算综合得分S^c
    composite_score = centers_normalized.mean(axis=1)

    # 步骤3：线性映射到[0.55, 0.95]
    benchmark_ability = 0.55 + 0.4 * composite_score

    # 整理结果表
    result_df = pd.DataFrame(
        {"综合得分S^c": composite_score, "基准能力值A_b": benchmark_ability}
    )
    return result_df

# ---------------------------------------------------------------------------
# 主流程函数
# ---------------------------------------------------------------------------
def evaluate_benchmark_driving_ability(config: Dict[str, Any]):
    """执行完整的基准驾驶能力评估流程"""
    out = config["output_path"]
    out.mkdir(parents=True, exist_ok=True)

    # 1. 加载标准化数据
    logging.info("正在加载标准化数据...")
    X = pd.read_pickle(config["input_path"])
    logging.info("数据加载完成，形状: %s", X.shape)

    # 2. 确定最佳聚类数目
    best_k, eval_df = find_best_k(X, out, config)
    logging.info("最佳聚类数目确定为: k=%d", best_k)

    # 3. 执行K-means++聚类
    logging.info("正在执行K-means++聚类...")
    centers, labels, kmeans = cluster_analysis(X, best_k, config)
    logging.info("聚类完成，迭代次数: %d", kmeans.n_iter_)

    # 4. 聚类结果统计
    sorted_cluster_names = centers.index.tolist()
    cluster_stats = pd.DataFrame(
        {
            "样本数量": labels.value_counts(),
            "占比": labels.value_counts(normalize=True).round(3) * 100,
        }
    ).reindex(sorted_cluster_names)
    cluster_stats["占比"] = cluster_stats["占比"].astype(str) + "%"

    # 5. 核心指标聚类中心
    key_centers = centers[config["key_indicators"] + ["综合得分"]].round(2)
    key_centers = pd.concat([key_centers, cluster_stats], axis=1)

    # 6. 基准能力量化
    benchmark_result = quantify_benchmark_ability(centers.drop("综合得分", axis=1), config)
    benchmark_result = pd.concat([cluster_stats, benchmark_result.round(2)], axis=1)

    # 7. 保存结果
    labels.to_csv(out / "Ab_cluster_labels.csv", encoding="utf-8-sig", header=["能力等级"])
    centers.to_csv(out / "Ab_cluster_centers_all_indicators.csv", encoding="utf-8-sig")
    key_centers.to_csv(out / "Ab_cluster_key_centers.csv", encoding="utf-8-sig")
    benchmark_result.to_csv(out / "Ab_quantification.csv", encoding="utf-8-sig")

    # 8. PCA可视化
    plot_cluster_metrics(eval_df, out, best_k=config["best_k"])
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

# ---------------------------------------------------------------------------
# 命令行参数解析 & 主函数
# ---------------------------------------------------------------------------
def main():
    # 1. 定义命令行参数
    parser = argparse.ArgumentParser(
        description="基准驾驶能力评估",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    # 配置文件参数
    parser.add_argument(
        "-c", "--config",
        default=BASE_DIR / "config" / "baseline_capability.yaml",
        help="YAML配置文件路径"
    )
    # 路径参数（可覆盖YAML）
    parser.add_argument(
        "-i", "--input",
        help="标准化问卷数据路径（PKL格式，覆盖YAML配置）"
    )
    parser.add_argument(
        "-o", "--output",
        help="输出目录（覆盖YAML配置）"
    )
    # 核心参数（可覆盖YAML）
    parser.add_argument(
        "--best_k", type=int,
        help="最佳聚类数（覆盖YAML配置）"
    )
    parser.add_argument(
        "--max_iter", type=int,
        help="KMeans最大迭代次数（覆盖YAML配置）"
    )
    parser.add_argument(
        "--random_state", type=int,
        help="随机种子（覆盖YAML配置）"
    )

    args = parser.parse_args()

    # 2. 配置初始化
    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
    )
    try:
        # 加载YAML配置
        default_config = load_yaml_config(args.config)
        # 合并配置（命令行参数优先级更高）
        final_config = merge_configs(default_config, args)
        logging.info("配置加载完成，使用的核心参数：")
        logging.info(f"  聚类范围: {final_config['cluster_range']}")
        logging.info(f"  最佳k值: {final_config['best_k']}")
        logging.info(f"  输入路径: {final_config['input_path']}")
        logging.info(f"  输出路径: {final_config['output_path']}")

        # 3. 执行评估流程
        evaluate_benchmark_driving_ability(final_config)
    except Exception as exc:
        logging.error("评估失败：%s", exc, exc_info=True)
        return

if __name__ == "__main__":
    main()
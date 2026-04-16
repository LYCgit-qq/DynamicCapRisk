# D:\Local\DynamicCapRisk\src\2_risk_assessment\risk_validator.py

"""
risk_validator.py
风险度 R 有效性验证模块 (已修改：直接读取 performance_metrics.csv)
"""

import os
import sys
import warnings
import argparse
import yaml
import numpy as np
import pandas as pd
from scipy import stats
from typing import Dict, List, Tuple, Optional

warnings.filterwarnings("ignore")

# ---------------------------------------------------------------------------
# 导入可视化模块
# ---------------------------------------------------------------------------
VISUALIZATION_DIR = os.path.join(os.path.dirname(__file__), "..", "visualization")
if VISUALIZATION_DIR not in sys.path:
    sys.path.insert(0, os.path.abspath(VISUALIZATION_DIR))

try:
    import src.visualization.plot_risk as plot_risk_module
    HAS_PLOT_MODULE = True
except ImportError:
    HAS_PLOT_MODULE = False
    print("⚠️  无法导入 plot_risk 模块，绘图功能将跳过。")


# =============================================================================
# 配置加载
# =============================================================================

DEFAULT_CONFIG = {
    "paths": {
        "performance_csv": "performance_metrics.csv",
        "risk_csv":    "risk_windows_all.csv",
        "output_dir":  "validation_output",
    },
    "risk_level_order": ["低风险", "中风险", "高风险"],
    "figure": {
        "dpi": 150,
        "figsize_roc":    [6, 5],
        "figsize_bar":    [6, 4],
        "figsize_box":    [5, 5],
        "color_low_risk":  "#4CAF50",
        "color_mid_risk":  "#FF9800",
        "color_high_risk": "#F44336",
        "color_no_event":  "#90CAF9",
        "color_event":     "#EF9A9A",
    },
}

def deep_merge_dict(base: dict, override: dict) -> dict:
    result = base.copy()
    for key, value in override.items():
        if isinstance(value, dict) and isinstance(result.get(key), dict):
            result[key] = deep_merge_dict(result[key], value)
        else:
            result[key] = value
    return result

def load_config(config_path: Optional[str] = None) -> dict:
    import copy
    config = copy.deepcopy(DEFAULT_CONFIG)
    if config_path and os.path.exists(config_path):
        with open(config_path, "r", encoding="utf-8") as f:
            user_config = yaml.safe_load(f) or {}
        config = deep_merge_dict(config, user_config)
        print(f"已加载配置: {config_path}")
    elif config_path:
        print(f"⚠️  配置文件不存在: {config_path}，使用内置默认值")
    else:
        print("未指定配置文件，使用内置默认值")
    return config


# =============================================================================
# 1. 数据加载 (已修改：直接加载 CSV)
# =============================================================================

def load_performance_data(csv_path: str) -> pd.DataFrame:
    """直接加载由 capability_validator 生成的客观绩效数据"""
    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"客观绩效文件不存在: {csv_path}")
    df = pd.read_csv(csv_path, encoding="utf-8-sig")
    print(f"  客观绩效加载完成：{len(df)} 个窗口")
    # 确保索引列存在且类型正确
    for col in ['sample_idx', 'window_idx']:
        if col not in df.columns:
            df[col] = df.get('index', df.index) # 兜底
        df[col] = df[col].astype(int)
    return df

def load_risk_data(csv_path: str) -> pd.DataFrame:
    """加载风险评估结果"""
    if not os.path.exists(csv_path):
        raise FileNotFoundError(
            f"风险结果文件不存在: {csv_path}\n"
            "请先运行 risk_evaluator.py 生成 risk_windows_all.csv"
        )
    df = pd.read_csv(csv_path, encoding="utf-8-sig")
    print(f"  风险结果加载完成：{len(df)} 行")
    return df


# =============================================================================
# 2. 合并风险结果与事件标注
# =============================================================================

def merge_risk_with_events(risk_df: pd.DataFrame, perf_df: pd.DataFrame) -> pd.DataFrame:
    # 风险数据保留列
    keep_risk_cols  = [col for col in ["sample_idx", "window_idx", "R", "risk_level", "risk_level_optimized",
                               "F_S", "Ad_norm", "A_d", "field_label"]
                  if col in risk_df.columns]
    
    # 绩效数据保留列 (包含你需要的 SDLP, steer_SASD 等)
    keep_perf_cols = [col for col in ["sample_idx", "window_idx", "abnormal_event",
                               "SDLP", "lane_stability", "steer_SASD"]
                  if col in perf_df.columns]

    merged_df = pd.merge(
        risk_df[keep_risk_cols], perf_df[keep_perf_cols],
        on=["sample_idx", "window_idx"], how="inner",
    )
    
    # 兼容性处理：如果 risk_level_optimized 存在，优先使用它作为主风险等级
    if "risk_level_optimized" in merged_df.columns:
        merged_df["risk_level"] = merged_df["risk_level_optimized"]

    print(
        f"  合并完成：{len(merged_df)} 个有效窗口  "
        f"（risk={len(risk_df)}，perf={len(perf_df)}）"
    )
    return merged_df


# =============================================================================
# 3. 统计验证 (已适配新列名 R 和 abnormal_event)
# =============================================================================

def calculate_pearson_correlation(merged_df: pd.DataFrame) -> pd.DataFrame:
    """R 与各客观绩效指标的 Pearson 相关系数。"""
    # 扩展了指标列表，充分利用你已有的 SDLP 和 SASD
    metric_mapping = {
        "异常驾驶事件(0/1)": "abnormal_event",
        "车道保持SDLP":       "SDLP",
        "车道保持稳定性":     "lane_stability",
        "方向盘转角SASD":     "steer_SASD",
    }
    row_list = []
    for label, col_name in metric_mapping.items():
        if col_name not in merged_df.columns:
            continue
        # 去除 NaN
        valid_data = merged_df[["R", col_name]].dropna()
        if len(valid_data) < 2:
            continue
            
        corr_coef, p_value = stats.pearsonr(valid_data["R"], valid_data[col_name])
        row_list.append({
            "绩效指标":   label,
            "相关系数r":  round(corr_coef, 3),
            "显著性p":    "<0.001" if p_value < 0.001 else round(p_value, 4),
            "相关方向":   "正相关" if corr_coef > 0 else "负相关",
        })
    result_df = pd.DataFrame(row_list)
    print("\n── Pearson 相关性验证 ────────────────────────")
    print(result_df.to_string(index=False))
    return result_df


def compute_roc_auc_analysis(merged_df: pd.DataFrame, config: dict, output_dir: str) -> Tuple[pd.DataFrame, float]:
    try:
        from sklearn.metrics import (roc_auc_score, roc_curve,
                                      f1_score, precision_score, recall_score)
    except ImportError:
        print("  ⚠️  sklearn 未安装，跳过 ROC 验证")
        return pd.DataFrame(), 0.0

    # 列名适配：使用 R 和 abnormal_event
    valid_data = merged_df[["R", "abnormal_event"]].dropna()
    true_labels    = valid_data["abnormal_event"].to_numpy(dtype=int)
    risk_scores    = valid_data["R"].to_numpy(dtype=float)
    
    if true_labels.sum() == 0 or true_labels.sum() == len(true_labels):
        print("  ⚠️  abnormal_event 全0或全1，无法计算 AUC")
        return pd.DataFrame(), 0.0

    roc_auc           = roc_auc_score(true_labels, risk_scores)
    false_pos_rate, true_pos_rate, thresholds = roc_curve(true_labels, risk_scores)
    youden_index = np.argmax(true_pos_rate - false_pos_rate)
    optimal_threshold = thresholds[youden_index]
    pred_labels = (risk_scores >= optimal_threshold).astype(int)

    summary_df = pd.DataFrame([{
        "AUC":       round(roc_auc, 3),
        "最优阈值":  round(optimal_threshold, 3),
        "F1":        round(f1_score(true_labels, pred_labels), 3),
        "Precision": round(precision_score(true_labels, pred_labels, zero_division=0), 3),
        "Recall":    round(recall_score(true_labels, pred_labels), 3),
        "Youden_J":  round(true_pos_rate[youden_index] - false_pos_rate[youden_index], 3),
    }])
    print(f"\n── ROC/AUC 验证 ─────────────────────────────")
    print(summary_df.to_string(index=False))

    if HAS_PLOT_MODULE:
        try:
            plot_risk_module.plot_roc_curve(
                fpr=false_pos_rate,
                tpr=true_pos_rate,
                auc=roc_auc,
                best_thr=optimal_threshold,
                best_fpr=float(false_pos_rate[youden_index]),
                best_tpr=float(true_pos_rate[youden_index]),
                cfg=config,
                fig_dir=output_dir,
            )
        except Exception as e:
            print(f"  ⚠️  ROC 图绘制失败: {e}")

    return summary_df, roc_auc


def analyze_risk_level_event_rate(merged_df: pd.DataFrame, config: dict, output_dir: str) -> pd.DataFrame:
    risk_level_order = config["risk_level_order"]
    row_list  = []
    
    # 确保只分析配置文件中存在的等级
    available_levels = [lvl for lvl in risk_level_order if lvl in merged_df["risk_level"].unique()]

    for level in available_levels:
        sub_df   = merged_df[merged_df["risk_level"] == level]
        total_count = len(sub_df)
        event_count  = int(sub_df["abnormal_event"].sum())
        row_list.append({
            "风险等级":   level,
            "窗口数":     total_count,
            "占比(%)":    round(total_count / len(merged_df) * 100, 1),
            "事件窗口数": event_count,
            "事件率(%)":  round(event_count / max(total_count, 1) * 100, 1),
        })
    result_df = pd.DataFrame(row_list)

    # 卡方检验
    contingency_table = pd.crosstab(merged_df["risk_level"], merged_df["abnormal_event"])
    contingency_table = contingency_table.reindex([level for level in available_levels if level in contingency_table.index])
    
    if len(contingency_table) > 1:
        chi2_stat, chi2_p, dof, _ = stats.chi2_contingency(contingency_table.values)
        chi2_result_str = (f"χ²={chi2_stat:.2f}, df={dof}, "
                    f"p={'<0.001' if chi2_p < 0.001 else round(chi2_p, 4)}")
    else:
        chi2_result_str = "N/A (等级不足)"

    print(f"\n── 风险等级 × 异常事件发生率 ─────────────────")
    print(result_df.to_string(index=False))
    print(f"  卡方检验: {chi2_result_str}")
    result_df["chi2检验"] = [chi2_result_str] + [""] * (len(result_df) - 1)

    if HAS_PLOT_MODULE:
        try:
            plot_risk_module.plot_risk_event_rate(
                level_df=result_df,
                chi2_str=chi2_result_str,
                cfg=config,
                fig_dir=output_dir,
            )
        except Exception as e:
            print(f"  ⚠️  风险等级图绘制失败: {e}")

    return result_df


def perform_mannwhitney_test(merged_df: pd.DataFrame, config: dict, output_dir: str) -> pd.DataFrame:
    valid_data = merged_df[["R", "abnormal_event"]].dropna()
    risk_no_event = valid_data[valid_data["abnormal_event"] == 0]["R"].values
    risk_with_event = valid_data[valid_data["abnormal_event"] == 1]["R"].values
    
    if len(risk_no_event) == 0 or len(risk_with_event) == 0:
        print("  ⚠️  缺少事件组或无事件组数据，跳过 U 检验")
        return pd.DataFrame()

    u_stat, p_value = stats.mannwhitneyu(risk_no_event, risk_with_event, alternative="less")
    p_value_str = "<0.001" if p_value < 0.001 else str(round(p_value, 4))

    summary_df = pd.DataFrame([
        {"组别": "无异常事件", "样本数": len(risk_no_event),
         "R均值": round(risk_no_event.mean(), 4), "R标准差": round(risk_no_event.std(), 4)},
        {"组别": "异常事件",   "样本数": len(risk_with_event),
         "R均值": round(risk_with_event.mean(), 4), "R标准差": round(risk_with_event.std(), 4)},
    ])
    print(f"\n── Mann-Whitney U 检验 ─────────────────────")
    print(summary_df.to_string(index=False))
    print(f"  U={u_stat:.1f}, p={p_value_str}（单侧：事件组 R > 无事件组）")

    if HAS_PLOT_MODULE:
        try:
            plot_risk_module.plot_r_star_boxplot( # 注意：如果绘图函数内部也硬编码了 R_star，可能需要改绘图函数，但这里先传数据
                ev0=risk_no_event,
                ev1=risk_with_event,
                p_str=p_value_str,
                cfg=config,
                fig_dir=output_dir,
            )
        except Exception as e:
            print(f"  ⚠️  R 箱线图绘制失败: {e}")

    return summary_df


# =============================================================================
# 4. 汇总报告
# =============================================================================

def write_validation_summary(merged_df, corr_df, roc_df, level_df, mwu_df, config, output_dir):
    lines = [
        "=" * 65,
        "风险度 R 有效性验证报告 (直接读取 Performance 版)",
        "=" * 65,
        f"有效验证窗口总数:  {len(merged_df)}",
        f"异常事件窗口数:    {int(merged_df['abnormal_event'].sum())}",
        f"异常事件发生率:    {merged_df['abnormal_event'].mean()*100:.1f}%",
        f"R 均值:           {merged_df['R'].mean():.4f}",
        f"R 标准差:         {merged_df['R'].std():.4f}",
        "",
        "── Pearson 相关性验证 ──────────────────────────────────",
        corr_df.to_string(index=False),
        "",
        "── ROC/AUC ─────────────────────────────────────────────",
        roc_df.to_string(index=False) if not roc_df.empty else "（sklearn 未安装，已跳过）",
        "",
        "── 风险等级 × 事件发生率（卡方检验）───────────────────",
        level_df.to_string(index=False),
        "",
        "── Mann-Whitney U 检验 ─────────────────────────────────",
        mwu_df.to_string(index=False) if not mwu_df.empty else "（数据不足，已跳过）",
        "",
        "=" * 65,
    ]

    report = "\n".join(lines)
    print("\n" + report)
    save_path = os.path.join(output_dir, "validation_summary.txt")
    with open(save_path, "w", encoding="utf-8") as f:
        f.write(report)
    print(f"\n验证报告 → {save_path}")


# =============================================================================
# 主函数 (精简流程)
# =============================================================================

def main():
    parser = argparse.ArgumentParser(
        description="R 有效性验证 (直接读取 performance_metrics.csv)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("-c", "--config",      type=str,   default="config/risk_validator.yaml",
                        help="YAML 配置文件路径")
    parser.add_argument("--performance_csv",   type=str,   default=None, help="覆盖绩效文件路径")
    parser.add_argument("--risk_csv",          type=str,   default=None, help="覆盖风险文件路径")
    parser.add_argument("--output_dir",        type=str,   default=None, help="覆盖输出目录")
    args = parser.parse_args()

    config = load_config(args.config)

    # 命令行覆盖
    if args.performance_csv: config["paths"]["performance_csv"] = args.performance_csv
    if args.risk_csv:        config["paths"]["risk_csv"]   = args.risk_csv
    if args.output_dir:      config["paths"]["output_dir"] = args.output_dir

    output_directory = config["paths"]["output_dir"]
    os.makedirs(output_directory, exist_ok=True)

    print("=" * 65)
    print("TCI 风险度 R 有效性验证 (直接读取绩效版)")
    print(f"  绩效数据:  {config['paths']['performance_csv']}")
    print(f"  风险结果:  {config['paths']['risk_csv']}")
    print(f"  输出目录:  {output_directory}")
    print("=" * 65)

    # Step 1：加载数据
    print("\n[1/4] 加载客观绩效数据...")
    perf_df = load_performance_data(config["paths"]["performance_csv"])

    print("\n[2/4] 加载 R 风险结果...")
    risk_result_df = load_risk_data(config["paths"]["risk_csv"])

    # Step 2：合并
    print("\n[3/4] 合并风险结果与事件标注...")
    merged_risk_event_df = merge_risk_with_events(risk_result_df, perf_df)
    merged_risk_event_df.to_csv(os.path.join(output_directory, "merged_risk_events.csv"),
                  index=False, encoding="utf-8-sig")

    # Step 3：统计验证
    print("\n[4/4] 统计验证...")
    correlation_df       = calculate_pearson_correlation(merged_risk_event_df)
    roc_summary_df, _     = compute_roc_auc_analysis(merged_risk_event_df, config, output_directory)
    risk_level_df      = analyze_risk_level_event_rate(merged_risk_event_df, config, output_directory)
    mannwhitney_df        = perform_mannwhitney_test(merged_risk_event_df, config, output_directory)

    # Step 4：保存 CSV
    print("\n[保存结果]...")
    correlation_df.to_csv(os.path.join(output_directory, "validation_correlation.csv"),
                   index=False, encoding="utf-8-sig")
    if not roc_summary_df.empty:
        roc_summary_df.to_csv(os.path.join(output_directory, "validation_roc.csv"),
                      index=False, encoding="utf-8-sig")
    risk_level_df.to_csv(os.path.join(output_directory, "validation_chisquare.csv"),
                    index=False, encoding="utf-8-sig")
    if not mannwhitney_df.empty:
        mannwhitney_df.to_csv(os.path.join(output_directory, "validation_mannwhitney.csv"),
                      index=False, encoding="utf-8-sig")

    write_validation_summary(merged_risk_event_df, correlation_df, roc_summary_df, risk_level_df, mannwhitney_df, config, output_directory)

    print(f"\n{'='*65}")
    print(f"验证完成！所有结果 → {output_directory}/")
    print(f"{'='*65}")


if __name__ == "__main__":
    main()
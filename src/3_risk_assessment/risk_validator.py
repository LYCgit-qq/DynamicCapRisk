# D:\Local\DynamicCapRisk\src\3_risk_assessment\risk_validator.py

"""
risk_validator.py
风险度 R 有效性验证模块

功能：
  1. 从原始驾驶数据（act）按非重叠窗口自动提取客观异常事件标注
  2. 基于事件标注对 R 进行多维度统计验证：
       - Pearson 相关性（R vs. 各绩效指标）
       - ROC / AUC 判别分析（R 预测异常事件能力）
       - 卡方检验（风险等级分组 × 事件发生率一致性）
       - Mann-Whitney U 检验（事件组 vs. 无事件组 R 分布）
       - 子事件类型贡献分析
"""

import os
import pickle
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
# 导入可视化模块（plot_risk.py 与本脚本位于不同子目录，动态加入 sys.path）
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
        "raw_pkl":     "data/processed/raw_data.pkl",
        "risk_csv":    "output_csv/risk_windows_all.csv",
        "summary_csv": "output_csv/risk_summary_by_sample.csv",
        "output_dir":  "validation_output",
    },
    "window": {
        "window_seconds": 3,
        "act_hz": 60,
    },
    "event_thresholds": {
        "hard_brake_thresh":   0.30,
        "hard_accel_thresh":   0.75,
        "sharp_steer_thresh":  15.0,
        "lane_depart_thresh":  0.30,
        "lat_accel_thresh":    1.50,
        "lon_accel_thresh":    3.00,
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
# 1. 原始数据加载
# =============================================================================

def load_original_act_data(pkl_path: str) -> List:
    if not os.path.exists(pkl_path):
        raise FileNotFoundError(f"原始数据不存在: {pkl_path}")
    with open(pkl_path, "rb") as f:
        data = pickle.load(f)
    act_data = data.get("act", [])
    print(f"  原始 act 加载完成：{len(act_data)} 个样本")
    return act_data


# =============================================================================
# 2. 客观事件标注
# =============================================================================

def detect_driving_events_per_sample(
    act_signal: np.ndarray,
    sample_index: int,
    window_size: int,
    threshold_config: dict,
) -> pd.DataFrame:
    """
    对单样本 act 信号按非重叠窗口检测异常驾驶事件。

    act 维度（60 Hz）：
      0: 油门, 1: 刹车, 2: 方向盘转角, 3: 车速,
      4: 纵向加速度, 5: 横向加速度, 6: X, 7: Y,
      8: 横向偏移, 9: 路段类型

    has_abnormal_event = 1 表示该窗口内至少触发一类子事件。
    """
    signal_array = np.asarray(act_signal, dtype=float)
    total_windows = signal_array.shape[0] // window_size
    if total_windows == 0:
        return pd.DataFrame()

    sample_interval = 1.0 / 60.0   # act 采样间隔（秒）
    row_list = []
    for window_idx in range(total_windows):
        signal_segment = signal_array[window_idx * window_size: (window_idx + 1) * window_size]

        throttle = signal_segment[:, 0]
        brake    = signal_segment[:, 1]
        steer    = signal_segment[:, 2]
        lon_acc  = signal_segment[:, 4]
        lat_acc  = signal_segment[:, 5]
        lat_off  = signal_segment[:, 8]

        steering_angular_vel = np.abs(np.diff(steer, prepend=steer[0])) / sample_interval

        is_hard_brake  = int(np.abs(brake).mean()   > threshold_config["hard_brake_thresh"])
        is_hard_accel  = int(throttle.mean()         > threshold_config["hard_accel_thresh"])
        is_sharp_steer = int(steering_angular_vel.mean()        > threshold_config["sharp_steer_thresh"])
        is_lane_depart = int(np.abs(lat_off).mean()  > threshold_config["lane_depart_thresh"])
        is_lat_accel   = int(np.abs(lat_acc).mean()  > threshold_config["lat_accel_thresh"])
        is_lon_accel   = int(np.abs(lon_acc).mean()  > threshold_config["lon_accel_thresh"])

        has_abnormal_event = int(
            is_hard_brake | is_hard_accel | is_sharp_steer |
            is_lane_depart | is_lat_accel | is_lon_accel
        )

        row_list.append({
            "sample_idx":      sample_index,
            "window_idx":      window_idx,
            "event_label":     has_abnormal_event,
            "ev_hard_brake":   is_hard_brake,
            "ev_hard_accel":   is_hard_accel,
            "ev_sharp_steer":  is_sharp_steer,
            "ev_lane_depart":  is_lane_depart,
            "ev_lat_accel":    is_lat_accel,
            "ev_lon_accel":    is_lon_accel,
            # 连续绩效指标（用于 Pearson 相关性）
            "mean_brake":      round(float(np.abs(brake).mean()),   4),
            "mean_lat_off":    round(float(np.abs(lat_off).mean()), 4),
            "mean_lon_acc":    round(float(np.abs(lon_acc).mean()), 4),
            "mean_steer_vel":  round(float(steering_angular_vel.mean()),       4),
            "mean_lat_acc":    round(float(np.abs(lat_acc).mean()), 4),
        })

    return pd.DataFrame(row_list)


def generate_event_annotations(act_sample_list: List, config: dict) -> pd.DataFrame:
    window_sample_count = config["window"]["act_hz"] * config["window"]["window_seconds"]
    event_threshold = config["event_thresholds"]
    data_frame_list = []
    for sample_idx, act_signal in enumerate(act_sample_list):
        event_df = detect_driving_events_per_sample(np.asarray(act_signal), sample_idx, window_sample_count, event_threshold)
        if not event_df.empty:
            data_frame_list.append(event_df)
    if not data_frame_list:
        raise RuntimeError(
            "未提取到任何事件标注窗口，请检查 raw_pkl 路径或 event_thresholds 阈值配置。"
        )
    result_df = pd.concat(data_frame_list, ignore_index=True)
    event_count = int(result_df["event_label"].sum())
    print(
        f"  事件标注完成：{len(result_df)} 个窗口，"
        f"异常窗口 {event_count}（{event_count/len(result_df)*100:.1f}%）"
    )
    return result_df


# =============================================================================
# 3. 合并风险结果与事件标注
# =============================================================================

def merge_risk_with_events(risk_df: pd.DataFrame, event_df: pd.DataFrame) -> pd.DataFrame:
    keep_risk_cols  = [col for col in ["sample_idx", "window_idx", "R_star", "risk_level",
                               "F_S", "Ad_norm", "A_d", "field_label"]
                  if col in risk_df.columns]
    keep_event_cols = [col for col in ["sample_idx", "window_idx", "event_label",
                               "mean_brake", "mean_lat_off", "mean_lon_acc",
                               "mean_steer_vel", "mean_lat_acc",
                               "ev_hard_brake", "ev_hard_accel", "ev_sharp_steer",
                               "ev_lane_depart", "ev_lat_accel", "ev_lon_accel"]
                  if col in event_df.columns]
    merged_df = pd.merge(
        risk_df[keep_risk_cols], event_df[keep_event_cols],
        on=["sample_idx", "window_idx"], how="inner",
    )
    print(
        f"  合并完成：{len(merged_df)} 个有效窗口  "
        f"（risk={len(risk_df)}，event={len(event_df)}）"
    )
    return merged_df


# =============================================================================
# 4. 统计验证
# =============================================================================

def calculate_pearson_correlation(merged_df: pd.DataFrame) -> pd.DataFrame:
    """R 与各客观绩效指标的 Pearson 相关系数。"""
    metric_mapping = {
        "异常事件标注(0/1)": "event_label",
        "平均刹车强度":       "mean_brake",
        "横向偏移均值(m)":    "mean_lat_off",
        "纵向加速度均值":     "mean_lon_acc",
        "方向盘角速度均值":   "mean_steer_vel",
        "横向加速度均值":     "mean_lat_acc",
    }
    row_list = []
    for label, col_name in metric_mapping.items():
        if col_name not in merged_df.columns:
            continue
        corr_coef, p_value = stats.pearsonr(merged_df["R_star"], merged_df[col_name])
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
    """
    ROC/AUC：以 event_label 为真值，R 为预测得分。
    绘图委托给 plot_risk.plot_roc_curve()。
    """
    try:
        from sklearn.metrics import (roc_auc_score, roc_curve,
                                      f1_score, precision_score, recall_score)
    except ImportError:
        print("  ⚠️  sklearn 未安装，跳过 ROC 验证（pip install scikit-learn）")
        return pd.DataFrame(), 0.0

    true_labels    = merged_df["event_label"].to_numpy(dtype=int)
    risk_scores = merged_df["R_star"].to_numpy(dtype=float)
    if true_labels.sum() == 0 or true_labels.sum() == len(true_labels):
        print("  ⚠️  event_label 全0或全1，无法计算 AUC，请调整 event_thresholds")
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

    # 绘图：委托给 plot_risk 模块
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
    else:
        print("  ⚠️  plot_risk 未加载，跳过 ROC 曲线图")

    return summary_df, roc_auc


def analyze_risk_level_event_rate(merged_df: pd.DataFrame, config: dict, output_dir: str) -> pd.DataFrame:
    """
    卡方检验：风险等级分组 × 异常事件发生率。
    绘图委托给 plot_risk.plot_risk_event_rate()。
    """
    risk_level_order = config["risk_level_order"]
    row_list  = []
    for level in risk_level_order:
        sub_df   = merged_df[merged_df["risk_level"] == level]
        total_count = len(sub_df)
        event_count  = int(sub_df["event_label"].sum())
        row_list.append({
            "风险等级":   level,
            "窗口数":     total_count,
            "占比(%)":    round(total_count / len(merged_df) * 100, 1),
            "事件窗口数": event_count,
            "事件率(%)":  round(event_count / max(total_count, 1) * 100, 1),
        })
    result_df = pd.DataFrame(row_list)

    contingency_table = pd.crosstab(merged_df["risk_level"], merged_df["event_label"])
    contingency_table = contingency_table.reindex([level for level in risk_level_order if level in contingency_table.index])
    chi2_stat, chi2_p, dof, _ = stats.chi2_contingency(contingency_table.values)
    chi2_result_str = (f"χ²={chi2_stat:.2f}, df={dof}, "
                f"p={'<0.001' if chi2_p < 0.001 else round(chi2_p, 4)}")

    print(f"\n── 风险等级 × 异常事件发生率 ─────────────────")
    print(result_df.to_string(index=False))
    print(f"  卡方检验: {chi2_result_str}")
    result_df["chi2检验"] = [chi2_result_str] + [""] * (len(result_df) - 1)

    # 绘图：委托给 plot_risk 模块
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
    else:
        print("  ⚠️  plot_risk 未加载，跳过风险等级图")

    return result_df


def perform_mannwhitney_test(merged_df: pd.DataFrame, config: dict, output_dir: str) -> pd.DataFrame:
    """
    Mann-Whitney U 检验 + R 分布箱线图。
    绘图委托给 plot_risk.plot_r_star_boxplot()。
    """
    risk_no_event = merged_df[merged_df["event_label"] == 0]["R_star"].values
    risk_with_event = merged_df[merged_df["event_label"] == 1]["R_star"].values
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

    # 绘图：委托给 plot_risk 模块
    if HAS_PLOT_MODULE:
        try:
            plot_risk_module.plot_r_star_boxplot(
                ev0=risk_no_event,
                ev1=risk_with_event,
                p_str=p_value_str,
                cfg=config,
                fig_dir=output_dir,
            )
        except Exception as e:
            print(f"  ⚠️  R 箱线图绘制失败: {e}")
    else:
        print("  ⚠️  plot_risk 未加载，跳过 R 箱线图")

    return summary_df


def breakdown_event_type_contribution(merged_df: pd.DataFrame) -> pd.DataFrame:
    """各子事件类型的触发率及对应 R 均值对比。"""
    event_type_mapping = {
        "急刹车":     "ev_hard_brake",
        "急加速":     "ev_hard_accel",
        "急打方向盘": "ev_sharp_steer",
        "车道偏离":   "ev_lane_depart",
        "横向超载":   "ev_lat_accel",
        "纵向超载":   "ev_lon_accel",
    }
    total_windows = len(merged_df)
    row_list  = []
    for event_name, col_name in event_type_mapping.items():
        if col_name not in merged_df.columns:
            continue
        trigger_count     = int(merged_df[col_name].sum())
        risk_mean_trigger  = merged_df[merged_df[col_name] == 1]["R_star"].mean()
        risk_mean_no_trigger = merged_df[merged_df[col_name] == 0]["R_star"].mean()
        row_list.append({
            "事件类型":        event_name,
            "触发窗口数":      trigger_count,
            "占总窗口比(%)":   round(trigger_count / total_windows * 100, 1),
            "R均值(触发)":    round(risk_mean_trigger,  4) if not np.isnan(risk_mean_trigger)  else "N/A",
            "R均值(未触发)":  round(risk_mean_no_trigger, 4) if not np.isnan(risk_mean_no_trigger) else "N/A",
        })
    result_df = pd.DataFrame(row_list).sort_values("触发窗口数", ascending=False)
    print(f"\n── 子事件类型贡献分析 ──────────────────────")
    print(result_df.to_string(index=False))
    return result_df


# =============================================================================
# 5. 汇总报告
# =============================================================================

def write_validation_summary(merged_df, corr_df, roc_df, level_df, mwu_df, config, output_dir):
    lines = [
        "=" * 65,
        "风险度 R 有效性验证报告",
        "=" * 65,
        f"有效验证窗口总数:  {len(merged_df)}",
        f"异常事件窗口数:    {int(merged_df['event_label'].sum())}",
        f"异常事件发生率:    {merged_df['event_label'].mean()*100:.1f}%",
        f"R 均值:           {merged_df['R_star'].mean():.4f}",
        f"R 标准差:         {merged_df['R_star'].std():.4f}",
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
        mwu_df.to_string(index=False),
        "",
        "── 事件检测阈值（来自 yaml）───────────────────────────",
    ]
    for key, value in config["event_thresholds"].items():
        lines.append(f"  {key}: {value}")
    lines += ["", "=" * 65]

    report = "\n".join(lines)
    print("\n" + report)
    save_path = os.path.join(output_dir, "validation_summary.txt")
    with open(save_path, "w", encoding="utf-8") as f:
        f.write(report)
    print(f"\n验证报告 → {save_path}")


# =============================================================================
# 主函数
# =============================================================================

def main():
    parser = argparse.ArgumentParser(
        description="R 有效性验证（自动事件标注 + 统计检验）",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例:
  python risk_validator.py
  python risk_validator.py -c config/risk_validator.yaml
  python risk_validator.py --brake_thresh 0.25 --lat_off_thresh 0.35
        """,
    )
    parser.add_argument("-c", "--config",      type=str,   default="config/risk_validator.yaml",
                        help="YAML 配置文件路径（默认: config/risk_validator.yaml）")
    parser.add_argument("--raw_pkl",           type=str,   default=None,
                        help="原始数据路径（覆盖 yaml paths.raw_pkl）")
    parser.add_argument("--risk_csv",          type=str,   default=None,
                        help="R 结果 CSV 路径（覆盖 yaml paths.risk_csv）")
    parser.add_argument("--output_dir",        type=str,   default=None,
                        help="输出目录（覆盖 yaml paths.output_dir）")
    parser.add_argument("--brake_thresh",      type=float, default=None,
                        help="急刹车阈值（覆盖 yaml event_thresholds.hard_brake_thresh）")
    parser.add_argument("--lat_off_thresh",    type=float, default=None,
                        help="车道偏离阈值（覆盖 yaml event_thresholds.lane_depart_thresh）")
    parser.add_argument("--steer_thresh",      type=float, default=None,
                        help="急打方向盘阈值（覆盖 yaml event_thresholds.sharp_steer_thresh）")
    args = parser.parse_args()

    config = load_config(args.config)

    # 命令行覆盖
    if args.raw_pkl:         config["paths"]["raw_pkl"]    = args.raw_pkl
    if args.risk_csv:        config["paths"]["risk_csv"]   = args.risk_csv
    if args.output_dir:      config["paths"]["output_dir"] = args.output_dir
    if args.brake_thresh   is not None:
        config["event_thresholds"]["hard_brake_thresh"]  = args.brake_thresh
    if args.lat_off_thresh is not None:
        config["event_thresholds"]["lane_depart_thresh"] = args.lat_off_thresh
    if args.steer_thresh   is not None:
        config["event_thresholds"]["sharp_steer_thresh"] = args.steer_thresh

    output_directory = config["paths"]["output_dir"]
    os.makedirs(output_directory, exist_ok=True)

    print("=" * 65)
    print("TCI 风险度 R 有效性验证")
    print(f"  原始数据:  {config['paths']['raw_pkl']}")
    print(f"  风险结果:  {config['paths']['risk_csv']}")
    print(f"  输出目录:  {output_directory}")
    print("=" * 65)

    # Step 1：加载原始数据
    print("\n[1/6] 加载原始数据...")
    original_act_samples = load_original_act_data(config["paths"]["raw_pkl"])

    # Step 2：生成客观事件标注
    print("\n[2/6] 生成客观事件标注...")
    event_annotation_df = generate_event_annotations(original_act_samples, config)
    event_annotation_df.to_csv(os.path.join(output_directory, "event_labels.csv"),
                    index=False, encoding="utf-8-sig")
    print(f"  事件标注 → {os.path.join(output_directory, 'event_labels.csv')}")

    # Step 3：加载 R 风险结果
    print("\n[3/6] 加载 R 风险结果...")
    risk_file_path = config["paths"]["risk_csv"]
    if not os.path.exists(risk_file_path):
        raise FileNotFoundError(
            f"风险结果文件不存在: {risk_file_path}\n"
            "请先运行 risk_evaluator.py 生成 risk_windows_all.csv"
        )
    risk_result_df = pd.read_csv(risk_file_path, encoding="utf-8-sig")
    print(f"  R 数据加载：{len(risk_result_df)} 行")

    # Step 4：合并
    print("\n[4/6] 合并风险结果与事件标注...")
    merged_risk_event_df = merge_risk_with_events(risk_result_df, event_annotation_df)
    merged_risk_event_df.to_csv(os.path.join(output_directory, "merged_risk_events.csv"),
                  index=False, encoding="utf-8-sig")

    # Step 5：统计验证
    print("\n[5/6] 统计验证...")
    correlation_df       = calculate_pearson_correlation(merged_risk_event_df)
    roc_summary_df, _     = compute_roc_auc_analysis(merged_risk_event_df, config, output_directory)
    risk_level_df      = analyze_risk_level_event_rate(merged_risk_event_df, config, output_directory)
    mannwhitney_df        = perform_mannwhitney_test(merged_risk_event_df, config, output_directory)
    event_type_detail_df = breakdown_event_type_contribution(merged_risk_event_df)

    # Step 6：保存 CSV
    print("\n[6/6] 保存结果...")
    correlation_df.to_csv(os.path.join(output_directory, "validation_correlation.csv"),
                   index=False, encoding="utf-8-sig")
    if not roc_summary_df.empty:
        roc_summary_df.to_csv(os.path.join(output_directory, "validation_roc.csv"),
                      index=False, encoding="utf-8-sig")
    risk_level_df.to_csv(os.path.join(output_directory, "validation_chisquare.csv"),
                    index=False, encoding="utf-8-sig")
    event_type_detail_df.to_csv(os.path.join(output_directory, "validation_event_types.csv"),
                          index=False, encoding="utf-8-sig")
    mannwhitney_df.to_csv(os.path.join(output_directory, "validation_mannwhitney.csv"),
                  index=False, encoding="utf-8-sig")

    write_validation_summary(merged_risk_event_df, correlation_df, roc_summary_df, risk_level_df, mannwhitney_df, config, output_directory)

    print(f"\n{'='*65}")
    print(f"验证完成！所有结果 → {output_directory}/")
    print(f"{'='*65}")


if __name__ == "__main__":
    main()
# D:\Local\DynamicCapRisk\src\3_risk_assessment\risk_validator.py

"""
risk_validator.py
风险度 R* 有效性验证模块

功能：
  1. 从原始驾驶数据（act）按非重叠窗口自动提取客观异常事件标注
  2. 基于事件标注对 R* 进行多维度统计验证：
       - Pearson 相关性（R* vs. 各绩效指标）
       - ROC / AUC 判别分析（R* 预测异常事件能力）
       - 卡方检验（风险等级分组 × 事件发生率一致性）
       - Mann-Whitney U 检验（事件组 vs. 无事件组 R* 分布）
       - 子事件类型贡献分析

默认配置文件：config/risk_validator.yaml
用法示例：
  python risk_validator.py
  python risk_validator.py -c config/risk_validator.yaml
  python risk_validator.py --brake_thresh 0.25 --lat_off_thresh 0.35

输出（→ paths.output_dir）：
  event_labels.csv              每窗口事件标注明细
  merged_risk_events.csv        R* 与事件标注合并表
  validation_correlation.csv    Pearson 相关性结果
  validation_roc.csv            ROC/AUC 结果
  validation_chisquare.csv      卡方检验（风险等级 × 事件率）
  validation_event_types.csv    子事件类型贡献分析
  validation_mannwhitney.csv    Mann-Whitney U 结果
  validation_summary.txt        文字汇总报告
  fig_roc_curve.png             → 由 plot_risk.plot_roc_curve() 生成
  fig_risk_event_rate.png       → 由 plot_risk.plot_risk_event_rate() 生成
  fig_r_star_boxplot.png        → 由 plot_risk.plot_r_star_boxplot() 生成
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
_VIS_DIR = os.path.join(os.path.dirname(__file__), "..", "visualization")
if _VIS_DIR not in sys.path:
    sys.path.insert(0, os.path.abspath(_VIS_DIR))

try:
    import plot_risk as _pr
    _HAS_PLOT = True
except ImportError:
    _HAS_PLOT = False
    print("⚠️  无法导入 plot_risk 模块，绘图功能将跳过。")


# =============================================================================
# 配置加载
# =============================================================================

_FALLBACK = {
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


def _deep_merge(base: dict, override: dict) -> dict:
    result = base.copy()
    for k, v in override.items():
        if isinstance(v, dict) and isinstance(result.get(k), dict):
            result[k] = _deep_merge(result[k], v)
        else:
            result[k] = v
    return result


def load_config(path: Optional[str] = None) -> dict:
    import copy
    cfg = copy.deepcopy(_FALLBACK)
    if path and os.path.exists(path):
        with open(path, "r", encoding="utf-8") as f:
            user = yaml.safe_load(f) or {}
        cfg = _deep_merge(cfg, user)
        print(f"已加载配置: {path}")
    elif path:
        print(f"⚠️  配置文件不存在: {path}，使用内置默认值")
    else:
        print("未指定配置文件，使用内置默认值")
    return cfg


# =============================================================================
# 1. 原始数据加载
# =============================================================================

def load_raw(pkl_path: str) -> List:
    if not os.path.exists(pkl_path):
        raise FileNotFoundError(f"原始数据不存在: {pkl_path}")
    with open(pkl_path, "rb") as f:
        data = pickle.load(f)
    act = data.get("act", [])
    print(f"  原始 act 加载完成：{len(act)} 个样本")
    return act


# =============================================================================
# 2. 客观事件标注
# =============================================================================

def detect_events_single_sample(
    act: np.ndarray,
    sample_idx: int,
    window_size: int,
    thr: dict,
) -> pd.DataFrame:
    """
    对单样本 act 信号按非重叠窗口检测异常驾驶事件。

    act 维度（60 Hz）：
      0: 油门, 1: 刹车, 2: 方向盘转角, 3: 车速,
      4: 纵向加速度, 5: 横向加速度, 6: X, 7: Y,
      8: 横向偏移, 9: 路段类型

    event_label = 1 表示该窗口内至少触发一类子事件。
    """
    arr = np.asarray(act, dtype=float)
    n_windows = arr.shape[0] // window_size
    if n_windows == 0:
        return pd.DataFrame()

    dt = 1.0 / 60.0   # act 采样间隔（秒）
    rows = []
    for w in range(n_windows):
        seg = arr[w * window_size: (w + 1) * window_size]

        throttle = seg[:, 0]
        brake    = seg[:, 1]
        steer    = seg[:, 2]
        lon_acc  = seg[:, 4]
        lat_acc  = seg[:, 5]
        lat_off  = seg[:, 8]

        steer_vel = np.abs(np.diff(steer, prepend=steer[0])) / dt

        ev_hard_brake  = int(np.abs(brake).mean()   > thr["hard_brake_thresh"])
        ev_hard_accel  = int(throttle.mean()         > thr["hard_accel_thresh"])
        ev_sharp_steer = int(steer_vel.mean()        > thr["sharp_steer_thresh"])
        ev_lane_depart = int(np.abs(lat_off).mean()  > thr["lane_depart_thresh"])
        ev_lat_accel   = int(np.abs(lat_acc).mean()  > thr["lat_accel_thresh"])
        ev_lon_accel   = int(np.abs(lon_acc).mean()  > thr["lon_accel_thresh"])

        event_label = int(
            ev_hard_brake | ev_hard_accel | ev_sharp_steer |
            ev_lane_depart | ev_lat_accel | ev_lon_accel
        )

        rows.append({
            "sample_idx":      sample_idx,
            "window_idx":      w,
            "event_label":     event_label,
            "ev_hard_brake":   ev_hard_brake,
            "ev_hard_accel":   ev_hard_accel,
            "ev_sharp_steer":  ev_sharp_steer,
            "ev_lane_depart":  ev_lane_depart,
            "ev_lat_accel":    ev_lat_accel,
            "ev_lon_accel":    ev_lon_accel,
            # 连续绩效指标（用于 Pearson 相关性）
            "mean_brake":      round(float(np.abs(brake).mean()),   4),
            "mean_lat_off":    round(float(np.abs(lat_off).mean()), 4),
            "mean_lon_acc":    round(float(np.abs(lon_acc).mean()), 4),
            "mean_steer_vel":  round(float(steer_vel.mean()),       4),
            "mean_lat_acc":    round(float(np.abs(lat_acc).mean()), 4),
        })

    return pd.DataFrame(rows)


def build_event_labels(act_list: List, cfg: dict) -> pd.DataFrame:
    win = cfg["window"]["act_hz"] * cfg["window"]["window_seconds"]
    thr = cfg["event_thresholds"]
    frames = []
    for i, act in enumerate(act_list):
        df = detect_events_single_sample(np.asarray(act), i, win, thr)
        if not df.empty:
            frames.append(df)
    if not frames:
        raise RuntimeError(
            "未提取到任何事件标注窗口，请检查 raw_pkl 路径或 event_thresholds 阈值配置。"
        )
    result = pd.concat(frames, ignore_index=True)
    n_ev = int(result["event_label"].sum())
    print(
        f"  事件标注完成：{len(result)} 个窗口，"
        f"异常窗口 {n_ev}（{n_ev/len(result)*100:.1f}%）"
    )
    return result


# =============================================================================
# 3. 合并风险结果与事件标注
# =============================================================================

def merge_risk_events(risk_df: pd.DataFrame,
                      event_df: pd.DataFrame) -> pd.DataFrame:
    keep_risk  = [c for c in ["sample_idx", "window_idx", "R_star", "risk_level",
                               "F_S", "Ad_norm", "A_d", "field_label"]
                  if c in risk_df.columns]
    keep_event = [c for c in ["sample_idx", "window_idx", "event_label",
                               "mean_brake", "mean_lat_off", "mean_lon_acc",
                               "mean_steer_vel", "mean_lat_acc",
                               "ev_hard_brake", "ev_hard_accel", "ev_sharp_steer",
                               "ev_lane_depart", "ev_lat_accel", "ev_lon_accel"]
                  if c in event_df.columns]
    merged = pd.merge(
        risk_df[keep_risk], event_df[keep_event],
        on=["sample_idx", "window_idx"], how="inner",
    )
    print(
        f"  合并完成：{len(merged)} 个有效窗口  "
        f"（risk={len(risk_df)}，event={len(event_df)}）"
    )
    return merged


# =============================================================================
# 4. 统计验证
# =============================================================================

def validate_correlation(df: pd.DataFrame) -> pd.DataFrame:
    """R* 与各客观绩效指标的 Pearson 相关系数。"""
    metrics = {
        "异常事件标注(0/1)": "event_label",
        "平均刹车强度":       "mean_brake",
        "横向偏移均值(m)":    "mean_lat_off",
        "纵向加速度均值":     "mean_lon_acc",
        "方向盘角速度均值":   "mean_steer_vel",
        "横向加速度均值":     "mean_lat_acc",
    }
    rows = []
    for label, col in metrics.items():
        if col not in df.columns:
            continue
        r, p = stats.pearsonr(df["R_star"], df[col])
        rows.append({
            "绩效指标":   label,
            "相关系数r":  round(r, 3),
            "显著性p":    "<0.001" if p < 0.001 else round(p, 4),
            "相关方向":   "正相关" if r > 0 else "负相关",
        })
    result = pd.DataFrame(rows)
    print("\n── Pearson 相关性验证 ────────────────────────")
    print(result.to_string(index=False))
    return result


def validate_roc(df: pd.DataFrame,
                 cfg: dict,
                 out_dir: str) -> Tuple[pd.DataFrame, float]:
    """
    ROC/AUC：以 event_label 为真值，R* 为预测得分。
    绘图委托给 plot_risk.plot_roc_curve()。
    """
    try:
        from sklearn.metrics import (roc_auc_score, roc_curve,
                                      f1_score, precision_score, recall_score)
    except ImportError:
        print("  ⚠️  sklearn 未安装，跳过 ROC 验证（pip install scikit-learn）")
        return pd.DataFrame(), 0.0

    y    = df["event_label"].to_numpy(dtype=int)
    pred = df["R_star"].to_numpy(dtype=float)
    if y.sum() == 0 or y.sum() == len(y):
        print("  ⚠️  event_label 全0或全1，无法计算 AUC，请调整 event_thresholds")
        return pd.DataFrame(), 0.0

    auc           = roc_auc_score(y, pred)
    fpr, tpr, thr = roc_curve(y, pred)
    j_idx         = np.argmax(tpr - fpr)
    best_thr      = thr[j_idx]
    y_pred        = (pred >= best_thr).astype(int)

    summary = pd.DataFrame([{
        "AUC":       round(auc, 3),
        "最优阈值":  round(best_thr, 3),
        "F1":        round(f1_score(y, y_pred), 3),
        "Precision": round(precision_score(y, y_pred, zero_division=0), 3),
        "Recall":    round(recall_score(y, y_pred), 3),
        "Youden_J":  round(tpr[j_idx] - fpr[j_idx], 3),
    }])
    print(f"\n── ROC/AUC 验证 ─────────────────────────────")
    print(summary.to_string(index=False))

    # 绘图：委托给 plot_risk 模块
    if _HAS_PLOT:
        try:
            _pr.plot_roc_curve(
                fpr=fpr,
                tpr=tpr,
                auc=auc,
                best_thr=best_thr,
                best_fpr=float(fpr[j_idx]),
                best_tpr=float(tpr[j_idx]),
                cfg=cfg,
                fig_dir=out_dir,
            )
        except Exception as e:
            print(f"  ⚠️  ROC 图绘制失败: {e}")
    else:
        print("  ⚠️  plot_risk 未加载，跳过 ROC 曲线图")

    return summary, auc


def validate_risk_level_event_rate(df: pd.DataFrame,
                                    cfg: dict,
                                    out_dir: str) -> pd.DataFrame:
    """
    卡方检验：风险等级分组 × 异常事件发生率。
    绘图委托给 plot_risk.plot_risk_event_rate()。
    """
    order = cfg["risk_level_order"]
    rows  = []
    for lvl in order:
        sub   = df[df["risk_level"] == lvl]
        total = len(sub)
        n_ev  = int(sub["event_label"].sum())
        rows.append({
            "风险等级":   lvl,
            "窗口数":     total,
            "占比(%)":    round(total / len(df) * 100, 1),
            "事件窗口数": n_ev,
            "事件率(%)":  round(n_ev / max(total, 1) * 100, 1),
        })
    result = pd.DataFrame(rows)

    contingency = pd.crosstab(df["risk_level"], df["event_label"])
    contingency = contingency.reindex([l for l in order if l in contingency.index])
    chi2, p, dof, _ = stats.chi2_contingency(contingency.values)
    chi2_str = (f"χ²={chi2:.2f}, df={dof}, "
                f"p={'<0.001' if p < 0.001 else round(p, 4)}")

    print(f"\n── 风险等级 × 异常事件发生率 ─────────────────")
    print(result.to_string(index=False))
    print(f"  卡方检验: {chi2_str}")
    result["chi2检验"] = [chi2_str] + [""] * (len(result) - 1)

    # 绘图：委托给 plot_risk 模块
    if _HAS_PLOT:
        try:
            _pr.plot_risk_event_rate(
                level_df=result,
                chi2_str=chi2_str,
                cfg=cfg,
                fig_dir=out_dir,
            )
        except Exception as e:
            print(f"  ⚠️  风险等级图绘制失败: {e}")
    else:
        print("  ⚠️  plot_risk 未加载，跳过风险等级图")

    return result


def plot_r_star_by_event(df: pd.DataFrame,
                          cfg: dict,
                          out_dir: str) -> pd.DataFrame:
    """
    Mann-Whitney U 检验 + R* 分布箱线图。
    绘图委托给 plot_risk.plot_r_star_boxplot()。
    """
    ev0 = df[df["event_label"] == 0]["R_star"].values
    ev1 = df[df["event_label"] == 1]["R_star"].values
    stat, p = stats.mannwhitneyu(ev0, ev1, alternative="less")
    p_str = "<0.001" if p < 0.001 else str(round(p, 4))

    summary = pd.DataFrame([
        {"组别": "无异常事件", "样本数": len(ev0),
         "R*均值": round(ev0.mean(), 4), "R*标准差": round(ev0.std(), 4)},
        {"组别": "异常事件",   "样本数": len(ev1),
         "R*均值": round(ev1.mean(), 4), "R*标准差": round(ev1.std(), 4)},
    ])
    print(f"\n── Mann-Whitney U 检验 ─────────────────────")
    print(summary.to_string(index=False))
    print(f"  U={stat:.1f}, p={p_str}（单侧：事件组 R* > 无事件组）")

    # 绘图：委托给 plot_risk 模块
    if _HAS_PLOT:
        try:
            _pr.plot_r_star_boxplot(
                ev0=ev0,
                ev1=ev1,
                p_str=p_str,
                cfg=cfg,
                fig_dir=out_dir,
            )
        except Exception as e:
            print(f"  ⚠️  R* 箱线图绘制失败: {e}")
    else:
        print("  ⚠️  plot_risk 未加载，跳过 R* 箱线图")

    return summary


def analyze_event_types(df: pd.DataFrame) -> pd.DataFrame:
    """各子事件类型的触发率及对应 R* 均值对比。"""
    ev_cols = {
        "急刹车":     "ev_hard_brake",
        "急加速":     "ev_hard_accel",
        "急打方向盘": "ev_sharp_steer",
        "车道偏离":   "ev_lane_depart",
        "横向超载":   "ev_lat_accel",
        "纵向超载":   "ev_lon_accel",
    }
    total = len(df)
    rows  = []
    for name, col in ev_cols.items():
        if col not in df.columns:
            continue
        n     = int(df[col].sum())
        m_ev  = df[df[col] == 1]["R_star"].mean()
        m_nev = df[df[col] == 0]["R_star"].mean()
        rows.append({
            "事件类型":        name,
            "触发窗口数":      n,
            "占总窗口比(%)":   round(n / total * 100, 1),
            "R*均值(触发)":    round(m_ev,  4) if not np.isnan(m_ev)  else "N/A",
            "R*均值(未触发)":  round(m_nev, 4) if not np.isnan(m_nev) else "N/A",
        })
    result = pd.DataFrame(rows).sort_values("触发窗口数", ascending=False)
    print(f"\n── 子事件类型贡献分析 ──────────────────────")
    print(result.to_string(index=False))
    return result


# =============================================================================
# 5. 汇总报告
# =============================================================================

def write_summary(merged, corr_df, roc_df, level_df, mwu_df, cfg, out_dir):
    lines = [
        "=" * 65,
        "风险度 R* 有效性验证报告",
        "=" * 65,
        f"有效验证窗口总数:  {len(merged)}",
        f"异常事件窗口数:    {int(merged['event_label'].sum())}",
        f"异常事件发生率:    {merged['event_label'].mean()*100:.1f}%",
        f"R* 均值:           {merged['R_star'].mean():.4f}",
        f"R* 标准差:         {merged['R_star'].std():.4f}",
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
    for k, v in cfg["event_thresholds"].items():
        lines.append(f"  {k}: {v}")
    lines += ["", "=" * 65]

    report = "\n".join(lines)
    print("\n" + report)
    path = os.path.join(out_dir, "validation_summary.txt")
    with open(path, "w", encoding="utf-8") as f:
        f.write(report)
    print(f"\n验证报告 → {path}")


# =============================================================================
# 主函数
# =============================================================================

def main():
    parser = argparse.ArgumentParser(
        description="R* 有效性验证（自动事件标注 + 统计检验）",
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
                        help="R* 结果 CSV 路径（覆盖 yaml paths.risk_csv）")
    parser.add_argument("--output_dir",        type=str,   default=None,
                        help="输出目录（覆盖 yaml paths.output_dir）")
    parser.add_argument("--brake_thresh",      type=float, default=None,
                        help="急刹车阈值（覆盖 yaml event_thresholds.hard_brake_thresh）")
    parser.add_argument("--lat_off_thresh",    type=float, default=None,
                        help="车道偏离阈值（覆盖 yaml event_thresholds.lane_depart_thresh）")
    parser.add_argument("--steer_thresh",      type=float, default=None,
                        help="急打方向盘阈值（覆盖 yaml event_thresholds.sharp_steer_thresh）")
    args = parser.parse_args()

    cfg = load_config(args.config)

    # 命令行覆盖
    if args.raw_pkl:         cfg["paths"]["raw_pkl"]    = args.raw_pkl
    if args.risk_csv:        cfg["paths"]["risk_csv"]   = args.risk_csv
    if args.output_dir:      cfg["paths"]["output_dir"] = args.output_dir
    if args.brake_thresh   is not None:
        cfg["event_thresholds"]["hard_brake_thresh"]  = args.brake_thresh
    if args.lat_off_thresh is not None:
        cfg["event_thresholds"]["lane_depart_thresh"] = args.lat_off_thresh
    if args.steer_thresh   is not None:
        cfg["event_thresholds"]["sharp_steer_thresh"] = args.steer_thresh

    out_dir = cfg["paths"]["output_dir"]
    os.makedirs(out_dir, exist_ok=True)

    print("=" * 65)
    print("TCI 风险度 R* 有效性验证")
    print(f"  原始数据:  {cfg['paths']['raw_pkl']}")
    print(f"  风险结果:  {cfg['paths']['risk_csv']}")
    print(f"  输出目录:  {out_dir}")
    print("=" * 65)

    # Step 1：加载原始数据
    print("\n[1/6] 加载原始数据...")
    act_list = load_raw(cfg["paths"]["raw_pkl"])

    # Step 2：生成客观事件标注
    print("\n[2/6] 生成客观事件标注...")
    event_df = build_event_labels(act_list, cfg)
    event_df.to_csv(os.path.join(out_dir, "event_labels.csv"),
                    index=False, encoding="utf-8-sig")
    print(f"  事件标注 → {os.path.join(out_dir, 'event_labels.csv')}")

    # Step 3：加载 R* 风险结果
    print("\n[3/6] 加载 R* 风险结果...")
    risk_path = cfg["paths"]["risk_csv"]
    if not os.path.exists(risk_path):
        raise FileNotFoundError(
            f"风险结果文件不存在: {risk_path}\n"
            "请先运行 risk_evaluator.py 生成 risk_windows_all.csv"
        )
    risk_df = pd.read_csv(risk_path, encoding="utf-8-sig")
    print(f"  R* 数据加载：{len(risk_df)} 行")

    # Step 4：合并
    print("\n[4/6] 合并风险结果与事件标注...")
    merged = merge_risk_events(risk_df, event_df)
    merged.to_csv(os.path.join(out_dir, "merged_risk_events.csv"),
                  index=False, encoding="utf-8-sig")

    # Step 5：统计验证
    print("\n[5/6] 统计验证...")
    corr_df       = validate_correlation(merged)
    roc_df, _     = validate_roc(merged, cfg, out_dir)
    level_df      = validate_risk_level_event_rate(merged, cfg, out_dir)
    mwu_df        = plot_r_star_by_event(merged, cfg, out_dir)
    event_type_df = analyze_event_types(merged)

    # Step 6：保存 CSV
    print("\n[6/6] 保存结果...")
    corr_df.to_csv(os.path.join(out_dir, "validation_correlation.csv"),
                   index=False, encoding="utf-8-sig")
    if not roc_df.empty:
        roc_df.to_csv(os.path.join(out_dir, "validation_roc.csv"),
                      index=False, encoding="utf-8-sig")
    level_df.to_csv(os.path.join(out_dir, "validation_chisquare.csv"),
                    index=False, encoding="utf-8-sig")
    event_type_df.to_csv(os.path.join(out_dir, "validation_event_types.csv"),
                          index=False, encoding="utf-8-sig")
    mwu_df.to_csv(os.path.join(out_dir, "validation_mannwhitney.csv"),
                  index=False, encoding="utf-8-sig")

    write_summary(merged, corr_df, roc_df, level_df, mwu_df, cfg, out_dir)

    print(f"\n{'='*65}")
    print(f"验证完成！所有结果 → {out_dir}/")
    print(f"{'='*65}")


if __name__ == "__main__":
    main()

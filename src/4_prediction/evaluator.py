# D:\Local\DynamicCapRisk\src\4_prediction\evaluator.py

"""
MT-JP 模型评估器

评估指标（论文 §5.3）：
  能力预测（回归）：R²、MAE、RMSE
  风险度预测（回归）：R²、MAE、RMSE、高风险召回率/精确率
  风险等级预测（分类）：准确率、宏平均F1、Kappa系数、混淆矩阵、
                        各类别 Precision / Recall / F1

输出：
  evaluation_report.txt   文字报告（含混淆矩阵）
  evaluation_metrics.csv  数值指标（可供绘图）
  predictions.csv         每条测试样本的真实值与预测值

用法：
  python evaluator.py
  python evaluator.py -c config/evaluator.yaml
  python evaluator.py --ckpt output/3_prediction/checkpoints/best_model.pt
"""

import os
import sys
import pickle
import argparse
import warnings
import yaml
import copy
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from typing import Dict, Optional, Tuple

warnings.filterwarnings("ignore")

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from mtjp_model import build_model
from trainer import MTJPDataset


# =============================================================================
# 默认配置
# =============================================================================

_DEFAULTS = {
    "paths": {
        "dataset_pkl":  "output/3_prediction/mtjp_dataset.pkl",
        "ckpt":         "output/3_prediction/checkpoints/best_model.pt",
        "output_dir":   "output/3_prediction/evaluation",
    },
    "eval": {
        "batch_size":   256,
        "split":        "test",      # train / val / test
        "risk_thresh_high": 0.1,     # R* > 0.1 视为高风险（与论文图5.8阈值线一致）
        "risk_thresh_low": -0.1,     # R* < -0.1 视为低风险
    },
    "class_names": ["低风险", "中风险", "高风险"],
}

_DEFAULT_CONFIG_PATH = os.path.join(
    os.path.dirname(os.path.abspath(__file__)), "..", "..", "config", "evaluator.yaml"
)


def _deep_merge(base: dict, override: dict) -> dict:
    result = base.copy()
    for k, v in override.items():
        if isinstance(v, dict) and isinstance(result.get(k), dict):
            result[k] = _deep_merge(result[k], v)
        else:
            result[k] = v
    return result


def load_config(path: Optional[str] = None) -> dict:
    cfg = copy.deepcopy(_DEFAULTS)
    candidate = path or _DEFAULT_CONFIG_PATH
    if candidate and os.path.exists(candidate):
        with open(candidate, "r", encoding="utf-8") as f:
            user = yaml.safe_load(f) or {}
        cfg = _deep_merge(cfg, user)
        print(f"已加载配置文件: {os.path.abspath(candidate)}")
    else:
        print("使用内置默认配置")
    return cfg


# =============================================================================
# 推理：收集全部预测结果
# =============================================================================

@torch.no_grad()
def collect_predictions(
    model:    nn.Module,
    loader:   DataLoader,
    device:   torch.device,
) -> Dict[str, np.ndarray]:
    """
    对整个 DataLoader 执行推理，返回：
      ability_true / ability_pred   : (N,)  float32
      risk_reg_true / risk_reg_pred : (N,)  float32
      risk_cls_true / risk_cls_pred : (N,)  int
      risk_cls_prob                 : (N, 3) float32  softmax 概率
      f_s                           : (N,)  float32
    """
    model.eval()

    ability_true, ability_pred   = [], []
    risk_reg_true, risk_reg_pred = [], []
    risk_cls_true, risk_cls_pred = [], []
    risk_cls_prob_all            = []
    f_s_all                      = []

    for X, ya, yr, yc in loader:
        X = X.to(device)
        out = model(X)

        f_s = X[:, -1, 16].cpu().numpy()   # (B,)

        a_hat  = out["ability"].squeeze(-1).cpu().numpy()
        rr_hat = out["risk_reg"].squeeze(-1).cpu().numpy()
        rc_logits = out["risk_cls"].cpu().numpy()            # (B,3)
        rc_prob   = _softmax(rc_logits)                      # (B,3)
        rc_hat    = rc_prob.argmax(axis=-1)                  # (B,)

        ability_true.append(ya.numpy())
        ability_pred.append(a_hat)
        risk_reg_true.append(yr.numpy())
        risk_reg_pred.append(rr_hat)
        risk_cls_true.append(yc.numpy())
        risk_cls_pred.append(rc_hat)
        risk_cls_prob_all.append(rc_prob)
        f_s_all.append(f_s)

    return {
        "ability_true":  np.concatenate(ability_true),
        "ability_pred":  np.concatenate(ability_pred),
        "risk_reg_true": np.concatenate(risk_reg_true),
        "risk_reg_pred": np.concatenate(risk_reg_pred),
        "risk_cls_true": np.concatenate(risk_cls_true).astype(int),
        "risk_cls_pred": np.concatenate(risk_cls_pred).astype(int),
        "risk_cls_prob": np.concatenate(risk_cls_prob_all),
        "f_s":           np.concatenate(f_s_all),
    }


def _softmax(x: np.ndarray) -> np.ndarray:
    e = np.exp(x - x.max(axis=-1, keepdims=True))
    return e / e.sum(axis=-1, keepdims=True)


# =============================================================================
# 指标计算
# =============================================================================

def r2_score(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    ss_res = np.sum((y_true - y_pred) ** 2)
    ss_tot = np.sum((y_true - y_true.mean()) ** 2)
    return float(1.0 - ss_res / max(ss_tot, 1e-12))


def mae(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return float(np.mean(np.abs(y_true - y_pred)))


def rmse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return float(np.sqrt(np.mean((y_true - y_pred) ** 2)))


def confusion_matrix(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    n_classes: int = 3,
) -> np.ndarray:
    cm = np.zeros((n_classes, n_classes), dtype=int)
    for t, p in zip(y_true, y_pred):
        if 0 <= t < n_classes and 0 <= p < n_classes:
            cm[t, p] += 1
    return cm


def per_class_metrics(cm: np.ndarray) -> Dict[str, np.ndarray]:
    """从混淆矩阵计算各类别 Precision / Recall / F1。"""
    n = cm.shape[0]
    precision = np.zeros(n)
    recall    = np.zeros(n)
    f1        = np.zeros(n)
    for c in range(n):
        tp = cm[c, c]
        fp = cm[:, c].sum() - tp
        fn = cm[c, :].sum() - tp
        precision[c] = tp / max(tp + fp, 1)
        recall[c]    = tp / max(tp + fn, 1)
        f1[c]        = (2 * precision[c] * recall[c]
                        / max(precision[c] + recall[c], 1e-9))
    return {"precision": precision, "recall": recall, "f1": f1}


def kappa_score(cm: np.ndarray) -> float:
    """Cohen's Kappa 系数。"""
    n   = cm.sum()
    po  = cm.diagonal().sum() / max(n, 1)
    pe  = (cm.sum(axis=1) * cm.sum(axis=0)).sum() / max(n * n, 1)
    return float((po - pe) / max(1 - pe, 1e-9))


def high_risk_metrics(
    risk_reg_true: np.ndarray,
    risk_reg_pred: np.ndarray,
    threshold: float = 0.1,
) -> Tuple[float, float]:
    """
    以 R* > threshold 定义高风险正例，
    返回 (召回率, 精确率)。
    """
    true_pos = risk_reg_true > threshold
    pred_pos = risk_reg_pred > threshold
    tp = int(( true_pos &  pred_pos).sum())
    fn = int(( true_pos & ~pred_pos).sum())
    fp = int((~true_pos &  pred_pos).sum())
    recall_hr    = tp / max(tp + fn, 1)
    precision_hr = tp / max(tp + fp, 1)
    return float(recall_hr), float(precision_hr)


# =============================================================================
# 报告生成
# =============================================================================

def _fmt_cm(cm: np.ndarray, class_names) -> str:
    """将混淆矩阵格式化为对齐文本。"""
    header = "真实\\预测   " + "   ".join(f"{n:^8s}" for n in class_names) + "   召回率"
    lines  = [header, "-" * len(header)]
    row_sum = cm.sum(axis=1)
    for i, name in enumerate(class_names):
        cells  = "   ".join(f"{cm[i,j]:^8d}" for j in range(len(class_names)))
        recall = cm[i, i] / max(row_sum[i], 1) * 100
        lines.append(f"{name:^8s}     {cells}   {recall:.1f}%")
    lines.append("-" * len(header))
    col_sum = cm.sum(axis=0)
    prec_row = "   ".join(
        f"{cm[j,j]/max(col_sum[j],1)*100:^7.1f}%" for j in range(len(class_names))
    )
    lines.append(f"{'精确率':^8s}     {prec_row}")
    return "\n".join(lines)


def build_report(
    preds:       Dict[str, np.ndarray],
    cfg:         dict,
) -> Tuple[str, Dict, pd.DataFrame]:
    """
    生成三份输出：
      report_str : 可打印的文字报告
      metrics    : 结构化指标字典
      pred_df    : 逐样本预测 DataFrame
    """
    ec   = cfg["eval"]
    cnames = cfg["class_names"]

    # ── 能力预测指标 ─────────────────────────────────────────
    a_r2   = r2_score(preds["ability_true"], preds["ability_pred"])
    a_mae  = mae(preds["ability_true"],  preds["ability_pred"])
    a_rmse = rmse(preds["ability_true"], preds["ability_pred"])

    # ── 风险度回归指标 ────────────────────────────────────────
    r_r2   = r2_score(preds["risk_reg_true"], preds["risk_reg_pred"])
    r_mae  = mae(preds["risk_reg_true"],  preds["risk_reg_pred"])
    r_rmse = rmse(preds["risk_reg_true"], preds["risk_reg_pred"])
    hr_recall, hr_prec = high_risk_metrics(
        preds["risk_reg_true"], preds["risk_reg_pred"], ec["risk_thresh_high"]
    )

    # ── 风险等级分类指标 ──────────────────────────────────────
    cm     = confusion_matrix(preds["risk_cls_true"], preds["risk_cls_pred"], 3)
    pcm    = per_class_metrics(cm)
    kappa  = kappa_score(cm)
    acc    = cm.diagonal().sum() / max(cm.sum(), 1)
    macro_f1 = float(pcm["f1"].mean())

    # ── 组装指标字典 ─────────────────────────────────────────
    metrics = {
        "ability_R2":       round(a_r2,   4),
        "ability_MAE":      round(a_mae,  4),
        "ability_RMSE":     round(a_rmse, 4),
        "risk_reg_R2":      round(r_r2,   4),
        "risk_reg_MAE":     round(r_mae,  4),
        "risk_reg_RMSE":    round(r_rmse, 4),
        "high_risk_recall": round(hr_recall, 4),
        "high_risk_prec":   round(hr_prec,   4),
        "cls_accuracy":     round(float(acc),      4),
        "cls_macro_f1":     round(float(macro_f1), 4),
        "cls_kappa":        round(float(kappa),    4),
    }
    for i, name in enumerate(cnames):
        metrics[f"{name}_precision"] = round(float(pcm["precision"][i]), 4)
        metrics[f"{name}_recall"]    = round(float(pcm["recall"][i]),    4)
        metrics[f"{name}_f1"]        = round(float(pcm["f1"][i]),        4)

    # ── 逐样本 DataFrame ─────────────────────────────────────
    pred_df = pd.DataFrame({
        "ability_true":   preds["ability_true"],
        "ability_pred":   preds["ability_pred"].round(4),
        "risk_reg_true":  preds["risk_reg_true"],
        "risk_reg_pred":  preds["risk_reg_pred"].round(4),
        "risk_cls_true":  preds["risk_cls_true"],
        "risk_cls_pred":  preds["risk_cls_pred"],
        "prob_low":       preds["risk_cls_prob"][:, 0].round(4),
        "prob_mid":       preds["risk_cls_prob"][:, 1].round(4),
        "prob_high":      preds["risk_cls_prob"][:, 2].round(4),
        "f_s":            preds["f_s"].round(4),
    })

    # ── 文字报告 ─────────────────────────────────────────────
    n = len(preds["ability_true"])
    sep = "=" * 60
    lines = [
        sep,
        "MT-JP 联合预测模型 — 评估报告",
        f"评估样本数: {n}",
        sep,
        "",
        "【一】动态驾驶能力预测（回归）",
        f"  R²   = {a_r2:.4f}  （>0.8 为良好）",
        f"  MAE  = {a_mae:.4f}",
        f"  RMSE = {a_rmse:.4f}",
        "",
        "【二】风险度预测（回归）",
        f"  R²             = {r_r2:.4f}",
        f"  MAE            = {r_mae:.4f}",
        f"  RMSE           = {r_rmse:.4f}",
        f"  高风险召回率   = {hr_recall*100:.1f}%  （阈值 R*>{ec['risk_thresh_high']}）",
        f"  高风险精确率   = {hr_prec*100:.1f}%",
        "",
        "【三】风险等级分类",
        f"  总体准确率     = {acc*100:.1f}%",
        f"  宏平均 F1      = {macro_f1:.4f}",
        f"  Kappa 系数     = {kappa:.4f}",
        "",
        "  各类别指标:",
    ]
    for i, name in enumerate(cnames):
        lines.append(
            f"    {name}: Precision={pcm['precision'][i]*100:.1f}%  "
            f"Recall={pcm['recall'][i]*100:.1f}%  "
            f"F1={pcm['f1'][i]:.4f}"
        )
    lines += [
        "",
        "  混淆矩阵:",
        _fmt_cm(cm, cnames),
        "",
        sep,
    ]
    report_str = "\n".join(lines)

    return report_str, metrics, pred_df


# =============================================================================
# 分组分析（按能力等级）
# =============================================================================

def group_analysis(
    preds: Dict[str, np.ndarray],
) -> str:
    """
    将测试样本按真实能力三等分（高/中/低能力组），
    分别统计能力预测 R²，对应论文表5.4 的分组结果。
    """
    at = preds["ability_true"]
    ap = preds["ability_pred"]
    idx_sorted = np.argsort(at)
    n = len(at)
    groups = {
        "低能力组": idx_sorted[:n//3],
        "中能力组": idx_sorted[n//3: 2*n//3],
        "高能力组": idx_sorted[2*n//3:],
    }
    lines = [
        "【附】分组能力预测性能（按真实能力三等分）",
        f"  {'组别':8s}  {'N':>6s}  {'R²':>8s}  {'MAE':>8s}  {'RMSE':>8s}",
        "-" * 52,
    ]
    for name, idx in groups.items():
        if len(idx) == 0:
            continue
        g_r2   = r2_score(at[idx], ap[idx])
        g_mae  = mae(at[idx],  ap[idx])
        g_rmse = rmse(at[idx], ap[idx])
        lines.append(
            f"  {name:8s}  {len(idx):>6d}  {g_r2:>8.4f}  {g_mae:>8.4f}  {g_rmse:>8.4f}"
        )
    return "\n".join(lines)


# =============================================================================
# 主评估函数
# =============================================================================

def evaluate(cfg: dict) -> None:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"  设备: {device}")

    ckpt_path = cfg["paths"]["ckpt"]
    if not os.path.exists(ckpt_path):
        raise FileNotFoundError(f"检查点不存在: {ckpt_path}")

    # ── 加载检查点 ────────────────────────────────────────────
    print(f"\n[1/4] 加载模型检查点: {ckpt_path}")
    ckpt = torch.load(ckpt_path, map_location=device)
    model_cfg = ckpt.get("cfg", cfg)
    model = build_model(model_cfg).to(device)
    model.load_state_dict(ckpt["model_state"])
    print(f"  训练轮数: {ckpt.get('epoch','?')} | "
          f"验证损失: {ckpt.get('val_loss', float('nan')):.6f}")

    # ── 加载数据集 ────────────────────────────────────────────
    print(f"\n[2/4] 加载数据集: {cfg['paths']['dataset_pkl']}")
    with open(cfg["paths"]["dataset_pkl"], "rb") as f:
        data = pickle.load(f)
    split_name = cfg["eval"]["split"]
    ds     = MTJPDataset(data[split_name])
    loader = DataLoader(
        ds,
        batch_size  = cfg["eval"]["batch_size"],
        shuffle     = False,
        num_workers = 0,
    )
    print(f"  评估集: {split_name}，样本数={len(ds)}")

    # ── 推理 ──────────────────────────────────────────────────
    print("\n[3/4] 推理中...")
    preds = collect_predictions(model, loader, device)

    # ── 生成报告 ──────────────────────────────────────────────
    print("\n[4/4] 计算评估指标...")
    report_str, metrics, pred_df = build_report(preds, cfg)
    group_str = group_analysis(preds)

    full_report = report_str + "\n" + group_str + "\n"
    print("\n" + full_report)

    # ── 保存结果 ──────────────────────────────────────────────
    out_dir = cfg["paths"]["output_dir"]
    os.makedirs(out_dir, exist_ok=True)

    # 文字报告
    rpt_path = os.path.join(out_dir, "evaluation_report.txt")
    with open(rpt_path, "w", encoding="utf-8") as f:
        f.write(full_report)
    print(f"  文字报告 → {rpt_path}")

    # 指标 CSV
    met_path = os.path.join(out_dir, "evaluation_metrics.csv")
    pd.DataFrame([metrics]).to_csv(met_path, index=False, encoding="utf-8-sig")
    print(f"  指标 CSV  → {met_path}")

    # 逐样本预测 CSV
    pred_path = os.path.join(out_dir, "predictions.csv")
    pred_df.to_csv(pred_path, index=False, encoding="utf-8-sig")
    print(f"  预测结果  → {pred_path}")


# =============================================================================
# 命令行入口
# =============================================================================

def main():
    parser = argparse.ArgumentParser(
        description="MT-JP 联合预测模型评估器",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("-c", "--config", type=str, default=None,
                        help="YAML 配置文件路径（默认: config/evaluator.yaml）")
    parser.add_argument("--ckpt",    type=str, default=None,
                        help="模型检查点路径（覆盖 yaml paths.ckpt）")
    parser.add_argument("--split",   type=str, default=None,
                        help="评估集名称 train/val/test（覆盖 yaml eval.split）")
    parser.add_argument("--output_dir", type=str, default=None)
    args = parser.parse_args()

    cfg = load_config(args.config)
    if args.ckpt:       cfg["paths"]["ckpt"]       = args.ckpt
    if args.split:      cfg["eval"]["split"]        = args.split
    if args.output_dir: cfg["paths"]["output_dir"]  = args.output_dir

    print("=" * 60)
    print("MT-JP 联合预测模型 — 评估器")
    print(f"  检查点: {cfg['paths']['ckpt']}")
    print(f"  评估集: {cfg['eval']['split']}")
    print("=" * 60)

    evaluate(cfg)


if __name__ == "__main__":
    main()

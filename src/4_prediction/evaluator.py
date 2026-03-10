# D:\Local\DynamicCapRisk\src\4_prediction\evaluator.py

"""
MT-JP 单模型评估器

评估指标（论文 §5.3）：
  能力预测（回归）：R²、MAE、RMSE
  风险度预测（回归）：R²、MAE、RMSE、高风险召回率/精确率/F1
  风险等级预测（分类）：准确率、宏平均F1、Kappa系数、混淆矩阵、
                        各类别 Precision / Recall / F1

输出：
  evaluation_report.txt   文字报告（含混淆矩阵及分组分析）
  evaluation_metrics.csv  数值指标（可供绘图）
  predictions.csv         每条测试样本的真实值与预测值

用法：
  python evaluator.py
  python evaluator.py -c config/evaluator.yaml
  python evaluator.py --ckpt output/3_prediction/runs/20260310_155225_mtjp/best_model.pt
  python evaluator.py --ckpt ... --split val
"""

import os
import sys
import pickle
import argparse
import warnings
import yaml
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from typing import Dict, Optional, Tuple

warnings.filterwarnings("ignore")

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from src.models.mtjp_model import build_model
from trainer_dl import TorchDataset


# =============================================================================
# 配置加载
# =============================================================================

_DEFAULT_CONFIG_PATH = os.path.join(
    os.path.dirname(os.path.abspath(__file__)), "..", "..", "config", "evaluator.yaml"
)


def load_config(path: Optional[str] = None) -> dict:
    candidate = path or _DEFAULT_CONFIG_PATH
    if not os.path.exists(candidate):
        raise FileNotFoundError(
            f"配置文件不存在！\n"
            f"指定路径: {path}\n"
            f"默认路径: {_DEFAULT_CONFIG_PATH}\n"
            "请确保 YAML 配置文件存在后再运行"
        )
    with open(candidate, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f) or {}
    print(f"已加载配置文件: {os.path.abspath(candidate)}")
    return cfg


# =============================================================================
# 推理
# =============================================================================

@torch.no_grad()
def collect_predictions(
    model:  nn.Module,
    loader: DataLoader,
    device: torch.device,
) -> Dict[str, np.ndarray]:
    model.eval()
    ability_true, ability_pred   = [], []
    risk_reg_true, risk_reg_pred = [], []
    risk_cls_true, risk_cls_pred = [], []
    risk_cls_prob_all, f_s_all   = [], []

    for X, ya, yr, yc in loader:
        X   = X.to(device)
        out = model(X)

        f_s       = X[:, -1, 16].cpu().numpy()
        a_hat     = out["ability"].squeeze(-1).cpu().numpy()
        rr_hat    = out["risk_reg"].squeeze(-1).cpu().numpy()
        rc_logits = out["risk_cls"].cpu().numpy()
        rc_prob   = _softmax(rc_logits)
        rc_hat    = rc_prob.argmax(axis=-1)

        ability_true.append(ya.numpy());   ability_pred.append(a_hat)
        risk_reg_true.append(yr.numpy());  risk_reg_pred.append(rr_hat)
        risk_cls_true.append(yc.numpy());  risk_cls_pred.append(rc_hat)
        risk_cls_prob_all.append(rc_prob); f_s_all.append(f_s)

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


def confusion_matrix(y_true: np.ndarray, y_pred: np.ndarray, n_classes: int = 3) -> np.ndarray:
    cm = np.zeros((n_classes, n_classes), dtype=int)
    for t, p in zip(y_true, y_pred):
        if 0 <= t < n_classes and 0 <= p < n_classes:
            cm[t, p] += 1
    return cm


def per_class_metrics(cm: np.ndarray) -> Dict[str, np.ndarray]:
    n = cm.shape[0]
    precision = np.zeros(n); recall = np.zeros(n); f1 = np.zeros(n)
    for c in range(n):
        tp = cm[c, c]
        fp = cm[:, c].sum() - tp
        fn = cm[c, :].sum() - tp
        precision[c] = tp / max(tp + fp, 1)
        recall[c]    = tp / max(tp + fn, 1)
        f1[c]        = 2 * precision[c] * recall[c] / max(precision[c] + recall[c], 1e-9)
    return {"precision": precision, "recall": recall, "f1": f1}


def kappa_score(cm: np.ndarray) -> float:
    n  = cm.sum()
    po = cm.diagonal().sum() / max(n, 1)
    pe = (cm.sum(axis=1) * cm.sum(axis=0)).sum() / max(n * n, 1)
    return float((po - pe) / max(1 - pe, 1e-9))


def high_risk_metrics(
    risk_reg_true: np.ndarray,
    risk_reg_pred: np.ndarray,
    threshold: float = 0.1,
) -> Tuple[float, float]:
    """返回 (召回率, 精确率)。"""
    true_pos = risk_reg_true > threshold
    pred_pos = risk_reg_pred > threshold
    tp = int(( true_pos &  pred_pos).sum())
    fn = int(( true_pos & ~pred_pos).sum())
    fp = int((~true_pos &  pred_pos).sum())
    return float(tp / max(tp + fn, 1)), float(tp / max(tp + fp, 1))


def high_risk_f1(risk_reg_true: np.ndarray, risk_reg_pred: np.ndarray, threshold: float = 0.1) -> float:
    rec, prec = high_risk_metrics(risk_reg_true, risk_reg_pred, threshold)
    return float(2 * prec * rec / max(prec + rec, 1e-9))


# =============================================================================
# 报告生成
# =============================================================================

def _fmt_cm(cm: np.ndarray, class_names) -> str:
    header  = "真实\\预测   " + "   ".join(f"{n:^8s}" for n in class_names) + "   召回率"
    lines   = [header, "-" * len(header)]
    row_sum = cm.sum(axis=1)
    for i, name in enumerate(class_names):
        cells  = "   ".join(f"{cm[i,j]:^8d}" for j in range(len(class_names)))
        recall = cm[i, i] / max(row_sum[i], 1) * 100
        lines.append(f"{name:^8s}     {cells}   {recall:.1f}%")
    lines.append("-" * len(header))
    col_sum  = cm.sum(axis=0)
    prec_row = "   ".join(f"{cm[j,j]/max(col_sum[j],1)*100:^7.1f}%" for j in range(len(class_names)))
    lines.append(f"{'精确率':^8s}     {prec_row}")
    return "\n".join(lines)


def build_report(
    preds: Dict[str, np.ndarray],
    cfg:   dict,
) -> Tuple[str, Dict, pd.DataFrame]:
    ec     = cfg["eval"]
    cnames = cfg["class_names"]

    a_r2   = r2_score(preds["ability_true"], preds["ability_pred"])
    a_mae_ = mae(preds["ability_true"],  preds["ability_pred"])
    a_rmse_= rmse(preds["ability_true"], preds["ability_pred"])

    r_r2   = r2_score(preds["risk_reg_true"], preds["risk_reg_pred"])
    r_mae_ = mae(preds["risk_reg_true"],  preds["risk_reg_pred"])
    r_rmse_= rmse(preds["risk_reg_true"], preds["risk_reg_pred"])

    hr_recall, hr_prec = high_risk_metrics(
        preds["risk_reg_true"], preds["risk_reg_pred"], ec["risk_thresh_high"]
    )
    hr_f1_val = high_risk_f1(
        preds["risk_reg_true"], preds["risk_reg_pred"], ec["risk_thresh_high"]
    )

    cm       = confusion_matrix(preds["risk_cls_true"], preds["risk_cls_pred"], 3)
    pcm      = per_class_metrics(cm)
    kappa    = kappa_score(cm)
    acc      = cm.diagonal().sum() / max(cm.sum(), 1)
    macro_f1 = float(pcm["f1"].mean())

    metrics = {
        "ability_R2":       round(a_r2,     4),
        "ability_MAE":      round(a_mae_,   4),
        "ability_RMSE":     round(a_rmse_,  4),
        "risk_reg_R2":      round(r_r2,     4),
        "risk_reg_MAE":     round(r_mae_,   4),
        "risk_reg_RMSE":    round(r_rmse_,  4),
        "high_risk_recall": round(hr_recall,  4),
        "high_risk_prec":   round(hr_prec,    4),
        "high_risk_f1":     round(hr_f1_val,  4),
        "cls_accuracy":     round(float(acc),      4),
        "cls_macro_f1":     round(float(macro_f1), 4),
        "cls_kappa":        round(float(kappa),    4),
    }
    for i, name in enumerate(cnames):
        metrics[f"{name}_precision"] = round(float(pcm["precision"][i]), 4)
        metrics[f"{name}_recall"]    = round(float(pcm["recall"][i]),    4)
        metrics[f"{name}_f1"]        = round(float(pcm["f1"][i]),        4)

    pred_df = pd.DataFrame({
        "ability_true":  preds["ability_true"],
        "ability_pred":  preds["ability_pred"].round(4),
        "risk_reg_true": preds["risk_reg_true"],
        "risk_reg_pred": preds["risk_reg_pred"].round(4),
        "risk_cls_true": preds["risk_cls_true"],
        "risk_cls_pred": preds["risk_cls_pred"],
        "prob_low":      preds["risk_cls_prob"][:, 0].round(4),
        "prob_mid":      preds["risk_cls_prob"][:, 1].round(4),
        "prob_high":     preds["risk_cls_prob"][:, 2].round(4),
        "f_s":           preds["f_s"].round(4),
    })

    sep   = "=" * 60
    lines = [
        sep, "MT-JP 联合预测模型 — 评估报告",
        f"评估样本数: {len(preds['ability_true'])}", sep, "",
        "【一】动态驾驶能力预测（回归）",
        f"  R²   = {a_r2:.4f}  （>0.8 为良好）",
        f"  MAE  = {a_mae_:.4f}",
        f"  RMSE = {a_rmse_:.4f}", "",
        "【二】风险度预测（回归）",
        f"  R²             = {r_r2:.4f}",
        f"  MAE            = {r_mae_:.4f}",
        f"  RMSE           = {r_rmse_:.4f}",
        f"  高风险召回率   = {hr_recall*100:.1f}%  （阈值 R*>{ec['risk_thresh_high']}）",
        f"  高风险精确率   = {hr_prec*100:.1f}%",
        f"  高风险 F1      = {hr_f1_val:.4f}", "",
        "【三】风险等级分类",
        f"  总体准确率     = {acc*100:.1f}%",
        f"  宏平均 F1      = {macro_f1:.4f}",
        f"  Kappa 系数     = {kappa:.4f}", "",
        "  各类别指标:",
    ]
    for i, name in enumerate(cnames):
        lines.append(
            f"    {name}: Precision={pcm['precision'][i]*100:.1f}%  "
            f"Recall={pcm['recall'][i]*100:.1f}%  F1={pcm['f1'][i]:.4f}"
        )
    lines += ["", "  混淆矩阵:", _fmt_cm(cm, cnames), "", sep]
    return "\n".join(lines), metrics, pred_df


def group_analysis(preds: Dict[str, np.ndarray]) -> str:
    """按真实能力三等分，分组统计预测性能（对应论文表5.4）。"""
    at, ap     = preds["ability_true"], preds["ability_pred"]
    idx_sorted = np.argsort(at)
    n          = len(at)
    groups     = {
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
        lines.append(
            f"  {name:8s}  {len(idx):>6d}  "
            f"{r2_score(at[idx], ap[idx]):>8.4f}  "
            f"{mae(at[idx], ap[idx]):>8.4f}  "
            f"{rmse(at[idx], ap[idx]):>8.4f}"
        )
    return "\n".join(lines)


# =============================================================================
# 主评估函数
# =============================================================================

def evaluate(cfg: dict) -> None:
    device    = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ckpt_path = cfg["paths"]["ckpt"]
    print(f"  设备: {device}")

    if not os.path.exists(ckpt_path):
        raise FileNotFoundError(f"检查点不存在: {ckpt_path}")

    # 1. 加载模型
    print(f"\n[1/4] 加载模型检查点: {ckpt_path}")
    ckpt      = torch.load(ckpt_path, map_location=device)
    model_cfg = ckpt.get("cfg", cfg)
    model     = build_model(model_cfg).to(device)
    model.load_state_dict(ckpt["model_state"])
    print(f"  训练轮数: {ckpt.get('epoch','?')} | 验证损失: {ckpt.get('val_loss', float('nan')):.6f}")

    # 2. 加载数据集
    print(f"\n[2/4] 加载数据集: {cfg['paths']['dataset_pkl']}")
    with open(cfg["paths"]["dataset_pkl"], "rb") as f:
        data = pickle.load(f)
    split_name = cfg["eval"]["split"]
    loader = DataLoader(
        TorchDataset(data[split_name]),
        batch_size=cfg["eval"]["batch_size"], shuffle=False, num_workers=0,
    )
    print(f"  评估集: {split_name}，样本数={len(loader.dataset)}")

    # 3. 推理
    print("\n[3/4] 推理中...")
    preds = collect_predictions(model, loader, device)

    # 4. 指标计算 & 保存
    print("\n[4/4] 计算评估指标...")
    report_str, metrics, pred_df = build_report(preds, cfg)
    full_report = report_str + "\n" + group_analysis(preds) + "\n"
    print("\n" + full_report)

    out_dir = cfg["paths"]["output_dir"]
    os.makedirs(out_dir, exist_ok=True)

    rpt_path = os.path.join(out_dir, "evaluation_report.txt")
    with open(rpt_path, "w", encoding="utf-8") as f:
        f.write(full_report)
    print(f"  文字报告 → {rpt_path}")

    met_path = os.path.join(out_dir, "evaluation_metrics.csv")
    pd.DataFrame([metrics]).to_csv(met_path, index=False, encoding="utf-8-sig")
    print(f"  指标 CSV  → {met_path}")

    pred_path = os.path.join(out_dir, "predictions.csv")
    pred_df.to_csv(pred_path, index=False, encoding="utf-8-sig")
    print(f"  预测结果  → {pred_path}")


# =============================================================================
# 命令行入口
# =============================================================================

def main():
    parser = argparse.ArgumentParser(
        description="MT-JP 单模型评估器",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("-c", "--config",  type=str, default=None,
                        help="YAML 配置文件路径（默认: config/evaluator.yaml）")
    parser.add_argument("--ckpt",          type=str, default=None,
                        help="检查点 .pt 路径（覆盖 yaml paths.ckpt）")
    parser.add_argument("--split",         type=str, default=None,
                        help="评估集：train / val / test（覆盖 yaml eval.split）")
    parser.add_argument("--output_dir",    type=str, default=None)
    args = parser.parse_args()

    cfg = load_config(args.config)
    if args.ckpt:       cfg["paths"]["ckpt"]      = args.ckpt
    if args.split:      cfg["eval"]["split"]       = args.split
    if args.output_dir: cfg["paths"]["output_dir"] = args.output_dir

    print("=" * 60)
    print("MT-JP 联合预测模型 — 单模型评估器")
    print(f"  检查点: {cfg['paths']['ckpt']}")
    print(f"  评估集: {cfg['eval']['split']}")
    print("=" * 60)

    evaluate(cfg)


if __name__ == "__main__":
    main()

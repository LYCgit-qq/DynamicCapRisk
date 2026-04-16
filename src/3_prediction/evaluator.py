# /root/autodl-tmp/DynamicCapRisk/src/3_prediction/evaluator.py
"""
评估指标：
  风险度预测（回归）：R²、MAE、RMSE、高风险召回率/精确率/F1
  风险等级预测（分类）：准确率、宏平均F1、Kappa系数、混淆矩阵、
                        各类别 Precision / Recall / F1

输出：
  evaluation_report.txt   文字报告（含混淆矩阵及分组分析）
  evaluation_metrics.csv  数值指标（可供绘图）
  predictions.csv         每条测试样本的真实值与预测值
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
import glob

warnings.filterwarnings("ignore")

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
# 🔥 修复1：导入MTRP和基线模型的构建函数
from src.models.mtrp_model import build_model as build_mtrp
from src.models.baseline_models import build_baseline_model
from torch.utils.data import Dataset

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
# 自动获取 runs 文件夹下最新的 best_model.pt
# =============================================================================
def get_latest_checkpoint(runs_root: str = "output/3_prediction/runs") -> str:
    """获取最新训练生成的模型文件夹中的 best_model.pt"""
    if not os.path.exists(runs_root):
        raise FileNotFoundError(f"模型根目录不存在: {runs_root}")
    
    # 获取所有子文件夹
    dirs = [d for d in glob.glob(os.path.join(runs_root, "*")) if os.path.isdir(d)]
    if not dirs:
        raise FileNotFoundError(f"runs 目录下没有找到任何模型文件夹")
    
    # 按修改时间排序，取最新的文件夹
    dirs.sort(key=lambda x: os.path.getmtime(x), reverse=True)
    latest_dir = dirs[0]
    ckpt_path = os.path.join(latest_dir, "best_model.pt")
    
    if not os.path.exists(ckpt_path):
        raise FileNotFoundError(f"最新模型文件夹中未找到 best_model.pt: {ckpt_path}")
    
    print(f"✅ 自动找到最新模型: {ckpt_path}")
    return ckpt_path

# =============================================================================
# 与训练脚本完全一致的Dataset，仅返回3个值
# =============================================================================
class TorchDataset(Dataset):
    def __init__(self, split_data: dict):
        self.X = torch.from_numpy(split_data["X"]).float()
        self.y_risk_reg = torch.from_numpy(split_data["y_risk_reg"]).float()
        self.y_risk_cls = torch.from_numpy(split_data["y_risk_cls"]).long()

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        # 与训练脚本完全一致：返回特征、风险回归、风险分类
        return self.X[idx], self.y_risk_reg[idx], self.y_risk_cls[idx]


@torch.no_grad()
def collect_predictions(
    model:  nn.Module,
    loader: DataLoader,
    device: torch.device,
) -> Dict[str, np.ndarray]:
    model.eval()
    risk_reg_true, risk_reg_pred = [], []
    risk_cls_true, risk_cls_pred = [], []
    risk_cls_prob_all = []

    # 匹配Dataset返回的3个值 (X, yr, yc)
    for X, yr, yc in loader:
        X   = X.to(device)
        out = model(X)

        # 仅保留风险预测
        rr_hat    = out["risk_reg"].squeeze(-1).cpu().numpy()
        rc_logits = out["risk_cls"].cpu().numpy()
        rc_prob   = _softmax(rc_logits)
        rc_hat    = rc_prob.argmax(axis=-1)

        risk_reg_true.append(yr.numpy());  risk_reg_pred.append(rr_hat)
        risk_cls_true.append(yc.numpy());  risk_cls_pred.append(rc_hat)
        risk_cls_prob_all.append(rc_prob)

    return {
        "risk_reg_true": np.concatenate(risk_reg_true),
        "risk_reg_pred": np.concatenate(risk_reg_pred),
        "risk_cls_true": np.concatenate(risk_cls_true).astype(int),
        "risk_cls_pred": np.concatenate(risk_cls_pred).astype(int),
        "risk_cls_prob": np.concatenate(risk_cls_prob_all),
    }


def _softmax(x: np.ndarray) -> np.ndarray:
    # ✅ 修复拼写错误：keepdims 是 numpy 标准参数
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

    # 风险回归指标
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

    # 风险相关指标
    metrics = {
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

    # 预测结果
    pred_df = pd.DataFrame({
        "risk_reg_true": preds["risk_reg_true"],
        "risk_reg_pred": preds["risk_reg_pred"].round(4),
        "risk_cls_true": preds["risk_cls_true"],
        "risk_cls_pred": preds["risk_cls_pred"],
        "prob_low":      preds["risk_cls_prob"][:, 0].round(4),
        "prob_mid":      preds["risk_cls_prob"][:, 1].round(4),
        "prob_high":     preds["risk_cls_prob"][:, 2].round(4),
    })

    sep   = "=" * 60
    lines = [
        sep, "MTRP 风险预测模型 — 评估报告",
        f"评估样本数: {len(preds['risk_reg_true'])}", sep, "",
        "【一】风险度预测（回归）",
        f"  R²             = {r_r2:.4f}",
        f"  MAE            = {r_mae_:.4f}",
        # ✅ 修复语法错误：添加冒号
        f"  RMSE           = {r_rmse_:.4f}",
        f"  高风险召回率   = {hr_recall*100:.1f}%  （阈值 R>{ec['risk_thresh_high']}）",
        f"  高风险精确率   = {hr_prec*100:.1f}%",
        f"  高风险 F1      = {hr_f1_val:.4f}", "",
        "【二】风险等级分类",
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


# =============================================================================
# 主评估函数 ✅【核心修复：自动加载训练时的模型配置 + 动态构建模型】
# =============================================================================

def evaluate(cfg: dict) -> None:
    device    = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ckpt_path = cfg["paths"]["ckpt"]
    
    if ckpt_path is None or ckpt_path == "null" or ckpt_path.strip() == "":
        ckpt_path = get_latest_checkpoint()
    print(f"  设备: {device}")

    if not os.path.exists(ckpt_path):
        raise FileNotFoundError(f"检查点不存在: {ckpt_path}")

    # ===================== 加载训练时的配置文件 =====================
    model_dir = os.path.dirname(ckpt_path)
    train_config_path = os.path.join(model_dir, "run_config.yaml")
    
    if os.path.exists(train_config_path):
        print(f"✅ 加载训练配置: {train_config_path}")
        with open(train_config_path, "r", encoding="utf-8") as f:
            train_cfg = yaml.safe_load(f)
        # 用训练时的模型参数覆盖评估配置，保证结构一致
        cfg["model"] = train_cfg["model"]
    # ==========================================================================

    # 🔥 修复2：动态构建模型（自动适配 MTRP / LSTM / GRU / CNN-LSTM）
    print(f"\n[1/4] 加载模型检查点: {ckpt_path}")
    model_type = cfg["model"]["model_type"].lower()
    if model_type == "mtrp":
        model = build_mtrp(cfg)
        print(f"✅ 构建模型: MTRP")
    else:
        model = build_baseline_model(cfg)
        print(f"✅ 构建基线模型: {model_type.upper()}")
    
    # 加载权重
    model.load_state_dict(torch.load(ckpt_path, map_location=device))
    model = model.to(device)
    print(f"  模型加载成功！")

    # 加载数据集
    print(f"\n[2/4] 加载数据集: {cfg['paths']['dataset_pkl']}")
    with open(cfg["paths"]["dataset_pkl"], "rb") as f:
        data = pickle.load(f)
    split_name = cfg["eval"]["split"]
    loader = DataLoader(
        TorchDataset(data[split_name]),
        batch_size=cfg["eval"]["batch_size"], shuffle=False, num_workers=0,
    )
    print(f"  评估集: {split_name}，样本数={len(loader.dataset)}")

    # 推理
    print("\n[3/4] 推理中...")
    preds = collect_predictions(model, loader, device)

    # 指标计算 & 保存
    print("\n[4/4] 计算评估指标...")
    report_str, metrics, pred_df = build_report(preds, cfg)
    full_report = report_str + "\n"
    print("\n" + full_report)

    run_dir = os.path.dirname(ckpt_path)
    eval_dir = os.path.join(run_dir, "eval")
    os.makedirs(eval_dir, exist_ok=True)
    out_dir = eval_dir
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
        description="MTRP 风险预测模型评估器",
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
    print("MTRP 风险预测模型 — 单模型评估器")
    print(f"  检查点: {cfg['paths']['ckpt']}")
    print(f"  评估集: {cfg['eval']['split']}")
    print("=" * 60)

    evaluate(cfg)


if __name__ == "__main__":
    main()
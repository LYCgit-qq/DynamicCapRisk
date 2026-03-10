# D:\Local\DynamicCapRisk\src\4_prediction\evaluator_compare.py

"""
MT-JP 多模型横向对比评估器

对比内容（论文 §5.3）：
  § 5.3.1  动态驾驶能力预测对比 → 表5.7（MAE / RMSE / R² / 训练时间 / 参数量）
  § 5.3.2  风险状态预测对比     → 表5.8（风险度MAE / R² / 分类准确率 / 高风险召回率 / 高风险F1）

支持模型：SVR、CART、LSTM、GRU、CNN-LSTM、MT-JP

输出（保存至 output_dir）：
  comparison_ability.csv    能力预测对比表（对应表5.7）
  comparison_risk.csv       风险预测对比表（对应表5.8）
  comparison_report.txt     完整文字对比报告
  predictions_<模型>.csv    每个模型的逐样本预测结果

用法：
  # 读取 YAML（推荐：在 compare_ckpt_list 或 compare_ckpt_dir 中填好路径）
  python evaluator_compare.py
  python evaluator_compare.py -c config/evaluator_compare.yaml

  # 命令行覆盖：自动扫描目录
  python evaluator_compare.py --ckpt_dir output/3_prediction/runs/

  # 命令行覆盖：手动指定各模型检查点
  python evaluator_compare.py \\
      --ckpt_list svr:runs/svr/model.pkl cart:runs/cart/model.pkl \\
                  lstm:runs/lstm/best_model.pt gru:runs/gru/best_model.pt \\
                  cnn_lstm:runs/cnn_lstm/best_model.pt mt_jp:runs/mtjp/best_model.pt

  # 命令行补充/覆盖 YAML 中的元信息
  python evaluator_compare.py \\
      --train_times svr:23.5,cart:8.7,lstm:45.2,gru:38.6,cnn_lstm:51.3,mt_jp:62.8 \\
      --param_counts lstm:3.2M,gru:2.5M,cnn_lstm:3.8M,mt_jp:2.8M
"""

import os
import sys
import pickle
import argparse
import warnings
import yaml
import time
import glob
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from typing import Dict, List, Optional, Tuple

warnings.filterwarnings("ignore")

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from src.models.mtjp_model import build_model
from trainer_dl import TorchDataset


# =============================================================================
# 配置加载
# =============================================================================

_DEFAULT_CONFIG_PATH = os.path.join(
    os.path.dirname(os.path.abspath(__file__)), "..", "..", "config", "evaluator_compare.yaml"
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
# 模型类型常量
# =============================================================================

_MODEL_TYPE_KEYWORDS = {
    "svr":      ["svr"],
    "cart":     ["cart"],
    "lstm":     ["lstm"],
    "gru":      ["gru"],
    "cnn_lstm": ["cnn_lstm", "cnnlstm"],
    "mt_jp":    ["mtjp", "mt_jp", "mt-jp"],
}

_DISPLAY_NAMES = {
    "svr":      "SVR",
    "cart":     "CART",
    "lstm":     "LSTM",
    "gru":      "GRU",
    "cnn_lstm": "CNN-LSTM",
    "mt_jp":    "MT-JP",
}

_MODEL_ORDER = ["svr", "cart", "lstm", "gru", "cnn_lstm", "mt_jp"]


def _fmt_params(n: int) -> str:
    if n >= 1_000_000:
        return f"{n/1_000_000:.1f}M"
    if n >= 1_000:
        return f"{n/1_000:.1f}K"
    return str(n)


def _infer_model_type(path: str) -> Optional[str]:
    lower = path.lower()
    for mtype, keywords in _MODEL_TYPE_KEYWORDS.items():
        if any(kw in lower for kw in keywords):
            return mtype
    return None


# =============================================================================
# ModelEntry 数据容器
# =============================================================================

class ModelEntry:
    """封装单个待对比模型的元信息与推理结果。"""

    def __init__(
        self,
        name:        str,
        ckpt_path:   str,
        train_time:  Optional[float] = None,
        param_count: Optional[str]   = None,
    ):
        self.name        = name
        self.ckpt_path   = ckpt_path
        self.train_time  = train_time
        self.param_count = param_count
        self.metrics: Dict                            = {}
        self.preds:   Optional[Dict[str, np.ndarray]] = None


# =============================================================================
# 检查点发现
# =============================================================================

def discover_checkpoints(ckpt_dir: str) -> List[Tuple[str, str]]:
    """自动扫描目录，按模型类型关键词匹配 best_model.pt / model.pkl。"""
    found: Dict[str, str] = {}
    for pt in glob.glob(os.path.join(ckpt_dir, "**", "best_model.pt"), recursive=True):
        mtype = _infer_model_type(pt)
        if mtype and mtype not in found:
            found[mtype] = pt
    for pkl in glob.glob(os.path.join(ckpt_dir, "**", "model.pkl"), recursive=True):
        mtype = _infer_model_type(pkl)
        if mtype and mtype not in found:
            found[mtype] = pkl
    result  = [(m, found[m]) for m in _MODEL_ORDER if m in found]
    result += [(m, p) for m, p in found.items() if m not in _MODEL_ORDER]
    return result


def resolve_ckpt_list_from_yaml(cfg: dict) -> Optional[List[Tuple[str, str]]]:
    """读取 YAML paths.compare_ckpt_list，过滤空路径条目。"""
    raw: Optional[Dict] = cfg.get("paths", {}).get("compare_ckpt_list")
    if not raw:
        return None
    result = []
    for mtype in _MODEL_ORDER:
        path = str(raw.get(mtype, "")).strip()
        if path:
            result.append((mtype, path))
    for mtype, path in raw.items():
        if mtype not in _MODEL_ORDER and path and str(path).strip():
            result.append((mtype, str(path).strip()))
    return result if result else None


# =============================================================================
# 指标计算（复用自 evaluator.py 的纯函数）
# =============================================================================

def _softmax(x: np.ndarray) -> np.ndarray:
    e = np.exp(x - x.max(axis=-1, keepdims=True))
    return e / e.sum(axis=-1, keepdims=True)


def r2_score(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    ss_res = np.sum((y_true - y_pred) ** 2)
    ss_tot = np.sum((y_true - y_true.mean()) ** 2)
    return float(1.0 - ss_res / max(ss_tot, 1e-12))


def mae(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return float(np.mean(np.abs(y_true - y_pred)))


def rmse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return float(np.sqrt(np.mean((y_true - y_pred) ** 2)))


def _confusion_matrix(y_true: np.ndarray, y_pred: np.ndarray, n: int = 3) -> np.ndarray:
    cm = np.zeros((n, n), dtype=int)
    for t, p in zip(y_true, y_pred):
        if 0 <= t < n and 0 <= p < n:
            cm[t, p] += 1
    return cm


def _high_risk_metrics(y_true: np.ndarray, y_pred: np.ndarray, thr: float) -> Tuple[float, float]:
    tp = int(((y_true > thr) & (y_pred > thr)).sum())
    fn = int(((y_true > thr) & (y_pred <= thr)).sum())
    fp = int(((y_true <= thr) & (y_pred > thr)).sum())
    return float(tp / max(tp + fn, 1)), float(tp / max(tp + fp, 1))


def _compute_all_metrics(preds: Dict[str, np.ndarray], threshold: float = 0.1) -> Dict:
    a_r2   = r2_score(preds["ability_true"], preds["ability_pred"])
    a_mae_ = mae(preds["ability_true"],  preds["ability_pred"])
    a_rmse_= rmse(preds["ability_true"], preds["ability_pred"])

    r_r2   = r2_score(preds["risk_reg_true"], preds["risk_reg_pred"])
    r_mae_ = mae(preds["risk_reg_true"],  preds["risk_reg_pred"])
    r_rmse_= rmse(preds["risk_reg_true"], preds["risk_reg_pred"])

    hr_rec, hr_prec = _high_risk_metrics(preds["risk_reg_true"], preds["risk_reg_pred"], threshold)
    hr_f1 = float(2 * hr_prec * hr_rec / max(hr_prec + hr_rec, 1e-9))

    cm  = _confusion_matrix(preds["risk_cls_true"], preds["risk_cls_pred"], 3)
    acc = cm.diagonal().sum() / max(cm.sum(), 1)

    # macro F1
    f1_list = []
    for c in range(3):
        tp = cm[c, c]; fp = cm[:, c].sum() - tp; fn = cm[c, :].sum() - tp
        p  = tp / max(tp + fp, 1)
        r  = tp / max(tp + fn, 1)
        f1_list.append(2 * p * r / max(p + r, 1e-9))
    macro_f1 = float(np.mean(f1_list))

    # Cohen's Kappa
    n  = cm.sum()
    po = cm.diagonal().sum() / max(n, 1)
    pe = (cm.sum(axis=1) * cm.sum(axis=0)).sum() / max(n * n, 1)
    kappa = float((po - pe) / max(1 - pe, 1e-9))

    return {
        "ability_MAE":      round(a_mae_,  4),
        "ability_RMSE":     round(a_rmse_, 4),
        "ability_R2":       round(a_r2,    4),
        "risk_reg_MAE":     round(r_mae_,  4),
        "risk_reg_RMSE":    round(r_rmse_, 4),
        "risk_reg_R2":      round(r_r2,    4),
        "cls_accuracy":     round(float(acc),      4),
        "high_risk_recall": round(hr_rec,           4),
        "high_risk_f1":     round(hr_f1,            4),
        "cls_macro_f1":     round(float(macro_f1),  4),
        "cls_kappa":        round(float(kappa),     4),
    }


# =============================================================================
# 推理：深度学习模型 & sklearn 模型
# =============================================================================

@torch.no_grad()
def _predict_dl(
    ckpt_path: str,
    loader:    DataLoader,
    device:    torch.device,
) -> Tuple[Dict[str, np.ndarray], int, float]:
    ckpt        = torch.load(ckpt_path, map_location=device)
    model       = build_model(ckpt.get("cfg", {})).to(device)
    model.load_state_dict(ckpt["model_state"])
    model.eval()

    param_count = sum(p.numel() for p in model.parameters())
    train_time  = float(ckpt.get("train_time_min", float("nan")))

    ability_true, ability_pred   = [], []
    risk_reg_true, risk_reg_pred = [], []
    risk_cls_true, risk_cls_pred = [], []
    risk_cls_prob_all            = []

    for X, ya, yr, yc in loader:
        X   = X.to(device)
        out = model(X)
        rc_prob = _softmax(out["risk_cls"].cpu().numpy())

        ability_true.append(ya.numpy())
        ability_pred.append(out["ability"].squeeze(-1).cpu().numpy())
        risk_reg_true.append(yr.numpy())
        risk_reg_pred.append(out["risk_reg"].squeeze(-1).cpu().numpy())
        risk_cls_true.append(yc.numpy())
        risk_cls_pred.append(rc_prob.argmax(axis=-1))
        risk_cls_prob_all.append(rc_prob)

    preds = {
        "ability_true":  np.concatenate(ability_true),
        "ability_pred":  np.concatenate(ability_pred),
        "risk_reg_true": np.concatenate(risk_reg_true),
        "risk_reg_pred": np.concatenate(risk_reg_pred),
        "risk_cls_true": np.concatenate(risk_cls_true).astype(int),
        "risk_cls_pred": np.concatenate(risk_cls_pred).astype(int),
        "risk_cls_prob": np.concatenate(risk_cls_prob_all),
    }
    return preds, param_count, train_time


def _predict_sklearn(
    pkl_path: str,
    loader:   DataLoader,
) -> Dict[str, np.ndarray]:
    with open(pkl_path, "rb") as f:
        bundle = pickle.load(f)
    models = bundle["models"]
    scaler = bundle["scaler"]

    X_list, ya_list, yr_list, yc_list = [], [], [], []
    for X_batch, ya, yr, yc in loader:
        X_np = X_batch.numpy()
        X_list.append(X_np.reshape(len(X_np), -1))
        ya_list.append(ya.numpy()); yr_list.append(yr.numpy()); yc_list.append(yc.numpy())

    X_all  = scaler.transform(np.concatenate(X_list, axis=0))
    ya_all = np.concatenate(ya_list)
    yr_all = np.concatenate(yr_list)
    yc_all = np.concatenate(yc_list)

    ability_pred  = models["ability"].predict(X_all)
    risk_reg_pred = models["risk_reg"].predict(X_all)
    risk_cls_pred = models["risk_cls"].predict(X_all).astype(int)

    n    = len(yc_all)
    prob = np.zeros((n, 3), dtype=np.float32)
    for i, c in enumerate(risk_cls_pred):
        if 0 <= c < 3:
            prob[i, c] = 1.0

    return {
        "ability_true":  ya_all,
        "ability_pred":  ability_pred.astype(np.float32),
        "risk_reg_true": yr_all,
        "risk_reg_pred": risk_reg_pred.astype(np.float32),
        "risk_cls_true": yc_all.astype(int),
        "risk_cls_pred": risk_cls_pred,
        "risk_cls_prob": prob,
    }


# =============================================================================
# 对比表 & 报告生成
# =============================================================================

def build_comparison_tables(
    entries: List[ModelEntry],
) -> Tuple[pd.DataFrame, pd.DataFrame, str]:
    """生成表5.7（能力预测）和表5.8（风险预测）的 DataFrame 及文字报告。"""

    # ── 表5.7 ──────────────────────────────────────────────
    rows_ability = []
    for e in entries:
        m = e.metrics
        rows_ability.append({
            "模型":          _DISPLAY_NAMES.get(e.name, e.name),
            "MAE":           m.get("ability_MAE",  float("nan")),
            "RMSE":          m.get("ability_RMSE", float("nan")),
            "R²":            m.get("ability_R2",   float("nan")),
            "训练时间(min)":  e.train_time  if e.train_time  is not None else float("nan"),
            "参数量":         e.param_count if e.param_count is not None else "-",
        })
    df_ability = pd.DataFrame(rows_ability)

    # ── 表5.8 ──────────────────────────────────────────────
    rows_risk = []
    for e in entries:
        m = e.metrics
        rows_risk.append({
            "模型":         _DISPLAY_NAMES.get(e.name, e.name),
            "风险度MAE":    m.get("risk_reg_MAE",    float("nan")),
            "风险度R²":     m.get("risk_reg_R2",     float("nan")),
            "分类准确率":   m.get("cls_accuracy",     float("nan")),
            "高风险召回率": m.get("high_risk_recall", float("nan")),
            "高风险F1":     m.get("high_risk_f1",     float("nan")),
        })
    df_risk = pd.DataFrame(rows_risk)

    # ── 文字报告 ────────────────────────────────────────────
    SEP  = "=" * 72
    sep2 = "-" * 72
    lines = [
        SEP, "多模型横向对比评估报告",
        f"参与对比模型数: {len(entries)}", SEP, "",
        "【表5.7】动态驾驶能力预测性能对比", sep2,
        f"  {'模型':^10s}  {'MAE':>8s}  {'RMSE':>8s}  {'R²':>8s}  "
        f"{'训练时间(min)':>14s}  {'参数量':>8s}", sep2,
    ]
    for e in entries:
        m     = e.metrics
        t_val = e.train_time
        t_str = f"{float(t_val):>14.1f}" if (t_val is not None and not np.isnan(float(t_val))) else f"{'N/A':>14s}"
        p_str = f"{e.param_count:>8s}"   if e.param_count else f"{'N/A':>8s}"
        lines.append(
            f"  {_DISPLAY_NAMES.get(e.name, e.name):^10s}"
            f"  {m.get('ability_MAE',  float('nan')):>8.4f}"
            f"  {m.get('ability_RMSE', float('nan')):>8.4f}"
            f"  {m.get('ability_R2',   float('nan')):>8.4f}"
            f"  {t_str}  {p_str}"
        )
    lines += [
        sep2, "",
        "【表5.8】风险状态预测性能对比", sep2,
        f"  {'模型':^10s}  {'风险度MAE':>10s}  {'风险度R²':>10s}  "
        f"{'分类准确率':>10s}  {'高风险召回率':>12s}  {'高风险F1':>10s}", sep2,
    ]
    for e in entries:
        m = e.metrics
        lines.append(
            f"  {_DISPLAY_NAMES.get(e.name, e.name):^10s}"
            f"  {m.get('risk_reg_MAE',     float('nan')):>10.4f}"
            f"  {m.get('risk_reg_R2',      float('nan')):>10.4f}"
            f"  {m.get('cls_accuracy',     float('nan')):>10.4f}"
            f"  {m.get('high_risk_recall', float('nan')):>12.4f}"
            f"  {m.get('high_risk_f1',     float('nan')):>10.4f}"
        )
    lines += [sep2, ""]

    def _best(key: str, higher: bool = True) -> str:
        vals = [(e, e.metrics.get(key, float("nan"))) for e in entries]
        vals = [(e, v) for e, v in vals if not np.isnan(float(v))]
        if not vals:
            return "N/A"
        best_e = max(vals, key=lambda x: x[1]) if higher else min(vals, key=lambda x: x[1])
        return _DISPLAY_NAMES.get(best_e[0].name, best_e[0].name)

    lines += [
        "【各指标最优模型汇总】",
        f"  能力预测 R²  最优: {_best('ability_R2',       True)}",
        f"  能力预测 MAE 最低: {_best('ability_MAE',      False)}",
        f"  能力预测 RMSE最低: {_best('ability_RMSE',     False)}",
        f"  风险度 MAE   最低: {_best('risk_reg_MAE',     False)}",
        f"  风险度 R²    最优: {_best('risk_reg_R2',      True)}",
        f"  分类准确率   最高: {_best('cls_accuracy',     True)}",
        f"  高风险召回率 最高: {_best('high_risk_recall', True)}",
        f"  高风险 F1    最高: {_best('high_risk_f1',     True)}",
        "", SEP,
    ]
    return df_ability, df_risk, "\n".join(lines)


# =============================================================================
# 主对比函数
# =============================================================================

def compare_models(
    cfg:          dict,
    ckpt_list:    Optional[List[Tuple[str, str]]] = None,
    ckpt_dir:     Optional[str]                   = None,
    train_times:  Optional[Dict[str, float]]      = None,
    param_counts: Optional[Dict[str, str]]        = None,
) -> None:
    """
    检查点来源优先级（高→低）：
      命令行 --ckpt_list  >  YAML paths.compare_ckpt_list
                          >  命令行 --ckpt_dir
                          >  YAML paths.compare_ckpt_dir

    元信息优先级（高→低）：
      命令行 > YAML compare_train_times / compare_param_counts > 检查点内置字段
    """
    # 1. 确定检查点列表
    if ckpt_list is None:
        ckpt_list = resolve_ckpt_list_from_yaml(cfg)
    if ckpt_list is None:
        scan_dir = ckpt_dir or str(cfg.get("paths", {}).get("compare_ckpt_dir", "")).strip()
        if not scan_dir:
            raise ValueError(
                "请通过以下任一方式提供检查点来源：\n"
                "  1. 命令行: --ckpt_list model:path ...\n"
                "  2. 命令行: --ckpt_dir <目录>\n"
                "  3. YAML  : paths.compare_ckpt_list\n"
                "  4. YAML  : paths.compare_ckpt_dir"
            )
        print(f"\n  自动扫描检查点目录: {scan_dir}")
        ckpt_list = discover_checkpoints(scan_dir)
        if not ckpt_list:
            raise FileNotFoundError(f"在 {scan_dir} 中未找到任何可识别的检查点")

    print(f"\n  已确定对比模型列表（共 {len(ckpt_list)} 个）:")
    for mtype, path in ckpt_list:
        print(f"    [{_DISPLAY_NAMES.get(mtype, mtype):^10s}]  {path}")

    # 2. 合并元信息（YAML 为底，命令行覆盖）
    yaml_tt = cfg.get("compare_train_times",  {}) or {}
    yaml_pc = cfg.get("compare_param_counts", {}) or {}
    merged_tt: Dict[str, float] = {k: float(v) for k, v in yaml_tt.items()}
    merged_pc: Dict[str, str]   = {k: str(v)   for k, v in yaml_pc.items()}
    if train_times:  merged_tt.update(train_times)
    if param_counts: merged_pc.update(param_counts)

    # 3. 准备数据集
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\n  设备: {device}")
    print(f"  加载数据集: {cfg['paths']['dataset_pkl']}")
    with open(cfg["paths"]["dataset_pkl"], "rb") as f:
        data = pickle.load(f)
    split_name = cfg["eval"].get("split", "test")
    loader = DataLoader(
        TorchDataset(data[split_name]),
        batch_size=cfg["eval"]["batch_size"], shuffle=False, num_workers=0,
    )
    print(f"  评估集: {split_name}，样本数={len(loader.dataset)}")

    # 4. 逐模型推理
    entries:   List[ModelEntry] = []
    threshold: float            = cfg["eval"].get("risk_thresh_high", 0.1)

    for idx, (mtype, ckpt_path) in enumerate(ckpt_list, 1):
        display = _DISPLAY_NAMES.get(mtype, mtype)
        print(f"\n  [{idx}/{len(ckpt_list)}] 推理: {display}  ({ckpt_path})")
        if not os.path.exists(ckpt_path):
            print(f"    ⚠ 文件不存在，跳过")
            continue

        t_meta = merged_tt.get(mtype)
        p_meta = merged_pc.get(mtype)
        if isinstance(p_meta, str) and p_meta.strip() in ("", "null", "None"):
            p_meta = None

        entry = ModelEntry(
            name        = mtype,
            ckpt_path   = ckpt_path,
            train_time  = float(t_meta) if t_meta is not None else None,
            param_count = str(p_meta)   if p_meta is not None else None,
        )

        t0 = time.time()
        try:
            if ckpt_path.endswith(".pkl"):
                preds = _predict_sklearn(ckpt_path, loader)
            else:
                preds, param_count_int, train_time_ckpt = _predict_dl(ckpt_path, loader, device)
                if entry.param_count is None and param_count_int > 0:
                    entry.param_count = _fmt_params(param_count_int)
                if entry.train_time is None and not np.isnan(train_time_ckpt):
                    entry.train_time = train_time_ckpt

            entry.metrics = _compute_all_metrics(preds, threshold)
            entry.preds   = preds
            entries.append(entry)
            print(
                f"    ✓ 完成  {time.time()-t0:.1f}s  "
                f"能力R²={entry.metrics['ability_R2']:.4f}  "
                f"分类准确率={entry.metrics['cls_accuracy']:.4f}"
            )
        except Exception as exc:
            print(f"    ✗ 失败: {exc}")

    if not entries:
        raise RuntimeError("没有任何模型成功完成推理，请检查路径与数据集")

    # 5. 生成对比报告
    print("\n  生成对比报告...")
    df_ability, df_risk, report_str = build_comparison_tables(entries)
    print("\n" + report_str)

    # 6. 保存
    out_dir = cfg["paths"]["output_dir"]
    os.makedirs(out_dir, exist_ok=True)

    df_ability.to_csv(os.path.join(out_dir, "comparison_ability.csv"), index=False, encoding="utf-8-sig")
    df_risk.to_csv(   os.path.join(out_dir, "comparison_risk.csv"),    index=False, encoding="utf-8-sig")
    with open(os.path.join(out_dir, "comparison_report.txt"), "w", encoding="utf-8") as f:
        f.write(report_str)

    print(f"\n  能力对比表 → {os.path.join(out_dir, 'comparison_ability.csv')}")
    print(f"  风险对比表 → {os.path.join(out_dir, 'comparison_risk.csv')}")
    print(f"  文字报告   → {os.path.join(out_dir, 'comparison_report.txt')}")

    for e in entries:
        if e.preds is not None:
            pd.DataFrame({
                "ability_true":  e.preds["ability_true"],
                "ability_pred":  e.preds["ability_pred"].round(4),
                "risk_reg_true": e.preds["risk_reg_true"],
                "risk_reg_pred": e.preds["risk_reg_pred"].round(4),
                "risk_cls_true": e.preds["risk_cls_true"],
                "risk_cls_pred": e.preds["risk_cls_pred"],
            }).to_csv(
                os.path.join(out_dir, f"predictions_{_DISPLAY_NAMES.get(e.name, e.name)}.csv"),
                index=False, encoding="utf-8-sig",
            )


# =============================================================================
# 命令行入口
# =============================================================================

def main():
    parser = argparse.ArgumentParser(
        description="MT-JP 多模型横向对比评估器",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("-c", "--config",  type=str, default=None,
                        help="YAML 配置文件路径（默认: config/evaluator_compare.yaml）")
    parser.add_argument("--split",         type=str, default=None,
                        help="评估集：train / val / test（覆盖 yaml eval.split）")
    parser.add_argument("--output_dir",    type=str, default=None)
    parser.add_argument("--ckpt_dir",      type=str, default=None,
                        help="自动扫描检查点根目录（覆盖 YAML paths.compare_ckpt_dir）")
    parser.add_argument("--ckpt_list",     type=str, nargs="+", default=None,
                        help="手动指定 model_type:ckpt_path 列表，空格分隔\n"
                             "例: svr:runs/svr/model.pkl mt_jp:runs/mtjp/best_model.pt")
    parser.add_argument("--train_times",   type=str, default=None,
                        help="训练时间(分)，格式: model:time,...\n例: svr:23.5,mt_jp:62.8")
    parser.add_argument("--param_counts",  type=str, default=None,
                        help="参数量，格式: model:count,...\n例: lstm:3.2M,mt_jp:2.8M")
    args = parser.parse_args()

    cfg = load_config(args.config)
    if args.split:      cfg["eval"]["split"]       = args.split
    if args.output_dir: cfg["paths"]["output_dir"] = args.output_dir

    # 解析命令行参数
    ckpt_list_cli = None
    if args.ckpt_list:
        ckpt_list_cli = []
        for item in args.ckpt_list:
            if ":" not in item:
                parser.error(f"--ckpt_list 格式错误（需 model_type:path）: {item}")
            mtype, path = item.split(":", 1)
            ckpt_list_cli.append((mtype.strip().lower(), path.strip()))

    train_times_cli = None
    if args.train_times:
        train_times_cli = {}
        for item in args.train_times.split(","):
            k, v = item.split(":", 1)
            train_times_cli[k.strip().lower()] = float(v.strip())

    param_counts_cli = None
    if args.param_counts:
        param_counts_cli = {}
        for item in args.param_counts.split(","):
            k, v = item.split(":", 1)
            param_counts_cli[k.strip().lower()] = v.strip()

    print("=" * 60)
    print("MT-JP 联合预测模型 — 多模型横向对比评估器")
    print(f"  数据集:   {cfg['paths']['dataset_pkl']}")
    print(f"  评估集:   {cfg['eval'].get('split', 'test')}")
    print(f"  输出目录: {cfg['paths']['output_dir']}")
    print("=" * 60)

    compare_models(
        cfg          = cfg,
        ckpt_list    = ckpt_list_cli,
        ckpt_dir     = args.ckpt_dir,
        train_times  = train_times_cli,
        param_counts = param_counts_cli,
    )


if __name__ == "__main__":
    main()

# src/3_prediction/trainer_svr_cart.py
"""
SVR / CART 基线模型训练器

传统机器学习基线，使用与深度学习模型相同的数据集，
将时间窗口 (T=5, D=17) 展平为 85 维特征向量后训练。

每个任务（ability 回归、risk 回归、risk 分类）分别训练独立模型。
评价指标：MAE、RMSE、R²（回归任务），Accuracy、F1（分类任务）。

输出目录：output/3_prediction/runs/{timestamp}_svr|cart/

用法：
  python trainer_svr_cart.py
  python trainer_svr_cart.py -c config/trainer_svr_cart.yaml
  python trainer_svr_cart.py --model svr
  python trainer_svr_cart.py --model cart
"""

import os
import sys
import pickle
import argparse
import warnings
import yaml
import time
from datetime import datetime
import numpy as np
import pandas as pd
from typing import Optional

warnings.filterwarnings("ignore")

from sklearn.svm import SVR
from sklearn.tree import DecisionTreeRegressor, DecisionTreeClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score,
    accuracy_score,
    f1_score,
    classification_report,
)


# =============================================================================
# 配置加载
# =============================================================================

_DEFAULT_CONFIG_PATH = os.path.join(
    os.path.dirname(os.path.abspath(__file__)), "..", "..", "config", "trainer_svr_cart.yaml"
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
# 运行目录管理
# =============================================================================

def make_run_dir(runs_root: str, timestamp: str, model_name: str) -> dict:
    run_dir = os.path.join(runs_root, f"{timestamp}_{model_name}")
    os.makedirs(run_dir, exist_ok=True)
    return {
        "run_dir":    run_dir,
        "log_csv":    os.path.join(run_dir, "train_log.csv"),
        "metrics":    os.path.join(run_dir, "metrics.csv"),
        "run_config": os.path.join(run_dir, "run_config.yaml"),
        "model_pkl":  os.path.join(run_dir, "model.pkl"),
    }


def save_run_config(cfg: dict, path: str) -> None:
    def _to_basic(obj):
        if isinstance(obj, dict):
            return {k: _to_basic(v) for k, v in obj.items()}
        if isinstance(obj, (list, tuple)):
            return [_to_basic(i) for i in obj]
        if isinstance(obj, (np.integer, np.floating)):
            return obj.item()
        return obj
    with open(path, "w", encoding="utf-8") as f:
        yaml.dump(_to_basic(cfg), f, allow_unicode=True, sort_keys=False)


# =============================================================================
# 数据加载 & 展平
# =============================================================================

def load_and_flatten(dataset_pkl: str):
    """
    加载 pkl 数据集并将 (N, T, D) 展平为 (N, T*D)。
    返回 train/val/test 各 split 的 (X_flat, y_ability, y_risk_reg, y_risk_cls)。
    """
    if not os.path.exists(dataset_pkl):
        raise FileNotFoundError(f"数据集不存在: {dataset_pkl}")
    with open(dataset_pkl, "rb") as f:
        data = pickle.load(f)

    splits = {}
    for name in ("train", "val", "test"):
        sd = data[name]
        X_flat = sd["X"].reshape(len(sd["X"]), -1).astype(np.float32)  # (N, T*D)
        splits[name] = {
            "X":          X_flat,
            "y_ability":  sd["y_ability"].astype(np.float32),
            "y_risk_reg": sd["y_risk_reg"].astype(np.float32),
            "y_risk_cls": sd["y_risk_cls"].astype(np.int64),
        }
        print(f"  {name}: {X_flat.shape[0]} 样本，特征维度 {X_flat.shape[1]}")
    return splits


# =============================================================================
# 评估函数
# =============================================================================

def regression_metrics(y_true: np.ndarray, y_pred: np.ndarray, task: str) -> dict:
    mae  = mean_absolute_error(y_true, y_pred)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    r2   = r2_score(y_true, y_pred)
    return {"task": task, "MAE": mae, "RMSE": rmse, "R2": r2}


def classification_metrics(y_true: np.ndarray, y_pred: np.ndarray, task: str) -> dict:
    acc = accuracy_score(y_true, y_pred)
    f1  = f1_score(y_true, y_pred, average="weighted")
    return {"task": task, "Accuracy": acc, "F1_weighted": f1}


# =============================================================================
# 模型构建
# =============================================================================

def build_svr_models(cfg: dict):
    """返回 3 个任务对应的 SVR/DecisionTree 模型字典。"""
    sc = cfg["svr"]
    return {
        "ability":  SVR(kernel=sc["kernel"], C=sc["C"], gamma=sc["gamma"], epsilon=sc["epsilon"]),
        "risk_reg": SVR(kernel=sc["kernel"], C=sc["C"], gamma=sc["gamma"], epsilon=sc["epsilon"]),
        "risk_cls": None,   # 分类任务用 SVC，此处暂用 CART 替代以简化依赖
    }


def build_cart_models(cfg: dict):
    cc = cfg["cart"]
    return {
        "ability":  DecisionTreeRegressor(
            max_depth         = cc["max_depth"],
            min_samples_split = cc["min_samples_split"],
            random_state      = cc["seed"],
        ),
        "risk_reg": DecisionTreeRegressor(
            max_depth         = cc["max_depth"],
            min_samples_split = cc["min_samples_split"],
            random_state      = cc["seed"],
        ),
        "risk_cls": DecisionTreeClassifier(
            max_depth         = cc["max_depth"],
            min_samples_split = cc["min_samples_split"],
            random_state      = cc["seed"],
            class_weight      = "balanced",
        ),
    }


# =============================================================================
# SVR 分类任务（用 SVC 代替）
# =============================================================================

def _build_svc(cfg: dict):
    from sklearn.svm import SVC
    sc = cfg["svr"]
    return SVC(
        kernel      = sc["kernel"],
        C           = sc["C"],
        gamma       = sc["gamma"],
        decision_function_shape = "ovr",
        random_state = 42,
    )


# =============================================================================
# 主训练流程
# =============================================================================

def train(cfg: dict, model_type: str) -> None:
    model_type = model_type.lower()
    assert model_type in ("svr", "cart"), f"未知模型类型: {model_type}"

    timestamp  = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_paths  = make_run_dir(cfg["paths"]["runs_root"], timestamp, model_type)
    print(f"\n  本次运行目录: {run_paths['run_dir']}")
    save_run_config(cfg, run_paths["run_config"])

    # ── 数据 ──────────────────────────────────────────────────
    print("\n[1/3] 加载数据集...")
    splits = load_and_flatten(cfg["paths"]["dataset_pkl"])

    # 用训练集均值/标准差标准化（SVR 对特征尺度敏感）
    scaler = StandardScaler()
    X_train = scaler.fit_transform(splits["train"]["X"])
    X_val   = scaler.transform(splits["val"]["X"])
    X_test  = scaler.transform(splits["test"]["X"])

    # 合并 train+val 一起训练（传统ML不依赖验证集早停）
    X_tv     = np.concatenate([X_train, X_val],                            axis=0)
    ya_tv    = np.concatenate([splits["train"]["y_ability"],  splits["val"]["y_ability"]],  axis=0)
    yreg_tv  = np.concatenate([splits["train"]["y_risk_reg"], splits["val"]["y_risk_reg"]], axis=0)
    ycls_tv  = np.concatenate([splits["train"]["y_risk_cls"], splits["val"]["y_risk_cls"]], axis=0)
    print(f"  合并 train+val: {X_tv.shape[0]} 样本 (test: {X_test.shape[0]})")

    # ── 构建模型 ──────────────────────────────────────────────
    print(f"\n[2/3] 训练 {model_type.upper()} 模型（共 3 个任务）...")
    if model_type == "svr":
        models = build_svr_models(cfg)
        models["risk_cls"] = _build_svc(cfg)
    else:
        models = build_cart_models(cfg)

    task_data = {
        "ability":  (ya_tv,   splits["test"]["y_ability"]),
        "risk_reg": (yreg_tv, splits["test"]["y_risk_reg"]),
        "risk_cls": (ycls_tv, splits["test"]["y_risk_cls"]),
    }

    all_metrics = []
    predictions = {}

    for task, (y_tv, y_te) in task_data.items():
        t0 = time.time()
        print(f"  → 任务 [{task}] 开始训练...", end=" ", flush=True)
        models[task].fit(X_tv, y_tv)
        elapsed = time.time() - t0
        print(f"完成 ({elapsed:.1f}s)")

        y_pred = models[task].predict(X_test)
        predictions[task] = y_pred

        if task in ("ability", "risk_reg"):
            m = regression_metrics(y_te, y_pred, task)
            print(f"     MAE={m['MAE']:.4f}  RMSE={m['RMSE']:.4f}  R²={m['R2']:.4f}")
        else:
            m = classification_metrics(y_te, y_pred.astype(int), task)
            print(f"     Accuracy={m['Accuracy']:.4f}  F1={m['F1_weighted']:.4f}")
            print(classification_report(y_te, y_pred.astype(int),
                                        target_names=["低", "中", "高"], zero_division=0))
        all_metrics.append(m)

    # ── 保存结果 ───────────────────────────────────────────────
    print(f"\n[3/3] 保存结果...")
    metrics_df = pd.DataFrame(all_metrics)
    metrics_df.to_csv(run_paths["metrics"], index=False, encoding="utf-8-sig")
    print(f"  评估指标 → {run_paths['metrics']}")

    with open(run_paths["model_pkl"], "wb") as f:
        pickle.dump({"models": models, "scaler": scaler, "cfg": cfg}, f)
    print(f"  模型文件 → {run_paths['model_pkl']}")

    print(f"\n{'='*60}")
    print(f"训练完成  [{model_type.upper()}  {timestamp}]")
    print(f"  运行目录: {run_paths['run_dir']}")
    print("=" * 60)


# =============================================================================
# 命令行入口
# =============================================================================

def main():
    parser = argparse.ArgumentParser(
        description="SVR / CART 基线模型训练器",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("-c", "--config", type=str, default=None,
                        help="YAML 配置文件路径（默认: config/trainer_svr_cart.yaml）")
    parser.add_argument("--model", type=str, default=None,
                        help="模型类型: svr | cart（覆盖配置文件中的 model_type）")
    args = parser.parse_args()

    cfg = load_config(args.config)
    model_type = args.model or cfg.get("model_type", "svr")

    print("=" * 60)
    print(f"传统机器学习基线 — {model_type.upper()} 训练器")
    print(f"  dataset_pkl = {cfg['paths']['dataset_pkl']}")
    print(f"  runs_root   = {cfg['paths']['runs_root']}")
    print("=" * 60)

    train(cfg, model_type)


if __name__ == "__main__":
    main()

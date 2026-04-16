# /root/autodl-tmp/DynamicCapRisk/src/3_prediction/trainer_dl.py
"""
MTRP / LSTM / GRU / CNN-LSTM 风险预测模型训练器
输出目录：output/3_prediction/runs/
"""

import os
import sys
import time
import pickle
import argparse
import warnings
import yaml
from datetime import datetime
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torch.utils.tensorboard import SummaryWriter
import torch.nn.functional as F
from typing import Dict, Optional, Tuple

# 忽略警告
warnings.filterwarnings("ignore")

# 添加项目路径
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
# 修正模型导入
from src.models.mtrp_model import mtrp, build_model as _build_mtrp
from src.models.baseline_models import build_baseline_model
from evaluator import evaluate as run_evaluation
from evaluator import load_config as load_eval_config

# =============================================================================
# 配置加载
# =============================================================================
_DEFAULT_CONFIG_PATH = os.path.join(
    os.path.dirname(os.path.abspath(__file__)),
    "..", "..",
    "config", "trainer_dl.yaml"
)

def load_config(path: Optional[str] = None) -> dict:
    candidate = path or _DEFAULT_CONFIG_PATH
    if not os.path.exists(candidate):
        raise FileNotFoundError(f"配置文件不存在: {candidate}")
    with open(candidate, encoding="utf-8") as f:
        cfg = yaml.safe_load(f) or {}
    print(f"已加载配置文件: {os.path.abspath(candidate)}")
    return cfg

# =============================================================================
# 模型工厂
# =============================================================================
_DL_BASELINES = {"lstm", "gru", "cnn_lstm"}

def build_model(cfg: dict) -> nn.Module:
    model_type = cfg["model"].get("model_type", "mtrp").lower()
    if model_type == "mtrp":
        return _build_mtrp(cfg)
    elif model_type in _DL_BASELINES:
        return build_baseline_model(cfg)
    else:
        raise ValueError(f"未知 model_type: {model_type}，可选值: mtrp / lstm / gru / cnn_lstm")

# =============================================================================
# 运行目录管理
# =============================================================================
def make_run_dir(runs_root: str, timestamp: str, model_type: str) -> dict:
    run_dir = os.path.join(runs_root, f"{timestamp}_{model_type}")
    os.makedirs(run_dir, exist_ok=True)
    
    paths = {
        "run_dir": run_dir,
        "tb_dir": os.path.join(run_dir, "tb_logs"),
        "best_ckpt": os.path.join(run_dir, "best_model.pt"),
        "final_ckpt": os.path.join(run_dir, "final_model.pt"),
        "log_csv": os.path.join(run_dir, "train_log.csv"),
        "run_config": os.path.join(run_dir, "run_config.yaml")
    }
    os.makedirs(paths["tb_dir"], exist_ok=True)
    return paths

def save_run_config(cfg: dict, path: str) -> None:
    def _to_basic(obj):
        if isinstance(obj, dict):
            return {k: _to_basic(v) for k, v in obj.items()}
        elif isinstance(obj, (list, tuple)):
            return [_to_basic(i) for i in obj]
        elif isinstance(obj, (int, float, str, bool)) or obj is None:
            return obj
        else:
            return str(obj)
    
    basic_cfg = _to_basic(cfg)
    with open(path, "w", encoding="utf-8") as f:
        yaml.dump(basic_cfg, f, sort_keys=False, allow_unicode=True)

# =============================================================================
# 数据集定义
# =============================================================================
class TorchDataset(Dataset):
    def __init__(self, split_data: dict):
        self.X = torch.from_numpy(split_data["X"]).float()
        self.y_risk_reg = torch.from_numpy(split_data["y_risk_reg"]).float()
        self.y_risk_cls = torch.from_numpy(split_data["y_risk_cls"]).long()

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.y_risk_reg[idx], self.y_risk_cls[idx]

# =============================================================================
# 数据加载器 + 加权采样
# =============================================================================
def build_dataloaders(
    dataset_pkl: str,
    batch_size: int,
    num_workers: int,
    seed: int
) -> Tuple[DataLoader, DataLoader, DataLoader]:
    
    with open(dataset_pkl, "rb") as f:
        data = pickle.load(f)

    g = torch.Generator()
    g.manual_seed(seed)

    # 训练集加权采样
    train_ds = TorchDataset(data["train"])
    train_labels = train_ds.y_risk_cls.numpy()
    class_counts = np.bincount(train_labels)
    class_weights = 1.0 / class_counts
    sample_weights = class_weights[train_labels]
    
    sampler = torch.utils.data.WeightedRandomSampler(
        weights=sample_weights,
        num_samples=len(train_ds),
        replacement=True,
        generator=g
    )

    train_loader = DataLoader(
        train_ds,
        batch_size=batch_size,
        sampler=sampler,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
        generator=g,
        drop_last=False
    )

    val_loader = DataLoader(
        TorchDataset(data["val"]),
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available()
    )
    test_loader = DataLoader(
        TorchDataset(data["test"]),
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available()
    )

    sample_batch = next(iter(train_loader))
    feat_dim = sample_batch[0].shape[-1]
    assert feat_dim == 18, f"输入维度错误！期望18维，实际{feat_dim}维"

    print(f"DataLoader: train={len(train_loader.dataset)} / val={len(val_loader.dataset)} / test={len(test_loader.dataset)}")
    return train_loader, val_loader, test_loader

# ===================== Focal Loss 定义 =====================
class FocalLoss(nn.Module):
    def __init__(self, alpha, gamma=2.0):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma

    def forward(self, logits, target):
        ce = F.cross_entropy(logits, target, weight=self.alpha, reduction='none')
        pt = torch.exp(-ce)
        return ((1 - pt) ** self.gamma * ce).mean()

# =============================================================================
# 损失函数
# =============================================================================
class MTRPLoss(nn.Module):
    def __init__(self, loss_cfg: dict, device: torch.device):
        super().__init__()
        self.lambda_reg = loss_cfg["lambda_risk_reg"]
        self.lambda_cls = loss_cfg["lambda_risk_cls"]
        
        self.mse_loss = nn.MSELoss()
        weights = torch.tensor(loss_cfg["cls_weights"], dtype=torch.float, device=device)
        # 保留FocalLoss
        self.ce_loss = FocalLoss(alpha=weights, gamma=2.0)

    def forward(
        self,
        outputs: Dict[str, torch.Tensor],
        y_reg: torch.Tensor,
        y_cls: torch.Tensor
    ) -> Tuple[torch.Tensor, Dict[str, float]]:
        
        pred_reg = outputs["risk_reg"].squeeze()
        pred_cls = outputs["risk_cls"]
        
        loss_reg = self.mse_loss(pred_reg, y_reg)
        loss_cls = self.ce_loss(pred_cls, y_cls)
        total_loss = self.lambda_reg * loss_reg + self.lambda_cls * loss_cls
        
        loss_dict = {
            "total": total_loss.item(),
            "risk_reg": loss_reg.item(),
            "risk_cls": loss_cls.item()
        }
        return total_loss, loss_dict

# =============================================================================
# 早停策略
# =============================================================================
class EarlyStopping:
    def __init__(self, patience: int = 15, min_delta: float = 1e-3):
        self.patience = patience
        self.min_delta = min_delta
        self.counter = 0
        self.best_loss = float("inf")
        self.early_stop = False

    def step(self, val_loss: float):
        if val_loss < self.best_loss - self.min_delta:
            self.best_loss = val_loss
            self.counter = 0
        else:
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True
        return self.early_stop

# =============================================================================
# 单轮训练 + 混合精度
# =============================================================================
def run_epoch(
    model: nn.Module,
    loader: DataLoader,
    criterion: MTRPLoss,
    optimizer: Optional[torch.optim.Optimizer],
    device: torch.device,
    scaler: Optional[torch.cuda.amp.GradScaler] = None,
    grad_clip: float = 1.0
) -> Dict[str, float]:
    
    is_train = optimizer is not None
    model.train() if is_train else model.eval()
    
    total_loss = 0.0
    total_reg = 0.0
    total_cls = 0.0
    num_batches = len(loader)

    with torch.set_grad_enabled(is_train):
        for X, y_reg, y_cls in loader:
            X = X.to(device, non_blocking=True)
            y_reg = y_reg.to(device, non_blocking=True)
            y_cls = y_cls.to(device, non_blocking=True)

            if is_train and scaler is not None:
                with torch.cuda.amp.autocast():
                    outputs = model(X)
                    loss, loss_dict = criterion(outputs, y_reg, y_cls)
                optimizer.zero_grad()
                scaler.scale(loss).backward()
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
                scaler.step(optimizer)
                scaler.update()
            else:
                outputs = model(X)
                loss, loss_dict = criterion(outputs, y_reg, y_cls)
                if is_train:
                    optimizer.zero_grad()
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
                    optimizer.step()

            total_loss += loss_dict["total"]
            total_reg += loss_dict["risk_reg"]
            total_cls += loss_dict["risk_cls"]

    avg_loss = total_loss / num_batches
    avg_reg = total_reg / num_batches
    avg_cls = total_cls / num_batches

    return {
        "total": avg_loss,
        "risk_reg": avg_reg,
        "risk_cls": avg_cls
    }

# =============================================================================
# 主训练函数
# =============================================================================
def train(cfg: dict) -> None:
    model_type = cfg["model"]["model_type"].lower()
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_paths = make_run_dir(cfg["paths"]["runs_root"], timestamp, model_type)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.manual_seed(cfg["train"]["seed"])
    np.random.seed(cfg["train"]["seed"])

    print(f"\n模型类型: {model_type.upper()}")
    print(f"运行目录: {run_paths['run_dir']}")
    print(f"训练设备: {device}")

    save_run_config(cfg, run_paths["run_config"])

    print("\n[1/4] 加载数据集...")
    train_loader, val_loader, test_loader = build_dataloaders(
        dataset_pkl=cfg["paths"]["dataset_pkl"],
        batch_size=cfg["train"]["batch_size"],
        num_workers=cfg["train"]["num_workers"],
        seed=cfg["train"]["seed"]
    )

    print(f"\n[2/4] 构建模型 [{model_type.upper()}]...")
    model = build_model(cfg).to(device)
    criterion = MTRPLoss(cfg["loss"], device)
    
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=cfg["optimizer"]["lr"],
        weight_decay=cfg["optimizer"]["weight_decay"],
        betas=cfg["optimizer"]["betas"]
    )
    
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=cfg["scheduler"]["T_max"], eta_min=cfg["scheduler"]["eta_min"]
    )
    early_stopping = EarlyStopping(patience=cfg["train"]["patience"])
    writer = SummaryWriter(run_paths["tb_dir"])
    log_list = []
    best_val_loss = float("inf")

    print("\n[3/4] 开始训练...")
    max_epochs = cfg["train"]["max_epochs"]
    scaler = torch.cuda.amp.GradScaler() if torch.cuda.is_available() else None

    for epoch in range(1, max_epochs + 1):
        start_time = time.time()
        
        train_metrics = run_epoch(
            model=model,
            loader=train_loader,
            criterion=criterion,
            optimizer=optimizer,
            device=device,
            scaler=scaler,
            grad_clip=cfg["train"]["grad_clip"]
        )
        
        val_metrics = run_epoch(
            model=model,
            loader=val_loader,
            criterion=criterion,
            optimizer=None,
            device=device,
            scaler=None
        )
        
        scheduler.step()
        current_lr = optimizer.param_groups[0]["lr"]
        epoch_time = time.time() - start_time

        if val_metrics["total"] < best_val_loss:
            best_val_loss = val_metrics["total"]
            torch.save(model.state_dict(), run_paths["best_ckpt"])

        if early_stopping.step(val_metrics["total"]):
            print(f"\n早停触发！Epoch: {epoch}")
            break

        log_entry = {
            "epoch": epoch,
            "lr": round(current_lr, 6),
            "train_total": round(train_metrics["total"], 4),
            "train_reg": round(train_metrics["risk_reg"], 4),
            "train_cls": round(train_metrics["risk_cls"], 4),
            "val_total": round(val_metrics["total"], 4),
            "val_reg": round(val_metrics["risk_reg"], 4),
            "val_cls": round(val_metrics["risk_cls"], 4),
            "time": round(epoch_time, 2)
        }
        log_list.append(log_entry)

        writer.add_scalar("Loss/train_total", train_metrics["total"], epoch)
        writer.add_scalar("Loss/val_total", val_metrics["total"], epoch)
        writer.add_scalar("LR/lr", current_lr, epoch)

        if epoch % 5 == 0 or epoch == 1:
            print(
                f"Epoch {epoch:3d}/{max_epochs} | "
                f"Train Loss: {train_metrics['total']:.4f} | "
                f"Val Loss: {val_metrics['total']:.4f} | "
                f"LR: {current_lr:.6f} | "
                f"Time: {epoch_time:.2f}s"
            )

    torch.save(model.state_dict(), run_paths["final_ckpt"])
    pd.DataFrame(log_list).to_csv(run_paths["log_csv"], index=False)
    writer.close()

    print(f"\n训练完成！最优验证损失: {best_val_loss:.6f}")

    print("\n[4/4] 测试集评估...")
    model.load_state_dict(torch.load(run_paths["best_ckpt"], map_location=device))
    test_metrics = run_epoch(model, test_loader, criterion, None, device)
    print(
        f"测试集结果 | "
        f"总损失: {test_metrics['total']:.6f} | "
        f"回归损失: {test_metrics['risk_reg']:.6f} | "
        f"分类损失: {test_metrics['risk_cls']:.6f}"
    )

    print("\n[5/5] 训练完成，自动启动模型评估...")
    try:
        eval_cfg = load_eval_config()
        eval_cfg["paths"]["ckpt"] = run_paths["best_ckpt"]
        eval_cfg["paths"]["dataset_pkl"] = cfg["paths"]["dataset_pkl"]
        eval_cfg["eval"]["split"] = "test"
        
        run_evaluation(eval_cfg)
        print("\n✅ 自动评估完成！结果已保存至模型目录的 eval 文件夹")
    except Exception as e:
        print(f"\n❌ 自动评估失败: {e}")

# =============================================================================
# 主入口
# =============================================================================
def main():
    parser = argparse.ArgumentParser(description="风险预测模型训练脚本")
    parser.add_argument("-c", "--config", type=str, default=None, help="配置文件路径")
    args = parser.parse_args()

    cfg = load_config(args.config)
    
    print("=" * 60)
    print(f"风险预测训练器 — {cfg['model']['model_type'].upper()}")
    print(f"batch_size={cfg['train']['batch_size']} | lr={cfg['optimizer']['lr']}")
    print("=" * 60)

    train(cfg)

if __name__ == "__main__":
    main()
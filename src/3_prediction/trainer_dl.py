# src/3_prediction/trainer_dl.py
"""
MT-JP / LSTM / GRU / CNN-LSTM 联合预测模型训练器

通过 YAML 配置文件中的 model.model_type 字段选择模型：
  - mtjp      : Multi-Task Joint Prediction（默认）
  - lstm      : 2-层 LSTM 多任务基线
  - gru       : 2-层 GRU 多任务基线
  - cnn_lstm  : CNN-LSTM 混合多任务基线

输出目录：output/3_prediction/runs/

用法：
  python trainer.py
  python trainer.py -c config/trainer_dl.yaml
  python trainer.py --model_type lstm
  python trainer.py --model_type gru   --lr 5e-4
  python trainer.py --model_type cnn_lstm
  多次训练对比：tensorboard --logdir output/3_prediction/runs
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
from typing import Dict, Optional, Tuple

warnings.filterwarnings("ignore")

# 允许从同级目录导入
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from src.models.mtjp_model import MTJP, build_model as _build_mtjp
from src.models.baseline_models import build_baseline_model


# =============================================================================
# 配置加载（仅从 YAML 加载，无内置默认）
# =============================================================================

_DEFAULT_CONFIG_PATH = os.path.join(
    os.path.dirname(os.path.abspath(__file__)), "..", "..", "config", "trainer_dl.yaml"
)


def load_config(path: Optional[str] = None) -> dict:
    """
    加载配置：
      1. 优先使用指定的 path
      2. 未指定则使用默认路径 config/trainer_dl.yaml
      3. 配置文件不存在则抛出异常（无内置默认配置）
    """
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
# 模型工厂（统一入口）
# =============================================================================

_DL_BASELINES = {"lstm", "gru", "cnn_lstm"}


def build_model(cfg: dict) -> nn.Module:
    """
    根据 cfg["model"]["model_type"] 分发到对应的模型构建函数。
    model_type 不区分大小写。
    """
    model_type = cfg["model"].get("model_type", "mtjp").lower()
    if model_type == "mtjp":
        return _build_mtjp(cfg)
    elif model_type in _DL_BASELINES:
        return build_baseline_model(cfg)
    else:
        raise ValueError(
            f"未知 model_type='{model_type}'，"
            f"可选值: mtjp / lstm / gru / cnn_lstm\n"
            "（SVR / CART 请使用独立脚本 trainer_svr_cart.py）"
        )


# =============================================================================
# 运行目录管理
# =============================================================================


def make_run_dir(runs_root: str, timestamp: str, model_type: str) -> dict:
    """
    在 runs_root 下创建本次运行的独立目录，目录名含模型类型便于区分。

    目录结构：
      runs_root/
      └── {timestamp}_{model_type}/
          ├── best_model.pt
          ├── final_model.pt
          ├── train_log.csv
          ├── run_config.yaml
          └── events.out.tfevents... (TensorBoard 日志)
    """
    run_dir = os.path.join(runs_root, f"{timestamp}_{model_type}")
    os.makedirs(run_dir, exist_ok=True)
    return {
        "run_dir": run_dir,
        "tb_dir": run_dir,
        "best_ckpt": os.path.join(run_dir, "best_model.pt"),
        "final_ckpt": os.path.join(run_dir, "final_model.pt"),
        "log_csv": os.path.join(run_dir, "train_log.csv"),
        "run_config": os.path.join(run_dir, "run_config.yaml"),
    }


def save_run_config(cfg: dict, path: str) -> None:
    """将本次运行的完整配置序列化为 YAML，方便事后复现。"""

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
# Dataset
# =============================================================================


class TorchDataset(Dataset):
    """从 dataset_aug-False.pkl 的单个 split 构建 PyTorch Dataset。"""

    def __init__(self, split_data: dict):
        self.X = torch.from_numpy(split_data["X"]).float()
        self.y_ability = torch.from_numpy(split_data["y_ability"]).float()
        self.y_risk_reg = torch.from_numpy(split_data["y_risk_reg"]).float()
        self.y_risk_cls = torch.from_numpy(split_data["y_risk_cls"]).long()

    def __len__(self):
        return len(self.y_ability)

    def __getitem__(self, idx):
        return (
            self.X[idx],
            self.y_ability[idx],
            self.y_risk_reg[idx],
            self.y_risk_cls[idx],
        )


def build_dataloaders(
    dataset_pkl: str,
    batch_size: int,
    seed: int,
) -> Tuple[DataLoader, DataLoader, DataLoader]:
    if not os.path.exists(dataset_pkl):
        raise FileNotFoundError(f"数据集不存在: {dataset_pkl}")
    with open(dataset_pkl, "rb") as f:
        data = pickle.load(f)

    g = torch.Generator()
    g.manual_seed(seed)

    def _loader(split_name, shuffle):
        ds = TorchDataset(data[split_name])
        return DataLoader(
            ds,
            batch_size=batch_size,
            shuffle=shuffle,
            num_workers=0,
            pin_memory=torch.cuda.is_available(),
            generator=g if shuffle else None,
        )

    tr = _loader("train", True)
    va = _loader("val", False)
    te = _loader("test", False)
    print(
        f"  DataLoader: train={len(tr.dataset)} / val={len(va.dataset)} / test={len(te.dataset)}"
    )
    return tr, va, te


# =============================================================================
# 损失函数
# =============================================================================


class MTJPLoss(nn.Module):
    """
    联合损失函数（论文 §5.2.2）。
    所有多任务深度学习模型（MT-JP / LSTM / GRU / CNN-LSTM）共用此损失，
    保证对比实验公平性。
    """

    def __init__(self, loss_cfg: dict, device: torch.device):
        super().__init__()
        lc = loss_cfg
        self.lambda_ability = lc["lambda_ability"]
        self.lambda_risk_reg = lc["lambda_risk_reg"]
        self.lambda_risk_cls = lc["lambda_risk_cls"]
        self.lambda_consistency = lc["lambda_consistency"]
        self.tci_alpha = lc["tci_alpha"]
        self.tci_beta = lc["tci_beta"]

        self.huber = nn.HuberLoss(delta=lc["huber_delta"], reduction="mean")
        self.mse = nn.MSELoss(reduction="mean")
        w = torch.tensor(lc["cls_weights"], dtype=torch.float32).to(device)
        self.ce = nn.CrossEntropyLoss(weight=w, reduction="mean")

    def forward(
        self,
        outputs: Dict[str, torch.Tensor],
        y_ability: torch.Tensor,
        y_risk_reg: torch.Tensor,
        y_risk_cls: torch.Tensor,
        f_s: torch.Tensor,
    ) -> Tuple[torch.Tensor, dict]:
        y_hat_ability = outputs["ability"].squeeze(-1)
        y_hat_risk_reg = outputs["risk_reg"].squeeze(-1)
        y_hat_risk_cls = outputs["risk_cls"]

        l_ability = self.huber(y_hat_ability, y_ability)
        l_risk_reg = self.mse(y_hat_risk_reg, y_risk_reg)
        l_risk_cls = self.ce(y_hat_risk_cls, y_risk_cls)

        r_tci = self.tci_alpha * f_s - self.tci_beta * (2.0 * y_hat_ability - 1.0)
        l_consist = self.mse(r_tci, y_hat_risk_reg)

        total = (
            self.lambda_ability * l_ability
            + self.lambda_risk_reg * l_risk_reg
            + self.lambda_risk_cls * l_risk_cls
            + self.lambda_consistency * l_consist
        )
        loss_dict = {
            "total": total.item(),
            "ability": l_ability.item(),
            "risk_reg": l_risk_reg.item(),
            "risk_cls": l_risk_cls.item(),
            "consistency": l_consist.item(),
        }
        return total, loss_dict


# =============================================================================
# 早停
# =============================================================================


class EarlyStopping:
    def __init__(self, patience: int = 15, min_delta: float = 1e-5):
        self.patience = patience
        self.min_delta = min_delta
        self.counter = 0
        self.best_loss = float("inf")
        self.triggered = False

    def step(self, val_loss: float) -> bool:
        if val_loss < self.best_loss - self.min_delta:
            self.best_loss = val_loss
            self.counter = 0
        else:
            self.counter += 1
        if self.counter >= self.patience:
            self.triggered = True
        return self.triggered


# =============================================================================
# 训练单个 epoch
# =============================================================================


def _run_epoch(
    model: nn.Module,
    loader: DataLoader,
    criterion: MTJPLoss,
    optimizer: Optional[torch.optim.Optimizer],
    grad_clip: float,
    device: torch.device,
    train: bool,
) -> dict:
    model.train() if train else model.eval()
    agg = {k: 0.0 for k in ["total", "ability", "risk_reg", "risk_cls", "consistency"]}
    n_steps = 0

    ctx = torch.enable_grad() if train else torch.no_grad()
    with ctx:
        for X, ya, yr, yc in loader:
            X, ya, yr, yc = X.to(device), ya.to(device), yr.to(device), yc.to(device)
            f_s = X[:, -1, 16]  # 环境特征 F_S（最后时间步，第17维）

            outputs = model(X)
            loss, loss_dict = criterion(outputs, ya, yr, yc, f_s)

            if train:
                optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
                optimizer.step()

            for k, v in loss_dict.items():
                agg[k] += v
            n_steps += 1

    return {k: v / max(n_steps, 1) for k, v in agg.items()}


# =============================================================================
# 主训练器
# =============================================================================


def train(cfg: dict) -> None:
    model_type = cfg["model"].get("model_type", "mtjp").lower()

    # ── 生成时间戳，创建本次运行独立目录 ─────────────────────
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_paths = make_run_dir(cfg["paths"]["runs_root"], timestamp, model_type)
    print(f"\n  模型类型    : {model_type.upper()}")
    print(f"  本次运行目录: {run_paths['run_dir']}")

    save_run_config(cfg, run_paths["run_config"])
    print(f"  配置快照已保存: {run_paths['run_config']}")

    # ── 环境设置 ──────────────────────────────────────────────
    seed = cfg["train"]["seed"]
    torch.manual_seed(seed)
    np.random.seed(seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"  设备: {device}")

    # ── TensorBoard ───────────────────────────────────────────
    writer = SummaryWriter(log_dir=run_paths["tb_dir"])
    print(f"  TensorBoard 日志: {run_paths['tb_dir']}")

    # ── 数据 ──────────────────────────────────────────────────
    print("\n[1/4] 加载数据集...")
    train_loader, val_loader, test_loader = build_dataloaders(
        cfg["paths"]["dataset_pkl"],
        cfg["train"]["batch_size"],
        seed,
    )

    # ── 模型 ──────────────────────────────────────────────────
    print(f"\n[2/4] 构建模型 [{model_type.upper()}]...")
    model = build_model(cfg).to(device)
    print(f"  总参数量: {model.count_parameters():,}")
    try:
        sample_X = next(iter(train_loader))[0][:2].to(device)
        writer.add_graph(model, sample_X)
    except Exception:
        pass

    # ── 损失 / 优化器 / 调度器 ────────────────────────────────
    criterion = MTJPLoss(cfg["loss"], device)
    opt_cfg = cfg["optimizer"]
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=opt_cfg["lr"],
        weight_decay=opt_cfg["weight_decay"],
        betas=tuple(opt_cfg["betas"]),
    )
    sched_cfg = cfg["scheduler"]
    scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
        optimizer,
        T_0=sched_cfg["T_0"],
        eta_min=sched_cfg["eta_min"],
    )
    early_stop = EarlyStopping(patience=cfg["train"]["patience"])

    # ── 训练循环 ──────────────────────────────────────────────
    print("\n[3/4] 开始训练...")
    best_val_loss = float("inf")
    log_rows = []

    for epoch in range(1, cfg["train"]["max_epochs"] + 1):
        t0 = time.time()

        tr_losses = _run_epoch(
            model,
            train_loader,
            criterion,
            optimizer,
            cfg["train"]["grad_clip"],
            device,
            train=True,
        )
        va_losses = _run_epoch(
            model,
            val_loader,
            criterion,
            None,
            cfg["train"]["grad_clip"],
            device,
            train=False,
        )

        scheduler.step()
        elapsed = time.time() - t0
        lr_now = scheduler.get_last_lr()[0]

        # ── TensorBoard 记录 ──────────────────────────────────
        writer.add_scalar("Loss_train/total", tr_losses["total"], epoch)
        writer.add_scalar("Loss_val/total", va_losses["total"], epoch)
        writer.add_scalar("Loss_train/ability", tr_losses["ability"], epoch)
        writer.add_scalar("Loss_val/ability", va_losses["ability"], epoch)
        writer.add_scalar("Loss_train/risk_reg", tr_losses["risk_reg"], epoch)
        writer.add_scalar("Loss_val/risk_reg", va_losses["risk_reg"], epoch)
        writer.add_scalar("Loss_train/risk_cls", tr_losses["risk_cls"], epoch)
        writer.add_scalar("Loss_val/risk_cls", va_losses["risk_cls"], epoch)
        writer.add_scalar("Loss_train/consistency", tr_losses["consistency"], epoch)
        writer.add_scalar("Loss_val/consistency", va_losses["consistency"], epoch)
        writer.add_scalar("LR", lr_now, epoch)

        grad_norm = (
            sum(
                p.grad.data.norm(2).item() ** 2
                for p in model.parameters()
                if p.grad is not None
            )
            ** 0.5
        )
        writer.add_scalar("Grad/norm", grad_norm, epoch)

        # ── 保存最优权重 ──────────────────────────────────────
        if va_losses["total"] < best_val_loss:
            best_val_loss = va_losses["total"]
            torch.save(
                {
                    "epoch": epoch,
                    "timestamp": timestamp,
                    "model_type": model_type,
                    "model_state": model.state_dict(),
                    "val_loss": best_val_loss,
                    "cfg": cfg,
                },
                run_paths["best_ckpt"],
            )
            ckpt_mark = " ✓ best"
        else:
            ckpt_mark = ""

        # ── CSV 日志 ──────────────────────────────────────────
        log_rows.append(
            {
                "epoch": epoch,
                "lr": f"{lr_now:.2e}",
                "tr_total": f"{tr_losses['total']:.6f}",
                "tr_ability": f"{tr_losses['ability']:.6f}",
                "tr_risk_reg": f"{tr_losses['risk_reg']:.6f}",
                "tr_risk_cls": f"{tr_losses['risk_cls']:.6f}",
                "tr_consist": f"{tr_losses['consistency']:.6f}",
                "va_total": f"{va_losses['total']:.6f}",
                "va_ability": f"{va_losses['ability']:.6f}",
                "va_risk_reg": f"{va_losses['risk_reg']:.6f}",
                "va_risk_cls": f"{va_losses['risk_cls']:.6f}",
                "va_consist": f"{va_losses['consistency']:.6f}",
                "time_s": f"{elapsed:.1f}",
            }
        )
        pd.DataFrame(log_rows).to_csv(
            run_paths["log_csv"], index=False, encoding="utf-8-sig"
        )

        if epoch % 5 == 0 or epoch == 1:
            print(
                f"  Epoch {epoch:3d}/{cfg['train']['max_epochs']} | "
                f"tr={tr_losses['total']:.4f} | "
                f"va={va_losses['total']:.4f} | "
                f"lr={lr_now:.2e} | {elapsed:.1f}s{ckpt_mark}"
            )

        if early_stop.step(va_losses["total"]):
            print(
                f"\n  ⚡ 早停触发（Epoch {epoch}，patience={cfg['train']['patience']}）"
            )
            break

    # ── 保存最终权重 ───────────────────────────────────────────
    torch.save(
        {
            "epoch": epoch,
            "timestamp": timestamp,
            "model_type": model_type,
            "model_state": model.state_dict(),
            "cfg": cfg,
        },
        run_paths["final_ckpt"],
    )
    print(f"\n  训练日志  → {run_paths['log_csv']}")
    print(f"  最优权重  → {run_paths['best_ckpt']}  (val_loss={best_val_loss:.6f})")
    print(f"  最终权重  → {run_paths['final_ckpt']}")

    # ── 测试集快速评估 ─────────────────────────────────────────
    print("\n[4/4] 测试集快速损失评估...")
    ckpt = torch.load(run_paths["best_ckpt"], map_location=device)
    model.load_state_dict(ckpt["model_state"])
    te_losses = _run_epoch(
        model,
        test_loader,
        criterion,
        None,
        cfg["train"]["grad_clip"],
        device,
        train=False,
    )
    print(
        f"  测试集总损失: {te_losses['total']:.6f} | "
        f"ability={te_losses['ability']:.6f} | "
        f"risk_reg={te_losses['risk_reg']:.6f} | "
        f"risk_cls={te_losses['risk_cls']:.6f}"
    )

    # ── TensorBoard 超参数 & 最终指标 ─────────────────────────
    writer.add_hparams(
        hparam_dict={
            "model_type": model_type,
            "lr": cfg["optimizer"]["lr"],
            "batch_size": cfg["train"]["batch_size"],
            "d_model": cfg["model"].get("d_model", 0),
            "num_layers": cfg["model"].get(
                "num_layers", cfg["model"].get("baseline_num_layers", 0)
            ),
            "dropout": cfg["model"]["dropout"],
            "lambda_ability": cfg["loss"]["lambda_ability"],
            "lambda_risk_reg": cfg["loss"]["lambda_risk_reg"],
            "lambda_risk_cls": cfg["loss"]["lambda_risk_cls"],
            "lambda_consistency": cfg["loss"]["lambda_consistency"],
        },
        metric_dict={
            "hparam/best_val_loss": best_val_loss,
            "hparam/test_total": te_losses["total"],
            "hparam/test_ability": te_losses["ability"],
            "hparam/test_risk_reg": te_losses["risk_reg"],
            "hparam/test_risk_cls": te_losses["risk_cls"],
        },
        run_name=".",
    )
    writer.close()

    print(f"\n{'='*60}")
    print(f"训练完成  [{model_type.upper()}  {timestamp}]")
    print(f"  本次运行目录  : {run_paths['run_dir']}")
    print(f"  最优验证集损失: {best_val_loss:.6f}")
    print(f"  查看本次曲线  : tensorboard --logdir {run_paths['tb_dir']}")
    print(f"  多次运行对比  : tensorboard --logdir {cfg['paths']['runs_root']}")
    print("=" * 60)


# =============================================================================
# 命令行入口
# =============================================================================


def main():
    parser = argparse.ArgumentParser(
        description="MT-JP / LSTM / GRU / CNN-LSTM 多任务联合预测模型训练器",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例:
  python trainer.py                              # 使用默认 YAML，默认模型(mtjp)
  python trainer.py --model_type lstm            # 切换为 LSTM 基线
  python trainer.py --model_type gru  --lr 5e-4  # GRU + 自定义学习率
  python trainer.py --model_type cnn_lstm
        """,
    )
    parser.add_argument(
        "-c",
        "--config",
        type=str,
        default=None,
        help="YAML 配置文件路径（默认: config/trainer_dl.yaml）",
    )
    parser.add_argument(
        "--model_type",
        type=str,
        default=None,
        help="模型类型: mtjp | lstm | gru | cnn_lstm（覆盖 YAML）",
    )
    parser.add_argument("--batch_size", type=int, default=None)
    parser.add_argument("--max_epochs", type=int, default=None)
    parser.add_argument("--lr", type=float, default=None)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--runs_root", type=str, default=None)
    args = parser.parse_args()

    cfg = load_config(args.config)

    # 命令行覆盖 YAML
    if args.model_type is not None:
        cfg["model"]["model_type"] = args.model_type
    if args.batch_size is not None:
        cfg["train"]["batch_size"] = args.batch_size
    if args.max_epochs is not None:
        cfg["train"]["max_epochs"] = args.max_epochs
    if args.lr is not None:
        cfg["optimizer"]["lr"] = args.lr
    if args.seed is not None:
        cfg["train"]["seed"] = args.seed
    if args.runs_root is not None:
        cfg["paths"]["runs_root"] = args.runs_root

    model_type = cfg["model"].get("model_type", "mtjp")

    print("=" * 60)
    print(f"联合预测模型训练器 — {model_type.upper()}")
    print(f"  batch_size  = {cfg['train']['batch_size']}")
    print(f"  max_epochs  = {cfg['train']['max_epochs']}")
    print(f"  lr          = {cfg['optimizer']['lr']}")
    print(f"  patience    = {cfg['train']['patience']}")
    print(f"  runs_root   = {cfg['paths']['runs_root']}")
    print("=" * 60)

    train(cfg)


if __name__ == "__main__":
    main()

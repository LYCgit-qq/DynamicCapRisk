# src/models/baseline_models.py
"""
MT-JP 对比基线模型（深度学习部分）
包含：LSTM、GRU、CNN-LSTM
均采用多任务输出头，与 MT-JP 保持相同的损失函数接口。

输入: X  shape=(N, T=5, D=17)  float32
输出: dict
  ability   → (N, 1)   能力预测（回归）
  risk_reg  → (N, 1)   风险度预测（回归）
  risk_cls  → (N, 3)   风险等级预测（分类 logits）
"""

import torch
import torch.nn as nn
from typing import Dict


# =============================================================================
# 共用的多任务输出头
# =============================================================================

class MultiTaskHead(nn.Module):
    """三任务输出头，被所有基线模型复用。"""

    def __init__(self, in_dim: int, n_classes: int = 3, dropout: float = 0.2):
        super().__init__()
        self.head_ability = nn.Sequential(
            nn.Linear(in_dim, 64),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(64, 1),
            nn.Sigmoid(),          # 能力值 ∈ [0, 1]
        )
        self.head_risk_reg = nn.Sequential(
            nn.Linear(in_dim, 64),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(64, 1),
            nn.Tanh(),             # 风险度 ∈ [-1, 1]
        )
        self.head_risk_cls = nn.Sequential(
            nn.Linear(in_dim, 64),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(64, n_classes),  # logits，CrossEntropyLoss 自带 softmax
        )

    def forward(self, h: torch.Tensor) -> Dict[str, torch.Tensor]:
        return {
            "ability":  self.head_ability(h),
            "risk_reg": self.head_risk_reg(h),
            "risk_cls": self.head_risk_cls(h),
        }


# =============================================================================
# LSTM
# =============================================================================

class LSTMModel(nn.Module):
    """
    2 层 LSTM + 多任务输出头。
    论文对比配置：隐藏层维度 128，Dropout 0.2。
    """

    def __init__(
        self,
        seq_len:   int   = 5,
        input_dim: int   = 17,
        hidden:    int   = 128,
        num_layers:int   = 2,
        dropout:   float = 0.2,
        n_classes: int   = 3,
    ):
        super().__init__()
        self.lstm = nn.LSTM(
            input_size  = input_dim,
            hidden_size = hidden,
            num_layers  = num_layers,
            batch_first = True,
            dropout     = dropout if num_layers > 1 else 0.0,
        )
        self.dropout = nn.Dropout(dropout)
        self.head    = MultiTaskHead(hidden, n_classes, dropout)

    def forward(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        # x: (N, T, D)
        out, _ = self.lstm(x)        # (N, T, hidden)
        h      = self.dropout(out[:, -1, :])   # 取最后时间步
        return self.head(h)

    def count_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


# =============================================================================
# GRU
# =============================================================================

class GRUModel(nn.Module):
    """
    2 层 GRU + 多任务输出头。
    结构与 LSTM 对称，参数量更少。
    """

    def __init__(
        self,
        seq_len:   int   = 5,
        input_dim: int   = 17,
        hidden:    int   = 128,
        num_layers:int   = 2,
        dropout:   float = 0.2,
        n_classes: int   = 3,
    ):
        super().__init__()
        self.gru = nn.GRU(
            input_size  = input_dim,
            hidden_size = hidden,
            num_layers  = num_layers,
            batch_first = True,
            dropout     = dropout if num_layers > 1 else 0.0,
        )
        self.dropout = nn.Dropout(dropout)
        self.head    = MultiTaskHead(hidden, n_classes, dropout)

    def forward(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        out, _ = self.gru(x)
        h      = self.dropout(out[:, -1, :])
        return self.head(h)

    def count_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


# =============================================================================
# CNN-LSTM
# =============================================================================

class CNNLSTMModel(nn.Module):
    """
    2 层 1D-CNN（局部特征提取）+ 1 层 LSTM（时序建模）+ 多任务输出头。
    卷积核大小 3，通道数 64；LSTM 隐藏层维度 128。
    """

    def __init__(
        self,
        seq_len:     int   = 5,
        input_dim:   int   = 17,
        cnn_channels:int   = 64,
        cnn_layers:  int   = 2,
        kernel_size: int   = 3,
        lstm_hidden: int   = 128,
        dropout:     float = 0.2,
        n_classes:   int   = 3,
    ):
        super().__init__()

        # ── 1D-CNN 特征提取 ────────────────────────────────────
        cnn_blocks = []
        in_ch = input_dim
        for _ in range(cnn_layers):
            cnn_blocks += [
                nn.Conv1d(in_ch, cnn_channels, kernel_size=kernel_size, padding=kernel_size // 2),
                nn.BatchNorm1d(cnn_channels),
                nn.ReLU(),
                nn.Dropout(dropout),
            ]
            in_ch = cnn_channels
        self.cnn = nn.Sequential(*cnn_blocks)

        # ── LSTM 时序建模 ─────────────────────────────────────
        self.lstm = nn.LSTM(
            input_size  = cnn_channels,
            hidden_size = lstm_hidden,
            num_layers  = 1,
            batch_first = True,
        )
        self.dropout = nn.Dropout(dropout)
        self.head    = MultiTaskHead(lstm_hidden, n_classes, dropout)

    def forward(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        # x: (N, T, D) → Conv1d 需要 (N, D, T)
        c      = self.cnn(x.permute(0, 2, 1))     # (N, cnn_channels, T)
        c      = c.permute(0, 2, 1)               # (N, T, cnn_channels)
        out, _ = self.lstm(c)                     # (N, T, lstm_hidden)
        h      = self.dropout(out[:, -1, :])
        return self.head(h)

    def count_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


# =============================================================================
# 工厂函数
# =============================================================================

_MODEL_REGISTRY = {
    "lstm":     LSTMModel,
    "gru":      GRUModel,
    "cnn_lstm": CNNLSTMModel,
}


def build_baseline_model(cfg: dict) -> nn.Module:
    """
    根据 cfg["model"]["model_type"] 实例化对应的基线模型。
    所有模型共用 cfg["model"] 中的通用字段：
      - seq_len, dropout, n_classes
    LSTM/GRU 额外读取：baseline_hidden, baseline_num_layers
    CNN-LSTM  额外读取：baseline_cnn_channels, baseline_cnn_layers,
                        baseline_kernel_size, baseline_lstm_hidden
    """
    mc         = cfg["model"]
    model_type = mc["model_type"].lower()

    if model_type not in _MODEL_REGISTRY:
        raise ValueError(
            f"未知 model_type='{model_type}'，"
            f"可选值: {list(_MODEL_REGISTRY.keys())}"
        )

    ModelClass = _MODEL_REGISTRY[model_type]

    if model_type in ("lstm", "gru"):
        return ModelClass(
            seq_len    = mc.get("seq_len",             5),
            input_dim  = mc.get("input_dim",           17),
            hidden     = mc.get("baseline_hidden",     128),
            num_layers = mc.get("baseline_num_layers", 2),
            dropout    = mc.get("dropout",             0.2),
            n_classes  = mc.get("n_classes",           3),
        )
    else:  # cnn_lstm
        return ModelClass(
            seq_len      = mc.get("seq_len",                  5),
            input_dim    = mc.get("input_dim",                17),
            cnn_channels = mc.get("baseline_cnn_channels",   64),
            cnn_layers   = mc.get("baseline_cnn_layers",     2),
            kernel_size  = mc.get("baseline_kernel_size",    3),
            lstm_hidden  = mc.get("baseline_lstm_hidden",    128),
            dropout      = mc.get("dropout",                 0.2),
            n_classes    = mc.get("n_classes",               3),
        )

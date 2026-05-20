# src/models/mtrp_model.py
"""
MT-RP 风险预测模型（移除能力预测分支）
Multimodal Transformer for Risk Prediction

架构：
  多模态特征编码层  → 跨模态融合层  → 风险预测头
  ├─ BehaviorEncoder (8d  → 128d)
  ├─ EyeEncoder     (5d  → 128d)
  ├─ PhysioEncoder  (4d  → 128d)
  └─ EnvEncoder     (1d  → 128d)
           ↓  concat (4T×128)
     CrossModal SelfAttn → GlobalAvgPool → h_global (128)
           ↓
                      RiskBranch
                    (Tanh/Softmax)

输入  X : (B, T, 18)   — 18 = 8行为 + 5眼动 + 4生理 + 1F_S
输出  {
        "risk_reg" : (B, 1)   ∈ [-1,1]  风险度回归
        "risk_cls" : (B, 3)              风险等级 logits（三分类）
      }

严格实现指定的5种消融实验：none / no_cross_attn / single_modal / early_fusion / late_fusion
"""

import torch
import torch.nn as nn
from typing import Dict, Tuple

# =============================================================================
# 基础模块（无修改）
# =============================================================================
class LearnablePositionalEncoding(nn.Module):
    def __init__(self, seq_len: int, d_model: int, dropout: float = 0.1):
        super().__init__()
        self.pe = nn.Embedding(seq_len, d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        T = x.size(1)
        pos = torch.arange(T, device=x.device)
        x = x + self.pe(pos).unsqueeze(0)
        return self.dropout(x)

class ModalityEncoder(nn.Module):
    def __init__(
        self, in_dim: int, d_model: int = 32, nhead: int = 8, num_layers: int = 2,
        ffn_dim: int = 128, seq_len: int = 5, dropout: float = 0.25
    ):
        super().__init__()
        self.proj = nn.Linear(in_dim, d_model)
        self.pe = LearnablePositionalEncoding(seq_len, d_model, dropout)
        layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=nhead, dim_feedforward=ffn_dim,
            dropout=dropout, batch_first=True
        )
        self.encoder = nn.TransformerEncoder(layer, num_layers=num_layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.proj(x)
        x = self.pe(x)
        return self.encoder(x)

# =============================================================================
# 🔥 核心修复：统一RiskBranch结构，永不删除层！解决权重报错
# =============================================================================
class RiskBranch(nn.Module):
    def __init__(self, d_model=32, n_classes=3, dropout=0.25):
        super().__init__()
        # 固定保留所有层，结构完全统一！
        self.field_embed = nn.Linear(1, d_model)
        self.norm = nn.LayerNorm(d_model)
        self.fc = nn.Sequential(
            nn.Linear(d_model, d_model), nn.ReLU(), nn.Dropout(dropout)
        )
        self.reg_head = nn.Sequential(nn.Linear(d_model, 1), nn.Tanh())
        self.cls_head = nn.Linear(d_model, n_classes)

    def forward(self, h_global, f_s=None, use_simple_forward=False):
        # 极简前向（no_cross_attn专用）：不使用额外层，保证结构一致
        if use_simple_forward:
            return self.reg_head(h_global), self.cls_head(h_global)
        
        # 标准前向
        if f_s.dim() == 1:
            f_s = f_s.unsqueeze(-1)
        f_s_emb = self.field_embed(f_s)
        h = self.norm(h_global + f_s_emb)
        h = self.fc(h)
        return self.reg_head(h), self.cls_head(h)

# =============================================================================
# MT-RP 主模型（结构统一，消融仅控制前向逻辑）
# =============================================================================
class mtrp(nn.Module):
    MODAL_SLICES = {
        "behavior": slice(0, 8), "eye": slice(8, 13),
        "physio": slice(13, 17), "env": slice(17, 18)
    }

    def __init__(
        self, seq_len=5, d_model=32, nhead=8, num_layers=2, ffn_dim=128,
        dropout=0.25, n_classes=3, ablation="none"
    ):
        super().__init__()
        self.ablation = ablation.lower()

        # 固定所有编码器，不动态删除
        self.behavior_enc = ModalityEncoder(8, d_model, nhead, num_layers, ffn_dim, seq_len, dropout)
        self.eye_enc = ModalityEncoder(5, d_model, nhead, num_layers, ffn_dim, seq_len, dropout)
        self.physio_enc = ModalityEncoder(4, d_model, nhead, num_layers, ffn_dim, seq_len, dropout)
        self.env_enc = ModalityEncoder(1, d_model, nhead, num_layers, ffn_dim, seq_len, dropout)
        self.early_fusion_enc = ModalityEncoder(18, d_model, nhead, num_layers, ffn_dim, seq_len, dropout)
        self.cross_attn = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(d_model, nhead, ffn_dim, dropout, batch_first=True), 1
        )
        # 统一预测头，无结构变化
        self.risk_branch = RiskBranch(d_model, n_classes, dropout)
        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None: nn.init.zeros_(m.bias)

    # =========================================================================
    # 消融：仅修改前向逻辑，不修改模型层结构
    # =========================================================================
    def forward(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        B, T, _ = x.shape
        f_s = x[:, -1, 17]

        # 1. 完整模型
        if self.ablation == "none":
            enc_beh = self.behavior_enc(x[:, :, self.MODAL_SLICES["behavior"]])
            enc_eye = self.eye_enc(x[:, :, self.MODAL_SLICES["eye"]])
            enc_phy = self.physio_enc(x[:, :, self.MODAL_SLICES["physio"]])
            enc_env = self.env_enc(x[:, :, self.MODAL_SLICES["env"]])
            fused = self.cross_attn(torch.cat([enc_beh, enc_eye, enc_phy, enc_env], dim=1))
            h_global = fused.mean(dim=1)
            risk_reg, risk_cls = self.risk_branch(h_global, f_s)

        # 2. 移除跨注意力 + 极简前向（修复权重问题）
        elif self.ablation == "no_cross_attn":
            enc_beh = self.behavior_enc(x[:, :, self.MODAL_SLICES["behavior"]])
            enc_eye = self.eye_enc(x[:, :, self.MODAL_SLICES["eye"]])
            enc_phy = self.physio_enc(x[:, :, self.MODAL_SLICES["physio"]])
            enc_env = self.env_enc(x[:, :, self.MODAL_SLICES["env"]])
            h_global = torch.cat([enc_beh, enc_eye, enc_phy, enc_env], dim=1).mean(dim=1)
            # 关键：使用统一头，仅开启极简前向
            risk_reg, risk_cls = self.risk_branch(h_global, use_simple_forward=True)

        # 3. 仅行为单模态
        elif self.ablation == "single_modal":
            enc_beh = self.behavior_enc(x[:, :, self.MODAL_SLICES["behavior"]])
            h_global = enc_beh.mean(dim=1)
            risk_reg, risk_cls = self.risk_branch(h_global, f_s)

        # 4. 早期融合
        elif self.ablation == "early_fusion":
            fused_enc = self.early_fusion_enc(x)
            h_global = fused_enc.mean(dim=1)
            risk_reg, risk_cls = self.risk_branch(h_global, f_s)

        # 5. 晚期融合
        elif self.ablation == "late_fusion":
            encs = [
                self.behavior_enc(x[:, :, self.MODAL_SLICES["behavior"]]),
                self.eye_enc(x[:, :, self.MODAL_SLICES["eye"]]),
                self.physio_enc(x[:, :, self.MODAL_SLICES["physio"]]),
                self.env_enc(x[:, :, self.MODAL_SLICES["env"]])
            ]
            reg_list, cls_list = [], []
            for enc in encs:
                r_reg, r_cls = self.risk_branch(enc.mean(dim=1), f_s)
                reg_list.append(r_reg)
                cls_list.append(r_cls)
            risk_reg = torch.stack(reg_list).mean(dim=0)
            risk_cls = torch.stack(cls_list).mean(dim=0)

        return {"risk_reg": risk_reg, "risk_cls": risk_cls}

# =============================================================================
# 模型构建
# =============================================================================
def build_model(cfg: dict) -> mtrp:
    m = cfg["model"]
    return mtrp(
        seq_len=m.get("seq_len",5), d_model=m.get("d_model",32),
        nhead=m.get("nhead",8), num_layers=m.get("num_layers",2),
        ffn_dim=m.get("ffn_dim",128), dropout=m.get("dropout",0.25),
        n_classes=3, ablation=m.get("ablation","none")
    )
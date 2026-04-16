# /root/autodl-tmp/DynamicCapRisk/src/models/mtrp_model.py
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
"""

import torch
import torch.nn as nn
from typing import Dict, Tuple


# =============================================================================
# 工具模块
# =============================================================================

class LearnablePositionalEncoding(nn.Module):
    """可学习位置编码"""
    def __init__(self, seq_len: int, d_model: int, dropout: float = 0.1):
        super().__init__()
        self.pe = nn.Embedding(seq_len, d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        T = x.size(1)
        pos = torch.arange(T, device=x.device)
        x = x + self.pe(pos).unsqueeze(0)
        return self.dropout(x)


class ModalityEmbedding(nn.Module):
    """模态特征投影+位置编码"""
    def __init__(self, in_dim: int, d_model: int, seq_len: int, dropout: float = 0.1):
        super().__init__()
        self.proj = nn.Linear(in_dim, d_model)
        self.pe = LearnablePositionalEncoding(seq_len, d_model, dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.pe(self.proj(x))


# =============================================================================
# 单模态 Transformer 编码器
# =============================================================================

class ModalityEncoder(nn.Module):
    def __init__(
        self,
        in_dim:    int,
        d_model:   int   = 128,
        nhead:     int   = 8,
        num_layers:int   = 4,
        ffn_dim:   int   = 512,
        seq_len:   int   = 5,
        dropout:   float = 0.2,
    ):
        super().__init__()
        self.embed = ModalityEmbedding(in_dim, d_model, seq_len, dropout)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=nhead, dim_feedforward=ffn_dim,
            dropout=dropout, batch_first=True, norm_first=False
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.encoder(self.embed(x))


# =============================================================================
# 跨模态融合层
# =============================================================================

class CrossModalFusion(nn.Module):
    def __init__(self, d_model: int = 128, nhead: int = 8, ffn_dim: int = 512, dropout: float = 0.2):
        super().__init__()
        fusion_layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=nhead, dim_feedforward=ffn_dim,
            dropout=dropout, batch_first=True
        )
        self.fusion = nn.TransformerEncoder(fusion_layer, num_layers=1)

    def forward(self, *enc_list: torch.Tensor) -> torch.Tensor:
        x_cat = torch.cat(enc_list, dim=1)
        x_fuse = self.fusion(x_cat)
        return x_fuse.mean(dim=1)


# =============================================================================
# 风险预测分支（移除能力依赖）
# =============================================================================

class RiskBranch(nn.Module):
    def __init__(
        self,
        d_model:        int   = 128,
        nhead:          int   = 8,
        n_classes:      int   = 3,
        dropout:        float = 0.2,
    ):
        super().__init__()

        # 风险场强嵌入
        self.field_embed = nn.Linear(1, d_model)
        self.norm = nn.LayerNorm(d_model)

        # 特征融合
        self.fc = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.ReLU(),
            nn.Dropout(dropout),
        )

        # 输出层
        self.dropout = nn.Dropout(dropout)
        self.reg_head = nn.Sequential(nn.Linear(d_model, 1), nn.Tanh())
        self.cls_head = nn.Linear(d_model, n_classes)

    def forward(self, h_global: torch.Tensor, f_s: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        if f_s.dim() == 1:
            f_s = f_s.unsqueeze(-1)

        # 全局特征 + 环境场强融合
        f_s_emb = self.field_embed(f_s)
        h_risk = self.norm(h_global + f_s_emb)
        h_risk = self.fc(h_risk)
        h_risk = self.dropout(h_risk)

        # 风险输出
        risk_reg = self.reg_head(h_risk)
        risk_cls = self.cls_head(h_risk)
        return risk_reg, risk_cls


# =============================================================================
# MT-RP 主模型（纯风险预测）
# =============================================================================

class mtrp(nn.Module):
    MODAL_SLICES = {
        "behavior": slice(0, 8),
        "eye":      slice(8, 13),
        "physio":   slice(13, 17),
        "env":      slice(17, 18),
    }
    MODAL_DIMS = [("behavior", 8), ("eye", 5), ("physio", 4), ("env", 1)]

    def __init__(
        self,
        seq_len:    int   = 5,
        d_model:    int   = 128,
        nhead:      int   = 8,
        num_layers: int   = 4,
        ffn_dim:    int   = 512,
        dropout:    float = 0.2,
        n_classes:  int   = 3,
        ablation:   str   = "none",
    ):
        super().__init__()
        self.seq_len = seq_len
        self.ablation = ablation.lower()
        enc_kwargs = dict(d_model=d_model, nhead=nhead, num_layers=num_layers,
                          ffn_dim=ffn_dim, seq_len=seq_len, dropout=dropout)

        # 标准模型：四路编码 + 融合 + 风险分支
        self.behavior_enc = ModalityEncoder(in_dim=8, **enc_kwargs)
        self.eye_enc = ModalityEncoder(in_dim=5, **enc_kwargs)
        self.physio_enc = ModalityEncoder(in_dim=4, **enc_kwargs)
        self.env_enc = ModalityEncoder(in_dim=1, **enc_kwargs)
        self.fusion = CrossModalFusion(d_model, nhead, ffn_dim, dropout)
        self.risk_branch = RiskBranch(d_model, nhead, n_classes, dropout)

        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.LayerNorm):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Embedding):
                nn.init.normal_(m.weight, std=0.02)

    def forward(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        f_s = x[:, -1, 17]
        x_beh = x[:, :, self.MODAL_SLICES["behavior"]]
        x_eye = x[:, :, self.MODAL_SLICES["eye"]]
        x_phy = x[:, :, self.MODAL_SLICES["physio"]]
        x_env = x[:, :, self.MODAL_SLICES["env"]]

        enc_beh = self.behavior_enc(x_beh)
        enc_eye = self.eye_enc(x_eye)
        enc_phy = self.physio_enc(x_phy)
        enc_env = self.env_enc(x_env)

        h_global = self.fusion(enc_beh, enc_eye, enc_phy, enc_env)
        risk_reg, risk_cls = self.risk_branch(h_global, f_s)

        return {
            "risk_reg": risk_reg,
            "risk_cls": risk_cls,
            "h_global": h_global,
        }

    def count_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


# =============================================================================
# 模型构建函数
# =============================================================================

def build_model(cfg: dict) -> "mtrp":
    m = cfg.get("model", cfg)
    model = mtrp(
        seq_len=m.get("seq_len", 5), d_model=m.get("d_model", 128),
        nhead=m.get("nhead", 8), num_layers=m.get("num_layers", 4),
        ffn_dim=m.get("ffn_dim", 512), dropout=m.get("dropout", 0.2),
        n_classes=m.get("n_classes", 3), ablation=m.get("ablation", "none"),
    )
    return model
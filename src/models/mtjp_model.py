# D:\Local\DynamicCapRisk\src\4_prediction\mtjp_model.py

"""
MT-JP 多模态Transformer联合预测模型
Multimodal Transformer for Joint Prediction (MT-JP)

架构（论文 §5.2.1）：
  多模态特征编码层  → 跨模态融合层  → 双分支预测头
  ├─ BehaviorEncoder (8d  → 128d)
  ├─ EyeEncoder     (3d  → 128d)
  ├─ PhysioEncoder  (5d  → 128d)
  └─ EnvEncoder     (1d  → 128d)
           ↓  concat (4T×128)
     CrossModal SelfAttn → GlobalAvgPool → h_global (128)
           ↓
  ┌────────┴────────┐
  AbilityBranch   RiskBranch
  (FFN + Sigmoid) (CrossAttn + Tanh/Softmax)

输入  X : (B, T, 17)   — 17 = 8行为 + 3眼动 + 5生理 + 1F_S
         其中 F_S 为最后一维，在 RiskBranch 中额外使用
输出  {
        "ability"  : (B, 1)   ∈ [0,1]   归一化能力值
        "risk_reg" : (B, 1)   ∈ [-1,1]  风险度回归
        "risk_cls" : (B, 3)              风险等级 logits（三分类）
      }
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Tuple


# =============================================================================
# 工具模块
# =============================================================================

class LearnablePositionalEncoding(nn.Module):
    """可学习位置编码（论文 §5.2.1 采用可学习方式）。"""
    def __init__(self, seq_len: int, d_model: int, dropout: float = 0.1):
        super().__init__()
        self.pe      = nn.Embedding(seq_len, d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x : (B, T, d_model)
        """
        T   = x.size(1)
        pos = torch.arange(T, device=x.device)          # (T,)
        x   = x + self.pe(pos).unsqueeze(0)              # broadcast (1,T,d)
        return self.dropout(x)


class ModalityEmbedding(nn.Module):
    """将单一模态原始特征线性投影到 d_model 维，并加位置编码。"""
    def __init__(self, in_dim: int, d_model: int, seq_len: int, dropout: float = 0.1):
        super().__init__()
        self.proj = nn.Linear(in_dim, d_model)
        self.pe   = LearnablePositionalEncoding(seq_len, d_model, dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x : (B, T, in_dim) → (B, T, d_model)"""
        return self.pe(self.proj(x))


# =============================================================================
# 单模态 Transformer 编码器
# =============================================================================

class ModalityEncoder(nn.Module):
    """
    单模态 Transformer 编码器（论文 §5.2.1 多模态特征编码层）。
    结构：ModalityEmbedding + N × TransformerEncoderLayer
    """
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
        self.embed   = ModalityEmbedding(in_dim, d_model, seq_len, dropout)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model         = d_model,
            nhead           = nhead,
            dim_feedforward = ffn_dim,
            dropout         = dropout,
            batch_first     = True,   # (B, T, d)
            norm_first      = False,  # Post-LN，与论文 §5.1.2 公式一致
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x : (B, T, in_dim) → (B, T, d_model)"""
        return self.encoder(self.embed(x))


# =============================================================================
# 跨模态融合层
# =============================================================================

class CrossModalFusion(nn.Module):
    """
    跨模态融合层（论文 §5.2.1 跨模态融合层）：
      1. 拼接四个模态编码 → (B, 4T, 128)
      2. 跨模态自注意力（MultiHead Self-Attention）
      3. 时间维度全局平均池化 → h_global (B, 128)
    """
    def __init__(
        self,
        d_model:  int   = 128,
        nhead:    int   = 8,
        ffn_dim:  int   = 512,
        dropout:  float = 0.2,
    ):
        super().__init__()
        fusion_layer = nn.TransformerEncoderLayer(
            d_model         = d_model,
            nhead           = nhead,
            dim_feedforward = ffn_dim,
            dropout         = dropout,
            batch_first     = True,
        )
        self.fusion = nn.TransformerEncoder(fusion_layer, num_layers=1)

    def forward(self, *enc_list: torch.Tensor) -> torch.Tensor:
        """
        enc_list : M 个 (B, T, d_model) 编码特征
        返回 h_global : (B, d_model)
        """
        x_cat  = torch.cat(enc_list, dim=1)    # (B, M*T, d_model)
        x_fuse = self.fusion(x_cat)             # (B, M*T, d_model)
        return x_fuse.mean(dim=1)               # (B, d_model)  全局均值池化


# =============================================================================
# 能力预测分支
# =============================================================================

class AbilityBranch(nn.Module):
    """
    能力预测分支（论文 §5.2.1 公式 5.10）：
      h_global (128) → Linear(128,64) → ReLU → Linear(64,1) → Sigmoid
    输出 ŷ_ability ∈ [0,1]
    """
    def __init__(self, d_model: int = 128, hidden: int = 64, dropout: float = 0.2):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(d_model, hidden),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, 1),
            nn.Sigmoid(),
        )

    def forward(self, h: torch.Tensor) -> torch.Tensor:
        """h : (B, d_model) → (B, 1)"""
        return self.net(h)


# =============================================================================
# 风险预测分支（含交叉注意力）
# =============================================================================

class CrossAttention(nn.Module):
    """
    缩放点积交叉注意力（论文 §5.1.3 公式 5.8）：
      CrossAttn(Q, K, V) = softmax(QW_Q (KW_K)^T / √d_k) (VW_V)
    此处 Q/K/V 均为 (B, 1, d) 的长度-1 序列（将向量视为单步序列）。
    """
    def __init__(self, d_model: int = 128, nhead: int = 8, dropout: float = 0.2):
        super().__init__()
        self.attn = nn.MultiheadAttention(
            embed_dim   = d_model,
            num_heads   = nhead,
            dropout     = dropout,
            batch_first = True,
        )

    def forward(
        self,
        query: torch.Tensor,   # (B, d_model)
        key:   torch.Tensor,   # (B, d_model)
        value: torch.Tensor,   # (B, d_model)
    ) -> torch.Tensor:
        """返回 (B, d_model)"""
        q = query.unsqueeze(1)   # (B, 1, d)
        k = key.unsqueeze(1)     # (B, 1, d)
        v = value.unsqueeze(1)   # (B, 1, d)
        out, _ = self.attn(q, k, v)
        return out.squeeze(1)    # (B, d)


class RiskBranch(nn.Module):
    """
    风险预测分支（论文 §5.2.1 公式 5.11）：

      h_ability  = h_global @ W_proj_ability      投影能力表示作为 K/V
      f_s_emb    = F_S * w_field                  风险场强嵌入
      h_risk_env = LayerNorm(h_global + f_s_emb)  融合场强的查询向量
      h_risk     = CrossAttn(Q=h_risk_env, K/V=h_ability)

      ŷ_risk_reg = Tanh(h_risk @ w_reg)           ∈ [-1,1]
      ŷ_risk_cls = h_risk @ W_cls                 (B,3) logits
    """
    def __init__(
        self,
        d_model:   int   = 128,
        proj_dim:  int   = 128,
        nhead:     int   = 8,
        n_classes: int   = 3,
        dropout:   float = 0.2,
    ):
        super().__init__()
        # 能力投影（作为 CrossAttn 的 K/V 来源）
        self.ability_proj = nn.Linear(d_model, proj_dim)

        # 风险场强嵌入：F_S 标量 → proj_dim 维
        self.field_embed  = nn.Linear(1, proj_dim)

        # 融合 h_global + F_S_emb 后的 LayerNorm（查询向量）
        self.query_norm   = nn.LayerNorm(proj_dim)

        # 交叉注意力
        self.cross_attn   = CrossAttention(proj_dim, nhead, dropout)

        # 输出层
        self.dropout      = nn.Dropout(dropout)
        self.reg_head     = nn.Sequential(
            nn.Linear(proj_dim, 1),
            nn.Tanh(),
        )
        self.cls_head     = nn.Linear(proj_dim, n_classes)

    def forward(
        self,
        h_global: torch.Tensor,   # (B, d_model)
        f_s:      torch.Tensor,   # (B,) 或 (B,1)  风险场强
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        返回：
          risk_reg : (B, 1)   Tanh 输出
          risk_cls : (B, 3)   logits（未经 Softmax，供 CrossEntropy 使用）
        """
        if f_s.dim() == 1:
            f_s = f_s.unsqueeze(-1)   # (B, 1)

        # 能力向量（K/V）
        h_ability = self.ability_proj(h_global)        # (B, proj_dim)

        # 风险查询向量（Q）= h_global 投影 + F_S 嵌入，残差后归一化
        f_s_emb   = self.field_embed(f_s)              # (B, proj_dim)
        h_risk_env = self.query_norm(
            self.ability_proj(h_global) + f_s_emb      # proj 后加场强嵌入
        )                                              # (B, proj_dim)

        # 交叉注意力：风险查询 感知 能力状态
        h_risk = self.cross_attn(h_risk_env, h_ability, h_ability)  # (B, proj_dim)
        h_risk = self.dropout(h_risk)

        risk_reg = self.reg_head(h_risk)               # (B, 1)
        risk_cls = self.cls_head(h_risk)               # (B, 3)
        return risk_reg, risk_cls


# =============================================================================
# MT-JP 主模型
# =============================================================================

class MTJP(nn.Module):
    """
    Multimodal Transformer for Joint Prediction (MT-JP)

    输入拆分（论文 §5.1 特征维度 D=17）：
      X[:, :, 0:8]   → behavior  (8d)
      X[:, :, 8:11]  → eye       (3d)
      X[:, :, 11:16] → physio    (5d)
      X[:, :, 16:17] → env F_S   (1d)

    参数（表5.3）：
      d_model=128, nhead=8, num_layers=4, ffn_dim=512, dropout=0.2
    """

    # 模态切分索引（对应 dataset.py feat_cols_17 顺序）
    MODAL_SLICES = {
        "behavior": slice(0, 8),
        "eye":      slice(8, 11),
        "physio":   slice(11, 16),
        "env":      slice(16, 17),
    }

    def __init__(
        self,
        seq_len:    int   = 5,     # 历史步数 T（15s）
        d_model:    int   = 128,
        nhead:      int   = 8,
        num_layers: int   = 4,
        ffn_dim:    int   = 512,
        dropout:    float = 0.2,
        n_classes:  int   = 3,
    ):
        super().__init__()
        self.seq_len = seq_len

        enc_kwargs = dict(
            d_model    = d_model,
            nhead      = nhead,
            num_layers = num_layers,
            ffn_dim    = ffn_dim,
            seq_len    = seq_len,
            dropout    = dropout,
        )

        # 四个独立的模态编码器
        self.behavior_enc = ModalityEncoder(in_dim=8, **enc_kwargs)
        self.eye_enc      = ModalityEncoder(in_dim=3, **enc_kwargs)
        self.physio_enc   = ModalityEncoder(in_dim=5, **enc_kwargs)
        self.env_enc      = ModalityEncoder(in_dim=1, **enc_kwargs)

        # 跨模态融合层
        self.fusion = CrossModalFusion(d_model, nhead, ffn_dim, dropout)

        # 双分支预测头
        self.ability_branch = AbilityBranch(d_model, d_model // 2, dropout)
        self.risk_branch    = RiskBranch(d_model, d_model, nhead, n_classes, dropout)

        # 权重初始化
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
        """
        x : (B, T, 17)

        Returns dict：
          "ability"  : (B, 1)   ∈ [0,1]
          "risk_reg" : (B, 1)   ∈ [-1,1]
          "risk_cls" : (B, 3)   logits
          "h_global" : (B, 128) 融合表示（供一致性损失使用）
        """
        # ── 1. 模态拆分 ─────────────────────────────────────────
        x_beh = x[:, :, self.MODAL_SLICES["behavior"]]   # (B, T, 8)
        x_eye = x[:, :, self.MODAL_SLICES["eye"]]        # (B, T, 3)
        x_phy = x[:, :, self.MODAL_SLICES["physio"]]     # (B, T, 5)
        x_env = x[:, :, self.MODAL_SLICES["env"]]        # (B, T, 1)

        # F_S：取最后一步的场强值用于风险分支（当前预测窗口的场景）
        f_s = x[:, -1, 16]                               # (B,)

        # ── 2. 各模态独立编码 ─────────────────────────────────
        enc_beh = self.behavior_enc(x_beh)   # (B, T, 128)
        enc_eye = self.eye_enc(x_eye)        # (B, T, 128)
        enc_phy = self.physio_enc(x_phy)     # (B, T, 128)
        enc_env = self.env_enc(x_env)        # (B, T, 128)

        # ── 3. 跨模态融合 ─────────────────────────────────────
        h_global = self.fusion(enc_beh, enc_eye, enc_phy, enc_env)  # (B, 128)

        # ── 4. 双分支预测 ──────────────────────────────────────
        ability  = self.ability_branch(h_global)              # (B, 1)
        risk_reg, risk_cls = self.risk_branch(h_global, f_s)  # (B,1), (B,3)

        return {
            "ability":  ability,
            "risk_reg": risk_reg,
            "risk_cls": risk_cls,
            "h_global": h_global,
        }

    def count_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


# =============================================================================
# 工厂函数（从 config dict 构建模型）
# =============================================================================

def build_model(cfg: dict) -> "MTJP":
    """
    从配置字典构建 MT-JP 模型。
    cfg 对应 config/mtjp_model.yaml 的 model 节。
    """
    m = cfg.get("model", cfg)
    model = MTJP(
        seq_len    = m.get("seq_len",    5),
        d_model    = m.get("d_model",    128),
        nhead      = m.get("nhead",      8),
        num_layers = m.get("num_layers", 4),
        ffn_dim    = m.get("ffn_dim",    512),
        dropout    = m.get("dropout",    0.2),
        n_classes  = m.get("n_classes",  3),
    )
    return model


# =============================================================================
# 快速自检
# =============================================================================

if __name__ == "__main__":
    torch.manual_seed(42)
    B, T, D = 4, 5, 17
    x = torch.randn(B, T, D)

    model = MTJP(seq_len=T)
    out   = model(x)

    print(f"参数量: {model.count_parameters():,}")
    print(f"ability  : {out['ability'].shape}  range [{out['ability'].min():.3f}, {out['ability'].max():.3f}]")
    print(f"risk_reg : {out['risk_reg'].shape}  range [{out['risk_reg'].min():.3f}, {out['risk_reg'].max():.3f}]")
    print(f"risk_cls : {out['risk_cls'].shape}")
    print(f"h_global : {out['h_global'].shape}")
    print("✓ 前向传播通过")

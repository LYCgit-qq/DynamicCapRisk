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

消融实验支持（通过 cfg["model"]["ablation"] 控制）：
  none            : 完整 MT-JP（默认）
  no_cross_attn   : 移除交叉注意力，改用线性层
  no_consistency  : 一致性损失置零（仅需在 YAML 中设 lambda_consistency=0.0，无需改此处）
  single_modal    : 仅使用行为特征，跳过融合层
  early_fusion    : 早期融合，17维特征直接输入单一编码器
  late_fusion     : 晚期融合，四路模态各自独立预测后取均值
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
    风险预测分支（论文 §5.2.1 公式 5.11）。

    use_cross_attn=True（默认，完整 MT-JP）：
      h_ability  = h_global @ W_proj_ability
      f_s_emb    = F_S * w_field
      h_risk_env = LayerNorm(h_global + f_s_emb)
      h_risk     = CrossAttn(Q=h_risk_env, K/V=h_ability)

    use_cross_attn=False（消融：移除交叉注意力）：
      交叉注意力替换为线性层，直接将 h_risk_env 投影为 h_risk，
      不再显式建模能力-风险耦合关系。

    输出：
      ŷ_risk_reg ∈ [-1,1]（Tanh）
      ŷ_risk_cls  (B,3)  logits
    """
    def __init__(
        self,
        d_model:        int   = 128,
        proj_dim:       int   = 128,
        nhead:          int   = 8,
        n_classes:      int   = 3,
        dropout:        float = 0.2,
        use_cross_attn: bool  = True,   # 消融开关
    ):
        super().__init__()
        self.use_cross_attn = use_cross_attn

        # 能力投影（作为 CrossAttn 的 K/V 来源）
        self.ability_proj = nn.Linear(d_model, proj_dim)

        # 风险场强嵌入：F_S 标量 → proj_dim 维
        self.field_embed  = nn.Linear(1, proj_dim)

        # 融合 h_global + F_S_emb 后的 LayerNorm（查询向量）
        self.query_norm   = nn.LayerNorm(proj_dim)

        if self.use_cross_attn:
            # ── 完整版：交叉注意力 ──────────────────────────────
            self.cross_attn = CrossAttention(proj_dim, nhead, dropout)
        else:
            # ── 消融版：线性层替代，不建模能力-风险耦合 ──────────
            self.linear_replace = nn.Sequential(
                nn.Linear(proj_dim, proj_dim),
                nn.ReLU(),
                nn.Dropout(dropout),
            )

        # 输出层（两版本共用）
        self.dropout  = nn.Dropout(dropout)
        self.reg_head = nn.Sequential(
            nn.Linear(proj_dim, 1),
            nn.Tanh(),
        )
        self.cls_head = nn.Linear(proj_dim, n_classes)

    def forward(
        self,
        h_global: torch.Tensor,   # (B, d_model)
        f_s:      torch.Tensor,   # (B,) 或 (B,1)
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        if f_s.dim() == 1:
            f_s = f_s.unsqueeze(-1)   # (B, 1)

        h_ability  = self.ability_proj(h_global)                    # (B, proj_dim)
        f_s_emb    = self.field_embed(f_s)                          # (B, proj_dim)
        h_risk_env = self.query_norm(self.ability_proj(h_global) + f_s_emb)

        if self.use_cross_attn:
            # 完整版：交叉注意力显式建模能力-风险耦合
            h_risk = self.cross_attn(h_risk_env, h_ability, h_ability)
        else:
            # 消融版：线性映射，无能力-风险交互
            h_risk = self.linear_replace(h_risk_env)

        h_risk   = self.dropout(h_risk)
        risk_reg = self.reg_head(h_risk)   # (B, 1)
        risk_cls = self.cls_head(h_risk)   # (B, 3)
        return risk_reg, risk_cls


# =============================================================================
# MT-JP 主模型（含消融实验支持）
# =============================================================================

class MTJP(nn.Module):
    """
    Multimodal Transformer for Joint Prediction (MT-JP)

    消融实验通过 ablation 参数控制：
      "none"          → 完整 MT-JP（默认）
      "no_cross_attn" → RiskBranch 中交叉注意力替换为线性层
      "no_consistency"→ 仅需 YAML 中 lambda_consistency=0.0，模型结构不变
      "single_modal"  → 仅 BehaviorEncoder，跳过 CrossModalFusion，直接均值池化
      "early_fusion"  → 17 维特征直接输入单一 Transformer 编码器
      "late_fusion"   → 四路模态各自独立预测后对输出取均值

    参数（表5.3）：
      d_model=128, nhead=8, num_layers=4, ffn_dim=512, dropout=0.2
    """

    MODAL_SLICES = {
        "behavior": slice(0, 8),
        "eye":      slice(8, 11),
        "physio":   slice(11, 16),
        "env":      slice(16, 17),
    }
    # (模态名, 特征维度) 顺序与 MODAL_SLICES 一致
    MODAL_DIMS = [("behavior", 8), ("eye", 3), ("physio", 5), ("env", 1)]

    def __init__(
        self,
        seq_len:    int   = 5,
        d_model:    int   = 128,
        nhead:      int   = 8,
        num_layers: int   = 4,
        ffn_dim:    int   = 512,
        dropout:    float = 0.2,
        n_classes:  int   = 3,
        ablation:   str   = "none",   # ← 消融开关
    ):
        super().__init__()
        self.seq_len  = seq_len
        self.ablation = ablation.lower()

        enc_kwargs = dict(
            d_model    = d_model,
            nhead      = nhead,
            num_layers = num_layers,
            ffn_dim    = ffn_dim,
            seq_len    = seq_len,
            dropout    = dropout,
        )

        # ── 根据消融类型初始化子模块 ──────────────────────────────
        if self.ablation == "early_fusion":
            # 所有模态特征直接拼接（17维）后输入单一编码器
            self.unified_enc    = ModalityEncoder(in_dim=17, **enc_kwargs)
            self.ability_branch = AbilityBranch(d_model, d_model // 2, dropout)
            self.risk_branch    = RiskBranch(d_model, d_model, nhead, n_classes, dropout,
                                             use_cross_attn=True)

        elif self.ablation == "late_fusion":
            # 四路独立编码器，每路各自配备独立预测头
            self.encoders = nn.ModuleList([
                ModalityEncoder(in_dim=dim, **enc_kwargs)
                for _, dim in self.MODAL_DIMS
            ])
            self.ability_branches = nn.ModuleList([
                AbilityBranch(d_model, d_model // 2, dropout) for _ in self.MODAL_DIMS
            ])
            self.risk_branches = nn.ModuleList([
                RiskBranch(d_model, d_model, nhead, n_classes, dropout,
                           use_cross_attn=True)
                for _ in self.MODAL_DIMS
            ])

        elif self.ablation == "single_modal":
            # 仅保留行为模态编码器，跳过 CrossModalFusion
            self.behavior_enc   = ModalityEncoder(in_dim=8, **enc_kwargs)
            self.ability_branch = AbilityBranch(d_model, d_model // 2, dropout)
            self.risk_branch    = RiskBranch(d_model, d_model, nhead, n_classes, dropout,
                                             use_cross_attn=True)

        else:
            # "none" 或 "no_cross_attn" 或 "no_consistency"：保留完整四路编码器 + 融合层
            self.behavior_enc = ModalityEncoder(in_dim=8, **enc_kwargs)
            self.eye_enc      = ModalityEncoder(in_dim=3, **enc_kwargs)
            self.physio_enc   = ModalityEncoder(in_dim=5, **enc_kwargs)
            self.env_enc      = ModalityEncoder(in_dim=1, **enc_kwargs)
            self.fusion       = CrossModalFusion(d_model, nhead, ffn_dim, dropout)
            self.ability_branch = AbilityBranch(d_model, d_model // 2, dropout)
            # no_cross_attn：RiskBranch 内部将交叉注意力替换为线性层
            use_ca = (self.ablation != "no_cross_attn")
            self.risk_branch = RiskBranch(d_model, d_model, nhead, n_classes, dropout,
                                          use_cross_attn=use_ca)

        self._init_weights()

    # -------------------------------------------------------------------------

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

    # -------------------------------------------------------------------------

    def forward(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        x : (B, T, 17)

        Returns dict：
          "ability"  : (B, 1)   ∈ [0,1]
          "risk_reg" : (B, 1)   ∈ [-1,1]
          "risk_cls" : (B, 3)   logits
          "h_global" : (B, 128) 融合表示（供一致性损失使用）
        """
        f_s = x[:, -1, 16]   # (B,) 风险场强，各变体均需使用

        # ── early_fusion ──────────────────────────────────────
        if self.ablation == "early_fusion":
            h_enc    = self.unified_enc(x)          # (B, T, d)
            h_global = h_enc.mean(dim=1)            # (B, d)  均值池化替代融合层
            ability  = self.ability_branch(h_global)
            risk_reg, risk_cls = self.risk_branch(h_global, f_s)

        # ── late_fusion ───────────────────────────────────────
        elif self.ablation == "late_fusion":
            slices = [self.MODAL_SLICES[name] for name, _ in self.MODAL_DIMS]
            ability_preds  = []
            risk_reg_preds = []
            risk_cls_preds = []
            h_global_list  = []

            for enc, ab_branch, rk_branch, sl in zip(
                self.encoders, self.ability_branches, self.risk_branches, slices
            ):
                h = enc(x[:, :, sl]).mean(dim=1)        # (B, d) 各模态独立均值池化
                h_global_list.append(h)
                ability_preds.append(ab_branch(h))
                rr, rc = rk_branch(h, f_s)
                risk_reg_preds.append(rr)
                risk_cls_preds.append(rc)

            # 四路预测结果均值融合
            ability  = torch.stack(ability_preds,  dim=0).mean(dim=0)   # (B,1)
            risk_reg = torch.stack(risk_reg_preds, dim=0).mean(dim=0)   # (B,1)
            risk_cls = torch.stack(risk_cls_preds, dim=0).mean(dim=0)   # (B,3)
            h_global = torch.stack(h_global_list,  dim=0).mean(dim=0)   # (B,d)
            return {
                "ability":  ability,
                "risk_reg": risk_reg,
                "risk_cls": risk_cls,
                "h_global": h_global,
            }

        # ── single_modal ──────────────────────────────────────
        elif self.ablation == "single_modal":
            x_beh    = x[:, :, self.MODAL_SLICES["behavior"]]   # (B, T, 8)
            enc_beh  = self.behavior_enc(x_beh)                 # (B, T, d)
            h_global = enc_beh.mean(dim=1)                      # (B, d) 跳过融合层
            ability  = self.ability_branch(h_global)
            risk_reg, risk_cls = self.risk_branch(h_global, f_s)

        # ── none / no_cross_attn / no_consistency（完整四路） ──
        else:
            x_beh = x[:, :, self.MODAL_SLICES["behavior"]]
            x_eye = x[:, :, self.MODAL_SLICES["eye"]]
            x_phy = x[:, :, self.MODAL_SLICES["physio"]]
            x_env = x[:, :, self.MODAL_SLICES["env"]]

            enc_beh = self.behavior_enc(x_beh)
            enc_eye = self.eye_enc(x_eye)
            enc_phy = self.physio_enc(x_phy)
            enc_env = self.env_enc(x_env)

            h_global = self.fusion(enc_beh, enc_eye, enc_phy, enc_env)
            ability  = self.ability_branch(h_global)
            risk_reg, risk_cls = self.risk_branch(h_global, f_s)

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
    cfg 对应 config/trainer_dl.yaml 的 model 节。
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
        ablation   = m.get("ablation",   "none"),   # ← 新增
    )
    return model


# =============================================================================
# 快速自检
# =============================================================================

if __name__ == "__main__":
    torch.manual_seed(42)
    B, T, D = 4, 5, 17
    x = torch.randn(B, T, D)

    ablation_list = ["none", "no_cross_attn", "single_modal", "early_fusion", "late_fusion"]

    for abl in ablation_list:
        model = MTJP(seq_len=T, ablation=abl)
        out   = model(x)
        print(
            f"[{abl:<16}] params={model.count_parameters():>9,} | "
            f"ability={tuple(out['ability'].shape)} "
            f"risk_reg={tuple(out['risk_reg'].shape)} "
            f"risk_cls={tuple(out['risk_cls'].shape)}"
        )
    print("\n✓ 所有消融变体前向传播通过")

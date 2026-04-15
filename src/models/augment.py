# D:\Local\DynamicCapRisk\src\3_prediction\augment.py

"""
augment.py
MT-JP 联合预测模型 — 时序数据增强模块

支持的增强策略（均针对输入序列 X: (N, T, D)，标签保持不变）：

  1. gaussian_noise    — 高斯噪声注入（对生理/行为信号的传感器噪声建模）
  2. time_warp         — 时间扭曲（模拟驾驶行为的时间弹性变化）
  3. window_slice      — 窗口裁剪后等长插值（随机选取历史片段）
  4. feature_dropout   — 特征随机置零（模拟传感器缺失/遮挡）
  5. magnitude_warp    — 幅值扭曲（对生理/行为幅度的个体差异建模）
  6. mixup             — 跨样本线性插值（适合回归标签，分类标签按比例软化后 argmax）

配置示例（config/dataset.yaml）：

augmentation:
  enabled: true                # 总开关
  only_on_train: true          # 仅对训练集增强
  methods:
    gaussian_noise:
      enabled: true
      std_scale: 0.05          # 噪声标准差 = 特征均值绝对值 × std_scale
      prob: 1.0                # 每个样本被选中增强的概率
    time_warp:
      enabled: true
      sigma: 0.2               # 扭曲幅度（比例）
      prob: 0.5
    window_slice:
      enabled: false
      slice_ratio: 0.9         # 保留的时间步比例
      prob: 0.3
    feature_dropout:
      enabled: true
      drop_prob: 0.1           # 每维特征被置零的概率
      prob: 0.5
    magnitude_warp:
      enabled: true
      sigma: 0.1
      prob: 0.5
    mixup:
      enabled: false
      alpha: 0.4               # Beta(alpha, alpha) 采样
      prob: 0.3

用法（在 dataset.py 中）：
  from augment import Augmentor
  aug = Augmentor(cfg["augmentation"])
  X_aug, ya_aug, yr_aug, yc_aug = aug.apply(X_train, ya_train, yr_train, yc_train)

注：
  - 增强后样本追加到原始训练集后（不替换），最终 N 增大
  - F_S（最后一维，索引 16）为环境标签，不参与噪声/扭曲，仅随样本复制
  - 所有增强在 Z-score 标准化之前执行（对原始特征尺度操作，效果更稳定）
"""

import numpy as np
from typing import Dict, Any, Tuple, Optional


# =============================================================================
# 单项增强函数（均接受/返回 (N, T, D) float32）
# =============================================================================

def _gaussian_noise(
    X: np.ndarray,
    std_scale: float = 0.05,
    rng: Optional[np.random.Generator] = None,
) -> np.ndarray:
    """
    对前 16 维（多模态特征）注入高斯噪声，F_S（第 17 维）不变。
    噪声标准差 = 各维全局均值绝对值 × std_scale，避免对小值特征过度扰动。
    """
    if rng is None:
        rng = np.random.default_rng()
    X_out = X.copy()
    N, T, D = X.shape
    feat_mean = np.abs(X[:, :, :16].reshape(-1, 16).mean(axis=0)) + 1e-6
    sigma     = feat_mean * std_scale   # (16,)
    noise     = rng.normal(0, sigma, size=(N, T, 16)).astype(np.float32)
    X_out[:, :, :16] += noise
    return X_out


def _time_warp(
    X: np.ndarray,
    sigma: float = 0.2,
    rng:   Optional[np.random.Generator] = None,
) -> np.ndarray:
    """
    时间扭曲：对每个样本独立生成平滑的时间轴扭曲，然后重采样到原始步长。
    使用简单的 1-D 分段线性插值（轻量、无外部依赖）。
    """
    if rng is None:
        rng = np.random.default_rng()
    N, T, D = X.shape
    X_out = np.empty_like(X)
    for i in range(N):
        # 生成随机扭曲锚点（在时间轴上加一个平滑扰动）
        t_orig   = np.linspace(0, T - 1, T)
        warp_pts = rng.normal(0, sigma * T, size=max(2, T // 4))
        warp_pts = np.clip(np.cumsum(warp_pts), 0, None)
        if warp_pts[-1] < 1e-6:
            warp_pts[-1] = 1.0
        warp_pts = warp_pts / warp_pts[-1] * (T - 1)
        knots    = np.linspace(0, T - 1, len(warp_pts))
        t_warp   = np.interp(t_orig, knots, warp_pts)
        t_warp   = np.clip(t_warp, 0, T - 1)
        # 对每一维插值
        for d in range(D):
            X_out[i, :, d] = np.interp(t_orig, t_warp, X[i, :, d])
    return X_out.astype(np.float32)


def _window_slice(
    X: np.ndarray,
    slice_ratio: float = 0.9,
    rng: Optional[np.random.Generator] = None,
) -> np.ndarray:
    """
    随机截取 slice_ratio 比例的连续片段，线性插值回原长度 T。
    """
    if rng is None:
        rng = np.random.default_rng()
    N, T, D = X.shape
    slice_len = max(2, int(T * slice_ratio))
    X_out = np.empty_like(X)
    t_out = np.linspace(0, slice_len - 1, T)
    for i in range(N):
        start = rng.integers(0, T - slice_len + 1)
        seg   = X[i, start: start + slice_len, :]   # (slice_len, D)
        t_in  = np.arange(slice_len, dtype=float)
        for d in range(D):
            X_out[i, :, d] = np.interp(t_out, t_in, seg[:, d])
    return X_out.astype(np.float32)


def _feature_dropout(
    X: np.ndarray,
    drop_prob: float = 0.1,
    rng: Optional[np.random.Generator] = None,
) -> np.ndarray:
    """
    对前 16 维特征，按 drop_prob 概率随机将整个维度置零（模拟传感器缺失）。
    F_S 不变。
    """
    if rng is None:
        rng = np.random.default_rng()
    X_out = X.copy()
    N, T, D = X.shape
    mask = rng.random(size=(N, 16)) < drop_prob    # (N, 16) bool
    # 将命中维度置零
    for i in range(N):
        for d in range(16):
            if mask[i, d]:
                X_out[i, :, d] = 0.0
    return X_out


def _magnitude_warp(
    X: np.ndarray,
    sigma: float = 0.1,
    rng:   Optional[np.random.Generator] = None,
) -> np.ndarray:
    """
    幅值扭曲：对每个样本、每个时间步，乘以一个从平滑曲线采样的缩放因子。
    F_S 不变。
    """
    if rng is None:
        rng = np.random.default_rng()
    N, T, D = X.shape
    X_out = X.copy()
    n_knots = max(2, T // 4)
    t_knots = np.linspace(0, T - 1, n_knots)
    t_full  = np.arange(T, dtype=float)
    for i in range(N):
        # 为前 16 维生成统一的幅值扭曲曲线
        knot_vals = 1.0 + rng.normal(0, sigma, size=n_knots)
        scale     = np.interp(t_full, t_knots, knot_vals)   # (T,)
        X_out[i, :, :16] = (X[i, :, :16] * scale[:, None]).astype(np.float32)
    return X_out


def _mixup(
    X:  np.ndarray,
    ya: np.ndarray,
    yr: np.ndarray,
    yc: np.ndarray,
    alpha: float = 0.4,
    rng:   Optional[np.random.Generator] = None,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Mixup：随机配对样本，按 Beta(alpha,alpha) 权重线性插值 X / ya / yr。
    yc（分类标签）取权重较大一方的标签（hard label）。
    """
    if rng is None:
        rng = np.random.default_rng()
    N = len(ya)
    lam    = rng.beta(alpha, alpha, size=N).astype(np.float32)
    idx    = rng.permutation(N)
    lam_x  = lam[:, None, None]   # broadcast for (N, T, D)
    X_mix  = lam_x * X + (1 - lam_x) * X[idx]
    ya_mix = (lam * ya + (1 - lam) * ya[idx]).astype(np.float32)
    yr_mix = (lam * yr + (1 - lam) * yr[idx]).astype(np.float32)
    yc_mix = np.where(lam >= 0.5, yc, yc[idx]).astype(np.int64)
    return X_mix.astype(np.float32), ya_mix, yr_mix, yc_mix


# =============================================================================
# Augmentor 类（统一接口）
# =============================================================================

class Augmentor:
    """
    数据增强器。

    Parameters
    ----------
    aug_cfg : dict
        来自 yaml 的 augmentation 节点，格式见模块文档。
    seed : int
        随机种子，保证可复现。
    """

    def __init__(self, aug_cfg: Dict[str, Any], seed: int = 42):
        self.cfg  = aug_cfg or {}
        self.rng  = np.random.default_rng(seed)
        self.enabled      = bool(self.cfg.get("enabled", False))
        self.only_on_train = bool(self.cfg.get("only_on_train", True))
        self._methods = self.cfg.get("methods", {})

    # ------------------------------------------------------------------
    def _method_cfg(self, name: str) -> Optional[Dict]:
        """返回已启用方法的配置，未启用返回 None。"""
        mc = self._methods.get(name, {})
        if mc.get("enabled", False):
            return mc
        return None

    def _sample_mask(self, N: int, prob: float) -> np.ndarray:
        """按概率 prob 随机选出要增强的样本下标。"""
        return np.where(self.rng.random(N) < prob)[0]

    # ------------------------------------------------------------------
    def apply(
        self,
        X:  np.ndarray,
        ya: np.ndarray,
        yr: np.ndarray,
        yc: np.ndarray,
        split: str = "train",
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """
        对输入样本执行所有已启用的增强策略，将增强样本追加到原始集合后返回。

        Parameters
        ----------
        X   : (N, T, D) float32
        ya  : (N,) float32   能力标签
        yr  : (N,) float32   风险回归标签
        yc  : (N,) int64     风险分类标签
        split : str          当前 split 名称，only_on_train=True 时非 train split 直接返回

        Returns
        -------
        X_out, ya_out, yr_out, yc_out — 增强后（原始 + 新增）样本
        """
        if not self.enabled:
            return X, ya, yr, yc
        if self.only_on_train and split != "train":
            return X, ya, yr, yc

        N = len(ya)
        X_aug_list  = []
        ya_aug_list = []
        yr_aug_list = []
        yc_aug_list = []

        # 1. Gaussian Noise
        mc = self._method_cfg("gaussian_noise")
        if mc:
            idx = self._sample_mask(N, mc.get("prob", 1.0))
            if len(idx):
                Xa = _gaussian_noise(X[idx], mc.get("std_scale", 0.05), self.rng)
                X_aug_list.append(Xa);  ya_aug_list.append(ya[idx])
                yr_aug_list.append(yr[idx]); yc_aug_list.append(yc[idx])
                print(f"    [augment] gaussian_noise  : +{len(idx)} 样本")

        # 2. Time Warp
        mc = self._method_cfg("time_warp")
        if mc:
            idx = self._sample_mask(N, mc.get("prob", 0.5))
            if len(idx):
                Xa = _time_warp(X[idx], mc.get("sigma", 0.2), self.rng)
                X_aug_list.append(Xa);  ya_aug_list.append(ya[idx])
                yr_aug_list.append(yr[idx]); yc_aug_list.append(yc[idx])
                print(f"    [augment] time_warp        : +{len(idx)} 样本")

        # 3. Window Slice
        mc = self._method_cfg("window_slice")
        if mc:
            idx = self._sample_mask(N, mc.get("prob", 0.3))
            if len(idx):
                Xa = _window_slice(X[idx], mc.get("slice_ratio", 0.9), self.rng)
                X_aug_list.append(Xa);  ya_aug_list.append(ya[idx])
                yr_aug_list.append(yr[idx]); yc_aug_list.append(yc[idx])
                print(f"    [augment] window_slice     : +{len(idx)} 样本")

        # 4. Feature Dropout
        mc = self._method_cfg("feature_dropout")
        if mc:
            idx = self._sample_mask(N, mc.get("prob", 0.5))
            if len(idx):
                Xa = _feature_dropout(X[idx], mc.get("drop_prob", 0.1), self.rng)
                X_aug_list.append(Xa);  ya_aug_list.append(ya[idx])
                yr_aug_list.append(yr[idx]); yc_aug_list.append(yc[idx])
                print(f"    [augment] feature_dropout  : +{len(idx)} 样本")

        # 5. Magnitude Warp
        mc = self._method_cfg("magnitude_warp")
        if mc:
            idx = self._sample_mask(N, mc.get("prob", 0.5))
            if len(idx):
                Xa = _magnitude_warp(X[idx], mc.get("sigma", 0.1), self.rng)
                X_aug_list.append(Xa);  ya_aug_list.append(ya[idx])
                yr_aug_list.append(yr[idx]); yc_aug_list.append(yc[idx])
                print(f"    [augment] magnitude_warp   : +{len(idx)} 样本")

        # 6. Mixup（对整个训练集执行，不按 prob 子采样——内部随机配对）
        mc = self._method_cfg("mixup")
        if mc:
            idx = self._sample_mask(N, mc.get("prob", 0.3))
            if len(idx):
                Xa, yam, yrm, ycm = _mixup(
                    X[idx], ya[idx], yr[idx], yc[idx],
                    mc.get("alpha", 0.4), self.rng,
                )
                X_aug_list.append(Xa);  ya_aug_list.append(yam)
                yr_aug_list.append(yrm); yc_aug_list.append(ycm)
                print(f"    [augment] mixup            : +{len(idx)} 样本")

        if not X_aug_list:
            return X, ya, yr, yc

        X_out  = np.concatenate([X]  + X_aug_list,  axis=0)
        ya_out = np.concatenate([ya] + ya_aug_list,  axis=0)
        yr_out = np.concatenate([yr] + yr_aug_list,  axis=0)
        yc_out = np.concatenate([yc] + yc_aug_list,  axis=0)

        n_new = len(ya_out) - N
        print(f"    [augment] 训练集扩充: {N} → {len(ya_out)} (+{n_new}, "
              f"+{n_new/N*100:.1f}%)")
        return X_out, ya_out, yr_out, yc_out

"""
build_mtjp_dataset.py
MT-JP 联合预测模型数据集构建模块

用法：
  python dataset.py                              # 使用默认 config/mtjp_dataset.yaml
  python dataset.py -c path/to/mtjp_dataset.yaml      # 指定配置文件
  python dataset.py --seq_len 5 --seed 42        # 命令行参数覆盖 yaml
  python dataset.py --no_augment                 # 禁用数据增强（覆盖 yaml）
"""

import os
import pickle
import argparse
import warnings
import yaml
import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional

warnings.filterwarnings("ignore")


# 默认配置文件路径（与脚本同级的 config/ 目录）
_DEFAULT_CONFIG_PATH = os.path.join(
    os.path.dirname(os.path.abspath(__file__)),
    "..",
    "..",
    "config",
    "mtjp_dataset.yaml",
)


def load_config(path: Optional[str] = None) -> dict:
    """
    加载配置：
      1. 优先使用指定的 path
      2. 未指定则使用默认路径 config/mtjp_dataset.yaml
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
# 1. 数据加载
# =============================================================================


def load_raw(pkl_path: str, n_samples: Optional[int] = None) -> Tuple[List, List, List]:
    """加载原始多模态信号，返回 act / eye / phy 列表。"""
    if not os.path.exists(pkl_path):
        raise FileNotFoundError(f"原始数据不存在: {pkl_path}")
    with open(pkl_path, "rb") as f:
        data = pickle.load(f)
    act = data.get("act", [])
    eye = data.get("eye", [])
    phy = data.get("phy", [])
    print(f"  原始数据加载：act={len(act)} / eye={len(eye)} / phy={len(phy)} 个样本")

    # 仅当传入 n_samples 时才校验
    if n_samples is not None and len(act) != n_samples:
        print(f"  ⚠️  样本数 {len(act)} 与配置 n_samples={n_samples} 不一致，以实际为准")
    return act, eye, phy


def load_capability(pkl_path: str) -> dict:
    """
    加载能力波动模块输出，提取：
      sample_fluctuations : List[np.ndarray]  各样本逐窗口 A_fl（已校准波动量）
      sample_field        : List[np.ndarray]  各样本逐窗口场景标签 (int)
      sample_window_counts: List[int]         各样本窗口数
    同时根据 A_fl 推导 Ã_d（归一化能力），供标签使用。
    """
    if not os.path.exists(pkl_path):
        raise FileNotFoundError(f"能力评估结果不存在: {pkl_path}")
    with open(pkl_path, "rb") as f:
        cap = pickle.load(f)
    print(f"  能力评估 pkl 键: {list(cap.keys())}")

    sample_fluctuations = cap.get("sample_fluctuations", [])
    sample_field = cap.get("sample_field", [])
    sample_window_counts = cap.get("sample_window_counts", [])

    # 推导 Ã_d：将 A_fl 线性映射到 [0,1]
    all_afl = np.concatenate([a for a in sample_fluctuations if len(a) > 0])
    afl_min, afl_max = float(all_afl.min()), float(all_afl.max())
    sample_ad_norm = []
    for afl in sample_fluctuations:
        if len(afl) == 0:
            sample_ad_norm.append(np.array([]))
        else:
            adn = np.clip((afl - afl_min) / max(afl_max - afl_min, 1e-8), 0.0, 1.0)
            sample_ad_norm.append(adn.astype(np.float32))

    print(
        f"  能力评估加载：{len(sample_fluctuations)} 个样本，"
        f"Ã_d ∈ [{afl_min:.4f}→0, {afl_max:.4f}→1]"
    )
    return {
        "sample_fluctuations": sample_fluctuations,
        "sample_field": sample_field,
        "sample_window_counts": sample_window_counts,
        "sample_ad_norm": sample_ad_norm,
    }


def load_risk(csv_path: str) -> pd.DataFrame:
    """加载 risk_windows_all.csv，提取每窗口的 R_star / risk_level / F_S / field_label。"""
    if not os.path.exists(csv_path):
        raise FileNotFoundError(
            f"风险评估结果不存在: {csv_path}\n"
            "请先运行 risk_evaluator.py 生成 risk_windows_all.csv"
        )
    df = pd.read_csv(csv_path, encoding="utf-8-sig")
    required = {
        "sample_idx",
        "window_idx",
        "R_star",
        "risk_level",
        "F_S",
        "field_label",
    }
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"risk_windows_all.csv 缺少列: {missing}")
    print(
        f"  风险评估加载：{len(df)} 行，"
        f"R* ∈ [{df['R_star'].min():.3f}, {df['R_star'].max():.3f}]"
    )
    return df


# =============================================================================
# 2. 逐窗口特征提取
# =============================================================================


def _sliding_window_mean(arr: np.ndarray, win: int) -> np.ndarray:
    """非重叠窗口均值（对齐 capability_fluctuation.py 的预处理逻辑）。"""
    n = arr.shape[0] // win
    if n == 0:
        return np.empty((0,) + arr.shape[1:])
    return arr[: n * win].reshape((n, win) + arr.shape[1:]).mean(axis=1)


def extract_window_features_single(
    act: np.ndarray,
    eye: np.ndarray,
    phy: np.ndarray,
    win_act: int,
    win_eye: int,
    win_phy: int,
) -> Tuple[Optional[np.ndarray], Optional[np.ndarray]]:
    """
    对单个样本提取逐窗口特征向量，维度对齐论文 §5.1：
      行为 8 维：steering_angle, steering_velocity, brake_pedal, throttle_pedal,
                 longitudinal_accel, lateral_offset, lateral_accel, vehicle_speed
      眼动 3 维：blink_frequency, gaze_x, gaze_y
      生理 5 维：bvp, ecg, resp, hr, hrv
      合计 16 维（F_S 来自 risk_csv，后续拼接为第 17 维）

    Returns:
        feat_arr : (n_windows, 16) float32
        field_arr: (n_windows,) int
    """
    act = np.asarray(act, dtype=float)
    eye = np.asarray(eye, dtype=float)
    phy = np.asarray(phy, dtype=float)

    n_win = act.shape[0] // win_act
    if n_win == 0:
        return None, None

    act_other = _sliding_window_mean(act[:, :-1], win_act)  # (n_win, 9)
    act_reshaped = act[: n_win * win_act, -1].reshape(n_win, win_act).astype(int)
    field_arr = np.array(
        [
            (lambda nz: int(np.bincount(nz).argmax()) if len(nz) > 0 else 0)(
                row[row != 0]
            )
            for row in act_reshaped
        ],
        dtype=int,
    )

    eye_win = _sliding_window_mean(eye, win_eye)
    phy_win = _sliding_window_mean(phy, win_phy)

    n_min = min(n_win, len(eye_win), len(phy_win))
    if n_min == 0:
        return None, None

    act_w = act_other[:n_min]
    eye_w = eye_win[:n_min]
    phy_w = phy_win[:n_min]
    field_w = field_arr[:n_min]

    steer_angle = np.abs(act_w[:, 2])
    steer_vel = np.abs(np.diff(steer_angle, prepend=steer_angle[0]) / 3.0)
    brake = np.abs(act_w[:, 1])
    throttle = np.abs(act_w[:, 0])
    lon_acc = np.abs(act_w[:, 4])
    lat_off = np.abs(act_w[:, 8])
    lat_acc = np.abs(act_w[:, 5])
    speed = act_w[:, 3]

    behavior = np.stack(
        [steer_angle, steer_vel, brake, throttle, lon_acc, lat_off, lat_acc, speed],
        axis=1,
    )

    eye_feat = eye_w[:, :3] if eye_w.shape[1] >= 3 else np.zeros((n_min, 3))

    phy_feat = phy_w[:, :5].copy() if phy_w.shape[1] >= 5 else np.zeros((n_min, 5))
    if phy_w.shape[1] >= 5:
        phy_feat[:, 4] = np.abs(phy_feat[:, 4])

    feat_arr = np.concatenate([behavior, eye_feat, phy_feat], axis=1).astype(np.float32)
    return feat_arr, field_w


# =============================================================================
# 3. 构建样本级对齐表
# =============================================================================


def build_per_window_table(
    act_list: List,
    eye_list: List,
    phy_list: List,
    cap_data: dict,
    risk_df: pd.DataFrame,
    cfg: dict,
) -> pd.DataFrame:
    """
    对每个样本、每个窗口，组装：
      sample_idx, window_idx, field_label,
      feat_00..feat_15  (16 维多模态特征),
      F_S               (来自 risk_csv，第 17 维),
      ad_norm           (归一化能力，标签),
      R_star            (风险度回归标签),
      risk_cls          (风险分类标签 0/1/2)
    """
    w = cfg["window"]
    win_act = w["act_hz"] * w["window_seconds"]
    win_eye = w["eye_hz"] * w["window_seconds"]
    win_phy = w["phy_hz"] * w["window_seconds"]
    rl_map = cfg["risk_level_map"]

    risk_dict = {
        (int(r.sample_idx), int(r.window_idx)): r
        for r in risk_df.itertuples(index=False)
    }
    ad_norm_list = cap_data["sample_ad_norm"]

    rows = []
    n_samples = len(act_list)
    for sidx in range(n_samples):
        feat_arr, field_arr = extract_window_features_single(
            act_list[sidx],
            eye_list[sidx],
            phy_list[sidx],
            win_act,
            win_eye,
            win_phy,
        )
        if feat_arr is None:
            continue

        ad_norm = ad_norm_list[sidx] if sidx < len(ad_norm_list) else np.array([])
        n_win = feat_arr.shape[0]

        for widx in range(n_win):
            key = (sidx, widx)
            if key not in risk_dict:
                continue

            rrow = risk_dict[key]
            fs_val = float(rrow.F_S)
            r_star = float(rrow.R_star)
            rl_str = str(rrow.risk_level)
            risk_cls = int(rl_map.get(rl_str, 1))

            if widx >= len(ad_norm):
                continue
            ad_val = float(ad_norm[widx])

            feat_row = feat_arr[widx].tolist()
            row = {
                "sample_idx": sidx,
                "window_idx": widx,
                "field_label": int(field_arr[widx]),
                "F_S": round(fs_val, 4),
                "ad_norm": round(ad_val, 4),
                "R_star": round(r_star, 4),
                "risk_cls": risk_cls,
            }
            for fi, fv in enumerate(feat_row):
                row[f"feat_{fi:02d}"] = round(float(fv), 6)
            rows.append(row)

        if (sidx + 1) % 10 == 0 or sidx == n_samples - 1:
            print(f"  特征提取进度: {sidx+1}/{n_samples}")

    df = pd.DataFrame(rows)
    print(f"\n  对齐窗口总数: {len(df)}")
    return df


# =============================================================================
# 4. 构建序列样本 (N, T, D) — 历史 15s，预测未来 3s
# =============================================================================


def build_sequences(
    window_df: pd.DataFrame,
    seq_len: int,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, pd.DataFrame]:
    """
    在同一个 sample_idx 内，以步长 1 的滑动窗口切出序列：
      - 输入 X    : 窗口 [t-seq_len+1 .. t]（共 seq_len 步，即 seq_len×3s = 15s 历史）
      - 预测目标  : 窗口 t+1（未来 3s）

    特征维度 D = 16 (多模态) + 1 (F_S) = 17

    Returns:
        X          : (N, seq_len, 17)   float32   历史输入
        y_ability  : (N,)               float32   未来步 Ã_d
        y_risk_reg : (N,)               float32   未来步 R*
        y_risk_cls : (N,)               int64     未来步风险等级
        meta_df    : (N, 3)             sample_idx / window_idx(目标步) / field_label(目标步)
    """
    feat_cols = sorted([c for c in window_df.columns if c.startswith("feat_")])
    all_feat_cols = feat_cols + ["F_S"]  # 17 维

    X_list, ya_list, yr_list, yc_list, meta_list = [], [], [], [], []

    for sidx, grp in window_df.groupby("sample_idx", sort=True):
        grp = grp.sort_values("window_idx").reset_index(drop=True)
        n = len(grp)
        if n < seq_len + 1:
            continue

        feat_mat = grp[all_feat_cols].to_numpy(dtype=np.float32)
        ya_arr = grp["ad_norm"].to_numpy(dtype=np.float32)
        yr_arr = grp["R_star"].to_numpy(dtype=np.float32)
        yc_arr = grp["risk_cls"].to_numpy(dtype=np.int64)
        fl_arr = grp["field_label"].to_numpy(dtype=np.int32)
        wi_arr = grp["window_idx"].to_numpy(dtype=np.int32)

        for t in range(seq_len - 1, n - 1):
            target = t + 1
            X_list.append(feat_mat[t - seq_len + 1 : t + 1])
            ya_list.append(ya_arr[target])
            yr_list.append(yr_arr[target])
            yc_list.append(yc_arr[target])
            meta_list.append(
                {
                    "sample_idx": int(sidx),
                    "window_idx": int(wi_arr[target]),
                    "field_label": int(fl_arr[target]),
                }
            )

    X = np.stack(X_list, axis=0).astype(np.float32)
    y_ability = np.array(ya_list, dtype=np.float32)
    y_risk_reg = np.array(yr_list, dtype=np.float32)
    y_risk_cls = np.array(yc_list, dtype=np.int64)
    meta_df = pd.DataFrame(meta_list)

    history_s = seq_len * 3
    print(f"  序列样本构建: X={X.shape} (历史{history_s}s → 预测未来3s)")
    print(f"  y_ability  ∈ [{y_ability.min():.3f}, {y_ability.max():.3f}]")
    print(f"  y_risk_reg ∈ [{y_risk_reg.min():.3f}, {y_risk_reg.max():.3f}]")
    cls_counts = {k: int((y_risk_cls == k).sum()) for k in [0, 1, 2]}
    print(
        f"  风险分类分布: 低={cls_counts[0]} / 中={cls_counts[1]} / 高={cls_counts[2]}"
    )

    return X, y_ability, y_risk_reg, y_risk_cls, meta_df


# =============================================================================
# 5. 分层数据集划分
# =============================================================================


def stratified_split(
    meta_df: pd.DataFrame,
    window_df: pd.DataFrame,
    train_ratio: float,
    val_ratio: float,
    seed: int,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    按 sample_idx（驾驶人）分层划分，保证三组基准能力比例在 train/val/test 中均衡。
    分层依据：各驾驶人平均 ad_norm 三等分为高/中/低组，组内独立随机分配。
    """
    rng = np.random.default_rng(seed)

    sid_mean = window_df.groupby("sample_idx")["ad_norm"].mean().reset_index()
    sid_mean = sid_mean.sort_values("ad_norm").reset_index(drop=True)
    n_s = len(sid_mean)

    low_sids = sid_mean.iloc[: n_s // 3]["sample_idx"].tolist()
    mid_sids = sid_mean.iloc[n_s // 3 : 2 * n_s // 3]["sample_idx"].tolist()
    high_sids = sid_mean.iloc[2 * n_s // 3 :]["sample_idx"].tolist()

    train_sids, val_sids, test_sids = [], [], []
    for group_sids in [low_sids, mid_sids, high_sids]:
        arr = np.array(group_sids)
        rng.shuffle(arr)
        n = len(arr)
        n_tr = max(1, int(n * train_ratio))
        n_va = max(1, int(n * val_ratio))
        train_sids.extend(arr[:n_tr].tolist())
        val_sids.extend(arr[n_tr : n_tr + n_va].tolist())
        test_sids.extend(arr[n_tr + n_va :].tolist())

    train_set = set(train_sids)
    val_set = set(val_sids)
    sample_arr = meta_df["sample_idx"].to_numpy()
    train_mask = np.isin(sample_arr, list(train_set))
    val_mask = np.isin(sample_arr, list(val_set))
    test_mask = ~train_mask & ~val_mask

    print(
        f"  驾驶人划分: train={len(train_sids)} / val={len(val_sids)} / test={len(test_sids)}"
    )
    print(
        f"  样本划分:   train={train_mask.sum()} / val={val_mask.sum()} / test={test_mask.sum()}"
    )
    return train_mask, val_mask, test_mask


# =============================================================================
# 6. Z-score 标准化（统计量来自增强后训练集）
# =============================================================================


def zscore_normalize(
    X_train: np.ndarray,
    X_val: np.ndarray,
    X_test: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    对特征维度 D 做 Z-score 标准化，μ/σ 仅从训练集计算。
    F_S（第 17 维，索引 16）已归一化至 [0,1]，不参与标准化。
    """
    _, _, D = X_train.shape
    X_tr_flat = X_train.reshape(-1, D)
    mu = X_tr_flat.mean(axis=0)
    sigma = X_tr_flat.std(axis=0) + 1e-8
    mu[-1] = 0.0
    sigma[-1] = 1.0

    def _apply(X):
        return ((X - mu) / sigma).astype(np.float32)

    return _apply(X_train), _apply(X_val), _apply(X_test), mu, sigma


# =============================================================================
# 7. 保存数据集
# =============================================================================


def save_dataset(
    splits: Dict,
    mu: np.ndarray,
    sigma: np.ndarray,
    feat_cols_17: List[str],
    output_pkl: str,
    output_stats: str,
) -> None:
    os.makedirs(os.path.dirname(output_pkl), exist_ok=True)
    os.makedirs(os.path.dirname(output_stats), exist_ok=True)

    payload = {
        "train": splits["train"],
        "val": splits["val"],
        "test": splits["test"],
        "norm": {"mu": mu, "sigma": sigma},
        "feature_names": feat_cols_17,
    }
    with open(output_pkl, "wb") as f:
        pickle.dump(payload, f)
    print(f"\n  数据集 pkl → {output_pkl}")

    stats_df = pd.DataFrame(
        {
            "feature": feat_cols_17,
            "train_mean": mu.tolist(),
            "train_std": sigma.tolist(),
        }
    )
    stats_df.to_csv(output_stats, index=False, encoding="utf-8-sig")
    print(f"  特征统计 CSV → {output_stats}")


# =============================================================================
# 主函数
# =============================================================================


def main():
    parser = argparse.ArgumentParser(
        description="MT-JP 联合预测模型数据集构建（历史15s → 预测未来3s）",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "-c",
        "--config",
        type=str,
        default=None,
        help="YAML 配置文件路径（默认: config/mtjp_dataset.yaml）",
    )
    parser.add_argument(
        "--seq_len",
        type=int,
        default=None,
        help="历史步数 T（覆盖 yaml model.seq_len）",
    )
    parser.add_argument(
        "--seed", type=int, default=None, help="随机种子（覆盖 yaml split.seed）"
    )
    parser.add_argument("--no_zscore", action="store_true", help="禁用 Z-score 标准化")
    parser.add_argument(
        "--no_augment",
        action="store_true",
        help="禁用数据增强（覆盖 yaml augmentation.enabled）",
    )
    parser.add_argument("--output_pkl", type=str, default=None, help="输出 pkl 路径")
    args = parser.parse_args()

    cfg = load_config(args.config)
    if args.seq_len is not None:
        cfg["model"]["seq_len"] = args.seq_len
    if args.seed is not None:
        cfg["split"]["seed"] = args.seed
    if args.no_zscore:
        cfg["normalization"]["zscore"] = False
    if args.no_augment:
        cfg["augmentation"]["enabled"] = False
    if args.output_pkl:
        cfg["paths"]["output_pkl"] = args.output_pkl

    # ========== 动态拼接增强参数到文件名 ==========
    if not args.output_pkl:  # 仅当用户未指定输出路径时自动拼接
        aug_cfg = cfg["augmentation"]
        aug_params = []

        # 核心增强开关
        aug_params.append(f"aug-{aug_cfg['enabled']}")
        if aug_cfg["enabled"]:
            # 仅训练集增强
            aug_params.append(f"onlyTrain-{aug_cfg['only_on_train']}")
            # 各增强方法的关键参数（仅启用的方法才加）
            methods = aug_cfg["methods"]
            if methods["gaussian_noise"]["enabled"]:
                aug_params.append(f"gaussStd-{methods['gaussian_noise']['std_scale']}")
            if methods["time_warp"]["enabled"]:
                aug_params.append(f"timeWarpSigma-{methods['time_warp']['sigma']}")
            if methods["feature_dropout"]["enabled"]:
                aug_params.append(f"featDrop-{methods['feature_dropout']['drop_prob']}")
            if methods["magnitude_warp"]["enabled"]:
                aug_params.append(f"magWarpSigma-{methods['magnitude_warp']['sigma']}")

        # 拼接参数并生成最终文件名
        param_suffix = "_".join(aug_params)
        cfg["paths"]["output_pkl"] = f"{cfg['paths']['output_pkl']}_{param_suffix}.pkl"
        cfg["paths"][
            "output_stats"
        ] = f"{cfg['paths']['output_stats']}_{param_suffix}.csv"
    # ==================================================

    seq_len = cfg["model"]["seq_len"]
    win_sec = cfg["window"]["window_seconds"]
    history_s = seq_len * win_sec
    do_aug = cfg["augmentation"].get("enabled", False)

    print("=" * 65)
    print("MT-JP 联合预测模型 — 数据集构建")
    print(f"  历史长度:  T={seq_len} 步 × {win_sec}s = {history_s}s")
    print(f"  预测目标:  未来 {win_sec}s（下一步）")
    print(f"  zscore  :  {cfg['normalization']['zscore']}")
    print(f"  数据增强:  {'开启' if do_aug else '关闭'}")
    print("=" * 65)

    # ── Step 1: 加载三路数据源 ─────────────────────────────────
    print("\n[1/6] 加载原始多模态数据...")
    act_list, eye_list, phy_list = load_raw(
        cfg["paths"]["raw_pkl"],
        cfg.get("n_samples", None),  # 兼容 YAML 中可能的缺失，或直接改为 None
    )

    print("\n[2/6] 加载能力评估结果...")
    cap_data = load_capability(cfg["paths"]["cap_pkl"])

    print("\n[3/6] 加载风险评估结果...")
    risk_df = load_risk(cfg["paths"]["risk_csv"])
    risk_df["risk_cls"] = (
        risk_df["risk_level"].map(cfg["risk_level_map"]).fillna(1).astype(int)
    )

    # ── Step 2: 逐窗口特征提取 + 三路对齐 ─────────────────────
    print("\n[4/6] 逐窗口特征提取与三路对齐...")
    window_df = build_per_window_table(
        act_list, eye_list, phy_list, cap_data, risk_df, cfg
    )
    mid_path = os.path.join(
        os.path.dirname(cfg["paths"]["output_pkl"]), "mtjp_window_aligned.csv"
    )
    os.makedirs(os.path.dirname(mid_path), exist_ok=True)
    window_df.to_csv(mid_path, index=False, encoding="utf-8-sig")
    print(f"  逐窗口对齐表 → {mid_path}")

    # ── Step 3: 构建序列样本（历史15s → 预测未来3s）──────────
    print(f"\n[5/6] 构建历史T={seq_len}步序列（→ 预测未来第T+1步）...")
    X, y_ability, y_risk_reg, y_risk_cls, meta_df = build_sequences(window_df, seq_len)

    # ── Step 4: 分层划分 ───────────────────────────────────────
    train_mask, val_mask, test_mask = stratified_split(
        meta_df,
        window_df,
        cfg["split"]["train_ratio"],
        cfg["split"]["val_ratio"],
        cfg["split"]["seed"],
    )

    X_tr = X[train_mask]
    ya_tr = y_ability[train_mask]
    yr_tr = y_risk_reg[train_mask]
    yc_tr = y_risk_cls[train_mask]
    meta_tr = meta_df[train_mask].reset_index(drop=True)

    X_va = X[val_mask]
    ya_va = y_ability[val_mask]
    yr_va = y_risk_reg[val_mask]
    yc_va = y_risk_cls[val_mask]
    meta_va = meta_df[val_mask].reset_index(drop=True)

    X_te = X[test_mask]
    ya_te = y_ability[test_mask]
    yr_te = y_risk_reg[test_mask]
    yc_te = y_risk_cls[test_mask]
    meta_te = meta_df[test_mask].reset_index(drop=True)

    # ── Step 4.5: 数据增强（在标准化前，仅对训练集）──────────
    if do_aug:
        print("\n[AUG] 数据增强（Z-score 标准化前）...")
        # 延迟导入，避免非增强模式也加载模块
        from src.models.augment import Augmentor

        aug = Augmentor(cfg["augmentation"], seed=cfg["split"]["seed"])
        # 保存原始训练集长度，用于后续meta扩充
        orig_len = len(X_tr)
        X_tr, ya_tr, yr_tr, yc_tr = aug.apply(X_tr, ya_tr, yr_tr, yc_tr, split="train")
        # meta_tr 扩充（修复核心逻辑）
        n_aug = len(ya_tr) - orig_len
        if n_aug > 0:
            # 方法1：为增强样本复制原始meta（循环填充至n_aug个）
            aug_meta_rows = []
            for i in range(n_aug):
                # 循环取原始meta的行，保证每个增强样本都有对应meta
                orig_idx = i % orig_len
                aug_meta_rows.append(meta_tr.iloc[orig_idx].to_dict())
            # 转为DataFrame并标记为增强样本
            aug_meta_df = pd.DataFrame(aug_meta_rows)
            aug_meta_df["augmented"] = True
            # 原始样本标记为非增强
            meta_tr["augmented"] = False
            # 合并原始meta和增强meta
            meta_tr = pd.concat([meta_tr, aug_meta_df], ignore_index=True)
        print(f"    增强后训练集: X_tr={X_tr.shape}")
    else:
        print("\n[AUG] 数据增强已关闭，跳过")

    # ── Step 5: 标准化 ─────────────────────────────────────────
    feat_cols_17 = [
        "steer_angle",
        "steer_vel",
        "brake",
        "throttle",
        "lon_acc",
        "lat_off",
        "lat_acc",
        "speed",  # 行为 8 维
        "blink_freq",
        "gaze_x",
        "gaze_y",  # 眼动 3 维
        "bvp",
        "ecg",
        "resp",
        "hr",
        "hrv",  # 生理 5 维
        "F_S",  # 环境 1 维
    ]

    if cfg["normalization"]["zscore"]:
        print("\n[6/6] Z-score 标准化（统计量来自训练集，F_S 不参与）...")
        X_tr, X_va, X_te, mu, sigma = zscore_normalize(X_tr, X_va, X_te)
    else:
        mu = np.zeros(X.shape[-1], dtype=np.float32)
        sigma = np.ones(X.shape[-1], dtype=np.float32)

    # ── Step 6: 打包并保存 ─────────────────────────────────────
    splits = {
        "train": {
            "X": X_tr,
            "y_ability": ya_tr,
            "y_risk_reg": yr_tr,
            "y_risk_cls": yc_tr,
            "meta": meta_tr,
        },
        "val": {
            "X": X_va,
            "y_ability": ya_va,
            "y_risk_reg": yr_va,
            "y_risk_cls": yc_va,
            "meta": meta_va,
        },
        "test": {
            "X": X_te,
            "y_ability": ya_te,
            "y_risk_reg": yr_te,
            "y_risk_cls": yc_te,
            "meta": meta_te,
        },
    }

    save_dataset(
        splits,
        mu,
        sigma,
        feat_cols_17,
        cfg["paths"]["output_pkl"],
        cfg["paths"]["output_stats"],
    )

    # ── 最终统计 ────────────────────────────────────────────────
    print(f"\n{'='*65}")
    print("数据集构建完成")
    print(f"{'='*65}")
    for split_name, sp in splits.items():
        n = len(sp["y_ability"])
        n0 = int((sp["y_risk_cls"] == 0).sum())
        n1 = int((sp["y_risk_cls"] == 1).sum())
        n2 = int((sp["y_risk_cls"] == 2).sum())
        print(
            f"  {split_name:5s}  N={n:6d}  "
            f"低风险={n0:5d}({n0/max(n,1)*100:4.1f}%)  "
            f"中风险={n1:5d}({n1/max(n,1)*100:4.1f}%)  "
            f"高风险={n2:5d}({n2/max(n,1)*100:4.1f}%)"
        )
    print(f"\n  输入维度:  X = (N, T={seq_len}, D=17)  [{history_s}s 历史]")
    print(f"  预测目标:  y_*(N,)  对应未来 {win_sec}s 窗口")
    print(f"  标签类型:  y_ability(float32) / y_risk_reg(float32) / y_risk_cls(int64)")
    print(f"  输出目录:  {os.path.dirname(cfg['paths']['output_pkl'])}")
    print("=" * 65)


if __name__ == "__main__":
    main()

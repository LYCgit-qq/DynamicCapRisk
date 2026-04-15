"""
风险场强量化计算模块
基于Driving Safety Field理论，计算施工区场景的综合风险场强
参考论文第4章：风险场强量化方法

权重确定：AHP（专家打分）× 熵权法（数据驱动）乘法合成
"""

import pandas as pd
import numpy as np
import argparse
import os
import yaml
from typing import Tuple, Dict, List
from scipy.ndimage import gaussian_filter1d

# 导入可视化函数
from src.visualization.plot_risk import (
    plot_radar_chart,
    plot_stacked_bar,
    plot_field_evolution,
)


# ============= 全局配置变量 =============


def load_config(config_path: str = "config/risk_field.yaml") -> Dict:
    with open(config_path, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)
    print(f"成功加载配置文件: {config_path}")
    return config


# =============================================================================
# 权重计算：AHP + 熵权法乘法合成
# =============================================================================


def load_ahp_weights_from_csv(csv_path: str) -> Dict[str, float]:
    """
    从 ahp_calculator.py 生成的权重 CSV 中读取 AHP 权重。

    CSV 格式（ahp_calculator 输出）：
        feature_name, ahp_weight, max_eigenvalue, CI, CR

    Args:
        csv_path: AHP 权重 CSV 路径

    Returns:
        {特征名: 归一化权重} 字典
    """
    if not os.path.exists(csv_path):
        raise FileNotFoundError(
            f"AHP权重文件不存在: {csv_path}\n"
            f"请先运行 ahp_calculator.py 生成该文件：\n"
            f"  python ahp_calculator.py -c <判断矩阵CSV> -s {csv_path}"
        )

    df = pd.read_csv(csv_path, encoding="utf-8-sig")

    if "feature_name" not in df.columns or "ahp_weight" not in df.columns:
        raise ValueError(
            f"AHP权重CSV格式错误，需含 'feature_name' 和 'ahp_weight' 列: {csv_path}"
        )

    weights = dict(zip(df["feature_name"], df["ahp_weight"].astype(float)))
    total = sum(weights.values())
    weights = {k: v / total for k, v in weights.items()}

    print(
        f"  已加载AHP权重 [{os.path.basename(csv_path)}]: "
        + ", ".join(f"{k}={v:.4f}" for k, v in weights.items())
    )
    return weights


def entropy_weights(data_dict: Dict[str, np.ndarray]) -> Dict[str, float]:
    """
    熵权法客观赋权。

    对各分量场强的原始值计算信息熵，差异越大的分量熵权越高。

    Args:
        data_dict: {特征名: 1-D numpy 数组}，各分量的原始场强值

    Returns:
        {特征名: 熵权} 字典
    """
    features = list(data_dict.keys())
    if not features:
        return {}

    X = np.column_stack([data_dict[f] for f in features]).astype(float)

    # 平移到严格正数
    col_min = X.min(axis=0)
    X = X - col_min + 1e-8

    col_sum = X.sum(axis=0)
    nonzero = col_sum != 0
    p = np.zeros_like(X)
    p[:, nonzero] = X[:, nonzero] / col_sum[nonzero]

    with np.errstate(divide="ignore", invalid="ignore"):
        e = -np.sum(p * np.log(p + 1e-12), axis=0) / np.log(len(X))

    d = 1 - e
    d[~nonzero] = 0.0

    if d.sum() == 0:
        w = np.ones(len(features)) / len(features)
    else:
        w = d / d.sum()

    ent_w = dict(zip(features, w))
    print(f"  熵权: " + ", ".join(f"{k}={v:.4f}" for k, v in ent_w.items()))
    return ent_w


def combine_weights(
    ahp_w: Dict[str, float], ent_w: Dict[str, float], features: List[str]
) -> Dict[str, float]:
    """
    AHP × 熵权 乘法合成，归一化输出。
    无公共特征时退化为纯熵权。

    Args:
        ahp_w:    AHP 权重字典
        ent_w:    熵权字典
        features: 输出特征列表

    Returns:
        {特征名: 组合权重} 字典（和为1）
    """
    common = [f for f in features if f in ahp_w and f in ent_w]

    if not common:
        print("  警告: AHP与熵权无公共特征，退化为纯熵权")
        w = {f: ent_w.get(f, 1 / len(features)) for f in features}
    else:
        raw = np.array([ahp_w[f] * ent_w[f] for f in common])
        raw = raw / raw.sum()
        w = dict(zip(common, raw))

    # 补全缺失特征（防御）
    for f in features:
        if f not in w:
            w[f] = 0.0

    total = sum(w.values())
    w = {k: v / total for k, v in w.items()}
    print(f"  组合权重(AHP×熵): " + ", ".join(f"{k}={v:.4f}" for k, v in w.items()))
    return w


def compute_dynamic_weights(
    scenario_dfs: List[pd.DataFrame], ahp_main_path: str, ahp_sign_path: str
) -> Dict[str, float]:
    """
    跨所有场景数据计算动态组合权重（AHP × 熵权法）。

    主层权重：  w_veh / w_geo / w_sign
    设施子层：  lambda_1(sign_density) / lambda_2(work_zone)

    熵权使用所有场景数据拼接后的原始场强值，
    反映各分量在完整数据集上的信息量差异。

    Args:
        scenario_dfs:  所有场景的原始 DataFrame 列表
        ahp_main_path: 主层 AHP 权重 CSV 路径
        ahp_sign_path: 设施子层 AHP 权重 CSV 路径

    Returns:
        含 w_veh / w_geo / w_sign / lambda_1 / lambda_2 的字典
    """
    print("\n" + "=" * 60)
    print("动态权重计算（AHP × 熵权法）")
    print("=" * 60)

    # ── 1. 加载 AHP 权重 ─────────────────────────────────────────────
    print("\n[1/4] 加载 AHP 权重...")
    ahp_main = load_ahp_weights_from_csv(ahp_main_path)  # {s_veh, s_geo, s_sign}
    ahp_sign = load_ahp_weights_from_csv(ahp_sign_path)  # {sign_density, work_zone}

    # ── 2. 计算各场景原始场强分量 ────────────────────────────────────
    print("\n[2/4] 计算各场景原始场强分量...")
    sign_cfg = CONFIG.get("sign_field", {})
    sigma_front = sign_cfg.get("sigma_front", 120.0)
    sigma_back = sign_cfg.get("sigma_back", 40.0)
    wz_sigma = sign_cfg.get("wz_smooth_sigma", 20.0)

    all_s_veh, all_s_geo, all_s_sign = [], [], []
    all_sign_density, all_wz = [], []

    for df in scenario_dfs:
        distances = df["距离 (m)"].values.astype(float)

        # 主层三分量
        all_s_geo.append(calculate_geo_field(df).values)
        all_s_sign.append(calculate_sign_field(df).values)
        all_s_veh.append(calculate_vehicle_field_static(df).values)

        # 设施子层：单独拆分 sign_density / work_zone
        has_sign = (df["标识牌类型"] != "-").values
        influence = np.zeros(len(distances))
        for sp in distances[has_sign]:
            d_vec = distances - sp
            kernel = np.where(
                d_vec <= 0, np.exp(d_vec / sigma_front), np.exp(-d_vec / sigma_back)
            )
            influence = np.maximum(influence, kernel)
        mx = influence.max()
        if mx > 1e-6:
            influence /= mx

        wz_raw = (df["施工区状态"] == "是").values.astype(float)
        wz_smooth = _smooth_mask(wz_raw, distances, wz_sigma)

        all_sign_density.append(influence)
        all_wz.append(wz_smooth)

    cat = np.concatenate
    main_data = {
        "s_veh": cat(all_s_veh),
        "s_geo": cat(all_s_geo),
        "s_sign": cat(all_s_sign),
    }
    sign_data = {"sign_density": cat(all_sign_density), "work_zone": cat(all_wz)}

    # ── 3. 熵权计算 ──────────────────────────────────────────────────
    print("\n[3/4] 熵权法...")
    print("  主层:")
    ent_main = entropy_weights(main_data)
    print("  设施子层:")
    ent_sign = entropy_weights(sign_data)

    # ── 4. 乘法合成 ──────────────────────────────────────────────────
    print("\n[4/4] AHP × 熵权 乘法合成...")
    print("  主层:")
    w_main = combine_weights(ahp_main, ent_main, ["s_veh", "s_geo", "s_sign"])
    print("  设施子层:")
    w_sign = combine_weights(ahp_sign, ent_sign, ["sign_density", "work_zone"])

    result = {
        "w_veh": w_main["s_veh"],
        "w_geo": w_main["s_geo"],
        "w_sign": w_main["s_sign"],
        "lambda_1": w_sign["sign_density"],
        "lambda_2": w_sign["work_zone"],
    }

    print(f"\n最终组合权重:")
    print(
        f"  主层:     w_veh={result['w_veh']:.4f},  "
        f"w_geo={result['w_geo']:.4f},  w_sign={result['w_sign']:.4f}"
    )
    print(
        f"  设施子层: λ1(sign_density)={result['lambda_1']:.4f},  "
        f"λ2(work_zone)={result['lambda_2']:.4f}"
    )
    print("=" * 60)

    return result


# =============================================================================
# 工具函数
# =============================================================================


def _get_dx(distances: np.ndarray) -> float:
    if len(distances) < 2:
        return 1.0
    dx = float(np.median(np.diff(distances)))
    return dx if dx > 1e-6 else 1.0


def _gaussian_smooth(
    values: np.ndarray, distances: np.ndarray, sigma_m: float
) -> np.ndarray:
    dx = _get_dx(distances)
    sigma_samples = max(sigma_m / dx, 0.5)
    return gaussian_filter1d(values.astype(float), sigma=sigma_samples, mode="nearest")


def _smooth_mask(
    binary_mask: np.ndarray, distances: np.ndarray, sigma_m: float
) -> np.ndarray:
    return _gaussian_smooth(binary_mask.astype(float), distances, sigma_m)


# =============================================================================
# 场强分量计算
# =============================================================================


def calculate_geo_field(df: pd.DataFrame) -> pd.Series:
    geo_cfg = CONFIG["geometry"]
    distances = df["距离 (m)"].values.astype(float)

    c_hat = df["道路几何类型"].map(geo_cfg["curvature_map"]).fillna(0.0).values
    w_hat = df["车道数"].map(geo_cfg["lane_width_map"]).fillna(0.0).values
    l_hat = df["车道数"].map(geo_cfg["lane_map"]).fillna(0.0).values

    raw_geo = (c_hat + w_hat + l_hat) / 3.0
    sigma_m = geo_cfg.get("smooth_sigma", 30.0)
    smoothed = _gaussian_smooth(raw_geo, distances, sigma_m)

    return pd.Series(smoothed, index=df.index)


def calculate_sign_field(df: pd.DataFrame) -> pd.Series:
    """
    s_sign = λ1 · influence_max(x) + λ2 · δ_wz_smooth(x)
    λ1/λ2 从 CONFIG['weights'] 读取（由动态权重计算覆盖）
    """
    sign_cfg = CONFIG.get("sign_field", {})
    sigma_front = sign_cfg.get("sigma_front", 120.0)
    sigma_back = sign_cfg.get("sigma_back", 40.0)
    wz_sigma = sign_cfg.get("wz_smooth_sigma", 20.0)

    weights = CONFIG["weights"]
    distances = df["距离 (m)"].values.astype(float)

    has_sign = (df["标识牌类型"] != "-").values
    sign_positions = distances[has_sign]
    sign_influence = np.zeros(len(distances))

    for sign_pos in sign_positions:
        d_vec = distances - sign_pos
        kernel = np.where(
            d_vec <= 0, np.exp(d_vec / sigma_front), np.exp(-d_vec / sigma_back)
        )
        sign_influence = np.maximum(sign_influence, kernel)

    max_inf = sign_influence.max()
    if max_inf > 1e-6:
        sign_influence /= max_inf

    delta_wz_raw = (df["施工区状态"] == "是").values.astype(float)
    delta_wz_smooth = _smooth_mask(delta_wz_raw, distances, wz_sigma)

    return pd.Series(
        weights["lambda_1"] * sign_influence + weights["lambda_2"] * delta_wz_smooth,
        index=df.index,
    )


def calculate_vehicle_field_static(df: pd.DataFrame) -> pd.Series:
    veh_cfg = CONFIG["vehicle_interaction"]
    distances = df["距离 (m)"].values.astype(float)

    baseline = veh_cfg["baseline"]
    wz_level = veh_cfg["work_zone"]
    cx_level = veh_cfg["complex_geometry"]
    bnd_sigma = veh_cfg.get("boundary_smooth_sigma", 25.0)

    wz_raw = (df["施工区状态"] == "是").values.astype(float)
    cx_raw = df["道路几何类型"].isin(["bend", "cross"]).values.astype(float)

    wz_smooth = _smooth_mask(wz_raw, distances, bnd_sigma)
    cx_smooth = _smooth_mask(cx_raw, distances, bnd_sigma)

    s_veh = (
        baseline
        + (wz_level - baseline) * wz_smooth
        + (cx_level - wz_level) * cx_smooth * wz_smooth
    )

    entry_peak = veh_cfg.get("entry_peak", 0.98)
    entry_sigma = veh_cfg.get("entry_sigma", 25.0)
    entry_offset = veh_cfg.get("entry_offset", -15.0)

    wz_int = wz_raw.astype(int)
    entry_indices = np.where(np.diff(wz_int) == 1)[0] + 1

    for idx in entry_indices:
        center = distances[idx] + entry_offset
        gaussian = entry_peak * np.exp(-0.5 * ((distances - center) / entry_sigma) ** 2)
        s_veh = np.maximum(s_veh, gaussian)

    return pd.Series(s_veh, index=df.index)


# =============================================================================
# 归一化 / 等级判定
# =============================================================================


def normalize_field(field: pd.Series) -> pd.Series:
    min_val, max_val = field.min(), field.max()
    if max_val - min_val < 1e-6:
        return pd.Series(np.zeros(len(field)), index=field.index)
    return (field - min_val) / (max_val - min_val)


def get_field_level(mean_field: float) -> str:
    levels = CONFIG["field_levels"]
    if mean_field < levels["medium"][0]:
        return "低"
    elif mean_field < levels["medium_high"][0]:
        return "中"
    elif mean_field < levels["high"][0]:
        return "中高"
    else:
        return "高"


# =============================================================================
# 综合场强
# =============================================================================


def calculate_comprehensive_field(df: pd.DataFrame) -> pd.DataFrame:
    """
    F_S = w_veh·s̃_veh + w_geo·s̃_geo + w_sign·s̃_sign
    权重从 CONFIG['weights'] 读取（已由 compute_dynamic_weights 覆盖）
    """
    weights = CONFIG["weights"]
    levels = CONFIG["field_levels"]
    final_sigma = CONFIG["calculation"].get("final_smooth_sigma", 8.0)

    result = df[["距离 (m)"]].copy()
    distances = df["距离 (m)"].values.astype(float)

    s_geo = calculate_geo_field(df)
    s_sign = calculate_sign_field(df)
    s_veh = calculate_vehicle_field_static(df)

    s_geo_norm = normalize_field(s_geo)
    s_sign_norm = normalize_field(s_sign)
    s_veh_norm = normalize_field(s_veh)

    F_S_raw = (
        weights["w_veh"] * s_veh_norm
        + weights["w_geo"] * s_geo_norm
        + weights["w_sign"] * s_sign_norm
    )

    F_S = pd.Series(
        _gaussian_smooth(F_S_raw.values, distances, final_sigma), index=df.index
    ).clip(0.0, 1.0)

    result["s_geo"] = s_geo
    result["s_geo_norm"] = s_geo_norm
    result["s_sign"] = s_sign
    result["s_sign_norm"] = s_sign_norm
    result["s_veh"] = s_veh
    result["s_veh_norm"] = s_veh_norm
    result["F_S"] = F_S

    result["field_level"] = pd.cut(
        F_S,
        bins=[
            levels["low"][0],
            levels["medium"][0],
            levels["medium_high"][0],
            levels["high"][0],
            levels["high"][1],
        ],
        labels=["低", "中", "中高", "高"],
        include_lowest=True,
    )

    return result


# =============================================================================
# 统计 / 场景处理 / 辅助输出
# =============================================================================


def get_scenario_statistics(result_df: pd.DataFrame, scenario_name: str) -> Dict:
    return {
        "scenario": scenario_name,
        "s_geo_mean": result_df["s_geo_norm"].mean(),
        "s_sign_mean": result_df["s_sign_norm"].mean(),
        "s_veh_mean": result_df["s_veh_norm"].mean(),
        "F_S_mean": result_df["F_S"].mean(),
        "F_S_max": result_df["F_S"].max(),
        "F_S_min": result_df["F_S"].min(),
        "F_S_std": result_df["F_S"].std(),
        "field_level": get_field_level(result_df["F_S"].mean()),
    }


def process_scenario(
    input_path: str, output_path: str, scenario_name: str
) -> Tuple[pd.DataFrame, Dict]:
    df = pd.read_csv(input_path, encoding="utf-8-sig")

    print(f"\n{'='*60}")
    print(f"处理场景: {scenario_name}  |  数据行数: {len(df)}")
    print(f"{'='*60}")

    result_df = calculate_comprehensive_field(df)
    stats = get_scenario_statistics(result_df, scenario_name)

    result_df.to_csv(output_path, index=False, encoding="utf-8-sig")
    print(f"结果已保存: {output_path}")
    print(f"\n场景统计:")
    print(f"  道路几何场强 (归一化): {stats['s_geo_mean']:.3f}")
    print(f"  道路设施场强 (归一化): {stats['s_sign_mean']:.3f}")
    print(f"  车辆交互场强 (归一化): {stats['s_veh_mean']:.3f}")
    print(
        f"  综合场强 F_S: {stats['F_S_mean']:.3f} "
        f"(范围: {stats['F_S_min']:.3f} - {stats['F_S_max']:.3f})"
    )
    print(f"  场强等级: {stats['field_level']}")

    return result_df, stats


def save_dynamic_weights(weights: Dict[str, float], output_dir: str) -> None:
    """将本次使用的组合权重保存到 CSV，方便复现。"""
    rows = [
        {"layer": "main", "feature": "s_veh", "combined_weight": weights["w_veh"]},
        {"layer": "main", "feature": "s_geo", "combined_weight": weights["w_geo"]},
        {"layer": "main", "feature": "s_sign", "combined_weight": weights["w_sign"]},
        {
            "layer": "sign",
            "feature": "sign_density",
            "combined_weight": weights["lambda_1"],
        },
        {
            "layer": "sign",
            "feature": "work_zone",
            "combined_weight": weights["lambda_2"],
        },
    ]
    save_path = os.path.join(output_dir, "risk_field_combined_weights.csv")
    pd.DataFrame(rows).to_csv(save_path, index=False, encoding="utf-8-sig")
    print(f"\n组合权重已保存: {save_path}")


# =============================================================================
# 主函数
# =============================================================================


def main():
    global CONFIG

    parser = argparse.ArgumentParser(
        description="计算施工区场景风险场强（AHP+熵权组合赋权）并生成可视化",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例用法:
  # 使用配置文件默认路径
  python risk_field.py

  # 指定 AHP 权重文件
  python risk_field.py \\
    --ahp_main output/2_risk_assessment/risk_field_main_weights.csv \\
    --ahp_sign output/2_risk_assessment/risk_field_sign_weights.csv

  # 指定场景和输出目录
  python risk_field.py -i data/processed -o output/2_risk_assessment -s work_zone_1 work_zone_2
        """,
    )

    parser.add_argument(
        "-c",
        "--config",
        type=str,
        default="config/risk_field.yaml",
        help="配置文件路径（默认: config/risk_field.yaml）",
    )
    parser.add_argument(
        "-i",
        "--input_dir",
        type=str,
        default=None,
        help="输入目录（含 *_continuous.csv）",
    )
    parser.add_argument(
        "-o", "--output_dir", type=str, default=None, help="输出目录（CSV + 可视化）"
    )
    parser.add_argument(
        "-v",
        "--vis_dir",
        type=str,
        default=None,
        help="可视化输出目录（默认同 output_dir）",
    )
    parser.add_argument(
        "-s", "--scenarios", type=str, nargs="+", default=None, help="要处理的场景列表"
    )
    parser.add_argument(
        "--ahp_main", type=str, default=None, help="主层 AHP 权重 CSV（覆盖配置文件）"
    )
    parser.add_argument(
        "--ahp_sign",
        type=str,
        default=None,
        help="设施子层 AHP 权重 CSV（覆盖配置文件）",
    )

    args = parser.parse_args()
    CONFIG = load_config(args.config)

    input_dir = args.input_dir or CONFIG["paths"]["input_dir"]
    output_dir = args.output_dir or CONFIG["paths"]["output_dir"]
    vis_dir = args.vis_dir or CONFIG["paths"].get("vis_dir", output_dir)
    scenarios = args.scenarios or CONFIG["scenarios"]["default"]
    ahp_main_path = args.ahp_main or CONFIG["paths"].get(
        "ahp_main_weights", "output/2_risk_assessment/risk_field_main_weights.csv"
    )
    ahp_sign_path = args.ahp_sign or CONFIG["paths"].get(
        "ahp_sign_weights", "output/2_risk_assessment/risk_field_sign_weights.csv"
    )

    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(vis_dir, exist_ok=True)

    print("=" * 60)
    print("风险场强计算与可视化（AHP + 熵权法）")
    print("=" * 60)
    print(f"输入目录:     {input_dir}")
    print(f"输出目录:     {output_dir}")
    print(f"可视化目录:   {vis_dir}")
    print(f"处理场景:     {', '.join(scenarios)}")
    print(f"AHP主层权重:  {ahp_main_path}")
    print(f"AHP设施权重:  {ahp_sign_path}")
    print("=" * 60)

    # ── 预加载所有场景数据（供跨场景熵权计算）───────────────────────
    scenario_dfs = []
    valid_scenarios = []
    for scenario in scenarios:
        input_file = os.path.join(input_dir, f"{scenario}_continuous.csv")
        if not os.path.exists(input_file):
            print(f"警告: 文件不存在，跳过 - {input_file}")
            continue
        try:
            scenario_dfs.append(pd.read_csv(input_file, encoding="utf-8-sig"))
            valid_scenarios.append(scenario)
        except Exception as e:
            print(f"错误: 读取 {scenario} 时出错 - {str(e)}")

    if not scenario_dfs:
        print("错误: 无有效场景数据，退出")
        return

    # ── 动态计算组合权重并写入 CONFIG ────────────────────────────────
    dynamic_w = compute_dynamic_weights(scenario_dfs, ahp_main_path, ahp_sign_path)
    CONFIG["weights"].update(dynamic_w)
    save_dynamic_weights(dynamic_w, output_dir)

    # ── 逐场景计算场强 ────────────────────────────────────────────────
    all_stats = []
    all_results = {}

    for scenario in valid_scenarios:
        input_file = os.path.join(input_dir, f"{scenario}_continuous.csv")
        output_file = os.path.join(output_dir, f"Fs_{scenario}.csv")
        try:
            result_df, stats = process_scenario(input_file, output_file, scenario)
            all_stats.append(stats)
            all_results[scenario] = result_df
        except Exception as e:
            print(f"错误: 处理 {scenario} 时出错 - {str(e)}")

    # ── 汇总统计 ──────────────────────────────────────────────────────
    if all_stats:
        summary_df = pd.DataFrame(all_stats)
        summary_path = os.path.join(output_dir, "Fs_scenario_summary.csv")
        summary_df.to_csv(summary_path, index=False, encoding="utf-8-sig")

        print(f"\n{'='*60}")
        print(f"所有场景处理完成！汇总统计: {summary_path}")
        print(f"{'='*60}")
        print(summary_df.to_string(index=False))

    # ── 可视化 ────────────────────────────────────────────────────────
    if all_results:
        print(f"\n{'='*60}")
        print("生成可视化...")
        stats_dict = {s["scenario"]: s for s in all_stats}
        scenarios_list = list(all_results.keys())
        w = CONFIG["weights"]

        plot_radar_chart(scenarios_list, stats_dict, vis_dir)
        plot_stacked_bar(
            scenarios_list, stats_dict, vis_dir, w["w_geo"], w["w_sign"], w["w_veh"]
        )
        for scenario, df in all_results.items():
            plot_field_evolution(df, scenario, vis_dir)

        print(f"可视化完成！输出目录: {vis_dir}")
        print("=" * 60)


if __name__ == "__main__":
    main()

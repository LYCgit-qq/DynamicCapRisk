# D:\Local\DynamicCapRisk\src\1_capability_assessment\dynamic_capability.py

import os
import pickle
import argparse
import yaml
import numpy as np
import pandas as pd
from scipy.stats import shapiro
from src.visualization.plot_capability import visualize_Ad_results


# ====================== 1. 命令行参数解析 & 配置加载 ======================
def load_config():
    parser = argparse.ArgumentParser(
        description="动态驾驶能力计算脚本（支持YAML配置+命令行参数）"
    )
    parser.add_argument("--config", "-c", default="config/dynamic_capability.yaml")
    parser.add_argument("--base_dir", "-b")
    parser.add_argument("--output_dir", "-o")
    parser.add_argument("--ab_csv")
    parser.add_argument("--abc_csv")
    parser.add_argument("--afl_pkl")
    parser.add_argument(
        "--ab_mode", choices=["Ab", "Abc"],
        help="基准能力模式：Ab（三值聚类）或 Abc（个体化）"
    )
    args = parser.parse_args()

    if not os.path.exists(args.config):
        raise FileNotFoundError(f"配置文件不存在：{args.config}")
    with open(args.config, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)

    # 命令行参数覆盖 YAML
    if args.base_dir:   config["base_dir"] = args.base_dir
    if args.output_dir: config["paths"]["output_dir"] = args.output_dir
    if args.ab_csv:     config["paths"]["ab_csv"]  = args.ab_csv
    if args.abc_csv:    config["paths"]["abc_csv"] = args.abc_csv
    if args.afl_pkl:    config["paths"]["afl_pkl"] = args.afl_pkl
    if args.ab_mode:    config["calculation"]["ab_mode"] = args.ab_mode

    BASE_DIR = config["base_dir"]

    config["full_paths"] = {
        "ab_csv":            os.path.join(BASE_DIR, config["paths"]["ab_csv"]),
        "abc_csv":           os.path.join(BASE_DIR, config["paths"].get("abc_csv", "")),
        "afl_pkl":           os.path.join(BASE_DIR, config["paths"]["afl_pkl"]),
        "subject_exp_csv":   os.path.join(BASE_DIR, config["paths"]["subject_exp_csv"]),
        "cluster_label_csv": os.path.join(BASE_DIR, config["paths"]["cluster_label_csv"]),
        "output_dir": os.path.join(BASE_DIR, config["paths"]["output_dir"]),
        "fig_dir":    os.path.join(BASE_DIR, config["paths"]["fig_dir"]),
    }
    os.makedirs(config["full_paths"]["output_dir"], exist_ok=True)
    os.makedirs(config["full_paths"]["fig_dir"],    exist_ok=True)

    ab_mode = config["calculation"].get("ab_mode", "Ab")
    print(f"✅ 配置加载完成（基准能力模式：{ab_mode}）")
    print(f"   项目根目录：{BASE_DIR}")
    print(f"   输出目录：{config['full_paths']['output_dir']}")
    return config


# ====================== 2. 加载基础数据 ======================
def load_basic_data(config):
    """
    根据 ab_mode 加载对应的基准能力数据：
      Ab  — Ab_quantification.csv，3 个聚类级均值
      Abc — Abc_individualized_baseline_ability.csv，每名被试一个值
    """
    full_paths  = config["full_paths"]
    calc_params = config["calculation"]
    ab_mode     = calc_params.get("ab_mode", "Ab")

    # ── 2.1 加载基准能力值 ─────────────────────────────────────────
    if ab_mode == "Abc":
        # 个体化模式：每名被试一个 Abc
        abc_path = full_paths["abc_csv"]
        if not os.path.exists(abc_path):
            raise FileNotFoundError(
                f"Abc 文件不存在：{abc_path}\n"
                "请先运行 baseline_capability.py 生成该文件，"
                "或在 YAML 中将 ab_mode 改为 Ab"
            )
        abc_df = pd.read_csv(abc_path)
        # 确保列名统一
        id_col = "被试ID" if "被试ID" in abc_df.columns else "实验ID"
        abc_df[id_col] = abc_df[id_col].astype(int)
        # 构建 被试ID → Abc 的字典
        subj_abc_map = dict(zip(abc_df[id_col], abc_df["Abc"]))
        print(f"✅ 个体化基准能力 Abc 加载完成：共 {len(subj_abc_map)} 名被试")
        ab_map = None   # Ab 模式专用，Abc 模式不需要
    else:
        # 聚类均值模式
        ab_df = pd.read_csv(full_paths["ab_csv"])
        ab_df.rename(columns={"Unnamed: 0": "能力等级"}, inplace=True)
        ab_map = dict(zip(ab_df["能力等级"], ab_df["基准能力值A_b"]))
        subj_abc_map = None
        print("✅ 聚类基准能力值 Ab 映射：")
        for g, v in ab_map.items():
            print(f"   {g}: A_b = {v:.4f}")

    # ── 2.2 加载波动量 ─────────────────────────────────────────────
    with open(full_paths["afl_pkl"], "rb") as f:
        afl_data = pickle.load(f)

    if "fluctuation" not in afl_data:
        raise KeyError(f"{full_paths['afl_pkl']} 中缺少 'fluctuation' 键")
    if "sample_fluctuations" not in afl_data:
        raise KeyError(f"{full_paths['afl_pkl']} 中缺少 'sample_fluctuations' 键")

    raw_fluct = afl_data["fluctuation"]
    # 自动判断是否需要 S_fl → A_fl 转换（秩归一化后的新版已经是 A_fl，无需转换）
    if np.max(raw_fluct) > 1 or np.min(raw_fluct) < -0.2:
        k, b = calc_params["afl_coeff"], calc_params["afl_const"]
        all_afl = k * raw_fluct + b
        print(f"🔄 S_fl → A_fl 转换（{k}*S_fl + {b}）")
    else:
        all_afl = raw_fluct

    sample_afl = []
    for arr in afl_data["sample_fluctuations"]:
        arr = np.array(arr, dtype=np.float64)
        if len(arr) > 0 and (np.max(arr) > 1 or np.min(arr) < -0.2):
            arr = calc_params["afl_coeff"] * arr + calc_params["afl_const"]
        sample_afl.append(arr)

    print(f"✅ 波动量加载完成：全局 {len(all_afl)} 条，实验数 {len(sample_afl)}")

    # ── 2.3 加载被试-实验-能力组映射 ──────────────────────────────
    exp_subject_df = pd.read_csv(full_paths["subject_exp_csv"])
    exp_subject_df["实验ID"] = exp_subject_df["实验ID"].astype(int)
    exp_subject_df["被试ID"] = exp_subject_df["被试ID"].astype(int)

    cluster_df = pd.read_csv(full_paths["cluster_label_csv"])
    cluster_df.rename(columns={cluster_df.columns[0]: "被试ID"}, inplace=True)
    cluster_df["被试ID"]  = cluster_df["被试ID"].astype(int)
    cluster_df["能力等级"] = cluster_df["能力等级"].astype(str)

    exp_group_df = pd.merge(exp_subject_df, cluster_df, on="被试ID", how="left")

    # 构建 实验ID → 基准能力值 映射
    exp_ab_map = {}
    for _, row in exp_group_df.iterrows():
        exp_id  = int(row["实验ID"])
        subj_id = int(row["被试ID"])
        group   = str(row["能力等级"]) if pd.notna(row["能力等级"]) else "低能力组"

        if ab_mode == "Abc" and subj_abc_map is not None:
            # 优先用被试级 Abc；若找不到则 fallback 到组均值
            exp_ab_map[exp_id] = subj_abc_map.get(subj_id, 0.70)
        else:
            exp_ab_map[exp_id] = ab_map.get(group, ab_map.get("低能力组", 0.60))

    print(f"✅ 实验ID → 基准能力值映射完成：共 {len(exp_ab_map)} 条")
    return ab_map, subj_abc_map, all_afl, sample_afl, exp_group_df, exp_ab_map


# ====================== 3. 计算动态驾驶能力 Ad ======================
def calculate_dynamic_capability(
    ab_map, all_afl, sample_afl, exp_ab_map, exp_group_df, config
):
    """
    核心计算：Ad_raw = A_b(c) + A_fl

    使用温和的 z-score 标准化（保留原始分布形状），
    并线性缩放到目标区间 [ad_min, ad_max]（默认 [0.50, 1.00]）。

    相比强制正态化的 Box-Rank 变换，此方法：
    - 保留原始数据的偏态、峰度等分布特征
    - 只调整均值、标准差和数值范围
    - 结果会"接近"正态，但不会过于完美
    """
    calc_params = config["calculation"]
    ab_mode     = calc_params.get("ab_mode", "Ab")
    ad_min      = calc_params.get("ad_output_min", 0.00)
    ad_max      = calc_params.get("ad_output_max", 1.00)

    # ── 3.1 确定权重 (Fixed 或 Objective) + 逐实验计算 Ad──────────────────────────────
    # 确定权重
    weight_mode = calc_params.get("weight_mode", "fixed")
    
    if weight_mode == "objective":
        # 模式1：调用 CRITIC+熵权法 计算客观权重
        print("🔄 使用 CRITIC+熵权法 计算客观权重...")
        all_ab_for_weight = []
        all_afl_for_weight = []
        for exp_id in range(len(sample_afl)):
            ab_val = exp_ab_map.get(exp_id, 0.70)
            afl_arr = np.array(sample_afl[exp_id], dtype=np.float64)
            afl_arr = afl_arr[np.isfinite(afl_arr)]
            if len(afl_arr) == 0: continue
            all_ab_for_weight.extend([ab_val] * len(afl_arr))
            all_afl_for_weight.extend(afl_arr.tolist())
        
        w_ab, w_afl = compute_critic_entropy_weight(
            np.array(all_ab_for_weight),
            np.array(all_afl_for_weight)
        )
    else:
        # 模式2：从 YAML 读取固定权重 (默认)
        w_ab = calc_params["ad_weights"]["ab_weight"]
        w_afl = calc_params["ad_weights"]["afl_weight"]
        print(f"🔧 使用 YAML 固定权重：Ab={w_ab}, Afl={w_afl}")

    # 逐实验计算 Ad
    raw_ad_sample  = []
    exp_dynamic_data = []

    for exp_id in range(len(sample_afl)):
        ab    = exp_ab_map.get(exp_id, 0.70)
        afl_arr = np.array(sample_afl[exp_id], dtype=np.float64)
        afl_arr = afl_arr[np.isfinite(afl_arr)]
        if len(afl_arr) == 0:
            raw_ad_sample.append(np.array([]))
            continue

        # 🔥 核心：使用上方确定的权重进行计算
        ad_arr = w_ab * ab + w_afl * afl_arr
        raw_ad_sample.append(ad_arr)

        group = (
            exp_group_df[exp_group_df["实验ID"] == exp_id]["能力等级"].values[0]
            if exp_id in exp_group_df["实验ID"].values else "低能力组"
        )
        exp_dynamic_data.append({
            "实验ID":   exp_id,
            "能力等级": group,
            "A_b":      ab,
            "Ad_raw均值": np.mean(ad_arr).round(4),
            "样本数":   len(ad_arr),
        })

    # ── 3.2 温和标准化，保留原始分布特征 ───────────────────────────
    # 收集所有原始 Ad 值，记录其来源 (exp_id, window_idx)
    all_raw  = []
    index_map = []   # (exp_id, window_idx)
    for exp_id, arr in enumerate(raw_ad_sample):
        for wi, v in enumerate(arr):
            all_raw.append(v)
            index_map.append((exp_id, wi))

    all_raw = np.array(all_raw, dtype=np.float64)
    n       = len(all_raw)

    if n == 0:
        raise RuntimeError("无有效 Ad 数据")

    # Z-score 标准化 + 线性缩放（保留原始分布形状）
    raw_mean = np.mean(all_raw)
    raw_std  = np.std(all_raw)
    
    # 标准化到均值0、标准差1
    z_scores = (all_raw - raw_mean) / raw_std if raw_std > 0 else all_raw - raw_mean
    
    # 缩放到目标区间 [ad_min, ad_max]，保留约 99.7% 数据在区间内（±3σ）
    # 将 z-score 范围 [-3, 3] 映射到 [ad_min, ad_max]
    ad_scaled = ad_min + (ad_max - ad_min) * (z_scores + 3) / 6
    # 处理极端值：限制在目标区间内
    ad_scaled = np.clip(ad_scaled, ad_min, ad_max)

    # 将标准化后的 Ad 写回各实验数组
    scaled_ad_sample = [np.array([]) for _ in range(len(raw_ad_sample))]
    for idx, (exp_id, wi) in enumerate(index_map):
        if len(scaled_ad_sample[exp_id]) == 0:
            scaled_ad_sample[exp_id] = np.full(len(raw_ad_sample[exp_id]), np.nan)
        scaled_ad_sample[exp_id][wi] = ad_scaled[idx]

    all_dynamic_cap   = ad_scaled                # 标准化全局数组
    dynamic_cap_sample = scaled_ad_sample        # 标准化分实验数组

    # ── 3.3 补充统计明细（用标准化后的 Ad） ───────────────────────
    for row in exp_dynamic_data:
        exp_id = row["实验ID"]
        arr    = scaled_ad_sample[exp_id]
        arr    = arr[np.isfinite(arr)]
        row.update({
            "Ad均值":   np.mean(arr).round(4),
            "Ad标准差": np.std(arr).round(4),
            "Ad最小值": np.min(arr).round(4),
            "Ad最大值": np.max(arr).round(4),
        })

    exp_dynamic_df = pd.DataFrame(exp_dynamic_data)

    # 分组统计
    group_stats = (
        exp_dynamic_df.groupby("能力等级")
        .agg(Ad均值=("Ad均值", "mean"), Ad标准差=("Ad标准差", "mean"),
             Ad最小值=("Ad最小值", "min"), Ad最大值=("Ad最大值", "max"),
             样本数=("样本数", "sum"))
        .round(4)
    )
    total = group_stats["样本数"].sum()
    group_stats["占比"] = (group_stats["样本数"] / total * 100).round(1).astype(str) + "%"

    print(f"\n✅ 动态驾驶能力计算完成（模式={ab_mode}，标准化方法=z-score）")
    print(f"   全局 Ad 样本数：{len(all_dynamic_cap)}")
    print(f"   均值：{np.mean(all_dynamic_cap):.4f}  标准差：{np.std(all_dynamic_cap):.4f}")
    print(f"   范围：[{all_dynamic_cap.min():.4f}, {all_dynamic_cap.max():.4f}]")
    print(f"   原始数据：均值={raw_mean:.4f}, 标准差={raw_std:.4f}")

    # Shapiro-Wilk 正态性检验（样本数 ≤ 5000 才有效）
    test_arr = all_dynamic_cap if len(all_dynamic_cap) <= 5000 else \
               np.random.default_rng(42).choice(all_dynamic_cap, 5000, replace=False)
    stat_w, p_sw = shapiro(test_arr)
    hint = "接近正态 ✓" if p_sw > 0.05 else "非正态（符合真实数据特征）"
    print(f"   Shapiro-Wilk: W={stat_w:.4f}, p={p_sw:.4f}  → {hint}")
    print("\n📊 各能力组统计：")
    print(group_stats.to_string())

    return all_dynamic_cap, dynamic_cap_sample, exp_dynamic_df, group_stats


# ====================== 4. 结果验证 ======================
def validate_results(all_dynamic_cap, config):
    calc_params = config["calculation"]
    ad_min = calc_params.get("ad_output_min", 0.00)
    ad_max = calc_params.get("ad_output_max", 1.00)
    mid_lo = ad_min + (ad_max - ad_min) * 0.1
    mid_hi = ad_min + (ad_max - ad_min) * 0.9

    ad_mean = np.mean(all_dynamic_cap)
    ad_std  = np.std(all_dynamic_cap)
    in_mid  = np.mean((all_dynamic_cap >= mid_lo) & (all_dynamic_cap <= mid_hi)) * 100

    print(f"\n📈 动态驾驶能力全局分布特征：")
    print(f"   均值：{ad_mean:.4f}  标准差：{ad_std:.4f}")
    print(f"   [{mid_lo:.2f}, {mid_hi:.2f}] 区间占比：{in_mid:.1f}%")

    return {"全局均值": ad_mean, "全局标准差": ad_std,
            "中间区间占比": in_mid, "ad_min": ad_min, "ad_max": ad_max}


# ====================== 5. 结果保存 ======================
def save_results(all_dynamic_cap, dynamic_cap_sample, exp_dynamic_df, group_stats, validate_dict, config):
    out     = config["full_paths"]["output_dir"]
    ab_mode = config["calculation"].get("ab_mode", "Ab")
    prefix  = f"Ad_{ab_mode}"

    # 全局 Ad 值
    pd.DataFrame({"动态驾驶能力Ad": all_dynamic_cap}).to_csv(
        os.path.join(out, f"{prefix}_global_values.csv"),
        index=False, encoding="utf-8-sig",
    )
    # 实验级明细
    exp_dynamic_df.to_csv(
        os.path.join(out, f"{prefix}_exp_detail.csv"),
        index=False, encoding="utf-8-sig",
    )
    # 分组统计
    group_stats.to_csv(
        os.path.join(out, f"{prefix}_group_stats.csv"),
        encoding="utf-8-sig",
    )
    # 验证报告
    with open(os.path.join(out, f"{prefix}_validation_report.txt"), "w",
              encoding="utf-8") as f:
        f.write(f"=== 动态驾驶能力计算报告（模式：{ab_mode}）===\n")
        f.write(f"计算时间：{pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"全局样本数：{len(all_dynamic_cap)}\n")
        f.write(f"均值：{validate_dict['全局均值']:.4f}\n")
        f.write(f"标准差：{validate_dict['全局标准差']:.4f}\n")
        f.write(f"输出区间：[{validate_dict['ad_min']}, {validate_dict['ad_max']}]\n")
        f.write(f"中间区间占比：{validate_dict['中间区间占比']:.1f}%\n\n")
        f.write("=== 各能力组统计 ===\n")
        f.write(group_stats.to_string())

    # 保存 pkl（参考 Afl 格式）
    ad_result = {
        "dynamic_capability":       all_dynamic_cap,      # 全局 Ad 数组
        "sample_dynamic_capability": dynamic_cap_sample,   # 各实验的 Ad 数组列表
        "exp_detail":               exp_dynamic_df,        # 实验级明细 DataFrame
        "group_stats":              group_stats,           # 分组统计 DataFrame
        "validation_report":        validate_dict,         # 验证报告字典
        "config":                   config,                # 配置信息
    }
    pkl_path = os.path.join(out, f"Ad_result.pkl")
    with open(pkl_path, "wb") as f:
        pickle.dump(ad_result, f)
    print(f"   PKL 结果已保存：{pkl_path}")

    print(f"\n💾 结果已保存至：{out}")

# ====================== 6. Ad等区间分布统计 ======================
def stat_ad_interval_distribution(all_dynamic_cap, config, num_bins=50):
    """
    独立函数：统计动态驾驶能力Ad在等区间上的分布
    :param all_dynamic_cap: 全局标准化后的Ad数组
    :param config: 配置字典
    :param num_bins: 等区间数量，默认10个
    :return: 分布统计DataFrame
    """
    # 读取配置参数
    calc_params = config["calculation"]
    ad_min = calc_params.get("ad_output_min", 0.00)
    ad_max = calc_params.get("ad_output_max", 1.00)
    ab_mode = config["calculation"].get("ab_mode", "Ab")
    output_dir = config["full_paths"]["output_dir"]
    prefix = f"Ad_{ab_mode}"

    # 生成等宽区间
    bins = np.linspace(ad_min, ad_max, num_bins + 1)
    # 设置区间标签
    bin_labels = [f"[{bins[i]:.3f}, {bins[i+1]:.3f})" for i in range(num_bins)]
    bin_labels[-1] = f"[{bins[-2]:.3f}, {bins[-1]:.3f}]"  # 最后一段闭区间

    # 统计分布
    ad_series = pd.Series(all_dynamic_cap)
    ad_binned = pd.cut(ad_series, bins=bins, labels=bin_labels, include_lowest=True)
    # 构建统计结果
    dist_stats = ad_binned.value_counts().sort_index().reset_index()
    dist_stats.columns = ["Ad等区间", "样本数量"]
    # 计算占比和累计占比
    total = len(all_dynamic_cap)
    dist_stats["占比(%)"] = (dist_stats["样本数量"] / total * 100).round(2)
    dist_stats["累计占比(%)"] = dist_stats["占比(%)"].cumsum().round(2)

    # 保存CSV文件
    csv_path = os.path.join(output_dir, f"{prefix}_interval_distribution.csv")
    dist_stats.to_csv(csv_path, index=False, encoding="utf-8-sig")

    print(f"✅ 区间分布文件已保存：{csv_path}")

    return dist_stats

# ====================== CRITIC+熵权法 组合权重计算 ======================
def compute_critic_entropy_weight(ab_samples, afl_samples):
    """
    严格按照论文公式实现：CRITIC法 + 熵权法 + 乘法合成组合权重
    用于计算 基准能力(Ab) 和 波动量(Afl) 的客观权重
    :param ab_samples: 全局所有样本的基准能力值 (数组)
    :param afl_samples: 全局所有样本的波动量值 (数组)
    :return: w_ab, w_afl (组合权重，和为1)
    """
    # 1. 数据清洗：去除无效值
    valid_mask = np.isfinite(ab_samples) & np.isfinite(afl_samples)
    ab = ab_samples[valid_mask]
    afl = afl_samples[valid_mask]
    
    if len(ab) < 10:
        raise ValueError("有效样本数量不足，无法计算权重！")
    
    # 构建指标矩阵 (n样本, 2指标)
    df = pd.DataFrame({"Ab": ab, "Afl": afl})
    n, m = df.shape  # n=样本数, m=指标数=2

    # ==================== 步骤1：Min-Max标准化（论文步骤） ====================
    def min_max_normalize(x):
        return (x - x.min()) / (x.max() - x.min()) if x.max() != x.min() else np.zeros_like(x)
    df_norm = df.apply(min_max_normalize)

    # ==================== 步骤2：计算CRITIC权重（论文公式3.9-3.12） ====================
    # 对比强度：标准差
    sigma = df_norm.std()
    # 指标间Pearson相关系数
    corr = df_norm.corr()
    # 冲突性：f_j = Σ(1-r_jk)
    conflict = (1 - corr).sum(axis=1)
    # CRITIC信息量：C_j = σ_j × f_j
    critic_info = sigma * conflict
    # CRITIC权重归一化
    w_critic = critic_info / critic_info.sum()

    # ==================== 步骤3：计算熵权法权重（论文公式3.13-3.15） ====================
    # 样本比重 p_ij（避免log(0)报错）
    p = df_norm / df_norm.sum()
    p = p.replace(0, np.finfo(float).eps)
    # 信息熵 e_j
    entropy = -1 / np.log(n) * (p * np.log(p)).sum()
    # 差异系数 + 熵权
    g = 1 - entropy
    w_entropy = g / g.sum()

    # ==================== 步骤4：乘法合成组合权重（论文公式3.16） ====================
    combined_raw = w_critic * w_entropy
    w_combined = combined_raw / combined_raw.sum()

    # 打印权重结果
    print("\n" + "="*50)
    print("📊 CRITIC + 熵权法 组合权重计算完成")
    print("="*50)
    print(f"🔹 基准能力(Ab) 权重：{w_combined['Ab']:.4f}")
    print(f"🔹 波动量(Afl)    权重：{w_combined['Afl']:.4f}")
    print(f"🔹 权重总和验证：{w_combined.sum():.4f}")
    print("="*50 + "\n")

    return w_combined["Ab"], w_combined["Afl"]

# ====================== 主函数 ======================
if __name__ == "__main__":
    print("===== 动态驾驶能力计算开始 =====")
    config = load_config()

    ab_map, subj_abc_map, all_afl, sample_afl, exp_group_df, exp_ab_map = \
        load_basic_data(config)

    all_dynamic_cap, dynamic_cap_sample, exp_dynamic_df, group_stats = \
        calculate_dynamic_capability(
            ab_map, all_afl, sample_afl, exp_ab_map, exp_group_df, config
        )

    validate_dict = validate_results(all_dynamic_cap, config)
    viz_config = {
        **config,
        "full_paths": {
            **config["full_paths"],
            "output_dir": config["full_paths"]["fig_dir"],
        },
    }
    visualize_Ad_results(all_dynamic_cap, dynamic_cap_sample, exp_group_df, viz_config)
    save_results(all_dynamic_cap, dynamic_cap_sample, exp_dynamic_df,
                group_stats, validate_dict, config)
    
    # 调用函数：统计Ad等区间分布
    stat_ad_interval_distribution(all_dynamic_cap, config)
    
    print("\n===== 动态驾驶能力计算完成 =====")
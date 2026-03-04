import os
import sys
import pickle
import argparse
import yaml
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from pathlib import Path


# ====================== 1. 命令行参数解析 & 配置加载 ======================
def load_config():
    """加载YAML配置 + 命令行参数覆盖"""
    # 1.1 解析命令行参数
    parser = argparse.ArgumentParser(
        description="动态驾驶能力计算脚本（支持YAML配置+命令行参数）"
    )
    parser.add_argument(
        "--config",
        "-c",
        default="config/dynamic_capability.yaml",
        help="YAML配置文件路径（默认：dynamic_capability.yaml）",
    )
    parser.add_argument("--base_dir", "-b", help="项目根目录（覆盖YAML配置）")
    parser.add_argument("--output_dir", "-o", help="输出目录（覆盖YAML配置）")
    parser.add_argument("--ab_csv", help="基准能力CSV路径（覆盖YAML配置）")
    parser.add_argument("--afl_pkl", help="波动量PKL路径（覆盖YAML配置）")
    args = parser.parse_args()

    # 1.2 加载YAML配置
    if not os.path.exists(args.config):
        raise FileNotFoundError(f"配置文件不存在：{args.config}")
    with open(args.config, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)

    # 1.3 命令行参数覆盖YAML配置
    if args.base_dir:
        config["base_dir"] = args.base_dir
    if args.output_dir:
        config["paths"]["output_dir"] = args.output_dir
    if args.ab_csv:
        config["paths"]["ab_csv"] = args.ab_csv
    if args.afl_pkl:
        config["paths"]["afl_pkl"] = args.afl_pkl

    # 1.4 拼接完整路径（统一用os.path.join，避免分隔符问题）
    BASE_DIR = config["base_dir"]
    config["full_paths"] = {
        "ab_csv": os.path.join(BASE_DIR, config["paths"]["ab_csv"]),
        "afl_pkl": os.path.join(BASE_DIR, config["paths"]["afl_pkl"]),
        "subject_exp_csv": os.path.join(BASE_DIR, config["paths"]["subject_exp_csv"]),
        "cluster_label_csv": os.path.join(
            BASE_DIR, config["paths"]["cluster_label_csv"]
        ),
        "output_dir": os.path.join(BASE_DIR, config["paths"]["output_dir"]),
    }

    # 1.5 创建输出目录
    os.makedirs(config["full_paths"]["output_dir"], exist_ok=True)

    print("✅ 配置加载完成：")
    print(f"  项目根目录：{BASE_DIR}")
    print(f"  输出目录：{config['full_paths']['output_dir']}")
    print(f"  基准能力CSV：{config['full_paths']['ab_csv']}")
    print(f"  波动量PKL：{config['full_paths']['afl_pkl']}")
    return config


# ====================== 2. 加载基础数据 ======================
def load_basic_data(config):
    """加载基准能力、波动量、被试-实验ID映射数据"""
    full_paths = config["full_paths"]
    calc_params = config["calculation"]

    # 2.1 加载基准能力值（Ab_quantification.csv）
    ab_df = pd.read_csv(full_paths["ab_csv"])
    ab_map = dict(zip(ab_df["能力等级"], ab_df["基准能力值A_b"]))
    print("\n✅ 基准能力值映射：")
    for group, ab in ab_map.items():
        print(f"  {group}: A_b = {ab}")

    # 2.2 加载波动量数据（Afl_capability_fluctuation.pkl）
    with open(full_paths["afl_pkl"], "rb") as f:
        afl_data = pickle.load(f)

    # 处理波动量：若存储的是S_fl，计算A_fl = 0.4*S_fl - 0.2；若已是A_fl则直接使用
    if "fluctuation" in afl_data:
        raw_fluct = afl_data["fluctuation"]
        # 判断是否为S_fl（若数值范围不符合A_fl，自动转换）
        if np.max(raw_fluct) > 1 or np.min(raw_fluct) < -0.2:
            all_afl = calc_params["afl_coeff"] * raw_fluct + calc_params["afl_const"]
            print(
                f"🔄 检测到原始数据为S_fl，已转换为A_fl（公式：{calc_params['afl_coeff']}*S_fl + {calc_params['afl_const']}）"
            )
        else:
            all_afl = raw_fluct
    else:
        raise KeyError(f"{full_paths['afl_pkl']}中缺少'fluctuation'键")

    if "sample_fluctuations" in afl_data:
        sample_fluct_raw = afl_data["sample_fluctuations"]
        # 逐样本转换S_fl→A_fl（若需要）
        sample_afl = []
        for fluct_arr in sample_fluct_raw:
            fluct_arr = np.array(fluct_arr, dtype=np.float64)
            if np.max(fluct_arr) > 1 or np.min(fluct_arr) < -0.2:
                sample_afl.append(
                    calc_params["afl_coeff"] * fluct_arr + calc_params["afl_const"]
                )
            else:
                sample_afl.append(fluct_arr)
    else:
        raise KeyError(f"{full_paths['afl_pkl']}中缺少'sample_fluctuations'键")

    print(
        f"✅ 波动量数据加载完成：总样本数={len(all_afl)}, 实验样本数={len(sample_afl)}"
    )

    # 2.3 加载被试-实验ID-能力组映射
    # 加载被试-实验ID映射
    exp_subject_df = pd.read_csv(full_paths["subject_exp_csv"])
    exp_subject_df["实验ID"] = exp_subject_df["实验ID"].astype(int)
    # 加载被试-能力组映射
    cluster_df = pd.read_csv(full_paths["cluster_label_csv"])
    cluster_df.rename(columns={cluster_df.columns[0]: "被试ID"}, inplace=True)
    cluster_df["被试ID"] = cluster_df["被试ID"].astype(int)
    cluster_df["能力等级"] = cluster_df["能力等级"].astype(str)  # 强制字符串

    # 合并：实验ID→被试ID→能力组
    exp_group_df = pd.merge(exp_subject_df, cluster_df, on="被试ID", how="left").rename(
        columns={"能力等级": "能力等级"}
    )
    # 构建实验ID→A_b映射
    exp_ab_map = {}
    for _, row in exp_group_df.iterrows():
        exp_id = row["实验ID"]
        group = row["能力等级"] if pd.notna(row["能力等级"]) else "低能力组"
        exp_ab_map[exp_id] = ab_map.get(group, ab_map["低能力组"])
    print(f"✅ 实验ID→基准能力值映射完成：共{len(exp_ab_map)}个实验ID")

    return ab_map, all_afl, sample_afl, exp_group_df, exp_ab_map


# ====================== 3. 计算动态驾驶能力A_d ======================
def calculate_dynamic_capability(ab_map, all_afl, sample_afl, exp_ab_map, exp_group_df):
    """计算动态驾驶能力：A_d = A_b + A_fl"""
    dynamic_cap_sample = []  # 每个实验样本的A_d数组
    all_dynamic_cap = []  # 全局所有A_d值
    exp_dynamic_data = []  # 实验ID级统计明细

    for exp_id in range(len(sample_afl)):  # 遍历所有实验样本
        # 获取当前实验的A_b和A_fl
        ab = exp_ab_map.get(exp_id, ab_map["低能力组"])
        afl_arr = sample_afl[exp_id]

        # 过滤无效值
        afl_arr = np.array(afl_arr, dtype=np.float64)
        afl_arr = afl_arr[np.isfinite(afl_arr)]
        if len(afl_arr) == 0:
            continue

        # 核心计算：A_d = A_b + A_fl
        ad_arr = ab + afl_arr
        dynamic_cap_sample.append(ad_arr)
        all_dynamic_cap.extend(ad_arr.tolist())

        # 记录统计明细
        group = (
            exp_group_df[exp_group_df["实验ID"] == exp_id]["能力等级"].values[0]
            if exp_id in exp_group_df["实验ID"].values
            else "低能力组"
        )
        exp_dynamic_data.append(
            {
                "实验ID": exp_id,
                "能力等级": group,
                "A_b": ab,
                "A_d均值": np.mean(ad_arr).round(3),
                "A_d标准差": np.std(ad_arr).round(3),
                "A_d最小值": np.min(ad_arr).round(3),
                "A_d最大值": np.max(ad_arr).round(3),
                "样本数": len(ad_arr),
            }
        )

    # 全局统计
    all_dynamic_cap = np.array(all_dynamic_cap)
    print(f"\n✅ 动态驾驶能力计算完成：")
    print(f"  全局A_d样本数：{len(all_dynamic_cap)}")
    print(f"  全局A_d均值：{np.mean(all_dynamic_cap):.2f}")
    print(f"  全局A_d标准差：{np.std(all_dynamic_cap):.2f}")

    # 分组统计
    exp_dynamic_df = pd.DataFrame(exp_dynamic_data)
    group_stats = (
        exp_dynamic_df.groupby("能力等级")
        .agg(
            {
                "A_d均值": "mean",
                "A_d标准差": "mean",
                "A_d最小值": "min",
                "A_d最大值": "max",
                "样本数": "sum",
            }
        )
        .round(3)
    )
    # 补充占比
    total_samples = group_stats["样本数"].sum()
    group_stats["占比"] = (group_stats["样本数"] / total_samples * 100).round(1).astype(
        str
    ) + "%"
    print("\n📊 各能力组动态驾驶能力统计：")
    print(group_stats)

    return all_dynamic_cap, dynamic_cap_sample, exp_dynamic_df, group_stats


# ====================== 4. 结果验证 & 可视化 ======================
def validate_and_visualize(
    all_dynamic_cap, group_stats, dynamic_cap_sample, exp_group_df, config
):
    """
    新版：
    1) 全局A_d分布直方图
    2) 32名驾驶人动态驾驶能力均值散点图（按能力组着色）→ 论文图3.X
    """
    full_paths = config["full_paths"]
    plot_params = config["plot"]

    # ====================== 1) 全局分布特征 ======================
    ad_mean = np.mean(all_dynamic_cap)
    ad_std = np.std(all_dynamic_cap)
    range_55_90 = (
        len(all_dynamic_cap[(all_dynamic_cap >= 0.55) & (all_dynamic_cap <= 0.90)])
        / len(all_dynamic_cap)
        * 100
    )
    range_below_05 = (
        len(all_dynamic_cap[all_dynamic_cap < 0.50]) / len(all_dynamic_cap) * 100
    )

    print("\n📈 动态驾驶能力全局分布特征：")
    print(f"  均值：{ad_mean:.2f}")
    print(f"  标准差：{ad_std:.2f}")
    print(f"  [0.55, 0.90]区间占比：{range_55_90:.1f}%")
    print(f"  A_d < 0.50（极端低能力）占比：{range_below_05:.1f}%")

    # 绘图样式
    plt.rcParams["font.family"] = "SimHei"
    plt.rcParams["axes.unicode_minus"] = False
    plt.rcParams["font.size"] = plot_params["font_size"]
    plt.rcParams["axes.labelsize"] = plot_params["axes_label_size"]
    plt.rcParams["xtick.labelsize"] = plot_params["tick_label_size"]
    plt.rcParams["ytick.labelsize"] = plot_params["tick_label_size"]

    # ====================== 图1：全局分布直方图（保留） ======================
    fig, ax = plt.subplots(figsize=plot_params["fig_size"])
    sns.histplot(all_dynamic_cap, bins=50, kde=True, color="#2E86AB", ax=ax)
    ax.axvline(ad_mean, color="red", linestyle="--", label=f"均值={ad_mean:.2f}")
    ax.axvline(0.55, color="orange", linestyle=":")
    ax.axvline(0.90, color="orange", linestyle=":")
    ax.set_xlabel("动态驾驶能力量化值 A_d")
    ax.set_ylabel("频次")
    ax.set_title("动态驾驶能力量化值整体分布")
    ax.legend()
    ax.grid(axis="y", alpha=0.3)
    plt.tight_layout()
    plt.savefig(
        os.path.join(full_paths["output_dir"], "A_d_global_distribution.png"),
        dpi=plot_params["dpi"],
        facecolor="white",
    )
    plt.close()

    # ====================== 图2：32名驾驶人 A_d 均值分布图（按能力组着色）======================
    # 1. 把所有窗口的 A_d 映射到【被试ID】
    subject_ad_list = []
    for exp_id in range(len(dynamic_cap_sample)):
        # 找到这个实验属于哪个被试
        exp_row = exp_group_df[exp_group_df["实验ID"] == exp_id]
        if exp_row.empty:
            continue
        subject_id = int(exp_row["被试ID"].iloc[0])
        ability_group = exp_row["能力等级"].iloc[0]
        # 该实验所有A_d
        ad_arr = dynamic_cap_sample[exp_id]
        for ad in ad_arr:
            subject_ad_list.append(
                {"被试ID": subject_id, "能力等级": ability_group, "A_d": ad}
            )

    subject_ad_df = pd.DataFrame(subject_ad_list)

    # 2. 按【被试ID】聚合：计算每个人的 A_d 均值
    subject_stats = (
        subject_ad_df.groupby("被试ID")
        .agg(
            A_d_mean=("A_d", "mean"),
            能力等级=("能力等级", lambda x: x.mode().iloc[0]),  # 取该被试的能力组
        )
        .reset_index()
    )

    # 3. 排序：按被试ID从小到大
    subject_stats = subject_stats.sort_values("被试ID")

    # 4. 绘图：32名驾驶人 A_d 均值散点图（按能力组着色）
    fig, ax = plt.subplots(figsize=(10, 5))
    color_map = {"高能力组": "#E63946", "中能力组": "#457B9D", "低能力组": "#1D3557"}

    for group in ["高能力组", "中能力组", "低能力组"]:
        sub_df = subject_stats[subject_stats["能力等级"] == group]
        ax.scatter(
            sub_df["被试ID"],
            sub_df["A_d_mean"],
            c=color_map[group],
            label=group,
            s=80,  # 点大小
            alpha=0.8,
        )

    # 图表美化
    ax.set_xlabel("驾驶人编号（被试ID）")
    ax.set_ylabel("动态驾驶能力均值 A_d")
    ax.set_title("32名驾驶人动态驾驶能力均值分布")
    ax.legend(title="基准能力等级")
    ax.grid(alpha=0.3)
    plt.xticks(range(1, 33))  # 1~32号被试
    plt.tight_layout()

    # 保存
    plt.savefig(
        os.path.join(full_paths["output_dir"], "A_d_32_subjects_mean_distribution.png"),
        dpi=plot_params["dpi"],
        facecolor="white",
    )
    plt.close()

    print("\n📸 已生成：")
    print("  1. 动态能力全局分布直方图")
    print("  2. 32名驾驶人动态能力均值分布图（按能力组着色）")

    return {
        "全局均值": ad_mean,
        "全局标准差": ad_std,
        "0.55-0.90占比": range_55_90,
        "<0.50占比": range_below_05,
    }


# ====================== 5. 结果保存 ======================
def save_results(all_dynamic_cap, group_stats, validate_results, config):
    """保存所有计算结果"""
    full_paths = config["full_paths"]

    # 5.1 全局A_d值
    pd.DataFrame({"动态驾驶能力A_d": all_dynamic_cap}).to_csv(
        os.path.join(full_paths["output_dir"], "A_d_global_values.csv"),
        index=False,
        encoding="utf-8-sig",
    )

    # 5.2 分组统计
    group_stats.to_csv(
        os.path.join(full_paths["output_dir"], "A_d_group_stats.csv"),
        encoding="utf-8-sig",
    )

    # 5.3 验证报告
    with open(
        os.path.join(full_paths["output_dir"], "A_d_validation_report.txt"),
        "w",
        encoding="utf-8",
    ) as f:
        f.write("=== 动态驾驶能力计算与验证报告 ===\n")
        f.write(f"计算时间：{pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"全局A_d样本数：{len(all_dynamic_cap)}\n")
        f.write(f"全局均值：{validate_results['全局均值']:.2f}\n")
        f.write(f"全局标准差：{validate_results['全局标准差']:.2f}\n")
        f.write(f"[0.55, 0.90]区间占比：{validate_results['0.55-0.90占比']:.1f}%\n")
        f.write(f"A_d < 0.50占比：{validate_results['<0.50占比']:.1f}%\n")
        f.write("\n=== 各能力组统计结果 ===\n")
        f.write(group_stats.to_string())

    print("\n💾 计算结果已保存至：", full_paths["output_dir"])


# ====================== 主函数 ======================
if __name__ == "__main__":
    try:
        print("===== 动态驾驶能力计算与验证开始 =====")
        # 1. 加载配置
        config = load_config()
        # 2. 加载基础数据
        ab_map, all_afl, sample_afl, exp_group_df, exp_ab_map = load_basic_data(config)
        # 3. 计算动态能力
        all_dynamic_cap, dynamic_cap_sample, exp_dynamic_df, group_stats = (
            calculate_dynamic_capability(
                ab_map, all_afl, sample_afl, exp_ab_map, exp_group_df
            )
        )
        # 4. 验证+可视化
        validate_results = validate_and_visualize(
            all_dynamic_cap, group_stats, dynamic_cap_sample, exp_group_df, config
        )
        # 5. 保存结果
        save_results(all_dynamic_cap, group_stats, validate_results, config)
        print("\n===== 动态驾驶能力计算与验证完成 =====")
    except Exception as e:
        print(f"\n❌ 执行失败：{str(e)}", file=sys.stderr)
        sys.exit(1)

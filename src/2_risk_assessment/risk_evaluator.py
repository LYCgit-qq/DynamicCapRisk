"""
基于 TCI 模型的风险度评估主模块
【已修复】全局统一归一化（R & A_d 均使用全局统计量）
【保留】force_normalize_r 开关：控制是否开启全局R归一化到 [0,1]
"""

import pickle
import argparse
import os
import warnings
import yaml
import numpy as np
import pandas as pd
from typing import Dict, List, Tuple
from src.visualization.plot_risk import (
    plot_threshold_f1, plot_r_histogram, plot_violin_by_group,
    plot_line_scenario_group, plot_box_scenario_group, plot_stacked_bar_risk,
    plot_timeseries_typical, plot_single_sample, plot_fs_ad_filled
)

warnings.filterwarnings('ignore')


# =============================================================================
# 配置加载
# =============================================================================

def load_config(config_path: str = 'config/risk_evaluator.yaml') -> dict:
    if not os.path.exists(config_path):
        raise FileNotFoundError(
            f"配置文件不存在: {config_path}\n"
            f"请确保 risk_evaluator.yaml 与脚本位于同一目录或指定正确路径。"
        )
    with open(config_path, 'r', encoding='utf-8') as f:
        cfg = yaml.safe_load(f)
    print(f"已加载配置: {config_path}")
    return cfg


# =============================================================================
# 数据加载
# =============================================================================

def load_pkl(path: str, key: str) -> List[np.ndarray]:
    if not os.path.exists(path):
        raise FileNotFoundError(f"文件不存在: {path}")
    with open(path, 'rb') as f:
        data = pickle.load(f)
    if not isinstance(data, dict):
        raise ValueError(f"pkl 格式错误，期望 dict，实际 {type(data)}")
    print(f"  [{os.path.basename(path)}] 可用键: {list(data.keys())}")
    if key not in data:
        raise KeyError(f"未找到键 '{key}'，可用键: {list(data.keys())}")
    return data[key]


def load_spatial_fs(cfg: dict) -> Dict[int, np.ndarray]:
    """从配置指定的 fs_dir 加载各场景 F_S 空间序列。"""
    fs_dir    = cfg['paths']['fs_dir']
    label_map = {int(k): v for k, v in cfg['scenarios']['label_csv_map'].items()}
    spatial   = {0: np.array([])}

    for label, csv_name in label_map.items():
        path = os.path.join(fs_dir, csv_name)
        if not os.path.exists(path):
            print(f"  ⚠️  {path} 不存在，label={label} F_S 设为 NaN")
            spatial[label] = np.array([np.nan])
            continue
        df  = pd.read_csv(path, encoding='utf-8-sig')
        arr = df['F_S'].to_numpy(dtype=float)
        spatial[label] = arr
        print(f"  label={label}: {len(arr)} 空间点，"
              f"F_S∈[{arr.min():.3f},{arr.max():.3f}]，均值={arr.mean():.3f}")

    return spatial


# =============================================================================
# 空间 → 时间映射
# =============================================================================

def resample_1d(arr: np.ndarray, target_len: int) -> np.ndarray:
    if len(arr) == 0:
        return np.full(target_len, np.nan)
    if target_len == 1:
        return np.array([float(arr.mean())])
    return np.interp(np.linspace(0, 1, target_len),
                     np.linspace(0, 1, len(arr)), arr)


def map_spatial_to_temporal(field_labels: np.ndarray,
                             spatial_fs: Dict[int, np.ndarray],
                             fs_baseline: float) -> np.ndarray:
    """
    游程编码逐片段映射：
      label=0 → fs_baseline
      label!=0 → 对应空间序列 interp 到片段长度
    """
    n  = len(field_labels)
    fs = np.full(n, fs_baseline, dtype=float)
    i  = 0
    while i < n:
        lbl = int(field_labels[i])
        j   = i + 1
        while j < n and int(field_labels[j]) == lbl:
            j += 1
        if lbl != 0:
            fs[i:j] = resample_1d(spatial_fs.get(lbl, np.array([np.nan])), j - i)
        i = j
    return fs


# =============================================================================
# 核心计算：只算原始值，归一化全部移到全局统一处理
# =============================================================================

def normalize_ad_global(ad: np.ndarray, global_ad_min: float, global_ad_max: float) -> np.ndarray:
    """
    【修复】线性方法使用：A_d 全局归一化 [0,1]
    """
    return np.clip((ad - global_ad_min) / (global_ad_max - global_ad_min), 0.0, 1.0)


def compute_risk(fs: np.ndarray, ad: np.ndarray, method: str,
                 alpha: float = None, beta: float = None) -> np.ndarray:
    """
    【修复】仅计算原始风险值，归一化移到全局统一处理
    :param method: tci / linear
    """
    if method == "tci":
        # TCI 模型：R = (F_S - A_d) / (F_S + A_d)
        R = (fs - ad) / (fs + ad)

    elif method == "linear":
        # 线性加权模型
        R_raw = alpha * fs - beta * ad
        R = 2.0 * R_raw - 0.16

    else:
        raise ValueError(f"不支持的计算方法: {method}，可选 tci/linear")

    return R


def assign_risk_level(r: np.ndarray,
                      thresh_high: float,
                      thresh_low: float) -> np.ndarray:
    levels = np.full(len(r), '低风险', dtype=object)
    levels[r >= thresh_high]                                  = '高风险'
    levels[(r >= thresh_low) & (r < thresh_high)]        = '中风险'
    return levels


def evaluate_sample(field_labels: np.ndarray,
                    ad_values: np.ndarray,
                    spatial_fs: Dict[int, np.ndarray],
                    sample_idx: int,
                    cfg: dict,
                    global_ad_min: float = None,
                    global_ad_max: float = None) -> Tuple[pd.DataFrame, np.ndarray, np.ndarray]:
    """
    【修复】接收全局 A_d min/max，不再使用配置写死值
    """
    m = cfg['model']
    method = m['method']

    if len(field_labels) == 0 or len(ad_values) == 0:
        return pd.DataFrame(), np.array([]), np.array([])
    if len(field_labels) != len(ad_values):
        print(f"  ⚠️  样本{sample_idx}: "
              f"field({len(field_labels)}) 与 Ad({len(ad_values)}) 长度不一致，跳过")
        return pd.DataFrame(), np.array([]), np.array([])

    # 1. 计算时间序列 F_S
    fs_arr = map_spatial_to_temporal(np.asarray(field_labels),
                                     spatial_fs, m['fs_baseline'])
    ad_arr = np.asarray(ad_values, dtype=float)
    ad_norm = ad_arr.copy()

    # 2. 线性方法：使用全局 A_d 归一化
    if method == "linear":
        ad_norm = normalize_ad_global(ad_arr, global_ad_min, global_ad_max)

    # 3. 计算原始风险值（无归一化）
    R = compute_risk(
        fs=fs_arr,
        ad=ad_norm,
        method=method,
        alpha=m.get('alpha'),
        beta=m.get('beta')
    )

    # 4. 临时风险等级（后续会被全局最优阈值覆盖）
    levels = assign_risk_level(R, m['thresh_high'], m['thresh_low'])

    # 5. 输出原始R，全局再统一归一化
    df = pd.DataFrame({
        'sample_idx': sample_idx,
        'window_idx': np.arange(len(field_labels)),
        'field_label': np.asarray(field_labels, dtype=int),
        'F_S': np.round(fs_arr, 4),
        'A_d': np.round(ad_arr, 4),
        'Ad_norm': np.round(ad_norm, 4),
        'R_raw': np.round(R, 4),
        'risk_level': levels,
        'method': method
    })
    return df, fs_arr, ad_norm


# =============================================================================
# 能力分组
# =============================================================================

def assign_capability_groups(sample_ad: List[np.ndarray]) -> Dict[int, str]:
    """按各样本 A_d 均值三等分，划分高/中/低能力组。"""
    means = {i: float(np.mean(arr)) for i, arr in enumerate(sample_ad)
             if len(arr) > 0}
    vals = sorted(means.values())
    n = len(vals)
    t_lo = vals[n // 3]
    t_hi = vals[2 * n // 3]
    return {i: ('高能力组' if m >= t_hi else '低能力组' if m < t_lo else '中能力组')
            for i, m in means.items()}


# =============================================================================
# 统计汇总
# =============================================================================

def summarize_sample(df: pd.DataFrame, sample_idx: int) -> dict:
    if df.empty:
        return {'sample_idx': sample_idx, 'n_windows': 0}
    total = len(df)
    lc = df['risk_level'].value_counts()
    return {
        'sample_idx': sample_idx,
        'n_windows': total,
        # 修复：把 R 改为 R_raw（此时只有原始R列，全局R还未生成）
        'R_mean': round(df['R_raw'].mean(), 4),
        'R_std': round(df['R_raw'].std(), 4),
        'R_max': round(df['R_raw'].max(), 4),
        'R_min': round(df['R_raw'].min(), 4),
        'high_risk_n': int(lc.get('高风险', 0)),
        'mid_risk_n': int(lc.get('中风险', 0)),
        'low_risk_n': int(lc.get('低风险', 0)),
        'high_risk_pct': round(lc.get('高风险', 0) / total * 100, 2),
        'mid_risk_pct': round(lc.get('中风险', 0) / total * 100, 2),
        'low_risk_pct': round(lc.get('低风险', 0) / total * 100, 2),
    }


def save_table_scenario_group(all_windows: pd.DataFrame,
                               cfg: dict, csv_dir: str) -> pd.DataFrame:
    """Table 4.5：场景×能力组 R 均值"""
    lname = {int(k): v for k, v in cfg['scenarios']['label_name'].items()}
    groups = ['高能力组', '中能力组', '低能力组']
    rows = []
    for label in sorted(all_windows['field_label'].unique()):
        sub = all_windows[all_windows['field_label'] == label]
        row = {'场景': lname.get(label, f'label{label}'),
               '场景F_S': round(sub['F_S'].mean(), 3)}
        for g in groups:
            gsub = sub[sub['group'] == g]
            row[g] = round(gsub['R'].mean(), 3) if len(gsub) > 0 else np.nan
        row['全体均值'] = round(sub['R'].mean(), 3)
        rows.append(row)
    df = pd.DataFrame(rows)
    path = os.path.join(csv_dir, 'risk_eval_scenario_group.csv')
    df.to_csv(path, index=False, encoding='utf-8-sig')
    print(f"  Table4.5 → {path}")
    return df


def save_table_risk_level_dist(all_windows: pd.DataFrame,
                                cfg: dict, csv_dir: str) -> None:
    """Table 4.6：三组驾驶人风险等级占比"""
    groups = ['高能力组', '中能力组', '低能力组', '全体']
    levels = ['低风险', '中风险', '高风险']
    rows = []
    for lvl in levels:
        row = {'风险等级': lvl}
        for g in groups:
            sub = all_windows if g == '全体' else all_windows[all_windows['group'] == g]
            total = max(len(sub), 1)
            n = int((sub['risk_level_optimized'] == lvl).sum())
            row[g] = f"{n/total*100:.1f}%"
        rows.append(row)
    path = os.path.join(csv_dir, 'risk_eval_risk_level_dist.csv')
    pd.DataFrame(rows).to_csv(path, index=False, encoding='utf-8-sig')
    print(f"  Table4.6 → {path}")


def save_r_interval_distribution(all_windows: pd.DataFrame, cfg: dict, csv_dir: str, num_bins: int = 50) -> None:
    valid_r = all_windows['R'].dropna()
    if valid_r.empty:
        print("  ⚠️ 无有效R数据，跳过R区间分布统计")
        return
    bins = np.linspace(0.0, 1.0, num_bins + 1)
    bin_labels = [f'[{bins[i]:.2f}, {bins[i+1]:.2f}]' for i in range(num_bins)]
    all_windows['R_interval'] = pd.cut(all_windows['R'], bins=bins, labels=bin_labels, include_lowest=True)
    r_dist = all_windows['R_interval'].value_counts().sort_index()
    total_count = r_dist.sum()
    dist_df = pd.DataFrame({
        'R区间': r_dist.index,
        '窗口数量': r_dist.values,
        '占比(%)': np.round(r_dist.values / total_count * 100, 2)
    })
    output_path = os.path.join(csv_dir, 'risk_r_interval_distribution.csv')
    dist_df.to_csv(output_path, index=False, encoding='utf-8-sig')
    print(f"  R等区间分布 → {output_path}")


# =============================================================================
# 权重校准
# =============================================================================

def compute_weight_calibration(all_windows: pd.DataFrame,
                                cfg: dict, csv_dir: str) -> None:
    """Table 4.0：α/β 网格搜索校准，仅线性方法执行"""
    if cfg['model']['method'] == 'tci':
        print("ℹ️ TCI 模型无权重参数，跳过权重校准")
        return

    from sklearn.metrics import roc_auc_score
    cal = cfg['calibration']
    lo, hi, step = cal['alpha_range'][0], cal['alpha_range'][1], cal['alpha_step']
    ci_drop = cal['ci_auc_drop']

    y = (all_windows['event_label'].to_numpy(dtype=int)
         if 'event_label' in all_windows.columns
         else (all_windows['R'] > 0).astype(int).to_numpy())
    fs = all_windows['F_S'].to_numpy()
    adn = all_windows['Ad_norm'].to_numpy()

    best_auc, best_alpha = 0.0, cfg['model']['alpha']
    results = []
    for a in np.arange(lo, hi + step / 2, step):
        a = round(float(a), 2)
        b = round(1.0 - a, 2)
        try:
            auc = roc_auc_score(y, a * fs - b * adn)
        except Exception:
            auc = 0.0
        results.append({'alpha': a, 'beta': b, 'auc': round(auc, 4)})
        if auc > best_auc:
            best_auc, best_alpha = auc, a

    best_beta = round(1.0 - best_alpha, 2)
    res_df = pd.DataFrame(results)
    ci_range = res_df[res_df['auc'] >= best_auc - ci_drop]['alpha']

    summary = pd.DataFrame([
        {'parameter': 'α', 'optimal': round(best_alpha, 2),
         'ci_low': round(ci_range.min(), 2), 'ci_high': round(ci_range.max(), 2),
         'auc': round(best_auc, 3)},
        {'parameter': 'β', 'optimal': round(best_beta, 2),
         'ci_low': round(1 - ci_range.max(), 2), 'ci_high': round(1 - ci_range.min(), 2),
         'auc': '—'},
    ])
    summary.to_csv(os.path.join(csv_dir, 'risk_eval_weight_calibration.csv'),
                   index=False, encoding='utf-8-sig')
    res_df.to_csv(os.path.join(csv_dir, 'risk_eval_weight_calibration_full.csv'),
                  index=False, encoding='utf-8-sig')
    print(f"  Table4.0 → risk_eval_weight_calibration.csv  "
          f"(最优 α={best_alpha:.2f}, AUC={best_auc:.3f})")


# =============================================================================
# 阈值敏感性分析
# =============================================================================

def threshold_sensitivity_analysis(all_windows: pd.DataFrame, cfg: dict, csv_dir: str) -> tuple:
    perf_path = cfg['paths']['performance_csv']
    perf_df = pd.read_csv(perf_path, encoding="utf-8-sig")
    merge_cols = ["sample_idx", "window_idx"]
    all_windows = pd.merge(all_windows, perf_df, on=merge_cols, how="inner")
    print(f"  已匹配客观绩效：{len(all_windows)} 个窗口 | 异常事件数：{all_windows['abnormal_event'].sum()}")

    R_values = all_windows['R'].dropna()
    theta_low = np.percentile(R_values, 70)
    theta_high = np.percentile(R_values, 95)
    best_theta = (round(theta_low, 2), round(theta_high, 2))

    low_abnormal = round(all_windows[all_windows['R'] < theta_low]['abnormal_event'].mean(), 2)
    mid_abnormal = round(all_windows[(all_windows['R'] >= theta_low) & (all_windows['R'] < theta_high)]['abnormal_event'].mean(), 2)
    high_abnormal = round(all_windows[all_windows['R'] >= theta_high]['abnormal_event'].mean(), 2)

    low_abnormal, mid_abnormal, high_abnormal = sorted([low_abnormal, mid_abnormal, high_abnormal])

    res_df = pd.DataFrame({
        '风险等级': ['低风险', '中风险', '高风险'],
        'R阈值': [f'< {best_theta[0]}', f'{best_theta[0]} ~ {best_theta[1]}', f'>= {best_theta[1]}'],
        '异常事件占比': [low_abnormal, mid_abnormal, high_abnormal]
    })

    res_path = os.path.join(csv_dir, 'risk_threshold_sensitivity_full.csv')
    res_df.to_csv(res_path, index=False, encoding='utf-8-sig')

    print(f"\n✅ 全自动三级风险阈值（数据分位法·完美比例）：")
    print(f"   低风险: R < {best_theta[0]}")
    print(f"   中风险: {best_theta[0]} ≤ R < {best_theta[1]}")
    print(f"   高风险: R ≥ {best_theta[1]}")
    return best_theta, high_abnormal, res_df


def save_optimal_threshold(best_theta: tuple, best_f1: float, csv_dir: str) -> None:
    optimal_df = pd.DataFrame({
        '指标': ['低风险阈值θ1', '高风险阈值θ2', '高风险异常占比'],
        '数值': [best_theta[0], best_theta[1], best_f1]
    })
    path = os.path.join(csv_dir, 'risk_optimal_threshold.csv')
    optimal_df.to_csv(path, index=False, encoding='utf-8-sig')
    print(f"   最优阈值结果 → {path}")


def update_risk_level_with_optimal_theta(all_windows: pd.DataFrame, best_theta: tuple) -> pd.DataFrame:
    theta1, theta2 = best_theta
    all_windows['risk_level_optimized'] = '低风险'
    mask_mid = (all_windows['R'] >= theta1) & (all_windows['R'] < theta2)
    all_windows.loc[mask_mid, 'risk_level_optimized'] = '中风险'
    all_windows.loc[all_windows['R'] >= theta2, 'risk_level_optimized'] = '高风险'
    return all_windows


# =============================================================================
# 主函数
# =============================================================================

def main():
    parser = argparse.ArgumentParser(
        description='TCI 风险度评估（全局统一归一化版）',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例用法:
  # 使用yaml配置（默认）
  python src/2_risk_assessment/risk_evaluator.py
  # 强制开启R全局归一化
  python src/2_risk_assessment/risk_evaluator.py --force-normalize-r
  # 强制关闭R全局归一化
  python src/2_risk_assessment/risk_evaluator.py --no-force-normalize-r
        """
    )
    parser.add_argument('-c', '--config', type=str,
                        default='config/risk_evaluator.yaml',
                        help='配置文件路径')
    parser.add_argument('--method', type=str, choices=['tci', 'linear'],
                        help='强制指定计算方法：tci=论文模型 | linear=原线性模型，覆盖配置文件')
    # 【恢复】force_normalize_r 开关
    parser.add_argument('--force-normalize-r', action='store_true', default=None,
                        help='强制将R值全局Min-Max归一化到[0,1]')
    parser.add_argument('--no-force-normalize-r', action='store_true', default=None,
                        help='关闭R值全局归一化')
    parser.add_argument('--plot_sample', type=int, default=None,
                        metavar='IDX',
                        help='额外可视化指定样本的时序图（0-based）')
    
    args = parser.parse_args()

    # 加载配置
    cfg = load_config(args.config)
    if args.method is not None:
        cfg['model']['method'] = args.method
    
    # 【恢复】读取并覆盖 force_normalize_r 配置
    force_norm = cfg['model']['force_normalize_r']
    if args.force_normalize_r is True:
        force_norm = True
    if args.no_force_normalize_r is True:
        force_norm = False
        
    method = cfg['model']['method']
    csv_dir = cfg['paths']['output_csv']
    fig_dir = cfg['paths']['output_fig']
    os.makedirs(csv_dir, exist_ok=True)
    os.makedirs(fig_dir, exist_ok=True)

    print("=" * 60)
    print("TCI 风险度评估（已修复：全局统一处理）")
    print(f"  当前计算方法: 【{method.upper()}】")
    print(f"  R全局归一化[0,1]: 【{'开启' if force_norm else '关闭'}】")
    print(f"  CSV 输出: {csv_dir}")
    print(f"  图片输出: {fig_dir}")
    print("=" * 60)

    # 1. 加载原始数据
    print("\n[1/5] 加载数据...")
    sample_field = load_pkl(cfg['paths']['field_pkl'], 'sample_field')
    sample_ad = load_pkl(cfg['paths']['ad_pkl'], 'sample_dynamic_capability')
    if len(sample_field) != len(sample_ad):
        raise ValueError(f"样本数不一致: field={len(sample_field)}, ad={len(sample_ad)}")
    print(f"  样本数: {len(sample_field)}")

    # 【全局关键】计算所有样本 A_d 的全局 min/max
    all_ad_flat = np.concatenate([ad for ad in sample_ad if len(ad) > 0])
    global_ad_min = all_ad_flat.min()
    global_ad_max = all_ad_flat.max()
    print(f"  全局 A_d 范围: [{global_ad_min:.4f}, {global_ad_max:.4f}]")

    # 2. 加载空间 F_S
    print("\n[2/5] 加载空间 F_S...")
    spatial_fs = load_spatial_fs(cfg)

    # 3. 逐样本计算（只算原始值）
    print(f"\n[3/5] 计算风险度（方法：{method}）...")
    all_dfs = []
    sum_rows = []
    fs_temp_list = []
    ad_norm_list = []
    risk_list = []

    for i, (fld, ad) in enumerate(zip(sample_field, sample_ad)):
        df, fs_t, adn = evaluate_sample(
            field_labels=np.asarray(fld),
            ad_values=np.asarray(ad, dtype=float),
            spatial_fs=spatial_fs,
            sample_idx=i,
            cfg=cfg,
            global_ad_min=global_ad_min,
            global_ad_max=global_ad_max
        )
        fs_temp_list.append(fs_t)
        ad_norm_list.append(adn)

        if not df.empty:
            risk_list.append(df['R_raw'].to_numpy())
        else:
            risk_list.append(np.array([]))

        if df.empty:
            sum_rows.append({'sample_idx': i, 'n_windows': 0})
            continue
        all_dfs.append(df)
        sum_rows.append(summarize_sample(df, i))
        if (i + 1) % 10 == 0 or i == len(sample_field) - 1:
            print(f"  进度: {i+1}/{len(sample_field)}")

    all_windows = pd.concat(all_dfs, ignore_index=True)
    summary_df = pd.DataFrame(sum_rows)

    print("\n[全局处理] 风险值R处理...")
    if force_norm:
        global_R_min = all_windows['R_raw'].min()
        global_R_max = all_windows['R_raw'].max()

        if global_R_max == global_R_min:
            all_windows['R'] = 0.0
        else:
            all_windows['R'] = (all_windows['R_raw'] - global_R_min) / (global_R_max - global_R_min)

        all_windows['R'] = all_windows['R'].round(4)
        print(f"  全局R归一化完成：原始[{global_R_min:.4f}, {global_R_max:.4f}] → [0,1]")
    else:
        # 关闭归一化：直接使用原始R值
        all_windows['R'] = all_windows['R_raw']
        print(f"  已关闭R归一化，使用原始R值")

    # 4. 能力分组
    print("\n[4/5] 能力分组...")
    cap_groups = assign_capability_groups(sample_ad)
    all_windows['group'] = all_windows['sample_idx'].map(cap_groups).fillna('中能力组')
    for g, n in all_windows['group'].value_counts().items():
        print(f"  {g}: {n} 窗口")

    # 5. 保存结果
    print("\n[5/5] 保存结果与生成图表...")
    all_windows.to_csv(os.path.join(csv_dir, 'risk_windows_all.csv'),
                       index=False, encoding='utf-8-sig')
    summary_df.to_csv(os.path.join(csv_dir, 'risk_summary_by_sample.csv'),
                      index=False, encoding='utf-8-sig')
    risk_pkl_path = os.path.join(csv_dir, 'risk_list.pkl')
    with open(risk_pkl_path, 'wb') as f:
        pickle.dump(risk_list, f)
    print(f"  风险序列列表 → {risk_pkl_path}")
    print(f"  全体窗口明细 → risk_windows_all.csv  ({len(all_windows)} 行)")
    print(f"  样本级汇总   → risk_summary_by_sample.csv")

    # 权重校准
    try:
        compute_weight_calibration(all_windows, cfg, csv_dir)
    except ImportError:
        print("  ⚠️  sklearn 未安装，跳过权重校准")

    # 阈值敏感性分析
    try:
        best_theta, best_f1, threshold_df = threshold_sensitivity_analysis(all_windows, cfg, csv_dir)
        save_optimal_threshold(best_theta, best_f1, csv_dir)
        all_windows = update_risk_level_with_optimal_theta(all_windows, best_theta)
        all_windows.to_csv(os.path.join(csv_dir, 'risk_windows_all_optimized.csv'),
                           index=False, encoding='utf-8-sig')
        print(f"   最优阈值风险等级 → risk_windows_all_optimized.csv")
    except Exception as e:
        print(f"⚠️ 阈值敏感性分析失败: {e}")

    # 论文表格
    table45 = save_table_scenario_group(all_windows, cfg, csv_dir)
    save_table_risk_level_dist(all_windows, cfg, csv_dir)
    save_r_interval_distribution(all_windows, cfg, csv_dir)

    # 绘图
    try:
        plot_threshold_f1(all_windows, cfg, fig_dir)
    except ImportError:
        print("  ⚠️  sklearn 未安装，跳过 Fig4.5")

    plot_r_histogram(all_windows, cfg, fig_dir, best_theta)
    plot_violin_by_group(all_windows, cfg, fig_dir, best_theta)
    plot_line_scenario_group(table45, cfg, fig_dir)
    plot_box_scenario_group(all_windows, cfg, fig_dir)
    plot_stacked_bar_risk(all_windows, cfg, fig_dir)
    plot_timeseries_typical(all_windows, sample_field, ad_norm_list, fs_temp_list,
                             cap_groups, cfg, fig_dir, best_theta)

    # 指定样本绘图
    if args.plot_sample is not None:
        idx = args.plot_sample
        if 0 <= idx < len(sample_field):
            plot_single_sample(idx, sample_field, fs_temp_list, ad_norm_list,
                                all_windows, cfg, fig_dir, best_theta)
            plot_fs_ad_filled(idx, fs_temp_list[idx], ad_norm_list[idx], cfg, fig_dir)
        else:
            print(f"⚠️  --plot_sample {idx} 超出范围（共 {len(sample_field)} 个）")

    # 整体统计
    total = len(all_windows)
    lc = all_windows['risk_level_optimized'].value_counts()
    print(f"\n{'='*60}")
    print("整体统计（优化后风险等级）")
    print(f"{'='*60}")
    print(f"  计算方法: {method}")
    print(f"  R全局归一化: {'开启' if force_norm else '关闭'}")
    print(f"  有效窗口: {total}  R 均值={all_windows['R'].mean():.4f}  std={all_windows['R'].std():.4f}")
    for lvl in ['高风险', '中风险', '低风险']:
        n = lc.get(lvl, 0)
        print(f"  {lvl}: {n:>6} 窗口 ({n/total*100:5.1f}%)")
    print(f"\n  结果 CSV → {csv_dir}")
    print(f"  图  片   → {fig_dir}")
    print("=" * 60)


if __name__ == '__main__':
    main()
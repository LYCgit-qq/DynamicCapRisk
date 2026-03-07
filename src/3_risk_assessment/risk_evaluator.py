# D:\Local\DynamicCapRisk\src\3_risk_assessment\risk_evaluator.py

"""
risk_evaluator.py
基于 TCI 模型的风险度评估主模块

核心公式（论文第4章）：
  Ã_d  = (A_d - ad_min) / (ad_max - ad_min)
  R    = α·F_S - β·Ã_d
  R*   = 2R - 0.16  ∈ [-1, 1]

输出目录：
  output_csv/   —— 所有 CSV 结果
  output_fig/   —— 所有图片（由 plot_risk_results.py 生成）
"""

import pickle
import argparse
import os
import warnings
import yaml
import numpy as np
import pandas as pd
from typing import Dict, List, Tuple

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
# 核心计算
# =============================================================================

def normalize_ad(ad: np.ndarray, ad_min: float, ad_max: float) -> np.ndarray:
    return np.clip((ad - ad_min) / (ad_max - ad_min), 0.0, 1.0)


def compute_risk(fs: np.ndarray, ad_norm: np.ndarray,
                 alpha: float, beta: float) -> Tuple[np.ndarray, np.ndarray]:
    R      = alpha * fs - beta * ad_norm
    R_star = np.clip(2.0 * R - 0.16, -1.0, 1.0)
    return R, R_star


def assign_risk_level(r_star: np.ndarray,
                      thresh_high: float,
                      thresh_low: float) -> np.ndarray:
    levels = np.full(len(r_star), '低风险', dtype=object)
    levels[r_star >= thresh_high]                                  = '高风险'
    levels[(r_star >= thresh_low) & (r_star < thresh_high)]        = '中风险'
    return levels


def evaluate_sample(field_labels: np.ndarray,
                    ad_values: np.ndarray,
                    spatial_fs: Dict[int, np.ndarray],
                    sample_idx: int,
                    cfg: dict) -> Tuple[pd.DataFrame, np.ndarray, np.ndarray]:
    """
    对单个样本计算风险度。

    Returns:
        (result_df, fs_temporal, ad_norm)
        fs_temporal 和 ad_norm 供可视化模块直接使用，避免重复计算。
    """
    m = cfg['model']
    if len(field_labels) == 0 or len(ad_values) == 0:
        return pd.DataFrame(), np.array([]), np.array([])
    if len(field_labels) != len(ad_values):
        print(f"  ⚠️  样本{sample_idx}: "
              f"field({len(field_labels)}) 与 Ad({len(ad_values)}) 长度不一致，跳过")
        return pd.DataFrame(), np.array([]), np.array([])

    fs_arr  = map_spatial_to_temporal(np.asarray(field_labels),
                                      spatial_fs, m['fs_baseline'])
    ad_arr  = np.asarray(ad_values, dtype=float)
    ad_norm = normalize_ad(ad_arr, m['ad_min'], m['ad_max'])
    R, R_star = compute_risk(fs_arr, ad_norm, m['alpha'], m['beta'])
    levels  = assign_risk_level(R_star, m['thresh_high'], m['thresh_low'])

    df = pd.DataFrame({
        'sample_idx':  sample_idx,
        'window_idx':  np.arange(len(field_labels)),
        'field_label': np.asarray(field_labels, dtype=int),
        'F_S':         np.round(fs_arr,   4),
        'A_d':         np.round(ad_arr,   4),
        'Ad_norm':     np.round(ad_norm,  4),
        'R':           np.round(R,        4),
        'R_star':      np.round(R_star,   4),
        'risk_level':  levels,
    })
    return df, fs_arr, ad_norm


# =============================================================================
# 能力分组
# =============================================================================

def assign_capability_groups(sample_ad: List[np.ndarray]) -> Dict[int, str]:
    """按各样本 A_d 均值三等分，划分高/中/低能力组。"""
    means = {i: float(np.mean(arr)) for i, arr in enumerate(sample_ad)
             if len(arr) > 0}
    vals  = sorted(means.values())
    n     = len(vals)
    t_lo  = vals[n // 3]
    t_hi  = vals[2 * n // 3]
    return {i: ('高能力组' if m >= t_hi else '低能力组' if m < t_lo else '中能力组')
            for i, m in means.items()}


# =============================================================================
# 统计汇总（生成 CSV 结果）
# =============================================================================

def summarize_sample(df: pd.DataFrame, sample_idx: int) -> dict:
    if df.empty:
        return {'sample_idx': sample_idx, 'n_windows': 0}
    total = len(df)
    lc    = df['risk_level'].value_counts()
    return {
        'sample_idx':    sample_idx,
        'n_windows':     total,
        'R_star_mean':   round(df['R_star'].mean(), 4),
        'R_star_std':    round(df['R_star'].std(),  4),
        'R_star_max':    round(df['R_star'].max(),  4),
        'R_star_min':    round(df['R_star'].min(),  4),
        'high_risk_n':   int(lc.get('高风险', 0)),
        'mid_risk_n':    int(lc.get('中风险', 0)),
        'low_risk_n':    int(lc.get('低风险', 0)),
        'high_risk_pct': round(lc.get('高风险', 0) / total * 100, 2),
        'mid_risk_pct':  round(lc.get('中风险', 0) / total * 100, 2),
        'low_risk_pct':  round(lc.get('低风险', 0) / total * 100, 2),
    }


def save_table_scenario_group(all_windows: pd.DataFrame,
                               cfg: dict, csv_dir: str) -> pd.DataFrame:
    """Table 4.5：场景×能力组 R* 均值"""
    lname  = {int(k): v for k, v in cfg['scenarios']['label_name'].items()}
    groups = ['高能力组', '中能力组', '低能力组']
    rows   = []
    for label in sorted(all_windows['field_label'].unique()):
        sub = all_windows[all_windows['field_label'] == label]
        row = {'场景': lname.get(label, f'label{label}'),
               '场景F_S': round(sub['F_S'].mean(), 3)}
        for g in groups:
            gsub   = sub[sub['group'] == g]
            row[g] = round(gsub['R_star'].mean(), 3) if len(gsub) > 0 else np.nan
        row['全体均值'] = round(sub['R_star'].mean(), 3)
        rows.append(row)
    df   = pd.DataFrame(rows)
    path = os.path.join(csv_dir, 'table4_5_scenario_group.csv')
    df.to_csv(path, index=False, encoding='utf-8-sig')
    print(f"  Table4.5 → {path}")
    return df


def save_table_risk_level_dist(all_windows: pd.DataFrame,
                                cfg: dict, csv_dir: str) -> None:
    """Table 4.6：三组驾驶人风险等级占比"""
    groups = ['高能力组', '中能力组', '低能力组', '全体']
    levels = ['低风险', '中风险', '高风险']
    rows   = []
    for lvl in levels:
        row = {'风险等级': lvl}
        for g in groups:
            sub   = all_windows if g == '全体' else all_windows[all_windows['group'] == g]
            total = max(len(sub), 1)
            n     = int((sub['risk_level'] == lvl).sum())
            row[g] = f"{n/total*100:.1f}%"
        rows.append(row)
    path = os.path.join(csv_dir, 'table4_6_risk_level_dist.csv')
    pd.DataFrame(rows).to_csv(path, index=False, encoding='utf-8-sig')
    print(f"  Table4.6 → {path}")


def compute_weight_calibration(all_windows: pd.DataFrame,
                                cfg: dict, csv_dir: str) -> None:
    """Table 4.0：α/β 网格搜索校准，保存完整结果和摘要。"""
    from sklearn.metrics import roc_auc_score
    cal   = cfg['calibration']
    lo, hi, step = cal['alpha_range'][0], cal['alpha_range'][1], cal['alpha_step']
    ci_drop = cal['ci_auc_drop']

    y  = (all_windows['event_label'].to_numpy(dtype=int)
          if 'event_label' in all_windows.columns
          else (all_windows['R_star'] > 0).astype(int).to_numpy())
    fs = all_windows['F_S'].to_numpy()
    adn = all_windows['Ad_norm'].to_numpy()

    best_auc, best_alpha = 0.0, cfg['model']['alpha']
    results = []
    for a in np.arange(lo, hi + step / 2, step):
        a   = round(float(a), 2)
        b   = round(1.0 - a, 2)
        try:
            auc = roc_auc_score(y, a * fs - b * adn)
        except Exception:
            auc = 0.0
        results.append({'alpha': a, 'beta': b, 'auc': round(auc, 4)})
        if auc > best_auc:
            best_auc, best_alpha = auc, a

    best_beta = round(1.0 - best_alpha, 2)
    res_df    = pd.DataFrame(results)
    ci_range  = res_df[res_df['auc'] >= best_auc - ci_drop]['alpha']

    summary = pd.DataFrame([
        {'parameter': 'α', 'optimal': round(best_alpha, 2),
         'ci_low': round(ci_range.min(), 2), 'ci_high': round(ci_range.max(), 2),
         'auc': round(best_auc, 3)},
        {'parameter': 'β', 'optimal': round(best_beta, 2),
         'ci_low': round(1 - ci_range.max(), 2), 'ci_high': round(1 - ci_range.min(), 2),
         'auc': '—'},
    ])
    summary.to_csv(os.path.join(csv_dir, 'weight_calibration.csv'),
                   index=False, encoding='utf-8-sig')
    res_df.to_csv(os.path.join(csv_dir, 'weight_calibration_full.csv'),
                  index=False, encoding='utf-8-sig')
    print(f"  Table4.0 → weight_calibration.csv  "
          f"(最优 α={best_alpha:.2f}, AUC={best_auc:.3f})")


# =============================================================================
# 主函数
# =============================================================================

def main():
    parser = argparse.ArgumentParser(
        description='TCI 风险度评估（结果→output_csv，图→output_fig）',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例用法:
  python risk_evaluator.py
  python risk_evaluator.py -c config/risk_evaluator.yaml
  python risk_evaluator.py --plot_sample 0
        """
    )
    parser.add_argument('-c', '--config', type=str,
                        default='config/risk_evaluator.yaml',
                        help='配置文件路径')
    parser.add_argument('--plot_sample', type=int, default=None,
                        metavar='IDX',
                        help='额外可视化指定样本的时序图（0-based）')
    args = parser.parse_args()

    # ── 加载配置 ─────────────────────────────────────────────────
    cfg     = load_config(args.config)
    csv_dir = cfg['paths']['output_csv']
    fig_dir = cfg['paths']['output_fig']
    os.makedirs(csv_dir, exist_ok=True)
    os.makedirs(fig_dir, exist_ok=True)

    print("=" * 60)
    print("TCI 风险度评估")
    print(f"  CSV 输出: {csv_dir}")
    print(f"  图片输出: {fig_dir}")
    print("=" * 60)

    # ── 1. 加载原始数据 ──────────────────────────────────────────
    print("\n[1/5] 加载数据...")
    sample_field = load_pkl(cfg['paths']['field_pkl'], 'sample_field')
    sample_ad    = load_pkl(cfg['paths']['ad_pkl'],    'sample_dynamic_capability')
    if len(sample_field) != len(sample_ad):
        raise ValueError(f"样本数不一致: field={len(sample_field)}, ad={len(sample_ad)}")
    print(f"  样本数: {len(sample_field)}")

    print("\n[2/5] 加载空间 F_S...")
    spatial_fs = load_spatial_fs(cfg)

    # ── 2. 逐样本计算 ────────────────────────────────────────────
    print(f"\n[3/5] 计算风险度...")
    all_dfs       = []
    sum_rows      = []
    fs_temp_list  = []   # 各样本的 F_S 时间序列（供可视化）
    ad_norm_list  = []   # 各样本的归一化 Ã_d（供可视化）
    risk_list     = []   # 各样本的 R_star 风险序列

    for i, (fld, ad) in enumerate(zip(sample_field, sample_ad)):
        df, fs_t, adn = evaluate_sample(
            np.asarray(fld), np.asarray(ad, dtype=float), spatial_fs, i, cfg
        )
        fs_temp_list.append(fs_t)
        ad_norm_list.append(adn)

        if not df.empty:
            risk_list.append(df['R_star'].to_numpy())
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
    summary_df  = pd.DataFrame(sum_rows)

    # ── 3. 能力分组 ──────────────────────────────────────────────
    print("\n[4/5] 能力分组...")
    cap_groups = assign_capability_groups(sample_ad)
    all_windows['group'] = all_windows['sample_idx'].map(cap_groups).fillna('中能力组')
    for g, n in all_windows['group'].value_counts().items():
        print(f"  {g}: {n} 窗口")

    # ── 4. 保存 CSV 结果 ─────────────────────────────────────────
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

    try:
        compute_weight_calibration(all_windows, cfg, csv_dir)    # Table 4.0
    except ImportError:
        print("  ⚠️  sklearn 未安装，跳过权重校准（pip install scikit-learn）")

    table45 = save_table_scenario_group(all_windows, cfg, csv_dir)  # Table 4.5
    save_table_risk_level_dist(all_windows, cfg, csv_dir)            # Table 4.6

    # ── 5. 生成图表（调用可视化模块） ────────────────────────────
    from src.visualization.plot_risk import (
        plot_threshold_f1, plot_r_histogram, plot_violin_by_group,
        plot_line_scenario_group, plot_box_scenario_group,
        plot_stacked_bar_risk, plot_timeseries_typical, plot_single_sample
    )

    try:
        plot_threshold_f1(all_windows, cfg, fig_dir)              # Fig 4.5
    except ImportError:
        print("  ⚠️  sklearn 未安装，跳过 Fig4.5")

    plot_r_histogram(all_windows, cfg, fig_dir)                   # Fig 4.6
    plot_violin_by_group(all_windows, cfg, fig_dir)               # Fig 4.7
    plot_line_scenario_group(table45, cfg, fig_dir)               # Fig 4.8
    plot_box_scenario_group(all_windows, cfg, fig_dir)            # Fig 4.9
    plot_stacked_bar_risk(all_windows, cfg, fig_dir)              # Fig 4.10
    plot_timeseries_typical(all_windows, sample_field,
                             ad_norm_list, fs_temp_list,
                             cap_groups, cfg, fig_dir)            # Fig 4.11

    if args.plot_sample is not None:
        idx = args.plot_sample
        if 0 <= idx < len(sample_field):
            plot_single_sample(idx, sample_field, fs_temp_list,
                                ad_norm_list, all_windows, cfg, fig_dir)
        else:
            print(f"⚠️  --plot_sample {idx} 超出范围（共 {len(sample_field)} 个）")

    # ── 整体统计打印 ─────────────────────────────────────────────
    total = len(all_windows)
    lc    = all_windows['risk_level'].value_counts()
    print(f"\n{'='*60}")
    print("整体统计")
    print(f"{'='*60}")
    print(f"  有效窗口: {total}  "
          f"R* 均值={all_windows['R_star'].mean():.4f}  "
          f"std={all_windows['R_star'].std():.4f}")
    for lvl in ['高风险', '中风险', '低风险']:
        n = int(lc.get(lvl, 0))
        print(f"  {lvl}: {n:>6} 窗口 ({n/total*100:5.1f}%)")
    print(f"\n  结果 CSV → {csv_dir}")
    print(f"  图  片   → {fig_dir}")
    print("=" * 60)


if __name__ == '__main__':
    main()

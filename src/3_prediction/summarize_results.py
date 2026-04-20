import os
import re
import yaml
import pandas as pd
import shutil
from typing import Dict, List, Optional, Tuple

# ===================== 【配置区域】 =====================
RUNS_DIR = "/root/autodl-tmp/DynamicCapRisk/output/3_prediction/runs"
OUTPUT_ROOT = "/root/autodl-tmp/DynamicCapRisk/output/3_prediction/results_summary"
# =======================================================

def parse_evaluation_report(file_path: str) -> Optional[Dict]:
    """解析 evaluation_report.txt"""
    metrics = {}
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            content = f.read()

        # 回归指标
        r2_match = re.search(r"R²\s*=\s*([0-9.]+)", content)
        metrics['risk_reg_R2'] = float(r2_match.group(1)) if r2_match else None
        mae_match = re.search(r"MAE\s*=\s*([0-9.]+)", content)
        metrics['risk_reg_MAE'] = float(mae_match.group(1)) if mae_match else None
        rmse_match = re.search(r"RMSE\s*=\s*([0-9.]+)", content)
        metrics['risk_reg_RMSE'] = float(rmse_match.group(1)) if rmse_match else None

        # 能力预测MAE（消融实验用）
        cap_mae_match = re.search(r"能力预测MAE\s*=\s*([0-9.]+)", content)
        metrics['cap_reg_MAE'] = float(cap_mae_match.group(1)) if cap_mae_match else None

        # 高风险指标
        hr_recall_match = re.search(r"高风险召回率\s*=\s*([0-9.]+)%", content)
        hr_prec_match = re.search(r"高风险精确率\s*=\s*([0-9.]+)%", content)
        hr_f1_match = re.search(r"高风险 F1\s*=\s*([0-9.]+)", content)
        metrics['high_risk_recall'] = float(hr_recall_match.group(1)) / 100 if hr_recall_match else None
        metrics['high_risk_prec'] = float(hr_prec_match.group(1)) / 100 if hr_prec_match else None
        metrics['high_risk_f1'] = float(hr_f1_match.group(1)) if hr_f1_match else None

        # 分类指标
        acc_match = re.search(r"总体准确率\s*=\s*([0-9.]+)%", content)
        macro_f1_match = re.search(r"宏平均 F1\s*=\s*([0-9.]+)", content)
        kappa_match = re.search(r"Kappa 系数\s*=\s*([0-9.]+)", content)
        metrics['cls_accuracy'] = float(acc_match.group(1)) / 100 if acc_match else None
        metrics['cls_macro_f1'] = float(macro_f1_match.group(1)) if macro_f1_match else None
        metrics['cls_kappa'] = float(kappa_match.group(1)) if kappa_match else None

        # 一致性偏差
        cd_match = re.search(r"一致性偏差\s*=\s*([0-9.]+)", content)
        metrics['consistency_deviation'] = float(cd_match.group(1)) if cd_match else None

        return metrics
    except Exception as e:
        print(f"[Error] 解析报告失败 {file_path}: {e}")
        return None

def parse_config(file_path: str) -> Optional[Dict]:
    """解析配置文件，提取模型类型、消融方案、实验路径"""
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            cfg = yaml.safe_load(f)
        return {
            'model_type': cfg['model']['model_type'],
            'ablation': cfg['model']['ablation'],
            'exp_path': os.path.dirname(file_path)
        }
    except Exception as e:
        print(f"[Error] 解析配置失败 {file_path}: {e}")
        return None

# ===================== 核心：筛选最优结果 =====================
def get_best_model_per_type(df: pd.DataFrame) -> pd.DataFrame:
    """表5.8：按模型分组 → 高风险召回率最高"""
    best_models = []
    for model_type in df['model_type'].unique():
        model_df = df[df['model_type'] == model_type].copy()
        best = model_df.sort_values('high_risk_recall', ascending=False).iloc[0]
        best_models.append(best)
    return pd.DataFrame(best_models).reset_index(drop=True)

def get_best_ablation_per_type(df: pd.DataFrame) -> pd.DataFrame:
    """表5.9：按消融方案分组 → 风险度MAE最小（最优）"""
    # 仅保留 MT-RP 模型的消融实验
    df_mtrp = df[df['model_type'] == 'mtrp'].copy()
    best_ablations = []
    for ablation_type in df_mtrp['ablation'].unique():
        abl_df = df_mtrp[df_mtrp['ablation'] == ablation_type].copy()
        # 消融实验：risk_reg_MAE 越小性能越好
        best = abl_df.sort_values('risk_reg_MAE', ascending=True).iloc[0]
        best_ablations.append(best)
    return pd.DataFrame(best_ablations).reset_index(drop=True)

# ===================== 复制最优报告 =====================
def copy_best_report(best_df: pd.DataFrame, prefix: str = "最优"):
    for _, row in best_df.iterrows():
        model_type = row['model_type']
        ablation = row['ablation']
        exp_path = row['exp_path']
        
        src_report = os.path.join(exp_path, "eval", "evaluation_report.txt")
        # 命名：最优报告_模型_消融方案.txt
        filename = f"{prefix}报告_{model_type}_{ablation}.txt" if ablation != 'none' else f"{prefix}报告_{model_type}.txt"
        dst_report = os.path.join(OUTPUT_ROOT, filename)
        
        if os.path.exists(src_report):
            shutil.copy(src_report, dst_report)
            print(f"✅ 已复制 -> {filename}")

def main():
    os.makedirs(OUTPUT_ROOT, exist_ok=True)
    all_results = []
    
    if not os.path.exists(RUNS_DIR):
        print(f"路径不存在: {RUNS_DIR}")
        return

    # 遍历解析所有实验
    exp_list = sorted(os.listdir(RUNS_DIR))
    print(f"发现 {len(exp_list)} 个实验文件夹，开始解析...")

    for exp_name in exp_list:
        exp_path = os.path.join(RUNS_DIR, exp_name)
        if not os.path.isdir(exp_path):
            continue

        report_path = os.path.join(exp_path, "eval", "evaluation_report.txt")
        config_path = os.path.join(exp_path, "run_config.yaml")

        if not os.path.exists(report_path) or not os.path.exists(config_path):
            continue

        metrics = parse_evaluation_report(report_path)
        config = parse_config(config_path)

        if metrics and config:
            combined = {'exp_id': exp_name, **config, **metrics}
            all_results.append(combined)

    if not all_results:
        print("未找到有效数据。")
        return

    df = pd.DataFrame(all_results)
    print(f"解析完成，共 {len(df)} 条有效实验")

    # ===================== 1. 生成 纯净版表5.8（模型对比） =====================
    print("\n📊 生成表5.8：每种模型最优高风险召回")
    df_best_model = get_best_model_per_type(df)
    table58_cols = ['model_type', 'risk_reg_MAE', 'risk_reg_R2', 'cls_accuracy', 'high_risk_recall', 'high_risk_f1']
    df_best_model[table58_cols].to_csv(os.path.join(OUTPUT_ROOT, "模型对比结果.csv"), index=False, encoding='utf-8-sig')

    # ===================== 2. 生成 纯净版表5.9（消融实验） =====================
    print("\n📊 生成表5.9：每种消融方案最优风险MAE")
    df_best_ablation = get_best_ablation_per_type(df)
    table59_cols = ['ablation', 'cap_reg_MAE', 'risk_reg_MAE', 'cls_macro_f1', 'consistency_deviation']
    df_best_ablation[table59_cols].to_csv(os.path.join(OUTPUT_ROOT, "消融实验结果.csv"), index=False, encoding='utf-8-sig')

    # ===================== 3. 复制所有最优实验报告 =====================
    print("\n📄 复制最优模型/消融实验报告...")
    copy_best_report(df_best_model, "模型")
    copy_best_report(df_best_ablation, "消融")

    # ===================== 终端预览 =====================
    print("\n" + "="*80)
    print("📈 表5.8 最终纯净数据（论文专用）")
    print(df_best_model[table58_cols].to_string(index=False))
    print("\n📈 表5.9 最终纯净数据（论文专用）")
    print(df_best_ablation[table59_cols].to_string(index=False))
    print("="*80)
    print(f"\n🎉 全部完成！文件保存在：{OUTPUT_ROOT}")

if __name__ == "__main__":
    main()
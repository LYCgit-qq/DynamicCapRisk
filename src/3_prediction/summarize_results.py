import os
import re
import yaml
import pandas as pd
from typing import Dict, List, Optional, Tuple

# ===================== 【配置区域】 =====================
# 实验结果根目录 (runs 文件夹所在路径)
RUNS_DIR = "/root/autodl-tmp/DynamicCapRisk/output/3_prediction/runs"
# 最终汇总报告输出根目录
OUTPUT_ROOT = "/root/autodl-tmp/DynamicCapRisk/output/3_prediction/results_summary"
# =======================================================

def parse_evaluation_report(file_path: str) -> Optional[Dict]:
    """解析 evaluation_report.txt，提取所有数值指标"""
    metrics = {}
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            content = f.read()

        # --- 1. 回归指标 ---
        r2_match = re.search(r"R²\s*=\s*([0-9.]+)", content)
        metrics['risk_reg_R2'] = float(r2_match.group(1)) if r2_match else None
        
        mae_match = re.search(r"MAE\s*=\s*([0-9.]+)", content)
        metrics['risk_reg_MAE'] = float(mae_match.group(1)) if mae_match else None
        
        rmse_match = re.search(r"RMSE\s*=\s*([0-9.]+)", content)
        metrics['risk_reg_RMSE'] = float(rmse_match.group(1)) if rmse_match else None

        # 高风险召回率/精确率/F1
        hr_recall_match = re.search(r"高风险召回率\s*=\s*([0-9.]+)%", content)
        hr_prec_match = re.search(r"高风险精确率\s*=\s*([0-9.]+)%", content)
        hr_f1_match = re.search(r"高风险 F1\s*=\s*([0-9.]+)", content)
        
        metrics['high_risk_recall'] = float(hr_recall_match.group(1)) / 100 if hr_recall_match else None
        metrics['high_risk_prec'] = float(hr_prec_match.group(1)) / 100 if hr_prec_match else None
        metrics['high_risk_f1'] = float(hr_f1_match.group(1)) if hr_f1_match else None

        # --- 2. 分类指标 ---
        acc_match = re.search(r"总体准确率\s*=\s*([0-9.]+)%", content)
        macro_f1_match = re.search(r"宏平均 F1\s*=\s*([0-9.]+)", content)
        kappa_match = re.search(r"Kappa 系数\s*=\s*([0-9.]+)", content)
        
        metrics['cls_accuracy'] = float(acc_match.group(1)) / 100 if acc_match else None
        metrics['cls_macro_f1'] = float(macro_f1_match.group(1)) if macro_f1_match else None
        metrics['cls_kappa'] = float(kappa_match.group(1)) if kappa_match else None

        # --- 3. 细分类别指标 ---
        low_pat = r"低风险:\s*Precision=([0-9.]+)%\s*Recall=([0-9.]+)%\s*F1=([0-9.]+)"
        low_match = re.search(low_pat, content)
        if low_match:
            metrics['低风险_precision'] = float(low_match.group(1))/100
            metrics['低风险_recall'] = float(low_match.group(2))/100
            metrics['低风险_f1'] = float(low_match.group(3))

        mid_pat = r"中风险:\s*Precision=([0-9.]+)%\s*Recall=([0-9.]+)%\s*F1=([0-9.]+)"
        mid_match = re.search(mid_pat, content)
        if mid_match:
            metrics['中风险_precision'] = float(mid_match.group(1))/100
            metrics['中风险_recall'] = float(mid_match.group(2))/100
            metrics['中风险_f1'] = float(mid_match.group(3))
            
        high_pat = r"高风险:\s*Precision=([0-9.]+)%\s*Recall=([0-9.]+)%\s*F1=([0-9.]+)"
        high_match = re.search(high_pat, content)
        if high_match:
            metrics['高风险_precision'] = float(high_match.group(1))/100
            metrics['高风险_recall'] = float(high_match.group(2))/100
            metrics['高风险_f1'] = float(high_match.group(3))

        return metrics

    except Exception as e:
        print(f"[Error] 解析报告失败 {file_path}: {e}")
        return None

def parse_config(file_path: str) -> Optional[Dict]:
    """解析 run_config.yaml，提取关键超参数"""
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            cfg = yaml.safe_load(f)
        
        return {
            'model_type': cfg['model']['model_type'],
            'ablation': cfg['model']['ablation'],
            'lr': cfg['optimizer']['lr'],
            'batch_size': cfg['train']['batch_size'],
            'dropout': cfg['model']['dropout'],
            'lambda_cls': cfg['loss']['lambda_risk_cls'],
            'd_model': cfg['model'].get('d_model'),
            'hidden': cfg['model'].get('baseline_hidden')
        }
    except Exception as e:
        print(f"[Error] 解析配置失败 {file_path}: {e}")
        return None

def save_sorted_df(df: pd.DataFrame, sort_by: List[str], ascending: List[bool], filename: str, output_dir: str):
    """排序并保存DataFrame"""
    # 检查排序列是否存在
    valid_cols = [c for c in sort_by if c in df.columns]
    if not valid_cols:
        return

    df_sorted = df.sort_values(by=valid_cols, ascending=ascending).reset_index(drop=True)
    out_path = os.path.join(output_dir, filename)
    df_sorted.to_csv(out_path, index=False, encoding='utf-8-sig')
    print(f"  -> 已生成: {filename}")

def main():
    # 1. 准备工作
    os.makedirs(OUTPUT_ROOT, exist_ok=True)
    all_results = []
    
    if not os.path.exists(RUNS_DIR):
        print(f"路径不存在: {RUNS_DIR}")
        return

    # 2. 遍历并解析数据
    exp_list = sorted(os.listdir(RUNS_DIR))
    print(f"发现 {len(exp_list)} 个实验文件夹，开始解析...")

    for exp_name in exp_list:
        exp_path = os.path.join(RUNS_DIR, exp_name)
        if not os.path.isdir(exp_path): continue

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
    
    # ===================== 【新增：去重逻辑开始】 =====================
    # 定义用于判断“参数配置完全相同”的列名
    config_cols = ['model_type', 'ablation', 'lr', 'batch_size', 'dropout', 'lambda_cls', 'd_model', 'hidden']
    # 过滤掉 DataFrame 中不存在的列（防止报错）
    config_cols = [c for c in config_cols if c in df.columns]

    if config_cols:
        # 步骤1：先按 'high_risk_f1' 从高到低排序（确保相同配置中，性能最好的排在最前面）
        if 'high_risk_f1' in df.columns:
            df = df.sort_values(by='high_risk_f1', ascending=False)
        else:
            # 如果没有 high_risk_f1，就用准确率
            if 'cls_accuracy' in df.columns:
                df = df.sort_values(by='cls_accuracy', ascending=False)

        # 步骤2：根据配置列去重，保留第一次出现的行（即性能最好的那一行）
        initial_count = len(df)
        df = df.drop_duplicates(subset=config_cols, keep='first').reset_index(drop=True)
        final_count = len(df)
        print(f"去重完成：原始 {initial_count} 个实验，保留 {final_count} 个唯一配置实验。")
    else:
        print("警告：未找到配置列，跳过去重步骤。")
    # ===================== 【新增：去重逻辑结束】 =====================

    # 调整列顺序 (把重要指标放前面)
    priority_cols = ['exp_id', 'model_type', 'ablation', 'high_risk_f1', 'cls_accuracy', 'risk_reg_R2']
    priority_cols = [c for c in priority_cols if c in df.columns]
    other_cols = [c for c in df.columns if c not in priority_cols]
    df = df[priority_cols + other_cols]

    # 3. 定义多种排序策略并批量保存
    print(f"\n开始生成多维度报告至: {OUTPUT_ROOT}")
    print("-" * 60)

    # 策略1: 综合核心指标 (高风险F1 -> 准确率)
    save_sorted_df(df, 
                   sort_by=['high_risk_f1', 'cls_accuracy'], 
                   ascending=[False, False], 
                   filename='01_综合推荐_高风险F1优先.csv',
                   output_dir=OUTPUT_ROOT)

    # 策略2: 安全优先 (高风险召回率 -> 宁错勿漏)
    save_sorted_df(df, 
                   sort_by=['high_risk_recall', 'high_risk_f1'], 
                   ascending=[False, False], 
                   filename='02_安全优先_高风险召回优先.csv',
                   output_dir=OUTPUT_ROOT)

    # 策略3: 精度优先 (高风险精确率 -> 减少误报)
    save_sorted_df(df, 
                   sort_by=['high_risk_prec', 'cls_accuracy'], 
                   ascending=[False, False], 
                   filename='03_精度优先_高风险精确率优先.csv',
                   output_dir=OUTPUT_ROOT)

    # 策略4: 回归性能 (R² -> 风险度预测准度)
    save_sorted_df(df, 
                   sort_by=['risk_reg_R2', 'risk_reg_RMSE'], 
                   ascending=[False, True], 
                   filename='04_回归优先_R2最大.csv',
                   output_dir=OUTPUT_ROOT)

    # 策略5: 分类性能 (总体准确率 -> Kappa)
    save_sorted_df(df, 
                   sort_by=['cls_accuracy', 'cls_kappa'], 
                   ascending=[False, False], 
                   filename='05_分类优先_总体准确率最高.csv',
                   output_dir=OUTPUT_ROOT)

    # 策略6: 按模型类型分组查看 (Model Type -> High Risk F1)
    save_sorted_df(df, 
                   sort_by=['model_type', 'high_risk_f1'], 
                   ascending=[True, False], 
                   filename='06_模型对比_按类型分组.csv',
                   output_dir=OUTPUT_ROOT)

    # 4. 终端快速预览 Top 1
    print("-" * 60)
    print("\n🏆 【01_综合推荐】榜单 Top 3 预览:")
    # 重新读取并打印，确保顺序一致
    df_top = pd.read_csv(os.path.join(OUTPUT_ROOT, '01_综合推荐_高风险F1优先.csv'))
    
    # 只打印关键列
    show_cols = ['exp_id', 'model_type', 'high_risk_f1', 'cls_accuracy', 'high_risk_recall', 'lr']
    # 过滤掉不存在的列
    show_cols = [c for c in show_cols if c in df_top.columns]
    
    print(df_top[show_cols].head(3).to_string(index=False))
    print(f"\n✅ 全部生成完毕！请前往 {OUTPUT_ROOT} 查看详细文件。")

if __name__ == "__main__":
    main()
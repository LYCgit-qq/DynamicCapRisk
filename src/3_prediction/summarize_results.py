import os
import re
import yaml
import pandas as pd
import shutil

# ===================== 全局配置 (列顺序定义) =====================
# 严格统一的列顺序 (核心指标部分)
UNIFIED_COLS = [
    'model_type', 
    'exp_id', 
    'high_risk_recall', 
    'high_risk_f1', 
    'risk_reg_MAE', 
    'cls_accuracy', 
    'cls_macro_f1'
]

# ===================== 配置读取 =====================
CONFIG_PATH = r"D:\Local\DynamicCapRisk\config\summarize_results.yaml"

def load_config(path):
    with open(path, 'r', encoding='utf-8') as f:
        return yaml.safe_load(f)

CFG = load_config(CONFIG_PATH)
RUNS_DIR = CFG['paths']['runs_dir']
OUTPUT_ROOT = CFG['paths']['output_root']

# ===================== 指标解析 =====================
def parse_txt_metrics(txt_path):
    try:
        with open(txt_path, 'r', encoding='utf-8') as f:
            c = f.read()
    except:
        return {}

    m = {}
    # 回归指标
    m['risk_reg_R2'] = float(g[1]) if (g := re.search(r'R²\s*=\s*([\d.]+)', c)) else None
    m['risk_reg_MAE'] = float(g[1]) if (g := re.search(r'MAE\s*=\s*([\d.]+)', c)) else None
    m['risk_reg_RMSE'] = float(g[1]) if (g := re.search(r'RMSE\s*=\s*([\d\.]+)', c)) else None

    # 高风险核心指标
    m['high_risk_recall'] = float(g[1])/100 if (g := re.search(r'高风险召回率\s*=\s*([\d.]+)%', c)) else None
    m['high_risk_prec'] = float(g[1])/100 if (g := re.search(r'高风险精确率\s*=\s*([\d.]+)%', c)) else None
    m['high_risk_f1'] = float(g[1]) if (g := re.search(r'高风险 F1\s*=\s*([\d.]+)', c)) else None

    # 整体分类指标
    m['cls_accuracy'] = float(g[1])/100 if (g := re.search(r'总体准确率\s*=\s*([\d.]+)%', c)) else None
    m['cls_macro_f1'] = float(g[1]) if (g := re.search(r'宏平均 F1\s*=\s*([\d.]+)', c)) else None
    m['cls_kappa'] = float(g[1]) if (g := re.search(r'Kappa 系数\s*=\s*([\d.]+)', c)) else None

    # 低/中/高风险细分指标
    m['低风险_precision'] = float(g[1])/100 if (g := re.search(r'低风险:\s*Precision=([\d.]+)%', c)) else None
    m['低风险_recall'] = float(g[1])/100 if (g := re.search(r'Recall=([\d.]+)%', c.split('低风险:')[-1].split('\n')[0])) else None
    m['低风险_f1'] = float(g[1]) if (g := re.search(r'F1=([\d.]+)', c.split('低风险:')[-1].split('\n')[0])) else None
    m['中风险_precision'] = float(g[1])/100 if (g := re.search(r'中风险:\s*Precision=([\d.]+)%', c)) else None
    m['中风险_recall'] = float(g[1])/100 if (g := re.search(r'Recall=([\d.]+)%', c.split('中风险:')[-1].split('\n')[0])) else None
    m['中风险_f1'] = float(g[1]) if (g := re.search(r'F1=([\d.]+)', c.split('中风险:')[-1].split('\n')[0])) else None
    m['高风险_precision'] = float(g[1])/100 if (g := re.search(r'高风险:\s*Precision=([\d.]+)%', c)) else None
    m['高风险_recall'] = float(g[1])/100 if (g := re.search(r'Recall=([\d.]+)%', c.split('高风险:')[-1].split('\n')[0])) else None
    m['高风险_f1'] = float(g[1]) if (g := re.search(r'F1=([\d.]+)', c.split('高风险:')[-1].split('\n')[0])) else None

    # 辅助指标
    m['consistency_deviation'] = float(g[1]) if (g := re.search(r'一致性偏差\s*=\s*([\d.]+)', c)) else None
    m['cap_reg_MAE'] = float(g[1]) if (g := re.search(r'能力预测MAE\s*=\s*([\d.]+)', c)) else None

    return m

def parse_csv_metrics(csv_path):
    try:
        return pd.read_csv(csv_path).iloc[0].to_dict()
    except:
        return None

def get_metrics(exp_path):
    csv = os.path.join(exp_path, 'eval', 'evaluation_metrics.csv')
    txt = os.path.join(exp_path, 'eval', 'evaluation_report.txt')
    return parse_csv_metrics(csv) or parse_txt_metrics(txt)

# ===================== 完整配置解析（所有超参数） =====================
def parse_exp_config(path):
    try:
        with open(path, 'r', encoding='utf-8') as f:
            y = yaml.safe_load(f)
        # 提取所有关键模型/训练参数，exp_path 仅内部使用
        return {
            # 基础信息
            'model_type': y['model']['model_type'],
            'ablation': y['model']['ablation'],
            'exp_path': os.path.dirname(path),  # 仅代码内部用，不进CSV
            # 模型结构参数
            'd_model': y['model'].get('d_model'),
            'nhead': y['model'].get('nhead'),
            'num_layers': y['model'].get('num_layers'),
            'ffn_dim': y['model'].get('ffn_dim'),
            'dropout': y['model'].get('dropout'),
            'input_dim': y['model'].get('input_dim'),
            'n_classes': y['model'].get('n_classes'),
            'seq_len': y['model'].get('seq_len'),
            # 训练参数
            'lr': y['optimizer'].get('lr'),
            'batch_size': y['train'].get('batch_size'),
            'weight_decay': y['optimizer'].get('weight_decay'),
            'max_epochs': y['train'].get('max_epochs'),
            'patience': y['train'].get('patience'),
            'grad_clip': y['train'].get('grad_clip'),
            'seed': y['train'].get('seed'),
            # 损失函数参数
            'lambda_risk_reg': y['loss'].get('lambda_risk_reg'),
            'lambda_risk_cls': y['loss'].get('lambda_risk_cls'),
        }
    except Exception as e:
        print(f"配置解析失败: {path} | {e}")
        return None

# ===================== 最优筛选 =====================
def select_best_models(df):
    sort_key = CFG['sorting']['model_comparison']['sort_by']
    asc = CFG['sorting']['model_comparison']['ascending']
    best = []
    for mt in df['model_type'].unique():
        sub = df[df['model_type'] == mt].dropna(subset=[sort_key])
        if sub.empty: continue
        best.append(sub.sort_values(sort_key, ascending=asc).iloc[0])
    return pd.DataFrame(best)

def select_best_ablation(df):
    sort_key = CFG['sorting']['ablation_study']['sort_by']
    asc = CFG['sorting']['ablation_study']['ascending']
    sub = df[(df['model_type'] == 'mtrp')].dropna(subset=[sort_key])
    best = []
    for ab in sub['ablation'].unique():
        g = sub[sub['ablation'] == ab]
        if g.empty: continue
        best.append(g.sort_values(sort_key, ascending=asc).iloc[0])
    return pd.DataFrame(best)

# ===================== 复制报告+追加原始配置 (已修改) =====================
def copy_best_with_params(df, prefix):
    if df.empty:
        return
    for _, r in df.iterrows():
        src_txt = os.path.join(r['exp_path'], 'eval', 'evaluation_report.txt')
        if not os.path.exists(src_txt):
            continue
        
        # 文件名
        mt, abl = r['model_type'], r['ablation']
        filename = f"{prefix}_{mt}_{abl}.txt" if abl != 'none' else f"{prefix}_{mt}.txt"
        dst_txt = os.path.join(OUTPUT_ROOT, filename)

        # 1. 读取原评估报告
        with open(src_txt, 'r', encoding='utf-8') as f:
            report_content = f.read()

        # 2. 读取对应的原始 run_config.yaml 全文
        cfg_src_path = os.path.join(r['exp_path'], 'run_config.yaml')
        cfg_content = ""
        if os.path.exists(cfg_src_path):
            with open(cfg_src_path, 'r', encoding='utf-8') as f:
                cfg_content = f.read()
        
        # 3. 拼接内容，保留YAML原始缩进和格式
        final_content = f"""{report_content}

============================================================
【附录：实验原始完整配置 run_config.yaml】
============================================================
{cfg_content}
"""
        # 写入最终文件
        with open(dst_txt, 'w', encoding='utf-8') as f:
            f.write(final_content)
        print(f"✅ 生成带完整配置报告：{filename}")

# ===================== 主流程 =====================
def main():
    os.makedirs(OUTPUT_ROOT, exist_ok=True)
    all_data = []

    # 遍历所有实验
    for exp_name in os.listdir(RUNS_DIR):
        exp_path = os.path.join(RUNS_DIR, exp_name)
        if not os.path.isdir(exp_path):
            continue
        
        cfg_path = os.path.join(exp_path, "run_config.yaml")
        if not os.path.exists(cfg_path):
            continue

        # 解析配置 + 指标
        cfg = parse_exp_config(cfg_path)
        metrics = get_metrics(exp_path)
        if cfg and metrics:
            # 合并：实验ID + 全部参数 + 全部指标
            all_data.append({'exp_id': exp_name, **cfg, **metrics})

    if not all_data:
        print("❌ 未找到有效实验数据")
        return

    # 构建 DataFrame
    df = pd.DataFrame(all_data)
    
    # ===================== 关键：删除 exp_path 列，不写入CSV =====================
    df = df.drop(columns=['exp_path'], errors='ignore')

    # ===================== 列排序逻辑 =====================
    # 1. 确定所有存在的统一列
    existing_unified_cols = [c for c in UNIFIED_COLS if c in df.columns]
    # 2. 确定剩下的列 (排除统一列)
    remaining_cols = [c for c in df.columns if c not in existing_unified_cols]
    # 3. 重新排列 DataFrame：统一列在前，其余在后
    df = df[existing_unified_cols + remaining_cols]

    # ===================== 全量实验按指定指标由好到差排序 =====================
    sort_key = CFG['sorting']['model_comparison']['sort_by']
    ascending = CFG['sorting']['model_comparison']['ascending']
    # 去除排序键为空的行并排序
    df = df.dropna(subset=[sort_key]).sort_values(by=sort_key, ascending=ascending)

    # 导出全量CSV (已按统一列排序 + 指标优劣排序)
    full_csv_path = os.path.join(OUTPUT_ROOT, "全量实验指标汇总.csv")
    df.to_csv(full_csv_path, index=False, encoding='utf-8-sig')
    print(f"✅ 全量实验参数+指标已保存：{full_csv_path}")

    # 生成模型对比表 (使用统一列)
    df_model = select_best_models(df)
    df_model[existing_unified_cols].to_csv(os.path.join(OUTPUT_ROOT, "模型对比结果.csv"), index=False, encoding='utf-8-sig')

    # 生成消融实验表 (统一列 + ablation列)
    df_ablate = select_best_ablation(df)
    # 构造消融表专用列：把 'ablation' 插在 'model_type' 后面
    ablate_cols = existing_unified_cols.copy()
    if 'ablation' in df_ablate.columns and 'ablation' not in ablate_cols:
        ablate_cols.insert(1, 'ablation')
    df_ablate[ablate_cols].to_csv(os.path.join(OUTPUT_ROOT, "消融实验结果.csv"), index=False, encoding='utf-8-sig')
    
    # 重新加载带exp_path的数据用于复制报告（不影响CSV）
    df_with_path = pd.DataFrame(all_data)
    df_model_with_path = select_best_models(df_with_path)
    df_ablate_with_path = select_best_ablation(df_with_path)

    # 生成带完整配置的最优报告
    print("\n📄 生成最优实验报告（含原始完整配置）")
    copy_best_with_params(df_model_with_path, "最优模型报告")
    copy_best_with_params(df_ablate_with_path, "最优消融报告")

    # 控制台预览
    print("\n" + "="*100)
    print("📌 模型对比结果")
    print(df_model[existing_unified_cols].to_string(index=False))
    print("\n📌 消融实验结果")
    print(df_ablate[ablate_cols].to_string(index=False))
    print("="*100)
    print(f"🎉 全部完成！所有文件保存在：{OUTPUT_ROOT}")

if __name__ == "__main__":
    main()
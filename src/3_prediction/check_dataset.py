import os
import pickle
import pandas as pd
import numpy as np

def main():
    # ================= 配置路径 =================
    pkl_path = r"D:\Local\DynamicCapRisk\data\dataset\dataset_aug-False.pkl"
    output_dir = r"D:\Local\DynamicCapRisk\output\3_prediction"
    
    # 确保输出目录存在
    os.makedirs(output_dir, exist_ok=True)
    
    # ================= 1. 加载数据 =================
    print(f"[1/5] 正在加载数据集: {pkl_path}")
    if not os.path.exists(pkl_path):
        raise FileNotFoundError(f"找不到文件: {pkl_path}")
        
    with open(pkl_path, 'rb') as f:
        data = pickle.load(f)
    
    # ================= 2. 检查顶层结构 =================
    print("\n[2/5] 检查数据集顶层结构...")
    top_level_keys = list(data.keys())
    print(f"  顶层键: {top_level_keys}")
    
    splits = ['train', 'val', 'test']
    available_splits = [s for s in splits if s in data]
    
    # 检查标准化参数
    if 'norm' in data:
        print(f"  标准化参数: mu shape={data['norm']['mu'].shape}, sigma shape={data['norm']['sigma'].shape}")
    
    # 检查特征名称
    feature_names = data.get('feature_names', [])
    print(f"  特征名称列表长度: {len(feature_names)}")
    if len(feature_names) > 0:
        print(f"  前5个特征: {feature_names[:5]}...")

    # ================= 3. 生成详细统计报告 (TXT) =================
    txt_report_path = os.path.join(output_dir, "dataset_check_report.txt")
    print(f"\n[3/5] 正在生成详细报告: {txt_report_path}")
    
    with open(txt_report_path, 'w', encoding='utf-8') as f:
        f.write("="*80 + "\n")
        f.write("MT-JP 数据集检查报告\n")
        f.write("="*80 + "\n\n")
        
        # --- 全局信息 ---
        f.write("[1. 全局信息]\n")
        f.write(f"源文件: {pkl_path}\n")
        f.write(f"特征维度 (D): {len(feature_names)}\n")
        if len(feature_names) > 0:
            f.write("特征列表:\n")
            for i, name in enumerate(feature_names):
                f.write(f"  [{i:2d}] {name}\n")
        f.write("\n")
        
        if 'norm' in data:
            f.write("[2. 标准化参数 (mu / sigma)]\n")
            mu = data['norm']['mu']
            sigma = data['norm']['sigma']
            for i, name in enumerate(feature_names):
                f.write(f"  {name:20s}: mu={mu[i]:.6f}, sigma={sigma[i]:.6f}\n")
            f.write("\n")

        # --- 各 Split 信息 ---
        f.write("[3. 数据子集详情]\n")
        for split_name in available_splits:
            split = data[split_name]
            f.write(f"\n--- {split_name.upper()} ---\n")
            
            X = split['X']
            f.write(f"  X shape: {X.shape} (N, T, D)\n")
            f.write(f"  X dtype: {X.dtype}\n")
            f.write(f"  X value range: [{X.min():.4f}, {X.max():.4f}]\n")
            
            # 标签统计
            ya = split['y_ability']
            yr = split['y_risk_reg']
            yc = split['y_risk_cls']
            
            f.write(f"  y_ability:  N={len(ya)}, range=[{ya.min():.4f}, {ya.max():.4f}], mean={ya.mean():.4f}\n")
            f.write(f"  y_risk_reg: N={len(yr)}, range=[{yr.min():.4f}, {yr.max():.4f}], mean={yr.mean():.4f}\n")
            
            cls_counts = pd.Series(yc).value_counts().sort_index()
            f.write(f"  y_risk_cls: Distribution -> {cls_counts.to_dict()}\n")
            
            # Meta 信息
            meta = split['meta']
            f.write(f"  meta columns: {list(meta.columns)}\n")
            if 'augmented' in meta.columns:
                f.write(f"  Augmented samples: {meta['augmented'].sum()} / {len(meta)}\n")

        # --- 样本抽查 ---
        f.write("\n\n[4. 样本抽查 (Train Set 前 3 个样本)]\n")
        if 'train' in data:
            X_tr = data['train']['X']
            ya_tr = data['train']['y_ability']
            yr_tr = data['train']['y_risk_reg']
            yc_tr = data['train']['y_risk_cls']
            meta_tr = data['train']['meta']
            
            n_samples_to_show = min(3, len(X_tr))
            for i in range(n_samples_to_show):
                f.write(f"\n  --- Sample Index {i} ---\n")
                f.write(f"  Meta: {meta_tr.iloc[i].to_dict()}\n")
                f.write(f"  Labels: Ability={ya_tr[i]:.4f}, Risk_Reg={yr_tr[i]:.4f}, Risk_Cls={yc_tr[i]}\n")
                f.write(f"  X (First time step, All features):\n")
                # 打印第一个时间步的所有特征值
                for j, val in enumerate(X_tr[i, 0, :]):
                    fname = feature_names[j] if j < len(feature_names) else f"feat_{j}"
                    f.write(f"    [{j:2d}] {fname:20s} = {val:.6f}\n")

    # ================= 4. 生成 Meta 汇总表 (CSV) =================
    csv_report_path = os.path.join(output_dir, "dataset_meta_summary.csv")
    print(f"[4/5] 正在生成 Meta 汇总表: {csv_report_path}")
    
    all_meta_list = []
    for split_name in available_splits:
        meta_df = data[split_name]['meta'].copy()
        meta_df['split'] = split_name
        meta_df['y_ability'] = data[split_name]['y_ability']
        meta_df['y_risk_reg'] = data[split_name]['y_risk_reg']
        meta_df['y_risk_cls'] = data[split_name]['y_risk_cls']
        all_meta_list.append(meta_df)
    
    if all_meta_list:
        full_meta = pd.concat(all_meta_list, ignore_index=True)
        full_meta.to_csv(csv_report_path, index=False, encoding='utf-8-sig')

    # ================= 5. 简单的完整性检查 =================
    print("\n[5/5] 执行完整性检查...")
    all_ok = True
    if 'train' in data:
        X = data['train']['X']
        # 检查是否有 NaN
        if np.isnan(X).any():
            print("  ⚠️  警告: 训练集 X 中存在 NaN 值!")
            all_ok = False
        else:
            print("  ✅ 训练集 X 无 NaN 值.")
            
        # 检查维度
        if X.shape[2] != len(feature_names):
            print(f"  ⚠️  警告: 特征维度不匹配! X_D={X.shape[2]}, Names_D={len(feature_names)}")

    print(f"\n检查完成! 结果已保存至: {output_dir}")
    if all_ok:
        print("数据集看起来是完整的。")

if __name__ == "__main__":
    main()
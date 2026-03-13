import numpy as np
import pandas as pd
import os
import argparse

def validate_feature_names_consistency(df):
    """校验CSV中特征名的一致性（第一列和第一行必须完全匹配）"""
    row_features = df.iloc[:, 0].dropna().tolist()
    col_features = df.columns[1:].tolist()
    
    if set(row_features) != set(col_features):
        raise ValueError(
            f"特征名不一致！\n行特征：{row_features}\n列特征：{col_features}\n请确保第一列和第一行的特征名完全相同"
        )
    return row_features

def read_judgment_matrix_from_csv(csv_path):
    """
    从CSV读取打分，自动生成判断矩阵（极简填写，自动补全）
    :param csv_path: CSV文件路径
    :return: (判断矩阵, 特征名列表)
    """
    df = pd.read_csv(csv_path, encoding="utf-8-sig")
    feature_names = validate_feature_names_consistency(df)
    n = len(feature_names)
    judgment_matrix = np.eye(n)
    feat2idx = {feat: i for i, feat in enumerate(feature_names)}
    
    for row_idx in range(n):
        row_feat = feature_names[row_idx]
        for col_idx in range(row_idx + 1, n):
            col_feat = feature_names[col_idx]
            cell_val = df.loc[df.iloc[:, 0] == row_feat, col_feat].values
            if len(cell_val) == 0:
                raise ValueError(f"CSV中未找到 {row_feat} 对 {col_feat} 的打分")
            val = cell_val[0]
            if pd.isna(val):
                continue
            try:
                val = float(val)
            except ValueError:
                raise ValueError(f"打分错误：{row_feat} 对 {col_feat} 的值 '{val}' 不是数字")
            judgment_matrix[row_idx, col_idx] = val
            judgment_matrix[col_idx, row_idx] = 1 / val if val != 0 else 0
    
    if np.any(judgment_matrix == 0):
        raise ValueError("判断矩阵存在0值！请检查CSV中的打分，重要度不能为0")
    
    print(f"成功从CSV读取判断矩阵（{n}×{n}），特征列表：{feature_names}")
    return judgment_matrix, feature_names

def validate_judgment_matrix(matrix):
    """
    验证判断矩阵的合法性：
    1. 方阵  2. 对角线为1  3. 互反性  4. 一致性检验（CR < 0.1）
    """
    n = matrix.shape[0]
    if matrix.shape[0] != matrix.shape[1]:
        raise ValueError("判断矩阵必须是方阵！")
    if not np.allclose(np.diag(matrix), np.ones(n)):
        raise ValueError("判断矩阵对角线必须全为1！")
    for i in range(n):
        for j in range(n):
            if i != j and not np.isclose(matrix[i,j], 1/matrix[j,i]):
                raise ValueError(f"判断矩阵不满足互反性：matrix[{i},{j}] = {matrix[i,j]}, matrix[{j},{i}] = {matrix[j,i]}")
    
    eig_vals, eig_vecs = np.linalg.eig(matrix)
    max_eig = np.real(max(eig_vals))
    ci = (max_eig - n) / (n - 1) if n > 1 else 0
    ri_table = {1:0, 2:0, 3:0.58, 4:0.90, 5:1.12, 6:1.24, 7:1.32, 8:1.41, 9:1.45, 10:1.49}
    ri = ri_table.get(n, 1.5)
    cr = ci / ri if ri != 0 else 0
    
    if cr > 0.1:
        print(f"⚠️ 警告：判断矩阵一致性检验不通过（CR={cr:.3f} > 0.1），建议调整打分！")
    else:
        print(f"✅ 判断矩阵一致性检验通过（CR={cr:.3f} ≤ 0.1）")
    
    return max_eig, ci, cr

def calculate_ahp_weights(judgment_matrix=None, feature_names=None, csv_path=None, save_path=None):
    """
    计算AHP权重（支持两种输入方式：直接传矩阵 / 读CSV）
    :param judgment_matrix: 二维数组，判断矩阵（优先级低于csv_path）
    :param feature_names:   列表，特征名（优先级低于csv_path）
    :param csv_path:        CSV打分文件路径（推荐使用）
    :param save_path:       权重保存的CSV路径（None则不保存）
    :return: 字典，{特征名: 权重}
    """
    if csv_path is not None:
        if not os.path.exists(csv_path):
            raise FileNotFoundError(f"CSV文件不存在：{csv_path}")
        judgment_matrix, feature_names = read_judgment_matrix_from_csv(csv_path)
    
    if judgment_matrix is None or feature_names is None:
        raise ValueError("必须传入 judgment_matrix+feature_names 或 csv_path！")
    
    max_eig, ci, cr = validate_judgment_matrix(judgment_matrix)
    
    eig_vals, eig_vecs = np.linalg.eig(judgment_matrix)
    max_eig_idx = np.argmax(np.real(eig_vals))
    eig_vec = np.real(eig_vecs[:, max_eig_idx])
    weights = eig_vec / np.sum(eig_vec)
    weight_dict = dict(zip(feature_names, weights))
    
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        weight_df = pd.DataFrame({
            "feature_name":   feature_names,
            "ahp_weight":     weights.round(4),
            "max_eigenvalue": [round(max_eig, 4)] * len(feature_names),
            "CI":             [round(ci, 4)]       * len(feature_names),
            "CR":             [round(cr, 4)]       * len(feature_names)
        })
        weight_df.to_csv(save_path, index=False, encoding="utf-8-sig")
        print(f"📁 AHP权重已保存至：{save_path}")
    
    print("\n📊 生成的AHP权重：")
    for feat, w in sorted(weight_dict.items(), key=lambda x: x[1], reverse=True):
        print(f"  {feat}: {w:.4f}")
    
    return weight_dict


# =============================================================================
# CLI 入口
# =============================================================================

def parse_args():
    parser = argparse.ArgumentParser(
        description="AHP权重计算器",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例用法:
python src/utils/ahp_calculator.py -c data/raw/ahp_afl_judgment_matrix.csv -s output/1_capability_assessment/results/Afl_ahp_weights.csv

# 主层权重：w_veh / w_geo / w_sign
python src/utils/ahp_calculator.py -c data/raw/ahp_risk_field_main.csv -s output/2_risk_assessment/results/risk_field_main_weights.csv

# 设施子层权重：lambda_1(sign_density) / lambda_2(work_zone)
python src/utils/ahp_calculator.py -c data/raw/ahp_risk_field_sign.csv -s output/2_risk_assessment/results/risk_field_sign_weights.csv
        """
    )
    parser.add_argument(
        "-c", "--csv_path",
        type=str,
        required=True,
        help="判断矩阵CSV文件路径（必填）"
    )
    parser.add_argument(
        "-s", "--save_path",
        type=str,
        default=None,
        help="权重输出CSV路径（不填则只打印不保存）"
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    calculate_ahp_weights(
        csv_path=args.csv_path,
        save_path=args.save_path
    )
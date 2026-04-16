import pandas as pd
import numpy as np

# ===================== 配置路径 =====================
QUESTIONNAIRE_PATH = r"D:\Local\DynamicCapRisk\data\processed\questionnaire_preprocessed.csv"
MAPPING_PATH = r"D:\Local\DynamicCapRisk\data\raw\被试-实验ID映射.csv"
# 🔥 专业文件名：问卷数据映射到67个实验样本
OUTPUT_PATH = r"D:\Local\DynamicCapRisk\data\processed\questionnaire_exp_mapped_67samples.csv"
# ====================================================

def main():
    # 1. 读取32个被试的问卷数据
    df_ques = pd.read_csv(QUESTIONNAIRE_PATH, index_col=0)
    df_ques = df_ques.reset_index(names="被试ID")
    print(f"✅ 读取问卷数据：{len(df_ques)} 个被试")

    # 2. 读取被试→实验ID映射（67个样本）
    df_map = pd.read_csv(MAPPING_PATH)
    print(f"✅ 读取映射关系：{len(df_map)} 个实验样本")

    # 3. 按ID匹配，复制问卷数据到对应实验样本
    df_result = pd.merge(df_map, df_ques, on="被试ID", how="left")

    # 🔥 小数保留 4 位
    for col in df_result.columns:
        if df_result[col].dtype in [np.float64, np.float32]:
            df_result[col] = df_result[col].round(4)

    # 4. 保存文件
    df_result.to_csv(OUTPUT_PATH, index=False, encoding="utf-8-sig")
    
    print(f"\n🎉 处理完成！")
    print(f"📄 输出文件：{OUTPUT_PATH}")
    print(f"📊 数据行数：{len(df_result)} 行（67个实验样本）")
    print(f"🔢 小数位数：统一保留 4 位")

if __name__ == "__main__":
    main()
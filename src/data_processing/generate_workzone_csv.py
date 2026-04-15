# D:\Local\DynamicCapRisk\src\data_processing\generate_workzone_csv.py

import pandas as pd
import argparse
import os

def process_sheet(df, sheet_name):
    """
    处理单个工作表，生成连续距离的CSV数据
    【最终修复】通过标识牌的"开始"/"结束"关键词识别区间，彻底解决中间行干扰问题
    """
    df = df.copy()
    
    # 1. 确保距离列是数值类型
    df['距离 (m)'] = pd.to_numeric(df['距离 (m)'], errors='coerce').fillna(0).astype(int)
    
    # 2. 生成连续距离序列
    max_dist = df['距离 (m)'].max()
    continuous_dist = pd.DataFrame({'距离 (m)': range(0, max_dist + 1)})
    
    # 3. 分离数据处理
    df_sorted = df.sort_values('距离 (m)').reset_index(drop=True)
    
    # 3.1 处理【需要连续填充】的列（车道数、限速、施工区状态、车道变化预告）
    fill_cols = ['车道数', '限速 (km/h)', '施工区状态', '车道变化预告']
    df_fill = df_sorted[['距离 (m)'] + fill_cols].copy()
    
    merged_fill = pd.merge_asof(
        continuous_dist, 
        df_fill, 
        on='距离 (m)', 
        direction='backward'
    )
    merged_fill[fill_cols] = merged_fill[fill_cols].ffill().bfill()
    
    # 3.2 处理【仅在原始点显示】的列（标识牌类型、标识牌编码）
    sign_cols = ['距离 (m)', '标识牌类型', '标识牌编码']
    df_sign = df_sorted[sign_cols].copy()
    
    merged_sign = pd.merge(
        continuous_dist,
        df_sign,
        on='距离 (m)',
        how='left'
    )
    merged_sign[['标识牌类型', '标识牌编码']] = merged_sign[['标识牌类型', '标识牌编码']].fillna('-')
    
    # 4. 合并基础数据
    merged = pd.merge(merged_fill, merged_sign, on='距离 (m)')
    
    # 5. 【核心修复】道路几何类型处理
    merged['道路几何类型'] = 'straight'  # 先默认全为straight
    
    # 5.1 优先通过"标识牌类型"的"开始"/"结束"关键词识别区间（适配work_zone_2）
    intervals = []
    current_start = None
    current_type = None
    
    for idx, row in df_sorted.iterrows():
        sign_type = str(row['标识牌类型'])
        geom_type = str(row['道路几何类型'])
        
        # 遇到"开始"标识：记录起点和类型
        if '开始' in sign_type:
            current_start = row['距离 (m)']
            current_type = geom_type
        # 遇到"结束"标识：记录终点，保存区间
        elif '结束' in sign_type and current_start is not None:
            current_end = row['距离 (m)']
            intervals.append((current_start, current_end, current_type))
            current_start = None
            current_type = None
    
    # 5.2 特殊处理：如果没有识别到开始/结束标识（如work_zone_3全段bend）
    if not intervals:
        # 检查原始数据第一行的道路几何类型，如果非straight则全段应用
        first_geom = str(df_sorted.iloc[0]['道路几何类型'])
        if first_geom != 'straight' and first_geom != 'nan':
            intervals.append((0, max_dist, first_geom))
    
    # 5.3 应用所有区间
    for start, end, geom_type in intervals:
        mask = (merged['距离 (m)'] >= start) & (merged['距离 (m)'] <= end)
        merged.loc[mask, '道路几何类型'] = geom_type
    
    # 6. 统一列顺序
    final_cols = [
        '距离 (m)', '车道数', '限速 (km/h)', '施工区状态', 
        '标识牌类型', '标识牌编码', '车道变化预告', '道路几何类型'
    ]
    return merged[final_cols]

def main(excel_path, output_dir='data/processed'):
    os.makedirs(output_dir, exist_ok=True)
    xl = pd.ExcelFile(excel_path)
    sheet_names = ['work_zone_1', 'work_zone_2', 'work_zone_3']
    
    for sheet in sheet_names:
        if sheet not in xl.sheet_names:
            print(f"警告：工作表 {sheet} 不存在，跳过")
            continue
        
        print(f"正在处理 {sheet} ...")
        df_raw = pd.read_excel(xl, sheet_name=sheet)
        df_processed = process_sheet(df_raw, sheet)
        
        output_path = os.path.join(output_dir, f"{sheet}_continuous.csv")
        df_processed.to_csv(output_path, index=False, encoding='utf-8-sig')
        print(f"已保存：{output_path}（共 {len(df_processed)} 行）")

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='施工区数据处理工具')
    parser.add_argument('-i', '--excel_path', type=str, default='data/raw/施工区元素.xlsx', help='输入Excel路径')
    parser.add_argument('-o', '--output_dir', type=str, default='data/processed', help='输出目录')
    args = parser.parse_args()
    main(args.excel_path, args.output_dir)
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""简洁的PKL文件读取脚本，通过命令行参数指定文件路径"""

import argparse
import pickle
import os
import pandas as pd

def read_pkl_file(file_path: str):
    """读取PKL文件，兼容pandas和原生pickle格式"""
    # 规范化路径（适配Windows反斜杠）
    file_path = os.path.normpath(file_path)
    
    # 检查文件是否存在
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"错误：文件不存在 -> {file_path}")
    
    # 检查文件后缀
    if not file_path.lower().endswith('.pkl'):
        raise ValueError(f"错误：文件不是PKL格式 -> {file_path}")
    
    # 读取文件（优先pandas，备用原生pickle）
    try:
        data = pd.read_pickle(file_path)
    except Exception:
        with open(file_path, 'rb') as f:
            data = pickle.load(f)
    
    return data

def main():
    # 解析命令行参数（修复required和default冲突）
    parser = argparse.ArgumentParser(description='读取指定路径的PKL文件')
    parser.add_argument('-p', '--path', 
                        type=str, 
                        # default='output/1_capability_assessment/Afl_capability_fluctuation.pkl',
                        # default='output/1_capability_assessment/Ad_result.pkl',
                        # default='output/2_risk_assessment/results/risk_list.pkl',
                        default='D:/Local/DynamicCapRisk/output/3_prediction/mtjp_dataset_aug-False.pkl',
                        help='PKL文件路径')
    args = parser.parse_args()
    
    # 读取并输出信息
    try:
        data = read_pkl_file(args.path)
        print(f"✅ 成功读取PKL文件：{args.path}")
        print(f"📊 数据类型：{type(data)}")
        
        if isinstance(data, pd.DataFrame):
            print(f"📏 数据形状：{data.shape}")
            print("\n🔍 前5行数据预览：")
            print(data.head())
    
    except Exception as e:
        print(f"❌ 读取失败：{str(e)}")
        exit(1)

if __name__ == '__main__':
    main()
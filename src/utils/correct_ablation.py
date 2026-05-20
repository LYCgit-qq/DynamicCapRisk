import os
import yaml

# ===================== 配置区域 =====================
# 目标根目录
ROOT_RUNS_DIR = r"D:\Local\DynamicCapRisk\output\3_prediction\runs"

# 需要匹配的文件夹后缀 (不区分大小写)
TARGET_SUFFIXES = ('lstm', 'gru', 'cnn_lstm')

# 配置文件名
CONFIG_FILENAME = "run_config.yaml"
# ===================================================

def modify_ablation_to_none():
    print(f"开始扫描目录: {ROOT_RUNS_DIR}")
    print(f"目标文件夹后缀: {TARGET_SUFFIXES}")
    print("-" * 60)
    
    modified_count = 0
    skipped_count = 0
    error_count = 0

    # 遍历根目录下的所有条目
    for item in os.listdir(ROOT_RUNS_DIR):
        item_path = os.path.join(ROOT_RUNS_DIR, item)
        
        # 1. 检查是否为文件夹
        if not os.path.isdir(item_path):
            continue

        # 2. 检查文件夹名是否以目标后缀结尾
        # (转换为小写进行比较，不区分大小写)
        if not item.lower().endswith(TARGET_SUFFIXES):
            continue

        # 3. 构造配置文件路径
        yaml_path = os.path.join(item_path, CONFIG_FILENAME)
        
        if not os.path.exists(yaml_path):
            print(f"⚠️  跳过 [{item}]: 未找到 {CONFIG_FILENAME}")
            skipped_count += 1
            continue

        # 4. 读取并修改 YAML
        try:
            with open(yaml_path, 'r', encoding='utf-8') as f:
                data = yaml.safe_load(f)

            # 获取旧值用于日志
            old_val = data.get('model', {}).get('ablation')

            # 5. 执行修改
            if data and 'model' in data:
                data['model']['ablation'] = 'none'
                
                # 写回文件
                with open(yaml_path, 'w', encoding='utf-8') as f:
                    # allow_unicode=True 防止中文乱码
                    # default_flow_style=False 保持块级样式 (不变成行内字典)
                    # sort_keys=False 保持原有的键顺序
                    yaml.dump(data, f, allow_unicode=True, default_flow_style=False, sort_keys=False)
                
                print(f"✅ 修改成功 [{item}]: ablation: {old_val} -> none")
                modified_count += 1
            else:
                print(f"⚠️  跳过 [{item}]: YAML结构异常 (未找到 'model' 键)")
                skipped_count += 1

        except Exception as e:
            print(f"❌ 处理失败 [{item}]: {str(e)}")
            error_count += 1

    # 6. 汇总报告
    print("-" * 60)
    print(f"处理完成！")
    print(f"  成功修改: {modified_count} 个")
    print(f"  跳过: {skipped_count} 个")
    print(f"  报错: {error_count} 个")

if __name__ == "__main__":
    confirm = input("⚠️  警告：即将批量修改配置文件。建议先备份 runs 文件夹。\n输入 'yes' 继续，其他键取消: ")
    if confirm.lower() == 'yes':
        modify_ablation_to_none()
    else:
        print("操作已取消。")
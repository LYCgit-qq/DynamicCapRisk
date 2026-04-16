import os
import yaml
import time
import subprocess
import itertools
from datetime import datetime
from typing import Dict, List

# ===================== 【核心配置】你只需要改这里的参数 =====================
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
# 基础配置文件路径（自动适配，不用改）
BASE_CONFIG_PATH = os.path.join(SCRIPT_DIR, "../../config/trainer_dl.yaml")

# 要遍历的参数网格（自由增删，脚本会自动全组合遍历）
PARAM_GRID = {
    # 1. 模型类型（必选）
    "model.model_type": ["mtrp", "lstm", "gru", "cnn_lstm"],
    # 2. 学习率
    "optimizer.lr": [5e-4, 8e-4, 1e-3],
    # 3. 批次大小
    "train.batch_size": [128, 256],
    # 4. Dropout
    "model.dropout": [0.2, 0.25, 0.3],
    # 5. 分类损失权重
    "loss.lambda_risk_cls": [1.0, 1.2, 1.5],
    # 6. 模型隐藏维度 (基线模型 + MTRP通用)
    "model.baseline_hidden": [64, 128],
    "model.d_model": [64, 128],
    # 7. MTRP专用消融实验（可选，只对mtrp生效）
    # "model.ablation": ["none", "no_cross_attn", "single_modal"]
}

# 临时配置文件存放目录
TEMP_CONFIG_DIR = os.path.join(SCRIPT_DIR, "temp_configs")
# 训练脚本路径（固定，不用改）
TRAIN_SCRIPT = os.path.join(SCRIPT_DIR, "trainer_dl.py")
# 日志文件（记录所有实验结果）
LOG_FILE = os.path.join(SCRIPT_DIR, f"grid_search_log_{datetime.now().strftime('%Y%m%d')}.txt")
# ======================================================================

def load_base_config() -> Dict:
    """加载基础配置文件"""
    with open(BASE_CONFIG_PATH, encoding="utf-8") as f:
        return yaml.safe_load(f)

def save_config(config: Dict, path: str):
    """保存配置文件"""
    with open(path, "w", encoding="utf-8") as f:
        yaml.dump(config, f, sort_keys=False, allow_unicode=True)

def set_nested_param(config: Dict, key: str, value):
    """递归修改嵌套配置（如 model.model_type → config['model']['model_type']）"""
    keys = key.split(".")
    d = config
    for k in keys[:-1]:
        d = d[k]
    d[keys[-1]] = value

def generate_param_combinations(param_grid: Dict) -> List[Dict]:
    """生成所有参数组合"""
    keys = param_grid.keys()
    values = param_grid.values()
    combinations = []
    for combo in itertools.product(*values):
        combo_dict = dict(zip(keys, combo))
        combinations.append(combo_dict)
    return combinations

def write_log(message: str):
    """写入日志"""
    time_str = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    log_msg = f"[{time_str}] {message}\n"
    print(log_msg.strip())
    with open(LOG_FILE, "a", encoding="utf-8") as f:
        f.write(log_msg)

def main():
    # 创建临时文件夹
    os.makedirs(TEMP_CONFIG_DIR, exist_ok=True)
    # 加载基础配置
    base_config = load_base_config()
    # 生成所有参数组合
    combos = generate_param_combinations(PARAM_GRID)
    total = len(combos)
    write_log(f"✅ 网格搜索启动！总实验次数：{total}")

    # 遍历所有参数组合
    for idx, combo in enumerate(combos, 1):
        start_time = time.time()
        exp_name = "_".join([f"{k.split('.')[-1]}_{v}" for k, v in combo.items()])
        write_log(f"\n===== 开始第 {idx}/{total} 个实验 | {exp_name} =====")

        # 复制基础配置
        curr_config = base_config.copy()
        # 应用当前参数组合
        for key, value in combo.items():
            set_nested_param(curr_config, key, value)

        # 生成临时配置文件
        config_path = os.path.join(TEMP_CONFIG_DIR, f"temp_{exp_name}.yaml")
        save_config(curr_config, config_path)

        # 启动训练（关键命令）
        try:
            # 调用你的训练脚本，指定配置文件
            cmd = [sys.executable, TRAIN_SCRIPT, "-c", config_path]
            # 指定工作目录为项目根目录，不修改yaml也能找到数据集
            project_root = "/root/autodl-tmp/DynamicCapRisk"
            subprocess.run(cmd, check=True, cwd=project_root)
            cost = round(time.time() - start_time, 2)
            write_log(f"✅ 实验成功！耗时：{cost}s | {exp_name}")
        except subprocess.CalledProcessError as e:
            write_log(f"❌ 实验失败！错误：{str(e)} | {exp_name}")
        except Exception as e:
            write_log(f"❌ 未知错误：{str(e)} | {exp_name}")

    write_log(f"\n🎉 所有实验完成！总次数：{total}")

if __name__ == "__main__":
    import sys
    main()
'''
src/3_prediction/run_grid_search.py
pkill -f "/root/autodl-tmp/DynamicCapRisk"
'''

import os
import sys
import json
import hashlib
import yaml
import time
import subprocess
import itertools
from datetime import datetime
from typing import Dict, List
from concurrent.futures import ProcessPoolExecutor, as_completed
from multiprocessing import Lock

# ===================== 【配置文件路径】唯一硬编码路径 =====================
GRID_SEARCH_CONFIG_PATH = "config/run_grid_search.yaml"
# ======================================================================

# 加载配置文件
def load_grid_search_config(config_path: str) -> Dict:
    """加载网格搜索外置配置"""
    with open(config_path, encoding="utf-8") as f:
        return yaml.safe_load(f)

# 加载所有配置
grid_config = load_grid_search_config(GRID_SEARCH_CONFIG_PATH)

# ===================== 从配置文件读取核心参数（支持多版本） =====================
PROJECT_ROOT = grid_config["project_root"]
BASE_CONFIG_PATH = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), grid_config["base_config_path"]))
TRAIN_SCRIPT = os.path.abspath(os.path.join(PROJECT_ROOT, grid_config["train_script"]))
MAX_PARALLEL_WORKERS = grid_config["max_parallel_workers"]

# 读取激活的参数版本 + 对应参数网格
ACTIVE_VERSION = grid_config["active_param_version"]
PARAM_GRID = grid_config["param_grids"][ACTIVE_VERSION]
# ======================================================================

# 自动生成的目录配置（无需修改）
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
OUTPUT_ROOT = os.path.join(PROJECT_ROOT, "output/3_prediction")
TEMP_CONFIG_DIR = os.path.join(OUTPUT_ROOT, "temp_configs")
LOG_DIR = os.path.join(OUTPUT_ROOT, "logs")
LOG_FILE = os.path.join(LOG_DIR, f"grid_search_log_{datetime.now().strftime('%Y%m%d')}.txt")
PROGRESS_FILE = os.path.join(OUTPUT_ROOT, "grid_search_progress.jsonl")

# 全局锁（多进程安全）
LOG_LOCK = Lock()
PROGRESS_LOCK = Lock()

def load_base_config() -> Dict:
    """加载基础配置文件"""
    with open(BASE_CONFIG_PATH, encoding="utf-8") as f:
        return yaml.safe_load(f)

def save_config(config: Dict, path: str):
    """保存配置文件"""
    with open(path, "w", encoding="utf-8") as f:
        yaml.dump(config, f, sort_keys=False, allow_unicode=True)

def set_nested_param(config: Dict, key: str, value):
    """递归修改嵌套配置"""
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

def get_exp_id(combo: Dict) -> str:
    """生成实验唯一ID (基于参数内容的Hash)"""
    combo_str = json.dumps(combo, sort_keys=True)
    return hashlib.md5(combo_str.encode()).hexdigest()[:8]

def load_finished_experiments() -> set:
    """
    【修改2】严格加载逻辑：
    只有状态为 'success' 的才跳过。
    'failed' (报错) 和 'running' (中断) 都会被重新加入队列。
    """
    finished_ids = set()
    if not os.path.exists(PROGRESS_FILE):
        return finished_ids
    
    with open(PROGRESS_FILE, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                data = json.loads(line)
                if data.get('status') == 'success':
                    finished_ids.add(data['exp_id'])
            except Exception:
                pass
    return finished_ids

def update_progress(exp_id: str, combo: Dict, status: str, msg: str = ""):
    """更新进度文件 (线程安全)"""
    with PROGRESS_LOCK:
        lines = []
        if os.path.exists(PROGRESS_FILE):
            with open(PROGRESS_FILE, "r", encoding="utf-8") as f:
                lines = f.readlines()
        
        log_entry = {
            "exp_id": exp_id,
            "status": status,
            "params": combo,
            "message": msg,
            "timestamp": datetime.now().isoformat(),
            "param_version": ACTIVE_VERSION  # 记录参数版本，可追溯
        }
        new_line = json.dumps(log_entry, ensure_ascii=False) + "\n"
        
        found = False
        for i, line in enumerate(lines):
            if exp_id in line:
                lines[i] = new_line
                found = True
                break
        if not found:
            lines.append(new_line)
        
        with open(PROGRESS_FILE, "w", encoding="utf-8") as f:
            f.writelines(lines)

def write_log(message: str):
    """写入日志"""
    time_str = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    log_msg = f"[{time_str}] {message}\n"
    print(log_msg.strip())
    with LOG_LOCK:
        with open(LOG_FILE, "a", encoding="utf-8") as f:
            f.write(log_msg)

def run_single_experiment(idx, combo, total, base_config):
    """单个实验的运行逻辑"""
    exp_id = get_exp_id(combo)
    exp_name = "_".join([f"{k.split('.')[-1]}_{v}" for k, v in combo.items()])
    start_time = time.time()
    
    update_progress(exp_id, combo, "running")
    write_log(f"===== [{idx}/{total}] 启动 | 版本:{ACTIVE_VERSION} | ID:{exp_id} | {exp_name} =====")

    curr_config = base_config.copy()
    for key, value in combo.items():
        set_nested_param(curr_config, key, value)
    
    config_path = os.path.join(TEMP_CONFIG_DIR, f"temp_{exp_id}.yaml")
    save_config(curr_config, config_path)

    try:
        cmd = [sys.executable, TRAIN_SCRIPT, "-c", config_path]
        subprocess.run(cmd, check=True, cwd=PROJECT_ROOT)
        
        cost = round(time.time() - start_time, 2)
        write_log(f"✅ [{idx}/{total}] 成功 | 耗时:{cost}s | {exp_name}")
        update_progress(exp_id, combo, "success", f"cost:{cost}s")
        
    except subprocess.CalledProcessError as e:
        err_msg = f"ProcessError: {e.returncode}"
        write_log(f"❌ [{idx}/{total}] 失败 | {err_msg} | {exp_name}")
        update_progress(exp_id, combo, "failed", err_msg)
    except Exception as e:
        err_msg = f"Exception: {str(e)}"
        write_log(f"❌ [{idx}/{total}] 异常 | {err_msg} | {exp_name}")
        update_progress(exp_id, combo, "failed", err_msg)

def main():
    os.makedirs(TEMP_CONFIG_DIR, exist_ok=True)
    os.makedirs(LOG_DIR, exist_ok=True)
    
    base_config = load_base_config()
    all_combos = generate_param_combinations(PARAM_GRID)
    total_count = len(all_combos)
    
    finished_ids = load_finished_experiments()
    task_queue = []
    for idx, combo in enumerate(all_combos, 1):
        eid = get_exp_id(combo)
        if eid in finished_ids:
            continue
        task_queue.append((idx, combo))
    
    remaining_count = len(task_queue)
    skip_count = total_count - remaining_count

    write_log(f"="*60)
    write_log(f"✅ 网格搜索初始化 | 当前参数版本: {ACTIVE_VERSION}")
    write_log(f"   总任务数: {total_count}")
    write_log(f"   已完成(跳过): {skip_count}")
    write_log(f"   待运行/重跑: {remaining_count}")
    write_log(f"   并行数: {MAX_PARALLEL_WORKERS}")
    write_log(f"="*60)

    if remaining_count == 0:
        write_log("🎉 所有实验均已成功完成！")
        return

    try:
        with ProcessPoolExecutor(max_workers=MAX_PARALLEL_WORKERS) as executor:
            futures = [executor.submit(run_single_experiment, idx, combo, total_count, base_config) for idx, combo in task_queue]
            for future in as_completed(futures):
                future.result()
    except KeyboardInterrupt:
        write_log("\n⚠️ 收到 Ctrl+C 信号，正在停止...")
        sys.exit(1)

    write_log(f"\n🎉 批次处理完成！")

if __name__ == "__main__":
    main()
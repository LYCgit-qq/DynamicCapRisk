import json
import os
from datetime import datetime, timedelta

# 固定你的进度文件路径（无需修改）
PROGRESS_FILE = "/root/autodl-tmp/DynamicCapRisk/output/3_prediction/grid_search_progress.jsonl"

def analyze_grid_progress():
    if not os.path.exists(PROGRESS_FILE):
        print("❌ 未找到进度文件，请先启动网格搜索")
        return

    # 存储所有实验
    experiments = []
    exp_ids = set()
    
    with open(PROGRESS_FILE, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                data = json.loads(line)
                exp_id = data["exp_id"]
                # 去重（避免重复记录）
                if exp_id not in exp_ids:
                    exp_ids.add(exp_id)
                    experiments.append(data)
            except:
                continue

    # ===================== 统计状态 =====================
    total = len(experiments)
    success = [e for e in experiments if e["status"] == "success"]
    failed = [e for e in experiments if e["status"] == "failed"]
    running = [e for e in experiments if e["status"] == "running"]
    remaining = total - len(success) - len(failed) - len(running)

    # ===================== 计算耗时 & 预估时间 =====================
    avg_cost = 0
    total_cost = 0
    estimate_remaining = "无法预估（暂无完成的实验）"
    estimate_end_time = "无法预估"

    if len(success) > 0:
        # 提取成功实验的耗时（秒）
        cost_list = []
        for e in success:
            try:
                cost = float(e["message"].split("cost:")[-1].replace("s", ""))
                cost_list.append(cost)
            except:
                continue
        
        if cost_list:
            avg_cost = sum(cost_list) / len(cost_list)
            total_cost = sum(cost_list)
            # 预估剩余时间 = 剩余数 * 平均耗时
            estimate_seconds = avg_cost * remaining
            estimate_remaining = str(timedelta(seconds=int(estimate_seconds)))
            # 预计完成时间
            now = datetime.now()
            end_time = now + timedelta(seconds=estimate_seconds)
            estimate_end_time = end_time.strftime("%Y-%m-%d %H:%M:%S")

    # ===================== 打印结果 =====================
    print("="*60)
    print("📊 网格搜索 进度统计 & 预估时间")
    print("="*60)
    print(f"🧪 总实验数：{total}")
    print(f"✅ 已完成：{len(success)}")
    print(f"❌ 已失败：{len(failed)}")
    print(f"🔄 运行中：{len(running)}")
    print(f"⏳ 待运行：{max(remaining, 0)}")
    print(f"📈 当前进度：{len(success)/total*100:.1f}%" if total>0 else "进度：0%")
    print("-"*60)
    print(f"⌛ 平均单个实验耗时：{avg_cost:.1f} 秒")
    print(f"⌛ 已累计运行：{str(timedelta(seconds=int(total_cost)))}")
    print(f"⏰ 预估剩余时间：{estimate_remaining}")
    print(f"🏁 预计完成时间：{estimate_end_time}")
    print("="*60)

if __name__ == "__main__":
    analyze_grid_progress()
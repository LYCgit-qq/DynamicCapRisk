"""问卷预处理工具模块
D:\Local\DynamicCapRisk\src\data_processing\questionnaire_processor.py
"""

import argparse
import logging
from pathlib import Path
from typing import Dict, Any, TypedDict, Literal

import pandas as pd
import numpy as np

from questionnaire_loader import load_questionnaire

# 项目根目录
BASE_DIR = Path(__file__).resolve().parents[2]

# ---------------------------------------------------------------------------
# 类型定义
# ---------------------------------------------------------------------------
DirectionType = Literal["正向", "负向", "中性"]
class IndicatorItem(TypedDict):
    ids: list[str]
    type: Literal["分类", "连续"]
    assign: dict | None
    direction: DirectionType | None
    custom: bool | None

IndicatorConfig = dict[str, IndicatorItem]

# ---------------------------------------------------------------------------
# 指标配置 【✅ 最终定稿：100%匹配Excel 0~95索引，无重复、无错误】
# 核心规则：
# 1. 分类题：实测数据为1起始（1=第一个选项）
# 2. 连续题：0-10分，负向=10-平均分，正向=直接平均分
# 3. 施工区经历：1/2=有经历(1)，3/4=无经历(0)
# ---------------------------------------------------------------------------
INDICATOR_CONFIG: IndicatorConfig = {
    # ===================== 一、个人基本信息（14项） =====================
    "性别": {"ids": ["81"], "type": "分类", "assign": {1: 1, 2: 0}, "direction": None, "custom": False},
    "年龄段": {"ids": ["82"], "type": "分类", "assign": {1:1,2:2,3:3,4:4,5:5}, "direction": None, "custom": False},
    "是否职业司机": {"ids": ["83"], "type": "分类", "assign": {1: 1, 2: 0}, "direction": None, "custom": False},
    "学历": {"ids": ["84"], "type": "分类", "assign": {1:1,2:2,3:3,4:4,5:5}, "direction": None, "custom": False},
    "年均收入": {"ids": ["85"], "type": "分类", "assign": {1:1,2:2,3:3,4:4}, "direction": None, "custom": False},
    "驾龄": {"ids": ["86"], "type": "分类", "assign": {1:1,2:2,3:3,4:4,5:5}, "direction": None, "custom": False},
    "每周开车频率": {"ids": ["87"], "type": "分类", "assign": {1:1,2:2,3:3,4:4}, "direction": None, "custom": False},
    "单次驾驶时长": {"ids": ["88"], "type": "分类", "assign": {1:1,2:2,3:3,4:4,5:5}, "direction": None, "custom": False},
    "是否本本族": {"ids": ["89"], "type": "分类", "assign": {1: 1, 2: 0}, "direction": None, "custom": False},
    "出行原因_通勤": {"ids": ["90"], "type": "分类", "assign": {0:0,1:1}, "direction": None, "custom": False},
    "出行原因_娱乐": {"ids": ["91"], "type": "分类", "assign": {0:0,1:1}, "direction": None, "custom": False},
    # "出行原因_谋生": {"ids": ["92"], "type": "分类", "assign": {0:0,1:1}, "direction": None, "custom": False},
    "出行原因_其他": {"ids": ["93"], "type": "分类", "assign": {0:0,1:1}, "direction": None, "custom": False},
    "事故经历": {"ids": ["94"], "type": "分类", "assign": {1:1,2:2}, "direction": None, "custom": False},
    "施工区经历": {"ids": ["95"], "type": "分类", "assign": {1:1,2:1,3:0,4:0}, "direction": None, "custom": False},

    # ===================== 二、驾驶行为特征（14项） =====================
    "跟车倾向": {"ids": ["0","1","2"], "type": "连续", "direction": "中性", "custom": False},
    "换道倾向": {"ids": ["3","4","5"], "type": "连续", "direction": "负向", "custom": False},
    "超车倾向": {"ids": ["6","7","8"], "type": "连续", "direction": "负向", "custom": False},
    "跟车距离过近": {"ids": ["9","10","11"], "type": "连续", "direction": "负向", "custom": False},
    "无意超速": {"ids": ["12","13","14"], "type": "连续", "direction": "负向", "custom": False},
    "右侧超车倾向": {"ids": ["15","16","17"], "type": "连续", "direction": "负向", "custom": False},
    "主动超速倾向": {"ids": ["18","19","20"], "type": "连续", "direction": "负向", "custom": False},
    "限速遵守度": {"ids": ["21","22","23"], "type": "连续", "direction": "正向", "custom": False},
    "无监控主动超速": {"ids": ["24","25","26"], "type": "连续", "direction": "负向", "custom": False},
    "提前变更车道": {"ids": ["27","28","29"], "type": "连续", "direction": "中性", "custom": False},
    "前车慢速强行超车": {"ids": ["30","31","32"], "type": "连续", "direction": "负向", "custom": False},
    "竞速斗气": {"ids": ["33","34","35"], "type": "连续", "direction": "负向", "custom": False},
    "换道观察充分性": {"ids": ["36","37","38"], "type": "连续", "direction": "正向", "custom": False},
    "减速避让及时性": {"ids": ["39","40","41"], "type": "连续", "direction": "正向", "custom": False},

    # ===================== 三、心理特征（11项） =====================
    "愤怒不耐烦": {"ids": ["42","43","44"], "type": "连续", "direction": "负向", "custom": False},
    "紧张焦虑": {"ids": ["45","46","47"], "type": "连续", "direction": "负向", "custom": False},
    "平静放松自信": {"ids": ["48","49","50"], "type": "连续", "direction": "正向", "custom": False},
    "压力程度": {"ids": ["51","52","53"], "type": "连续", "direction": "负向", "custom": False},
    "疲劳程度": {"ids": ["54","55","56"], "type": "连续", "direction": "负向", "custom": False},
    "脑力活动": {"ids": ["57","58","59","60"], "type": "连续", "direction": "中性", "custom": False},
    "体力活动": {"ids": ["61","62","63","64"], "type": "连续", "direction": "中性", "custom": False},
    "时间压力": {"ids": ["65","66","67","68"], "type": "连续", "direction": "负向", "custom": False},
    "任务完成度": {"ids": ["69","70","71","72"], "type": "连续", "direction": "正向", "custom": False},
    "付出努力": {"ids": ["73","74","75","76"], "type": "连续", "direction": "中性", "custom": False},
    "情绪感受": {"ids": ["77","78","79","80"], "type": "连续", "direction": "负向", "custom": False},

    # ===================== 四、自定义合成指标（论文专用） =====================
    "施工区安全感": {"ids": ["45","46","47","48","49","50"], "type": "连续", "direction": "正向", "custom": True},
    "情绪稳定性": {"ids": ["42","43","44"], "type": "连续", "direction": "正向", "custom": True},
    "压力应对": {"ids": ["51","52","53"], "type": "连续", "direction": "正向", "custom": True},
}

# ---------------------------------------------------------------------------
# 工具函数
# ---------------------------------------------------------------------------
def _build_id_to_question(df: pd.DataFrame) -> Dict[str, str]:
    """根据行索引生成编号->问题映射。"""
    questions = df.index.tolist()
    return {str(i): q for i, q in enumerate(questions)}


def preprocess_questionnaire(
    csv_path: Path | str,
    output_dir: Path | str,
    config: IndicatorConfig = INDICATOR_CONFIG,
) -> pd.DataFrame:
    """执行问卷数据的完整预处理流程。"""
    df = load_questionnaire(str(csv_path))
    df = df.drop(columns=[c for c in df.columns if c.startswith("Unnamed:")])
    
    # 脏数据清洗
    df = df.apply(pd.to_numeric, errors='coerce')
    df = df.dropna(how='all', axis=0)
    df = df.dropna(how='all', axis=1)

    id_to_question = _build_id_to_question(df)
    processed = pd.DataFrame(index=df.columns)

    for name, c in config.items():
        ids = c["ids"]
        rows = [id_to_question[i] for i in ids]
        sub = df.loc[rows]

        if c["type"] == "分类":
            raw = sub.iloc[0]
            processed[name] = raw.map(c.get("assign", {})).fillna(raw)
        else:
            if c.get("custom"):
                # 自定义心理指标计算
                if name == "施工区安全感":
                    calm = df.loc[[id_to_question[i] for i in ["48", "49", "50"]]].mean()
                    anxiety = df.loc[[id_to_question[i] for i in ["45", "46", "47"]]].mean()
                    processed[name] = (calm + (10 - anxiety)) / 2
                elif name == "情绪稳定性":
                    anger = df.loc[[id_to_question[i] for i in ["42", "43", "44"]]].mean()
                    processed[name] = 10 - anger
                elif name == "压力应对":
                    stress = df.loc[[id_to_question[i] for i in ["51", "52", "53"]]].mean()
                    processed[name] = 10 - stress
            else:
                mean_vals = sub.mean()
                if c.get("direction") == "负向":
                    processed[name] = 10 - mean_vals
                else:
                    processed[name] = mean_vals

    # 标准化
    standardized = (processed - processed.mean()) / processed.std()

    # 保存文件
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    processed.to_csv(out / "questionnaire_preprocessed.csv", encoding="utf-8-sig")
    standardized.to_csv(out / "questionnaire_standardized.csv", encoding="utf-8-sig")
    standardized.to_pickle(out / "questionnaire_standardized.pkl")

    return processed  # 返回原始分数据，用于统计


def augment_questionnaire_data(
    processed_df: pd.DataFrame,
    augment_times: int = 1,  # 增强倍数：1=原始数据×2，2=原始×3，以此类推
    noise_std: float = 0.17   # 高斯噪声标准差（越小越真实，建议0.1~0.3）
) -> pd.DataFrame:
    """
    问卷数据增强：增加样本数量
    规则：
    1. 分类指标（性别、驾龄等）完全保留，不修改
    2. 连续指标（驾驶行为、心理特征）添加微小高斯噪声
    3. 强制限制数值在量表有效范围（0~10分）
    """
    # 1. 获取所有连续型指标（仅增强这些列）
    continuous_indicators = [
        name for name, cfg in INDICATOR_CONFIG.items()
        if cfg["type"] == "连续"
    ]
    
    # 2. 初始化增强数据（原始数据为基础）
    augmented_data = processed_df.copy()
    
    # 3. 循环生成增强样本
    for _ in range(augment_times):
        new_samples = processed_df.copy()
        # 仅对连续指标添加噪声
        for col in continuous_indicators:
            noise = np.random.normal(loc=0, scale=noise_std, size=new_samples[col].shape)
            new_samples[col] = new_samples[col] + noise
            # 限制数值范围（问卷量表0-10分）
            new_samples[col] = new_samples[col].clip(0, 10)
        # 拼接新样本
        augmented_data = pd.concat([augmented_data, new_samples], ignore_index=True)
    
    logging.info(f"✅ 数据增强完成：原始样本{len(processed_df)}个 → 增强后{len(augmented_data)}个")
    return augmented_data


# ---------------------------------------------------------------------------
# 描述性统计（人口学特征）
# ---------------------------------------------------------------------------
def generate_descriptive_stats(csv_path: Path | str, output_dir: Path | str):
    """统计被试的人口学特征与驾驶经历分布。"""
    df = load_questionnaire(str(csv_path))
    df = df.drop(columns=[c for c in df.columns if c.startswith("Unnamed:")])
    df = df.apply(pd.to_numeric, errors='coerce')
    
    id_to_question = _build_id_to_question(df)

    stats_mapping = {
        "驾龄分布": {
            "col_id": "86",
            "mapping": {1: "正在考取驾照",2: "1年及以下",3: "1-5年",4: "5-20年",5: "20年以上"},
            "order": ["正在考取驾照", "1年及以下", "1-5年", "5-20年", "20年以上"]
        },
        "驾驶频率": {
            "col_id": "87",
            "mapping": {1: "每周不到1天",2: "每周1-3天",3: "每周4-6天",4: "每天"},
            "order": ["每周不到1天", "每周1-3天", "每周4-6天", "每天"]
        },
        "单次驾驶时长": {
            "col_id": "88",
            "mapping": {1: "少于30分钟",2: "30分钟-1小时",3: "1到2小时",4: "2到3小时",5: "超过3小时"},
            "order": ["少于30分钟", "30分钟-1小时", "1到2小时", "2到3小时", "超过3小时"]
        },
        "施工区经历": {
            "col_id": "95",
            "mapping": {1: "有经历", 2: "有经历", 3: "无经历", 4: "无经历"},
            "order": ["有经历", "无经历"]
        },
        "事故经历": {
            "col_id": "94",
            "mapping": {1: "无事故",2: "1-2次"},
            "order": ["无事故", "1-2次"]
        }
    }

    final_results = []
    for category, config in stats_mapping.items():
        q_text = id_to_question[config["col_id"]]
        raw_series = df.loc[q_text]
        grouped_series = raw_series.map(config["mapping"])
        counts = grouped_series.value_counts(dropna=False)
        total_people = len(grouped_series.dropna())
        full_index = pd.Index(config["order"], name="具体指标")
        counts_reindexed = counts.reindex(full_index, fill_value=0)
        percents_reindexed = (counts_reindexed / total_people * 100).round(2) if total_people > 0 else 0

        temp_df = pd.DataFrame({
            "指标类别": category, "人数": counts_reindexed.values, "占比(%)": percents_reindexed.values
        }, index=full_index).reset_index()
        final_results.append(temp_df)

    final_df = pd.concat(final_results, ignore_index=True)
    out_path = Path(output_dir) / "questionnaire_demographic_stats.csv"
    final_df.to_csv(out_path, index=False, encoding="utf-8-sig")
    logging.info(f"人口学统计已保存：{out_path}")


# ---------------------------------------------------------------------------
# 生成表3.2 驾驶行为与心理特征统计（均值+标准差）
# ---------------------------------------------------------------------------
def generate_behavior_mental_stats(processed_df: pd.DataFrame, output_dir: Path | str):
    """
    生成论文表3.2所需数据：关键驾驶行为与心理特征得分
    输出格式：特征类别、具体指标、均值、标准差
    """
    # 定义表3.2需要的指标（严格匹配论文表格）
    table_3_2 = {
        "安全行为": ["限速遵守度", "换道观察充分性", "减速避让及时性"],
        "心理状态": ["情绪稳定性", "施工区安全感", "压力应对"]
    }

    # 指标重命名（匹配论文表格文字）
    rename_map = {
        "情绪稳定性": "情绪不稳定性(逆向)",
        "压力应对": "施工区压力程度(逆向)"
    }

    # 计算均值和标准差
    stats_data = []
    for category, indicators in table_3_2.items():
        for ind in indicators:
            mean_val = processed_df[ind].mean().round(1)
            std_val = processed_df[ind].std().round(1)
            ind_name = rename_map.get(ind, ind)
            stats_data.append([category, ind_name, mean_val, std_val])

    # 构建表格DataFrame
    result_df = pd.DataFrame(stats_data, columns=["特征类别", "具体指标", "均值", "标准差"])
    
    # 保存文件
    out_path = Path(output_dir) / "questionnaire_behavior_mental_stats.csv"
    result_df.to_csv(out_path, index=False, encoding="utf-8-sig")
    
    # 打印表格（直接复制到论文）
    logging.info("-" * 50)
    logging.info("表3. 2 关键驾驶行为与心理特征得分")
    logging.info(result_df.to_string(index=False))
    logging.info("-" * 50)
    logging.info(f"表3.2数据已保存：{out_path}")


# ---------------------------------------------------------------------------
# 主函数
# ---------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(description="问卷预处理")
    parser.add_argument("-i", "--input", default=BASE_DIR / "data" / "raw" / "questionnaire.csv",
                        help="原始问卷CSV路径")
    parser.add_argument("-o", "--output", default=BASE_DIR / "data" / "processed",
                        help="输出目录")
    # ✅ 命令行参数 - 控制数据增强倍数
    parser.add_argument("-a", "--augment", type=int, default=0,
                        help="数据增强倍数（0=不增强，1=样本×2，2=样本×3）")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")

    try:
        # 1. 预处理问卷数据
        processed_data = preprocess_questionnaire(args.input, args.output)

        # ===================== 数据增强 =====================
        augmented_data = None
        if args.augment > 0:
            augmented_data = augment_questionnaire_data(processed_data, augment_times=args.augment)
            # 保存增强后的数据
            out_path = Path(args.output) / "questionnaire_augmented.csv"
            augmented_data.to_csv(out_path, encoding="utf-8-sig")
            # 标准化增强数据并保存
            aug_standardized = (augmented_data - augmented_data.mean()) / augmented_data.std()
            aug_standardized.to_csv(Path(args.output) / "questionnaire_augmented_standardized.csv", encoding="utf-8-sig")
            aug_standardized.to_pickle(Path(args.output) / "questionnaire_augmented_standardized.pkl")
        # ========================================================

        # 2. 生成人口学描述统计
        stats_output = BASE_DIR / "output" / "1_capability_assessment" / "results"
        generate_descriptive_stats(args.input, stats_output)
        # 3. 生成表3.2数据
        generate_behavior_mental_stats(processed_data, stats_output)

        logging.info("✅ 所有数据处理完成！")
        
    except Exception as exc:
        logging.error(f"❌ 处理失败：{exc}")
        return


if __name__ == "__main__":
    main()
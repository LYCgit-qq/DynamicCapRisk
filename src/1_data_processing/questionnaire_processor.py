"""问卷预处理工具模块

提供一个可复用的函数 `preprocess_questionnaire` 并支持命令行调用。

功能：
    * 从 CSV 读取问卷数据（依赖 `read_questionnaire.load_questionnaire`）
    * 根据配置算出 35 个指标
    * Z-score 标准化
    * 将结果保存为 CSV/PKL

使用方法：

    python src/1_data_processing/preprocess_questionnaire.py \ 
        --input data/raw/questionnaire.csv \
        --output data/processed

也可以在其它模块中通过导入 `preprocess_questionnaire` 函数。
"""

import argparse
import logging
from pathlib import Path
from typing import Dict, Any

import numpy as np
import pandas as pd

from read_questionnaire import load_questionnaire

# 项目根目录（假设本文件在 src/1_data_processing 中）
BASE_DIR = Path(__file__).resolve().parents[2]

# ---------------------------------------------------------------------------
# 指标配置
# ---------------------------------------------------------------------------
IndicatorConfig = Dict[str, Any]

INDICATOR_CONFIG: IndicatorConfig = {
    # -------- 个人基本信息（12项） --------
    "性别": {"ids": ["81"], "type": "分类", "assign": {1: 1, 2: 0}},
    "年龄段": {"ids": ["82"], "type": "分类", "assign": {1: 0, 2: 1, 3: 2, 4: 3, 5: 4}},
    "是否职业司机": {"ids": ["83"], "type": "分类", "assign": {1: 1, 2: 0}},
    "学历": {"ids": ["84"], "type": "分类", "assign": {1: 0, 2: 1, 3: 2, 4: 3, 5: 4}},
    "年均收入": {"ids": ["85"], "type": "分类", "assign": {1: 0, 2: 1, 3: 2, 4: 3, 5: 4}},
    "驾龄": {"ids": ["86"], "type": "分类", "assign": {1: 0, 2: 1, 3: 2, 4: 3, 5: 4}},
    "每周开车频率": {"ids": ["87"], "type": "分类", "assign": {1: 0, 2: 1, 3: 2, 4: 3}},
    "单次驾驶时长": {"ids": ["88"], "type": "分类", "assign": {1: 0, 2: 1, 3: 2}},
    "是否本本族": {"ids": ["89"], "type": "分类", "assign": {1: 1, 2: 0}},
    "出行原因_通勤": {"ids": ["90"], "type": "分类", "assign": {1: 1, 0: 0}},
    "事故经历": {"ids": ["94"], "type": "分类", "assign": {1: 0, 2: 1, 3: 2}},
    "施工区经历": {"ids": ["95"], "type": "分类", "assign": {1: 1, 2: 0, 3: 0}},

    # -------- 驾驶行为特征（9项） --------
    "跟车倾向": {"ids": ["00", "01", "02"], "type": "连续", "direction": "负向"},
    "换道倾向": {"ids": ["03", "04", "05", "27", "28", "29", "30", "31", "32"], "type": "连续", "direction": "负向"},
    "超车倾向": {"ids": ["06", "07", "08"], "type": "连续", "direction": "负向"},
    "跟车距离过近": {"ids": ["09", "10", "11"], "type": "连续", "direction": "负向"},
    "超速倾向": {"ids": ["12", "13", "14", "18", "19", "20", "24", "25", "26"], "type": "连续", "direction": "负向"},
    "右侧超车倾向": {"ids": ["15", "16", "17"], "type": "连续", "direction": "负向"},
    "限速遵守度": {"ids": ["21", "22", "23"], "type": "连续", "direction": "正向"},
    "换道观察充分性": {"ids": ["36", "37", "38"], "type": "连续", "direction": "正向"},
    "减速避让及时性": {"ids": ["39", "40", "41"], "type": "连续", "direction": "正向"},
    "竞速斗气": {"ids": ["33", "34", "35"], "type": "连续", "direction": "负向"},

    # -------- 心理特征（14项） --------
    "愤怒不耐烦": {"ids": ["42", "43", "44"], "type": "连续", "direction": "负向"},
    "紧张焦虑": {"ids": ["45", "46", "47"], "type": "连续", "direction": "负向"},
    "平静放松自信": {"ids": ["48", "49", "50"], "type": "连续", "direction": "正向"},
    "压力程度": {"ids": ["51", "52", "53"], "type": "连续", "direction": "负向"},
    "疲劳程度": {"ids": ["54", "55", "56"], "type": "连续", "direction": "负向"},
    "脑力活动": {"ids": ["57", "58", "59", "60"], "type": "连续", "direction": "中性"},
    "体力活动": {"ids": ["61", "62", "63", "64"], "type": "连续", "direction": "中性"},
    "时间压力": {"ids": ["65", "66", "67", "68"], "type": "连续", "direction": "负向"},
    "任务完成度": {"ids": ["69", "70", "71", "72"], "type": "连续", "direction": "正向"},
    "付出努力": {"ids": ["73", "74", "75", "76"], "type": "连续", "direction": "中性"},
    "情绪感受": {"ids": ["77", "78", "79", "80"], "type": "连续", "direction": "负向"},

    # 3 项自定义心理指标
    "施工区安全感": {"ids": ["45", "46", "47", "48", "49", "50"], "type": "连续", "direction": "正向", "custom": True},
    "情绪稳定性": {"ids": ["42", "43", "44"], "type": "连续", "direction": "正向", "custom": True},
    "压力应对": {"ids": ["51", "52", "53"], "type": "连续", "direction": "正向", "custom": True},
}


# ---------------------------------------------------------------------------
# helper routines
# ---------------------------------------------------------------------------

def _build_id_to_question(df: pd.DataFrame) -> Dict[str, str]:
    """根据行索引生成编号->问题映射。

    返回值使得 '00' 对应第一行问题文本，
    '01' 对应第二行，以此类推。
    """
    questions = df.index.tolist()
    return {str(i).zfill(2): q for i, q in enumerate(questions)}


def preprocess_questionnaire(
    csv_path: Path | str,
    output_dir: Path | str,
    config: IndicatorConfig = INDICATOR_CONFIG,
) -> pd.DataFrame:
    """执行问卷数据的完整预处理流程。

    Args:
        csv_path: 原始 CSV 路径。
        output_dir: 保存结果的目录。
        config: 指标配置字典。

    Returns:
        标准化后的 DataFrame（行=参与者，列=指标）。
    """
    df = load_questionnaire(str(csv_path))
    df = df.drop(columns=[c for c in df.columns if c.startswith("Unnamed:")])

    id_to_question = _build_id_to_question(df)
    processed = pd.DataFrame(index=df.columns)

    for name, c in config.items():
        ids = c["ids"]
        rows = [id_to_question[i] for i in ids]
        sub = df.loc[rows]

        if c["type"] == "分类":
            raw = sub.iloc[0]
            processed[name] = raw.map(c.get("assign", {})).fillna(raw)
        else:  # 连续
            if c.get("custom"):
                # 3 个自定义心理指标
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
                processed[name] = 10 - mean_vals if c.get("direction") == "负向" else mean_vals

    standardized = (processed - processed.mean()) / processed.std()

    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    processed.to_csv(out / "questionnaire_preprocessed.csv", encoding="utf-8-sig")
    standardized.to_csv(out / "questionnaire_standardized.csv", encoding="utf-8-sig")
    standardized.to_pickle(out / "questionnaire_standardized.pkl")

    return standardized


def main():
    parser = argparse.ArgumentParser(
        description="问卷预处理",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("-i", "--input", default=BASE_DIR / "data" / "raw" / "questionnaire.csv",
                        help="原始问卷 CSV 文件路径")
    parser.add_argument("-o", "--output", default=BASE_DIR / "data" / "processed",
                        help="输出目录")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")

    try:
        result = preprocess_questionnaire(args.input, args.output)
    except Exception as exc:
        logging.error("预处理失败：%s", exc)
        return

    logging.info("预处理完成，结果形状 %s", result.shape)
    logging.info("文件已保存到 %s", Path(args.output).resolve())


if __name__ == "__main__":
    main()

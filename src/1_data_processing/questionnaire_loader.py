"""读取问卷 CSV 并将第一列（问题）设为 key 的小工具。

功能：
- 使用 `pandas` 读取 CSV（默认路径 `data/raw/questionnaire.csv`）
- 将第一列设为索引（问题），其余列为不同受试/答卷的得分
- 提供将结果保存为 `pkl` 的函数

用法：
    python src/1_data_processing/read_questionnaire.py
或者在代码中：
    from src.1_data_processing.read_questionnaire import load_questionnaire, to_dict, save_pickle
"""
from pathlib import Path
from typing import Dict, Any, List

try:
    import pandas as pd
except Exception:
    pd = None


def load_questionnaire(csv_path: str = "data/raw/questionnaire.csv") -> pd.DataFrame:
    """加载问卷 CSV 并将第一列设为索引（问题）。

    Args:
        csv_path: CSV 文件路径

    Returns:
        pd.DataFrame: 索引为问题，每列为一位受试者/答卷的得分（尽量转换为数值）

    Raises:
        RuntimeError: 若未安装 `pandas`
        FileNotFoundError: 若文件不存在
    """
    if pd is None:
        raise RuntimeError("pandas is required to read CSV. Install with: pip install pandas")

    p = Path(csv_path)
    if not p.exists():
        raise FileNotFoundError(f"Questionnaire CSV not found: {p.resolve()}")

    df = pd.read_csv(p, header=0)
    # 将第一列设置为索引（问题文本或问题 ID）
    first_col = df.columns[0]
    df = df.set_index(first_col)

    # 尝试把每列转换为数值类型（若失败则保留原始）
    for col in df.columns:
        df[col] = pd.to_numeric(df[col], errors="ignore")

    return df


def to_dict(df: pd.DataFrame) -> Dict[str, List[Any]]:
    """把 DataFrame 转成字典：{ question: [scores...] }

    Args:
        df: 由 `load_questionnaire` 返回的 DataFrame
    """
    return {str(idx): df.loc[idx].tolist() for idx in df.index}


def save_pickle(df: pd.DataFrame, out_path: str = "data/processed/questionnaire.pkl") -> str:
    """将 DataFrame 保存为 pickle 文件（自动创建目录）。返回保存路径。"""
    out = Path(out_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    df.to_pickle(out)
    return str(out)


def main() -> None:
    csv_default = "data/raw/questionnaire.csv"
    try:
        df = load_questionnaire(csv_default)
    except FileNotFoundError:
        print(f"File not found: {csv_default}")
        return
    except RuntimeError as e:
        print(e)
        return

    print(f"Loaded questionnaire: questions={len(df.index)}, respondents={len(df.columns)}")
    print("First 5 questions:")
    for q in list(df.index)[:5]:
        row = df.loc[q]
        print(f"- {q}: sample -> {row.iloc[:3].tolist()}")

    out = save_pickle(df)
    print(f"Saved DataFrame as pickle: {out}")


if __name__ == "__main__":
    main()

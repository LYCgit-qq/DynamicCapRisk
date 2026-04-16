# D:\Local\DynamicCapRisk\src\data_processing\data_loader.py

"""读取 `data/raw` 下的原始数据文件。

当前实现：
- 扫描 `data/raw` 目录中的 `.mat` 文件并使用 `scipy.io.loadmat` 加载
- 返回一个以文件名（不含后缀）为键、mat 内容为值的字典

用法：
    python src/data_processing/load_raw_data.py
或者在代码中导入 `load_mat_files` 使用。
"""

from pathlib import Path
from typing import Dict, Any

try:
    import scipy.io as sio
except Exception:
    sio = None


def _unwrap_mat(obj: Any) -> Any:
    """递归展开只含单一公有键的 dict 包装，直到不再是单键 dict 为止。

    示例：{"act": {"act": ndarray}} -> ndarray
    保留多键字典或非 dict 类型不变。
    """
    while isinstance(obj, dict):
        public_keys = [k for k in obj.keys() if not k.startswith("__")]
        if len(public_keys) != 1:
            break
        obj = obj[public_keys[0]]
    return obj


def _simplify(obj: Any) -> Any:
    """对 mat 结构进行简化，尤其处理 object 类型数组。

    - numpy.ndarray(dtype=object) 会转换为 Python 列表，递归简化其中每一项
      并自动丢弃前导长度为1的维度（例如shape==(1,67) 会降为长度67）
    - dict 会递归处理其值
    - 其他类型保持不变

    处理目标是让用户能够以 ``data['act'][i]`` 直接得到 ndarray。
    """
    try:
        import numpy as _np
    except ImportError:
        _np = None

    if _np is not None and isinstance(obj, _np.ndarray):
        if obj.dtype == object:
            # squeeze leading singleton dims for convenience
            arr = obj
            if arr.ndim > 1 and arr.shape[0] == 1:
                arr = arr[0]
            # convert to list and recursively simplify
            return [_simplify(x) for x in arr]
        else:
            return obj
    elif isinstance(obj, dict):
        return {k: _simplify(v) for k, v in obj.items()}
    else:
        return obj


def load_mat_files(raw_dir: str = "data/raw") -> Dict[str, Any]:
    """加载 `raw_dir` 下所有 `.mat` 文件并返回字典。

    Args:
        raw_dir: 原始数据目录，默认 `data/raw`（相对仓库根目录）

    Returns:
        dict: { filename_without_ext: mat_contents }

    Raises:
        FileNotFoundError: 若目录不存在
        RuntimeError: 若缺少 `scipy` 依赖
    """
    if sio is None:
        raise RuntimeError(
            "scipy is required to load .mat files. Install with: pip install scipy"
        )

    p = Path(raw_dir)
    if not p.exists():
        raise FileNotFoundError(f"Raw data directory not found: {p.resolve()}")

    result: Dict[str, Any] = {}
    for path in sorted(p.glob("*.mat")):
        try:
            mat = sio.loadmat(path)
            # 初步展开包装
            mat = _unwrap_mat(mat)
            # 将 object 数组、长度1轴等简化为更直观的 Python 结构
            mat = _simplify(mat)
        except Exception as e:
            result[path.stem] = {"_load_error": str(e)}
        else:
            result[path.stem] = mat

    return result


def _summarize_mat(mat: Any) -> str:
    """简要描述对象类型或数组结构。"""
    if isinstance(mat, dict):
        keys = [k for k in mat.keys() if not k.startswith("__")]
        return f"keys={keys}" if keys else "(no public keys)"
    try:
        import numpy as _np

        if isinstance(mat, _np.ndarray):
            return f"ndarray, shape={mat.shape}, dtype={mat.dtype}"
    except Exception:
        pass
    # lists from simplified object arrays
    if isinstance(mat, list):
        return f"list(len={len(mat)})"
    return type(mat).__name__


def save_pickle(obj: Any, out_path: str) -> None:
    """将对象以 pickle 格式保存到指定路径。"""
    import pickle

    p = Path(out_path)
    p.parent.mkdir(parents=True, exist_ok=True)
    with open(p, "wb") as f:
        pickle.dump(obj, f)
    print(f"Saved pickle to {p}")


def main() -> None:
    data = load_mat_files()
    if not data:
        print("No .mat files found in data/raw")
        return

    print("Loaded files:", ", ".join(data.keys()))
    for name, mat in data.items():
        if isinstance(mat, dict) and mat.get("_load_error") is not None:
            print(f"{name}: load error -> {mat.get('_load_error')}")
        else:
            print(f"{name}: {_summarize_mat(mat)}")

    # 示例：将整个原始字典保存为 pickle
    save_pickle(data, "data/processed/raw_data.pkl")


if __name__ == "__main__":
    main()

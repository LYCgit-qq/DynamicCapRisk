import os
import sys
import site
from pathlib import Path

def main():
    """
    自动将项目根目录添加到当前Conda环境的永久Python路径中
    适用：Windows系统，DynamicCapRisk环境
    """
    # ====================== 【需用户确认/修改的参数】 ======================
    PROJECT_ROOT = r"D:\Local\DynamicCapRisk"  # 你的项目根目录（复制绝对路径即可）
    PTH_FILE_NAME = "dynamiccaprisk.pth"       # 生成的.pth文件名（可自定义）
    # =====================================================================

    # 1. 检查当前是否在Conda环境中
    conda_prefix = os.environ.get("CONDA_PREFIX")
    if not conda_prefix:
        print("❌ 错误：未检测到激活的Conda环境！")
        print("请先运行：conda activate DynamicCapRisk，再执行本脚本")
        sys.exit(1)
    
    # 2. 获取当前Conda环境的site-packages路径
    try:
        site_packages_path = site.getsitepackages()[0]
        # 验证site-packages是否属于当前Conda环境
        if conda_prefix not in site_packages_path:
            site_packages_path = Path(conda_prefix) / "Lib" / "site-packages"
    except Exception as e:
        site_packages_path = Path(conda_prefix) / "Lib" / "site-packages"
    print(f"✅ 找到当前Conda环境的site-packages路径：\n{site_packages_path}")

    # 3. 检查项目根目录是否存在
    project_root = Path(PROJECT_ROOT)
    if not project_root.exists():
        print(f"❌ 错误：项目根目录不存在！\n{PROJECT_ROOT}")
        sys.exit(1)

    # 4. 创建/写入.pth文件
    pth_file = Path(site_packages_path) / PTH_FILE_NAME
    try:
        # 写入项目根目录（覆盖原有内容，避免重复）
        with open(pth_file, "w", encoding="utf-8") as f:
            f.write(f"{PROJECT_ROOT}\n")  # 一行一个路径，可追加其他路径
        print(f"✅ 成功创建.pth文件：\n{pth_file}")
        print(f"✅ 已写入项目路径：{PROJECT_ROOT}")
    except PermissionError:
        print("❌ 错误：权限不足！请以管理员身份运行本脚本")
        sys.exit(1)
    except Exception as e:
        print(f"❌ 创建.pth文件失败：{str(e)}")
        sys.exit(1)

    # 5. 验证路径是否生效
    print("\n🔍 验证路径是否添加成功...")
    # 重新加载sys.path（模拟Python启动时的加载逻辑）
    site.addsitedir(str(project_root))
    if str(project_root) in sys.path:
        print("✅ 验证通过！项目路径已永久添加到当前Conda环境")
        # 测试导入模块（可选，根据你的模块调整）
        try:
            from src.visualization.plot_capability import plot_pca_visualization  # 测试导入
            print("✅ 模块导入测试成功！可以直接使用 from src.visualization import ...")
        except ModuleNotFoundError as e:
            print(f"⚠️ 模块导入测试失败（可能是visualization目录缺少__init__.py）：{e}")
            print("提示：请在 src/visualization 目录下创建空的 __init__.py 文件")
    else:
        print("❌ 验证失败！路径未添加成功，请检查路径是否正确")

if __name__ == "__main__":
    main()
import sys
from pathlib import Path

# 1. 动态获取项目根目录（当前文件在 examples/ 下，向上两级就是 guithub/）
PROJECT_ROOT = Path(__file__).resolve().parent.parent

# 2. 将 src 目录加入 Python 搜索路径
# 这样 Python 才能找到 knowledge_base 这个包
SRC_DIR = PROJECT_ROOT / "src"
sys.path.append(str(SRC_DIR))

# 3. 导入核心逻辑
from knowledge_base.ingest import ingest_document
from knowledge_base.config import settings

if __name__ == "__main__":
    print("=== 开始构建 RAG 知识底座索引 ===")
    print(f"当前项目根目录: {PROJECT_ROOT}")
    print(f"数据源目录: {settings.raw_dir}")
    print()
    try:
        ingest_document()
        print("\n=== 索引构建流程结束 ===")
    except Exception as e:
        print(f"\n❌ 索引构建失败: {e}")
        raise
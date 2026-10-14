import sys
import argparse
from pathlib import Path

# 1. 动态获取项目根目录（当前文件在 examples/ 下，向上两级就是 guithub/）
PROJECT_ROOT = Path(__file__).resolve().parent.parent

# 2. 将 src 目录加入 Python 搜索路径
SRC_DIR = PROJECT_ROOT / "src"
sys.path.append(str(SRC_DIR))

# 3. 导入核心逻辑
from knowledge_base.vector_store import load_vector_store
# 进程内缓存向量库实例：交互模式下不用反复加载
_VECTOR_STORE = None
# 交互模式下视为“退出”的输入
EXIT_COMMAND = {"q", "quit", "exit"}

def get_vector_store():
    """懒加载向量库，同一个进程里复用同一个实例。"""
    global _VECTOR_STORE
    if _VECTOR_STORE is None:
        _VECTOR_STORE = load_vector_store()
    return _VECTOR_STORE

def _format_source(source):
    """把绝对路径转成相对项目根目录的路径，读起来更清爽。"""
    if not source:
        return "未知来源"
    try:
        return str(Path(source).relative_to(PROJECT_ROOT))
    except (ValueError, TypeError):
        return str(source)

def print_results(docs_with_scores):
    """格式化打印检索结果"""
    if not docs_with_scores:
        print("没有检索到任何结果。")
        return

    for i, item in enumerate(docs_with_scores, 1):
        # Chroma 的 similarity_search_with_score 返回 (Document, score)
        # score 越小越相似（Chroma 默认用 L2 距离）
        if isinstance(item, tuple):
            doc, score = item
            score_str = f"{score:.4f}"
        else:
            doc, score_str = item, "N/A"

        source = _format_source(doc.metadata.get("source", "未知来源"))
        content = doc.page_content
        preview = content[:300] + ("..." if len(content) > 300 else "")

        print(f"\n[{i}] score={score_str}")
        print(f"    来源: {source}")
        print(f"    内容: {preview}")

    print("\n" + "-" * 60)

def search(query: str, k: int=4):
    """执行一次检索"""
    print(f"🔍 查询: {query}")
    print(f"   Top-K: {k}")

    try:
        vector_store = load_vector_store()
        results = vector_store.similarity_search_with_score(query, k=k)
    except Exception as ex:
        # 向量库加载失败 / 检索异常时不要整个程序崩掉，交互模式还能继续
        print(f"❌ 检索失败: {ex}")
        return
    print_results(results)

def interactive_loop(k: int = 4):
    """交互模式：反复输入查询并检索，支持 q / quit / exit 退出。"""
    print("输入查询语句，回车检索；输入 q / quit / exit 退出。\n")
    while True:
        try:
            query = input(">>> ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\n退出。")
            break

        if not query:
            continue
        if query.lower() in EXIT_COMMAND:
            print("退出。")
            break

        search(query, k=k)

def main():
    parser = argparse.ArgumentParser(description="RAG 知识库检索 Demo")
    parser.add_argument("query", nargs="?", help="查询语句；不传则进入交互模式")
    parser.add_argument("-k", type=int, default=4, help="返回结果数量，默认 4")
    args = parser.parse_args()

    print("=== RAG 知识库检索 Demo ===")
    print(f"项目根目录: {PROJECT_ROOT}\n")

    if args.query:
        search(args.query, k=args.k)
        return

    # 交互模式
    interactive_loop(k=args.k)

if __name__ == "__main__":
    main()
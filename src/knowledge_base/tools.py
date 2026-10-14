from langchain_core.tools import tool

# 1.引入你的检索类
from knowledge_base.retriever import Retriever
from knowledge_base.vector_store import load_vector_store
from knowledge_base.config import settings

# 2. 在模块级别初始化一次全局的 retriever（避免每次调用工具都重新加载向量库，极大提升性能）
# 如果你使用 web 服务，建议在这里做成单例模式
_vector_store = load_vector_store()
_retriever = Retriever(_vector_store)

@tool
def knowledge_search(query: str) -> str:
    """从知识底座检索相关文档片段。"""
    try:
        # 使用全局初始化的 _retriever
        results = _retriever.search(query, top_k=settings.top_k)

        if not results:
            return "知识库中未找到与该问题相关的内容。请根据你的常识回答，或告知用户你无法回答。"

        formatted_results = "\n\n---\n\n".join(
            f"【来源: {r.source}】\n{r.content}" for r in results
        )

        return f"以下是从知识库检索到的相关片段：\n\n{formatted_results}"
    except Exception as e:
        return f"检索知识库时发生错误：{str(e)}"
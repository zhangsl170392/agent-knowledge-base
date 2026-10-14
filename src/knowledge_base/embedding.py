from functools import lru_cache
from langchain_core.embeddings import Embeddings
from langchain_ollama import OllamaEmbeddings
from knowledge_base.config import settings

@lru_cache(maxsize=1)
def get_embeddings() -> Embeddings:
    """
    返回全局单例的 Ollama Embeddings 实例。
    使用 lru_cache 避免每次调用都重新创建客户端。
    """
    return OllamaEmbeddings(
        model=settings.embedding_model,
        base_url=settings.ollama_base_url,
    )
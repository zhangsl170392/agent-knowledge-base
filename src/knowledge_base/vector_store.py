from langchain_chroma import Chroma
from knowledge_base.config import settings
from knowledge_base.embedding import get_embeddings

def load_vector_store() -> Chroma:
    """
    加载 Chroma 向量库。
    如果目录不存在会自动创建。
    """
    embeddings = get_embeddings()

    return Chroma(
        collection_name="knowledge_base",
        persist_directory=settings.vector_store_path,
        embedding_function=embeddings,
    )



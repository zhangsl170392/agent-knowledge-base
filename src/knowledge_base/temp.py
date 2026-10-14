from knowledge_base.retriever import Retriever
from knowledge_base.vector_store import load_vector_store

vector_store = load_vector_store()
retriever = Retriever(vector_store)

try:
    print(f"当前向量库文档总数: {vector_store._collection.count()}")
except Exception as e:
    print(f"获取文档数量失败: {e}")

results = retriever.search("什么是废弃机制？", top_k=3)

for i, r in enumerate(results, 1):
    print(f"[{i}] score={r.score:.4f} source={r.source}")
    print(r.content[:200])
    print("-" * 60)
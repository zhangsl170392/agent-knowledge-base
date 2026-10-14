from knowledge_base.schemas import RetrievedChunk

class Retriever:
    def __init__(self, vector_store):
        self.vector_store = vector_store

    def search(self, query: str, top_k: int = 3) -> list[RetrievedChunk]:
        docs = self.vector_store.similarity_search_with_score(query, k=top_k)
        return [
            RetrievedChunk(
                content=doc.page_content,
                source=doc.metadata.get("source", "unknown"),
                score=float(score)
            )
            for doc, score in docs
        ]
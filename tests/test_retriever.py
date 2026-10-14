import pytest
from knowledge_base.retriever import Retriever
from knowledge_base.vector_store import load_vector_store

@pytest.mark.integration
def test_search_returns_results():
    vector_store = load_vector_store()
    retriever = Retriever(vector_store)
    results = retriever.search("如何让 API 只返回指定字段？", top_k=3)
    assert len(results) > 0
    assert all(r.content for r in results)
    assert all(r.source for r in results)

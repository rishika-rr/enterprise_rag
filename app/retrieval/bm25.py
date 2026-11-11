from typing import List

from app.config import settings
from app.models.retrieval import RetrievalResult
from app.retrieval.retrieval_index import RetrievalIndex
from app.retrieval.sparse_index import get_sparse_index


class BM25Retriever:
    """Lexical retrieval over the shared BM25 index."""

    def __init__(
        self,
        sparse_index: RetrievalIndex | None = None,
        top_k: int | None = None,
    ):
        self.top_k = top_k if top_k is not None else settings.TOP_K
        self.sparse_index = sparse_index if sparse_index is not None else get_sparse_index()

    def search(self, query: str, where: dict | None = None) -> List[RetrievalResult]:
        return self.sparse_index.search(query=query, top_k=self.top_k, where=where)

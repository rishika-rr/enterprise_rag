from typing import List

from app.config import settings
from app.database.chroma import get_vector_store
from app.database.interfaces import VectorStore
from app.models.retrieval import RetrievalResult
from app.services.embedding_service import embedding_service


class DenseRetriever:
    """
    Performs semantic retrieval using vector embeddings.
    """

    def __init__(
        self,
        vector_store: VectorStore | None = None,
        top_k: int | None = None,
    ):
        self.top_k = top_k if top_k is not None else settings.TOP_K
        self.vector_store = vector_store if vector_store is not None else get_vector_store()

    def search(self, query: str, where: dict | None = None) -> List[RetrievalResult]:
        query_embedding = embedding_service.embed_query(query)
        return self.vector_store.search(
            embedding=query_embedding, top_k=self.top_k, where=where
        )

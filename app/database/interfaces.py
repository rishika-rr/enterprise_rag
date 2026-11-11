from abc import ABC, abstractmethod
from typing import List, Tuple

from app.models.retrieval import RetrievalResult


class VectorStore(ABC):
    """
    Abstract interface for all vector databases.
    """

    @abstractmethod
    def add_documents(
        self,
        ids: List[str],
        documents: List[str],
        embeddings: List[List[float]],
        metadatas: List[dict],
    ) -> None:
        pass

    @abstractmethod
    def search(
        self,
        embedding: List[float],
        top_k: int = 10,
        where: dict | None = None,
    ) -> List[RetrievalResult]:
        """`where` = metadata equality filters, ANDed together."""
        pass

    @abstractmethod
    def count(self) -> int:
        pass

    @abstractmethod
    def delete(self) -> None:
        """Delete the whole collection."""

    # ---- document-level operations -------------------------------------

    @abstractmethod
    def document_chunk_count(self, doc_id: str) -> int:
        """Number of indexed chunks belonging to a document (0 = absent)."""

    @abstractmethod
    def delete_document(self, doc_id: str) -> int:
        """Remove every chunk of a document. Returns chunks removed."""

    @abstractmethod
    def list_documents(self) -> List[dict]:
        """One summary dict per indexed document."""

    @abstractmethod
    def get_all(self) -> Tuple[List[str], List[str], List[dict]]:
        """Return (ids, documents, metadatas) for every chunk."""

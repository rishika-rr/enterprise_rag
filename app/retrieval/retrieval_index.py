from abc import ABC, abstractmethod
from typing import List

from app.models.retrieval import RetrievalResult


class RetrievalIndex(ABC):
    """
    Interface for all sparse retrieval indexes.
    """

    @abstractmethod
    def add_documents(
        self,
        ids: List[str],
        documents: List[str],
        metadatas: List[dict],
    ) -> None:
        pass

    @abstractmethod
    def remove_ids(self, ids: List[str]) -> None:
        pass

    @abstractmethod
    def search(
        self,
        query: str,
        top_k: int = 10,
        where: dict | None = None,
    ) -> List[RetrievalResult]:
        pass

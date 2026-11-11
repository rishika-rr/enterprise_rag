from typing import List

from app.config import settings
from app.models.retrieval import RetrievalResult
from app.retrieval.bm25 import BM25Retriever
from app.retrieval.dense import DenseRetriever
from app.retrieval.fusion import WeightedReciprocalRankFusion


class HybridRetriever:
    """
    Combines dense and sparse retrieval using Weighted Reciprocal Rank Fusion.

    `mode` (config: RETRIEVAL_MODE) selects "hybrid", "dense" or "bm25"
    without code changes (RET-004, RET-005).
    """

    def __init__(
        self,
        dense_retriever: DenseRetriever | None = None,
        bm25_retriever: BM25Retriever | None = None,
        fusion: WeightedReciprocalRankFusion | None = None,
        mode: str | None = None,
    ):
        mode = mode if mode is not None else settings.RETRIEVAL_MODE
        if mode not in {"hybrid", "dense", "bm25"}:
            raise ValueError("mode must be one of: hybrid, dense, bm25")

        self.mode = mode
        self.dense = dense_retriever if dense_retriever else DenseRetriever()
        self.bm25 = bm25_retriever if bm25_retriever else BM25Retriever()
        self.fusion = fusion if fusion else WeightedReciprocalRankFusion()

    def search(self, query: str, where: dict | None = None) -> List[RetrievalResult]:
        if self.mode == "dense":
            return self.dense.search(query, where=where)
        if self.mode == "bm25":
            return self.bm25.search(query, where=where)

        return self.fusion.fuse(
            self.dense.search(query, where=where),
            self.bm25.search(query, where=where),
        )

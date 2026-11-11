from dataclasses import replace
from typing import List, Protocol

from app.config import settings
from app.models.retrieval import RetrievalResult
from app.services.cross_encoder_service import cross_encoder_service, sigmoid


class Reranker(Protocol):
    def rerank(self, query: str, results: List[RetrievalResult]) -> List[RetrievalResult]: ...


class NoReranker:
    """Reranking disabled: keep the fused order, bounded by top-K (RET-006)."""

    def __init__(self, top_k: int | None = None):
        self.top_k = top_k if top_k is not None else settings.RERANK_TOP_K

    def rerank(self, query: str, results: List[RetrievalResult]) -> List[RetrievalResult]:
        return [replace(r, rerank_score=None, relevance=None) for r in results[: self.top_k]]


class CrossEncoderReranker:
    """
    Reranks retrieved chunks using a Cross Encoder and keeps the top-K.
    `rerank_score` is the raw logit; `relevance` is sigmoid(logit).
    """

    def __init__(self, top_k: int | None = None):
        self.top_k = top_k if top_k is not None else settings.RERANK_TOP_K

    def rerank(self, query: str, results: List[RetrievalResult]) -> List[RetrievalResult]:
        if not results:
            return []

        scores = cross_encoder_service.score(query, [r.document for r in results])
        reranked = [
            replace(r, rerank_score=score, relevance=sigmoid(score))
            for r, score in zip(results, scores)
        ]
        reranked.sort(key=lambda x: x.rerank_score, reverse=True)
        return reranked[: self.top_k]


def build_reranker(kind: str | None = None) -> Reranker:
    """`RERANKER=cross_encoder` (default) or `none` to disable reranking."""
    kind = kind or settings.RERANKER
    if kind == "cross_encoder":
        return CrossEncoderReranker()
    if kind == "none":
        return NoReranker()
    raise ValueError("reranker must be one of: cross_encoder, none")

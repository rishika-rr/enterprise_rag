from dataclasses import dataclass
from typing import Any, Dict


@dataclass
class RetrievalResult:
    """
    Represents a single retrieved chunk.
    """

    id: str

    document: str

    metadata: Dict[str, Any]

    # Retriever-specific score (dense: distance, bm25: BM25 score, hybrid: RRF)
    score: float

    retriever: str

    # Raw reranker output (cross-encoder logit, unbounded)
    rerank_score: float | None = None

    # Normalised 0-1 relevance from the reranker; None when reranking is off
    relevance: float | None = None

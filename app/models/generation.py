from dataclasses import dataclass, field
from typing import List

from app.models.citation import CitationResult
from app.models.retrieval import RetrievalResult


@dataclass
class GenerationResult:
    """
    Represents the final output of the RAG pipeline.
    """

    answer: str

    retrieved_chunks: List[RetrievalResult]

    confidence: float | None = None

    citation_results: List[CitationResult] = field(default_factory=list)

    model: str | None = None

    latency_ms: float | None = None

    # True when no sufficiently relevant passage was found (RET-008)
    no_evidence: bool = False

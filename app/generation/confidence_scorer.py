from app.models.citation import CitationResult
from app.models.retrieval import RetrievalResult
from app.services.cross_encoder_service import sigmoid


class ConfidenceScorer:
    """
    Composite answer confidence in [0, 1]  (GEN-006).

        confidence = 0.5 * retrieval_relevance + 0.5 * citation_support

    retrieval_relevance
        Mean normalised (0-1) reranker relevance of the passages given to the
        LLM: sigmoid(cross-encoder logit).

    citation_support
        Fraction of (claim, citation) pairs the verifier judged supported.
        0 when the answer contains no checkable citations.

    When reranking is disabled there is no relevance signal, and confidence
    equals citation_support alone.

    It is a heuristic indicator, NOT a probability that the answer is
    correct. It must never be presented as verification (GEN-005).
    """

    RETRIEVAL_WEIGHT = 0.5
    CITATION_WEIGHT = 0.5

    @staticmethod
    def _relevance(chunk: RetrievalResult) -> float | None:
        if chunk.relevance is not None:
            return min(max(chunk.relevance, 0.0), 1.0)
        if chunk.rerank_score is not None:
            return sigmoid(chunk.rerank_score)
        return None

    def score(
        self,
        retrieved_chunks: list[RetrievalResult],
        citation_results: list[CitationResult],
    ) -> float:
        if not retrieved_chunks:
            return 0.0

        citation_score = (
            sum(1 for r in citation_results if r.supported) / len(citation_results)
            if citation_results
            else 0.0
        )

        relevances = [r for r in map(self._relevance, retrieved_chunks) if r is not None]
        if not relevances:
            return round(citation_score, 3)

        retrieval_score = sum(relevances) / len(relevances)
        confidence = (
            self.RETRIEVAL_WEIGHT * retrieval_score
            + self.CITATION_WEIGHT * citation_score
        )
        return round(min(max(confidence, 0.0), 1.0), 3)

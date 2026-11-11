from collections import defaultdict
from dataclasses import replace
from typing import List, Sequence

from app.models.retrieval import RetrievalResult


class WeightedReciprocalRankFusion:
    """
    Weighted Reciprocal Rank Fusion:  score(d) = sum_i  w_i / (k + rank_i(d))

    With the default equal weights this is plain RRF.
    """

    def __init__(self, k: int = 60, weights: Sequence[float] | None = None):
        self.k = k
        self.weights = list(weights) if weights is not None else None

    def fuse(self, *retrieval_lists: List[RetrievalResult]) -> List[RetrievalResult]:
        weights = self.weights or [1.0] * len(retrieval_lists)
        if len(weights) != len(retrieval_lists):
            raise ValueError("Number of weights must match number of result lists.")

        scores: dict[str, float] = defaultdict(float)
        lookup: dict[str, RetrievalResult] = {}

        for weight, results in zip(weights, retrieval_lists):
            for rank, chunk in enumerate(results, start=1):
                scores[chunk.id] += weight / (self.k + rank)
                lookup.setdefault(chunk.id, chunk)

        ranked = sorted(scores.items(), key=lambda item: item[1], reverse=True)

        # copies - never mutate the retrievers' own result objects
        return [
            replace(lookup[cid], score=score, retriever="hybrid")
            for cid, score in ranked
        ]

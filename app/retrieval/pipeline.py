from app.database.chroma import matches_nothing
from app.models.retrieval import RetrievalResult
from app.retrieval.hybrid import HybridRetriever
from app.retrieval.reranker import Reranker, build_reranker


class RetrievalPipeline:
    """
    Complete retrieval pipeline:

        dense + BM25 retrieval  ->  weighted reciprocal rank fusion
                                ->  reranking  ->  top-K passages

    `where` is applied inside both retrievers, so unauthorized chunks are
    excluded before fusion, reranking and generation (SEC-004, NFR-SEC-004).
    """

    def __init__(
        self,
        retriever: HybridRetriever | None = None,
        reranker: Reranker | None = None,
    ):
        self.retriever = retriever if retriever is not None else HybridRetriever()
        self.reranker = reranker if reranker is not None else build_reranker()

    def retrieve(self, query: str, where: dict | None = None) -> list[RetrievalResult]:
        if matches_nothing(where):  # nothing is authorized: skip model calls entirely
            return []
        retrieved = self.retriever.search(query, where=where)
        return self.reranker.rerank(query=query, results=retrieved)

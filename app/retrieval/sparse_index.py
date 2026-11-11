import logging
import re
import threading
from typing import List

from rank_bm25 import BM25Okapi

from app.models.retrieval import RetrievalResult
from app.retrieval.retrieval_index import RetrievalIndex

logger = logging.getLogger("rag.sparse")

_TOKEN = re.compile(r"\w+", re.UNICODE)


def tokenize(text: str) -> List[str]:
    """Lower-case word tokens, with punctuation stripped ("leave." -> "leave")."""
    return _TOKEN.findall(text.lower())


class SparseIndex(RetrievalIndex):
    """
    In-memory BM25 index. It is rebuilt from the vector store on start-up
    (see `get_sparse_index`) so it survives restarts.
    """

    def __init__(self):
        self.ids: List[str] = []
        self.documents: List[str] = []
        self.metadatas: List[dict] = []
        self.tokenized_documents: List[List[str]] = []
        self._token_sets: List[set] = []
        self.bm25: BM25Okapi | None = None
        self._lock = threading.RLock()

    def _rebuild(self) -> None:
        self.bm25 = BM25Okapi(self.tokenized_documents) if self.tokenized_documents else None

    def add_documents(self, ids: List[str], documents: List[str], metadatas: List[dict]) -> None:
        with self._lock:
            self.ids.extend(ids)
            self.documents.extend(documents)
            self.metadatas.extend(metadatas)
            tokenized = [tokenize(d) for d in documents]
            self.tokenized_documents.extend(tokenized)
            self._token_sets.extend(set(t) for t in tokenized)
            self._rebuild()

    def remove_ids(self, ids: List[str]) -> None:
        drop = set(ids)
        if not drop:
            return
        with self._lock:
            keep = [i for i, cid in enumerate(self.ids) if cid not in drop]
            self.ids = [self.ids[i] for i in keep]
            self.documents = [self.documents[i] for i in keep]
            self.metadatas = [self.metadatas[i] for i in keep]
            self.tokenized_documents = [self.tokenized_documents[i] for i in keep]
            self._token_sets = [self._token_sets[i] for i in keep]
            self._rebuild()

    def remove_document(self, doc_id: str) -> int:
        """Remove all chunks whose metadata doc_id matches."""
        with self._lock:
            ids = [i for i, m in zip(self.ids, self.metadatas) if m.get("doc_id") == doc_id]
            self.remove_ids(ids)
            return len(ids)

    @staticmethod
    def _compile(where: dict | None) -> list[tuple[str, object, bool]]:
        """
        Normalise a filter to (key, expected, is_set) triples. Supports equality
        and {"$in": [...]}. Other Chroma operators are rejected: failing loudly
        beats silently returning rows the filter should exclude.
        """
        compiled = []
        for key, expected in (where or {}).items():
            if isinstance(expected, dict):
                if set(expected) != {"$in"} or not isinstance(expected["$in"], (list, tuple, set)):
                    raise ValueError("SparseIndex supports equality and $in filters only.")
                compiled.append((key, frozenset(expected["$in"]), True))
            else:
                compiled.append((key, expected, False))
        return compiled

    @staticmethod
    def _matches(metadata: dict, compiled: list[tuple[str, object, bool]]) -> bool:
        for key, expected, is_set in compiled:
            value = metadata.get(key)
            if (value not in expected) if is_set else (value != expected):
                return False
        return True

    def search(
        self, query: str, top_k: int = 10, where: dict | None = None
    ) -> List[RetrievalResult]:
        compiled = self._compile(where)
        with self._lock:
            if self.bm25 is None:
                return []

            tokens = tokenize(query)
            if not tokens:
                return []

            scores = self.bm25.get_scores(tokens)
            query_terms = set(tokens)
            # Keep only chunks that share at least one term with the query.
            # (Filtering on `score > 0` is wrong: Okapi IDF is <= 0 for terms
            # present in half or more of the corpus, e.g. in tiny corpora.)
            matching = [
                (i, s)
                for i, s in enumerate(scores)
                if query_terms & self._token_sets[i]
                and self._matches(self.metadatas[i], compiled)
            ]
            ranked = sorted(matching, key=lambda x: x[1], reverse=True)[:top_k]

            return [
                RetrievalResult(
                    id=self.ids[index],
                    document=self.documents[index],
                    metadata=self.metadatas[index],
                    score=float(score),
                    retriever="bm25",
                )
                for index, score in ranked
            ]


_sparse_index: SparseIndex | None = None
_sparse_lock = threading.Lock()


def get_sparse_index() -> SparseIndex:
    """
    Process-wide shared BM25 index, hydrated from the persisted vector store.

    The indexer and the BM25 retriever MUST share this instance - previously
    each created its own empty index, so BM25 never returned anything.
    """
    global _sparse_index
    if _sparse_index is None:
        with _sparse_lock:
            if _sparse_index is None:
                index = SparseIndex()
                try:
                    from app.database.chroma import get_vector_store

                    ids, documents, metadatas = get_vector_store().get_all()
                    if ids:
                        index.add_documents(ids, documents, metadatas)
                    logger.info("sparse_index_hydrated chunks=%d", len(ids))
                except Exception:
                    logger.exception("sparse_index_hydration_failed")
                _sparse_index = index
    return _sparse_index

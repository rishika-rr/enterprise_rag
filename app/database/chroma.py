import threading
from collections import OrderedDict
from typing import List, Tuple

import chromadb

from app.config import settings
from app.database.interfaces import VectorStore
from app.models.retrieval import RetrievalResult

LEGACY_PREFIX = "legacy-"  # id prefix given to documents indexed before doc_id existed
_BATCH = 500


def matches_nothing(where: dict | None) -> bool:
    """True when a filter contains an empty `$in` list (nothing is authorized)."""
    return any(
        isinstance(value, dict) and "$in" in value and not value["$in"]
        for value in (where or {}).values()
    )


class ChromaVectorStore(VectorStore):
    """
    ChromaDB implementation of the VectorStore interface.
    """

    def __init__(self, collection_name: str = "enterprise_rag", path: str | None = None):
        self.client = chromadb.PersistentClient(path=path or settings.CHROMA_DB_PATH)
        self.collection = self.client.get_or_create_collection(name=collection_name)

    # ------------------------------------------------------------------

    @staticmethod
    def _where(doc_id: str) -> dict:
        return {"doc_id": doc_id}

    def adopt_legacy_chunks(self) -> int:
        """
        Give chunks indexed before `doc_id` existed a stable parent reference
        ("legacy-<source name>") so they can be registered, authorized and
        deleted like any other document (DATA-002). Idempotent.
        """
        found = self.collection.get(include=["metadatas"])
        ids, metadatas = [], []
        for chunk_id, metadata in zip(found["ids"], found["metadatas"]):
            metadata = dict(metadata or {})
            if metadata.get("doc_id"):
                continue
            source = metadata.get("source") or "Unknown Document"
            metadata["source"] = source
            metadata["doc_id"] = f"{LEGACY_PREFIX}{source}"
            ids.append(chunk_id)
            metadatas.append(metadata)
        for i in range(0, len(ids), _BATCH):
            self.collection.update(ids=ids[i:i + _BATCH], metadatas=metadatas[i:i + _BATCH])
        return len(ids)

    @staticmethod
    def to_chroma_where(where: dict | None) -> dict | None:
        """
        Chroma rejects a dict with several keys ("exactly one operator"),
        so AND them explicitly. Empty / None means "no filter".
        A value may be {"$in": [...]}; an empty list matches nothing and is
        handled by the caller (Chroma rejects it).
        """
        if not where:
            return None
        if len(where) == 1:
            return dict(where)
        return {"$and": [{key: value} for key, value in where.items()]}

    def add_documents(
        self,
        ids: List[str],
        documents: List[str],
        embeddings: List[List[float]],
        metadatas: List[dict],
    ) -> None:
        # Chroma limits the size of a single add(); batch large documents.
        for i in range(0, len(ids), _BATCH):
            # upsert: a retried ingestion after a crash can never fail/duplicate
            self.collection.upsert(
                ids=ids[i:i + _BATCH],
                documents=documents[i:i + _BATCH],
                embeddings=embeddings[i:i + _BATCH],
                metadatas=metadatas[i:i + _BATCH],
            )

    def search(
        self,
        embedding: List[float],
        top_k: int = 10,
        where: dict | None = None,
    ) -> List[RetrievalResult]:
        total = self.collection.count()
        if total == 0 or matches_nothing(where):
            return []

        query = {"query_embeddings": [embedding], "n_results": min(top_k, total)}
        chroma_where = self.to_chroma_where(where)
        if chroma_where:
            query["where"] = chroma_where

        results = self.collection.query(**query)

        return [
            RetrievalResult(
                id=chunk_id,
                document=document,
                metadata=metadata,
                score=distance,  # distance: lower = closer
                retriever="dense",
            )
            for chunk_id, document, metadata, distance in zip(
                results["ids"][0],
                results["documents"][0],
                results["metadatas"][0],
                results["distances"][0],
            )
        ]

    def count(self) -> int:
        return self.collection.count()

    def delete(self) -> None:
        self.client.delete_collection(self.collection.name)

    # ---- document-level --------------------------------------------------

    def document_chunk_count(self, doc_id: str) -> int:
        found = self.collection.get(where=self._where(doc_id), include=[])
        return len(found["ids"])

    def delete_document(self, doc_id: str) -> int:
        found = self.collection.get(where=self._where(doc_id), include=[])
        ids = found["ids"]
        for i in range(0, len(ids), _BATCH):
            self.collection.delete(ids=ids[i:i + _BATCH])
        return len(ids)

    def list_documents(self) -> List[dict]:
        found = self.collection.get(include=["metadatas"])
        docs: "OrderedDict[str, dict]" = OrderedDict()
        for metadata in found["metadatas"]:
            source = metadata.get("source", "Unknown Document")
            doc_id = metadata.get("doc_id") or f"{LEGACY_PREFIX}{source}"  # see adopt_legacy_chunks
            entry = docs.setdefault(
                doc_id, {"doc_id": doc_id, "name": source, "chunks": 0}
            )
            entry["chunks"] += 1
        return sorted(docs.values(), key=lambda d: d["name"].lower())

    def get_all(self) -> Tuple[List[str], List[str], List[dict]]:
        found = self.collection.get(include=["documents", "metadatas"])
        return found["ids"], found["documents"], found["metadatas"]


_store: ChromaVectorStore | None = None
_store_lock = threading.Lock()


def get_vector_store() -> ChromaVectorStore:
    """Process-wide shared store (one Chroma client)."""
    global _store
    if _store is None:
        with _store_lock:
            if _store is None:
                _store = ChromaVectorStore()
    return _store

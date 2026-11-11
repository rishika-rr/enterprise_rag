import hashlib
import logging
import time
from pathlib import Path

from langchain_core.documents import Document

from app.database.chroma import get_vector_store
from app.database.interfaces import VectorStore
from app.ingestion.chunker import Chunker
from app.ingestion.loader import DocumentLoader
from app.models.document import ChunkingStrategy, DocumentChunk, IngestResult
from app.retrieval.sparse_index import SparseIndex, get_sparse_index
from app.services.embedding_service import embedding_service

logger = logging.getLogger("rag.indexer")


def compute_doc_id(file_path: str, scope: str = "") -> str:
    """
    Stable document identifier (IF-004): SHA-256 of the scope and the file
    content. The scope (tenant, plus the owner for private documents) keeps
    identical files of different tenants apart, so one tenant can never
    detect or collide with another tenant's document.
    """
    digest = hashlib.sha256()
    if scope:
        digest.update(scope.encode("utf-8") + b"\x00")
    with open(file_path, "rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()[:16]


class Indexer:
    """
    Coordinates the complete document ingestion pipeline.

        Load -> Chunk -> Embed -> Dense Index -> Sparse Index

    Guarantees:
      * identical content is indexed once (duplicate policy, ING-007)
      * ingestion is all-or-nothing: on failure nothing is left behind
        (ING-009, ERR-005, NFR-REL-002)
    """

    def __init__(
        self,
        vector_store: VectorStore | None = None,
        sparse_index: SparseIndex | None = None,
    ):
        self.loader = DocumentLoader()
        self.chunker = Chunker()
        self.vector_store = vector_store if vector_store is not None else get_vector_store()
        self.sparse_index = sparse_index if sparse_index is not None else get_sparse_index()

    # --------------------------------------------------
    # Helpers
    # --------------------------------------------------

    @staticmethod
    def _create_document_chunk(
        chunk: Document, index: int, strategy: ChunkingStrategy, doc_id: str, source: str
    ) -> DocumentChunk:
        metadata = chunk.metadata
        page = metadata.get("page")

        return DocumentChunk(
            id=f"{doc_id}-{index}",
            text=chunk.page_content,
            source=source,
            page=page + 1 if isinstance(page, int) else None,  # PyMuPDF is 0-based
            section=metadata.get("section"),
            chunk_index=index,
            strategy=strategy,
            char_count=len(chunk.page_content),
            doc_id=doc_id,
        )

    @staticmethod
    def _prepare_metadata(chunk: DocumentChunk) -> dict:
        metadata = {
            "doc_id": chunk.doc_id,
            "source": str(chunk.source),
            "chunk_index": int(chunk.chunk_index),
            "strategy": str(chunk.strategy.value),
            "char_count": int(chunk.char_count),
        }
        if chunk.page is not None:
            metadata["page"] = int(chunk.page)
        if chunk.section is not None:
            metadata["section"] = str(chunk.section)
        return metadata

    # --------------------------------------------------
    # Public API
    # --------------------------------------------------

    def index_document(
        self,
        file_path: str,
        strategy: ChunkingStrategy = ChunkingStrategy.RECURSIVE,
        doc_id: str | None = None,
        source_name: str | None = None,
    ) -> IngestResult:
        """
        `doc_id` / `source_name` let the caller supply a tenant-scoped id and
        the original file name when the file is stored under another name.
        """
        start = time.time()

        documents = self.loader.load(file_path)  # validates existence / type / content
        doc_id = doc_id or compute_doc_id(file_path)
        source = source_name or Path(file_path).name

        existing = self.vector_store.document_chunk_count(doc_id)
        if existing:
            logger.info("ingest_duplicate doc_id=%s", doc_id)
            return IngestResult(doc_id, source, "duplicate", existing)

        chunks = [
            c for c in self.chunker.chunk(documents, strategy) if c.page_content.strip()
        ]
        if not chunks:
            raise ValueError(
                "No extractable text found (scanned/image-only or empty file)."
            )

        document_chunks = [
            self._create_document_chunk(chunk, i, strategy, doc_id, source)
            for i, chunk in enumerate(chunks)
        ]
        texts = [c.text for c in document_chunks]
        ids = [c.id for c in document_chunks]
        metadatas = [self._prepare_metadata(c) for c in document_chunks]

        # Embed everything first - the slow, failure-prone step - before
        # touching either index.
        embeddings = embedding_service.embed_documents(texts)

        try:
            self.vector_store.add_documents(
                ids=ids, documents=texts, embeddings=embeddings, metadatas=metadatas
            )
            self.sparse_index.add_documents(ids=ids, documents=texts, metadatas=metadatas)
        except Exception:
            logger.exception("ingest_failed_rolling_back doc_id=%s", doc_id)
            self._rollback(doc_id, ids)
            raise

        logger.info(
            "ingest_ok doc_id=%s chunks=%d seconds=%.2f",
            doc_id, len(document_chunks), time.time() - start,
        )
        return IngestResult(doc_id, source, "indexed", len(document_chunks))

    def _rollback(self, doc_id: str, ids: list[str]) -> None:
        try:
            self.vector_store.delete_document(doc_id)
        except Exception:
            logger.exception("rollback_vector_store_failed doc_id=%s", doc_id)
        try:
            self.sparse_index.remove_ids(ids)
        except Exception:
            logger.exception("rollback_sparse_index_failed doc_id=%s", doc_id)

    def delete_document(self, doc_id: str) -> int:
        """Remove a document and all derived chunks from both indexes (ING-008)."""
        removed = self.vector_store.delete_document(doc_id)
        self.sparse_index.remove_document(doc_id)
        return removed

    def list_documents(self) -> list[dict]:
        return self.vector_store.list_documents()

    def get_stats(self):
        return {"total_chunks": self.vector_store.count()}

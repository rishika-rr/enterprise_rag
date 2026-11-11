import logging
import os
import threading
from pathlib import Path

from app.config import settings
from app.database.state import StateStore, get_state_store
from app.generation.generator import Generator
from app.ingestion.indexer import Indexer, compute_doc_id
from app.ingestion.loader import DocumentLoader
from app.models.document import ChunkingStrategy, IngestResult
from app.models.generation import GenerationResult
from app.models.identity import DocumentRecord, DocumentStatus, User, Visibility

logger = logging.getLogger("rag.service")

_STRATEGIES = {
    "fixed": ChunkingStrategy.FIXED,
    "recursive": ChunkingStrategy.RECURSIVE,
}

INCOMING_PREFIX = ".incoming-"  # uploads that have not been accepted yet


def parse_strategy(strategy: str) -> ChunkingStrategy:
    try:
        return _STRATEGIES[strategy.lower()]
    except KeyError:
        raise ValueError(
            f"Unsupported chunking strategy '{strategy}'. "
            f"Use one of: {', '.join(_STRATEGIES)}."
        ) from None


def _diagnostic(error: Exception) -> str:
    """Failure reason safe to store and show: no paths, content or internals (ING-010)."""
    if isinstance(error, ValueError):
        return str(error)
    return f"Internal processing error ({type(error).__name__})."


class EnterpriseRAG:
    """
    High-level interface for the Enterprise RAG system.

    Responsibilities:
    - Document ingestion / deletion, kept consistent with the document
      registry (status, ownership, visibility)
    - Question answering restricted to the documents a user may read
    """

    def __init__(
        self,
        generator: Generator | None = None,
        indexer: Indexer | None = None,
        state: StateStore | None = None,
    ):
        self.state = state if state is not None else get_state_store()
        self._write_lock = threading.RLock()  # serialises ingest / delete
        self._generator = generator
        self._generator_lock = threading.Lock()

        if indexer is None:
            # Before the BM25 index is hydrated from the vector store, give
            # pre-doc_id chunks their parent reference.
            from app.database.chroma import get_vector_store

            adopted = get_vector_store().adopt_legacy_chunks()
            if adopted:
                logger.info("legacy_chunks_adopted chunks=%d", adopted)
            indexer = Indexer()
        self.indexer = indexer

        self.reconcile()
        self.apply_retention()

    @property
    def generator(self) -> Generator:
        """Created on first use, so document management works without an LLM key."""
        if self._generator is None:
            with self._generator_lock:
                if self._generator is None:
                    self._generator = Generator()
        return self._generator

    # ---- consistency ----------------------------------------------------

    def reconcile(self) -> None:
        """
        Bring registry and indexes back in line after a crash or restart
        (NFR-REL-002, DATA-003):

        * a job still `processing` was interrupted -> its chunks are removed
          and it is marked failed;
        * a document `indexed` in the registry but absent from the index is
          marked failed;
        * documents indexed before the registry existed ("legacy-" ids) are
          registered as tenant-visible documents of the default tenant;
        * any other unregistered chunks stay unreadable (fail closed).
        """
        with self._write_lock:
            upload_dir = Path(settings.UPLOAD_DIR)
            if upload_dir.is_dir():
                for stale in upload_dir.glob(f"{INCOMING_PREFIX}*"):
                    stale.unlink(missing_ok=True)

            for record in self.state.all_documents():
                if record.status == DocumentStatus.PROCESSING:
                    self.indexer.delete_document(record.doc_id)
                    self._remove_stored_file(record)
                    self.state.mark_failed(record.doc_id, "Ingestion was interrupted.")
                    logger.warning("interrupted_ingestion_rolled_back doc_id=%s", record.doc_id)

            indexed = {d["doc_id"]: d for d in self.indexer.list_documents()}
            registered = {r.doc_id: r for r in self.state.all_documents()}

            for doc_id, record in registered.items():
                if record.status == DocumentStatus.INDEXED and doc_id not in indexed:
                    self.state.mark_failed(
                        doc_id, "Indexed content is missing; ingest the document again."
                    )
                    logger.warning("registry_document_missing_from_index doc_id=%s", doc_id)

            orphans = 0
            for doc_id, summary in indexed.items():
                if doc_id in registered:
                    continue
                if not doc_id.startswith("legacy-"):
                    orphans += 1
                    continue
                stored = upload_dir / summary["name"]
                self.state.begin_document(
                    doc_id=doc_id,
                    tenant_id=settings.DEFAULT_TENANT,
                    owner_id=None,
                    name=summary["name"],
                    visibility=Visibility.TENANT,
                    size_bytes=stored.stat().st_size if stored.is_file() else None,
                    stored_path=str(stored) if stored.is_file() else None,
                )
                self.state.mark_indexed(doc_id, summary["chunks"])
            if orphans:
                logger.warning(
                    "unregistered_documents_in_index count=%d - not readable by any user", orphans
                )

    def apply_retention(self) -> None:
        """Enforce the configured retention periods (DATA-005, NFR-SEC-006)."""
        if settings.AUDIT_RETENTION_DAYS:
            purged = self.state.purge_audit_older_than(settings.AUDIT_RETENTION_DAYS)
            if purged:
                logger.info("audit_retention_purged events=%d", purged)
        if settings.DOCUMENT_RETENTION_DAYS:
            for record in self.state.documents_older_than(settings.DOCUMENT_RETENTION_DAYS):
                self.delete_document(record.doc_id)
                self.state.audit(
                    "document.retention_delete", "success",
                    target=record.doc_id, tenant_id=record.tenant_id, actor_name="system",
                )

    # ---- ingestion ------------------------------------------------------

    def ingest(
        self,
        document_path: str,
        strategy: str | None = None,
        *,
        tenant_id: str | None = None,
        owner_id: str | None = None,
        visibility: Visibility | str = Visibility.TENANT,
        source_name: str | None = None,
        take_ownership: bool = False,
    ) -> IngestResult:
        """
        Validate, register and index one file.

        `take_ownership=True` (uploads): the file is moved into the upload
        directory under its document id and removed again if indexing fails
        or the content is a duplicate.

        Duplicate policy (ING-007): identical content within the same access
        scope is indexed once; a repeat returns status "duplicate".
        """
        path = Path(document_path)
        if not path.is_file():
            raise FileNotFoundError(f"{document_path} not found.")

        chunking = parse_strategy(strategy or settings.CHUNK_STRATEGY)
        name = source_name or path.name
        if not DocumentLoader.is_supported(name):
            raise ValueError(f"Unsupported file type: {Path(name).suffix or '(none)'}")
        DocumentLoader.validate(path)  # rejected files are never registered

        tenant = tenant_id or settings.DEFAULT_TENANT
        visibility = Visibility(visibility)
        scope = tenant if visibility == Visibility.TENANT else f"{tenant}|{owner_id}"
        doc_id = compute_doc_id(str(path), scope)

        with self._write_lock:
            record = self.state.get_document(doc_id)
            indexed_chunks = self.indexer.vector_store.document_chunk_count(doc_id)

            if record and record.status == DocumentStatus.INDEXED and indexed_chunks:
                if take_ownership:
                    path.unlink(missing_ok=True)
                logger.info("ingest_duplicate doc_id=%s", doc_id)
                return IngestResult(doc_id, record.name, "duplicate", indexed_chunks)

            if indexed_chunks:  # leftovers of an earlier interrupted attempt
                self.indexer.delete_document(doc_id)

            size = path.stat().st_size
            stored_path = None
            if take_ownership:
                upload_dir = Path(settings.UPLOAD_DIR)
                upload_dir.mkdir(parents=True, exist_ok=True)
                final = upload_dir / f"{doc_id}{path.suffix.lower()}"
                os.replace(path, final)
                path, stored_path = final, str(final)

            self.state.begin_document(
                doc_id, tenant, owner_id, name, visibility, size, stored_path
            )
            try:
                result = self.indexer.index_document(
                    str(path), chunking, doc_id=doc_id, source_name=name
                )
            except Exception as e:
                if take_ownership:
                    path.unlink(missing_ok=True)
                self.state.mark_failed(doc_id, _diagnostic(e))
                logger.warning("ingest_failed doc_id=%s reason=%s", doc_id, _diagnostic(e))
                raise

            self.state.mark_indexed(doc_id, result.chunk_count)
            return IngestResult(doc_id, name, "indexed", result.chunk_count)

    def ingest_directory(self, directory: str, strategy: str | None = None, **owner) -> dict:
        """Index every supported file; returns accepted/processed/failed counts (ING-006)."""
        directory_path = Path(directory)
        if not directory_path.is_dir():
            raise FileNotFoundError(directory)

        summary = {
            "accepted": 0,
            "processed": 0,
            "indexed": 0,
            "duplicates": 0,
            "failed": 0,
            "skipped_unsupported": 0,
            "total_chunks": 0,
            "failures": [],
        }

        for file in sorted(p for p in directory_path.rglob("*") if p.is_file()):
            if not DocumentLoader.is_supported(file.name):
                summary["skipped_unsupported"] += 1
                continue

            summary["accepted"] += 1
            try:
                result = self.ingest(str(file), strategy, **owner)
            except Exception as e:
                summary["failed"] += 1
                summary["failures"].append({"file": file.name, "reason": _diagnostic(e)})
                continue

            summary["processed"] += 1
            if result.status == "duplicate":
                summary["duplicates"] += 1
            else:
                summary["indexed"] += 1
                summary["total_chunks"] += result.chunk_count

        return summary

    @staticmethod
    def _remove_stored_file(record: DocumentRecord) -> None:
        if record.stored_path:
            Path(record.stored_path).unlink(missing_ok=True)

    def delete_document(self, doc_id: str) -> int:
        """
        Remove a document everywhere: both indexes, the stored upload and the
        registry (ING-008, DATA-004). Takes effect immediately; nothing is
        cached. Returns the number of chunks removed.
        """
        with self._write_lock:
            record = self.state.get_document(doc_id)
            removed = self.indexer.delete_document(doc_id)
            if record is not None:
                self._remove_stored_file(record)
                self.state.delete_document(doc_id)
            return removed

    def list_documents(self) -> list[dict]:
        """Raw index contents (no access control) - for operators and tests."""
        return self.indexer.list_documents()

    # ---- question answering -------------------------------------------

    def access_filter(self, user: User) -> dict:
        """Retrieval filter limiting results to documents `user` may read (SEC-004)."""
        return {"doc_id": {"$in": self.state.readable_doc_ids(user)}}

    def ask(
        self, question: str, where: dict | None = None, user: User | None = None
    ) -> GenerationResult:
        """
        `user` restricts retrieval to the documents that user is authorized
        to read; `where` adds further metadata equality filters.
        """
        if user is not None:
            where = dict(where or {})
            allowed = self.state.readable_doc_ids(user)
            requested = where.get("doc_id")
            if isinstance(requested, str):  # caller narrowed to one document
                allowed = [requested] if requested in allowed else []
            where["doc_id"] = {"$in": allowed}
        return self.generator.generate(question, where=where)

    # ---- stats ----------------------------------------------------------

    def stats(self):
        return self.indexer.get_stats()

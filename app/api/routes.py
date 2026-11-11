import logging
import threading
import uuid
from pathlib import Path

from fastapi import APIRouter, Depends, File, Form, HTTPException, Request, UploadFile

from app.api.errors import (
    AUTH_ERRORS,
    DEPENDENCY_ERRORS,
    NOT_FOUND,
    RATE_LIMITED,
    UPLOAD_ERRORS,
    to_http_error,
)
from app.api.schemas import (
    AskRequest,
    AskResponse,
    BatchIngestResponse,
    BatchItem,
    CitationDetail,
    DeleteResponse,
    DocumentList,
    DocumentOut,
    DocumentUpdate,
    IngestRequest,
    IngestResponse,
    SourceResponse,
)
from app.config import settings
from app.database.state import StateStore
from app.ingestion.loader import DocumentLoader, FileValidationError
from app.models.document import IngestResult
from app.models.identity import (
    DocumentRecord,
    DocumentStatus,
    Permission,
    User,
    Visibility,
    can_manage,
    can_read,
)
from app.rag import INCOMING_PREFIX, EnterpriseRAG
from app.security.auth import get_state, require
from app.security.rate_limit import ask_limiter

logger = logging.getLogger("rag.api")

router = APIRouter(prefix="/v1", tags=["Enterprise RAG"], responses=AUTH_ERRORS)

# --------------------------------------------------
# Lazy initialisation
# --------------------------------------------------

rag: EnterpriseRAG | None = None
_rag_lock = threading.Lock()


def get_rag() -> EnterpriseRAG:
    global rag
    if rag is None:
        with _rag_lock:
            if rag is None:
                rag = EnterpriseRAG()
    return rag


def _request_id(request: Request) -> str | None:
    return getattr(request.state, "request_id", None)


def _is_allowed_ingest_path(path: Path) -> bool:
    resolved = path.resolve()
    for allowed in settings.INGEST_ALLOWED_DIRS:
        root = Path(allowed).resolve()
        if resolved == root or root in resolved.parents:
            return True
    return False


def _document_out(record: DocumentRecord, user: User) -> DocumentOut:
    # Ingestion diagnostics are only for users who manage documents (UI-004).
    manager = user.has(Permission.DOCUMENTS_WRITE)
    return DocumentOut(
        doc_id=record.doc_id,
        name=record.name,
        chunks=record.chunks,
        size=record.size_bytes,
        status=record.status,
        visibility=record.visibility,
        owner_id=record.owner_id if manager else None,
        error=record.error if manager else None,
        created_at=record.created_at,
        updated_at=record.updated_at,
    )


def _visible(record: DocumentRecord, user: User) -> bool:
    if can_read(user, record):
        # Readers only ever see documents that are actually searchable.
        return record.status == DocumentStatus.INDEXED or user.has(Permission.DOCUMENTS_WRITE)
    return can_manage(user, record)


# --------------------------------------------------
# Ask
# --------------------------------------------------


@router.post(
    "/ask",
    response_model=AskResponse,
    responses={**RATE_LIMITED, **DEPENDENCY_ERRORS},
    summary="Ask a question about the documents you are authorized to read",
)
def ask(
    payload: AskRequest,
    request: Request,
    user: User = Depends(require(Permission.QUERY)),
):
    # NB: plain `def` (not async) -> FastAPI runs this in a worker thread, so
    # the blocking retrieval/LLM calls do not freeze the event loop.
    retry_after = ask_limiter.check(user.id, settings.RATE_LIMIT_PER_MINUTE)
    if retry_after:
        raise HTTPException(
            429, "Rate limit exceeded. Please retry later.",
            headers={"Retry-After": str(retry_after)},
        )

    try:
        result = get_rag().ask(payload.question, user=user)

        verified = sum(1 for c in result.citation_results if c.supported)

        # citation number -> passage, in the order the passages were given to the model
        sources = [
            SourceResponse(
                title=chunk.metadata.get("source", "Unknown Document"),
                score=round(chunk.relevance, 4) if chunk.relevance is not None else None,
                citation=number,
                page=chunk.metadata.get("page"),
                snippet=chunk.document[:300],
                doc_id=chunk.metadata.get("doc_id"),
            )
            for number, chunk in enumerate(result.retrieved_chunks, start=1)
        ]

        return AskResponse(
            answer=result.answer,
            confidence=result.confidence,
            model=result.model,
            latency_ms=result.latency_ms,
            retrieved_chunks=len(result.retrieved_chunks),
            citations=len(result.citation_results),
            sources=sources,
            verified_citations=verified,
            citation_details=[
                CitationDetail(
                    claim=c.claim, citation=c.citation, supported=c.supported, reason=c.reason
                )
                for c in result.citation_results
            ],
            no_evidence=result.no_evidence,
            request_id=_request_id(request),
        )
    except Exception as e:
        raise to_http_error(e)


# --------------------------------------------------
# Upload
# --------------------------------------------------


def _receive_and_ingest(
    file: UploadFile, user: User, visibility: Visibility
) -> tuple[str, IngestResult]:
    """
    Validate one uploaded file, store it and index it.
    Raises HTTPException 413/415/422 for files that are not accepted.
    """
    # Strip any directory components -> blocks path traversal ("../../x.pdf")
    filename = Path(file.filename or "").name

    if not filename or not DocumentLoader.is_supported(filename):
        allowed = ", ".join(sorted(DocumentLoader.SUPPORTED_EXTENSIONS))
        raise HTTPException(415, f"Unsupported file type. Allowed: {allowed}")

    # Build the engine first: its start-up clean-up removes stale incoming files.
    engine = get_rag()

    upload_dir = Path(settings.UPLOAD_DIR)
    upload_dir.mkdir(parents=True, exist_ok=True)
    incoming = upload_dir / f"{INCOMING_PREFIX}{uuid.uuid4().hex}{Path(filename).suffix.lower()}"
    limit = settings.MAX_UPLOAD_MB * 1024 * 1024

    try:
        written = 0
        with incoming.open("wb") as buffer:
            while block := file.file.read(1024 * 1024):
                written += len(block)
                if written > limit:
                    raise HTTPException(
                        413, f"File exceeds the {settings.MAX_UPLOAD_MB} MB limit."
                    )
                buffer.write(block)

        if written == 0:
            raise HTTPException(422, "The uploaded file is empty.")

        result = engine.ingest(
            str(incoming),
            tenant_id=user.tenant_id,
            owner_id=user.id,
            visibility=visibility,
            source_name=filename,
            take_ownership=True,
        )
        return filename, result
    finally:
        incoming.unlink(missing_ok=True)  # no-op once the file has been accepted


def _ingest_response(filename: str, result: IngestResult) -> IngestResponse:
    return IngestResponse(
        message=(
            "Document already indexed."
            if result.status == "duplicate"
            else "Document indexed successfully."
        ),
        doc_id=result.doc_id,
        filename=filename,
        status=result.status,
        chunks=result.chunk_count,
    )


@router.post(
    "/upload",
    response_model=IngestResponse,
    responses=UPLOAD_ERRORS,
    summary="Upload and index one document",
)
def upload_document(
    request: Request,
    file: UploadFile = File(...),
    visibility: Visibility = Form(Visibility.TENANT),
    user: User = Depends(require(Permission.DOCUMENTS_WRITE)),
    state: StateStore = Depends(get_state),
):
    try:
        filename, result = _receive_and_ingest(file, user, visibility)
    except Exception as e:
        error = to_http_error(e)
        state.audit(
            "document.ingest", "failure", actor=user,
            detail=str(error.detail)[:200], request_id=_request_id(request),
        )
        raise error
    state.audit(
        "document.ingest", "success", actor=user, target=result.doc_id,
        detail=result.status, request_id=_request_id(request),
    )
    return _ingest_response(filename, result)


@router.post(
    "/upload/batch",
    response_model=BatchIngestResponse,
    responses=UPLOAD_ERRORS,
    summary="Upload several documents; reports accepted / processed / failed counts",
)
def upload_documents(
    request: Request,
    files: list[UploadFile] = File(...),
    visibility: Visibility = Form(Visibility.TENANT),
    user: User = Depends(require(Permission.DOCUMENTS_WRITE)),
    state: StateStore = Depends(get_state),
):
    if len(files) > settings.MAX_BATCH_FILES:
        raise HTTPException(413, f"At most {settings.MAX_BATCH_FILES} files per request.")

    results: list[BatchItem] = []
    accepted = processed = 0
    for file in files:
        shown_name = Path(file.filename or "").name or "(unnamed)"
        try:
            filename, result = _receive_and_ingest(file, user, visibility)
        except Exception as e:
            # Type / size / content validation failures are refused before
            # acceptance; anything else failed during processing.
            rejected = isinstance(e, (HTTPException, FileValidationError))
            error = to_http_error(e)
            accepted += 0 if rejected else 1
            results.append(
                BatchItem(
                    filename=shown_name,
                    status="rejected" if rejected else "failed",
                    reason=str(error.detail),
                )
            )
            continue

        accepted += 1
        processed += 1
        results.append(
            BatchItem(
                filename=filename, status=result.status,
                doc_id=result.doc_id, chunks=result.chunk_count,
            )
        )

    failed = len(files) - processed
    state.audit(
        "document.ingest_batch", "success" if not failed else "partial", actor=user,
        detail=f"received={len(files)} processed={processed} failed={failed}",
        request_id=_request_id(request),
    )
    return BatchIngestResponse(
        received=len(files), accepted=accepted, processed=processed, failed=failed,
        results=results,
    )


# --------------------------------------------------
# Ingest (server-side path, administrators only)
# --------------------------------------------------


@router.post(
    "/ingest",
    response_model=IngestResponse,
    responses={**NOT_FOUND, **UPLOAD_ERRORS},
    summary="Index a file that already exists in an allowed server directory",
)
def ingest(
    payload: IngestRequest,
    request: Request,
    user: User = Depends(require(Permission.SERVER_INGEST)),
    state: StateStore = Depends(get_state),
):
    try:
        path = Path(payload.document_path)
        # Only files inside the configured directories: otherwise any file
        # readable by the server could be indexed and then queried.
        if not _is_allowed_ingest_path(path):
            raise HTTPException(403, "Path is outside the allowed ingestion directories.")

        result = get_rag().ingest(
            str(path),
            payload.strategy,
            tenant_id=user.tenant_id,
            owner_id=user.id,
            visibility=payload.visibility,
        )
    except Exception as e:
        error = to_http_error(e)
        state.audit(
            "document.ingest", "failure", actor=user,
            detail=str(error.detail)[:200], request_id=_request_id(request),
        )
        raise error
    state.audit(
        "document.ingest", "success", actor=user, target=result.doc_id,
        detail=result.status, request_id=_request_id(request),
    )
    return _ingest_response(result.source, result)


# --------------------------------------------------
# Documents
# --------------------------------------------------


@router.get(
    "/documents",
    response_model=DocumentList,
    summary="List the documents visible to the caller",
)
def documents(
    user: User = Depends(require(Permission.DOCUMENTS_READ)),
    state: StateStore = Depends(get_state),
):
    """
    Readers see the indexed documents they may query. Document managers also
    see processing / failed ingestion jobs and their diagnostic reason.
    """
    try:
        get_rag()  # make sure registry and index have been reconciled
        return DocumentList(
            documents=[
                _document_out(record, user)
                for record in state.tenant_documents(user.tenant_id)
                if _visible(record, user)
            ]
        )
    except Exception as e:
        raise to_http_error(e)


def _get_visible_document(doc_id: str, user: User, state: StateStore) -> DocumentRecord:
    record = state.get_document(doc_id)
    # 404 (not 403) for documents the caller may not see: do not reveal existence.
    if record is None or not _visible(record, user):
        raise HTTPException(404, "Document not found.")
    return record


@router.get(
    "/documents/{doc_id}",
    response_model=DocumentOut,
    responses=NOT_FOUND,
    summary="Processing status of one document",
)
def document_status(
    doc_id: str,
    user: User = Depends(require(Permission.DOCUMENTS_READ)),
    state: StateStore = Depends(get_state),
):
    return _document_out(_get_visible_document(doc_id, user, state), user)


@router.patch(
    "/documents/{doc_id}",
    response_model=DocumentOut,
    responses=NOT_FOUND,
    summary="Change who may read a document",
)
def update_document(
    doc_id: str,
    payload: DocumentUpdate,
    request: Request,
    user: User = Depends(require(Permission.DOCUMENTS_WRITE)),
    state: StateStore = Depends(get_state),
):
    record = _get_visible_document(doc_id, user, state)
    if not can_manage(user, record):
        raise HTTPException(404, "Document not found.")
    if payload.visibility != record.visibility:
        state.set_visibility(doc_id, payload.visibility)
        state.audit(
            "document.visibility_change", "success", actor=user, target=doc_id,
            detail=f"{record.visibility.value} -> {payload.visibility.value}",
            request_id=_request_id(request),
        )
    return _document_out(state.get_document(doc_id), user)


@router.delete(
    "/documents/{doc_id}",
    response_model=DeleteResponse,
    responses=NOT_FOUND,
    summary="Delete a document and everything derived from it",
)
def delete_document(
    doc_id: str,
    request: Request,
    user: User = Depends(require(Permission.DOCUMENTS_DELETE)),
    state: StateStore = Depends(get_state),
):
    try:
        record = state.get_document(doc_id)
        if record is None or not can_manage(user, record):
            raise HTTPException(404, "Document not found.")
        removed = get_rag().delete_document(doc_id)
    except HTTPException:
        raise
    except Exception as e:
        state.audit(
            "document.delete", "failure", actor=user, target=doc_id,
            request_id=_request_id(request),
        )
        raise to_http_error(e)

    state.audit(
        "document.delete", "success", actor=user, target=doc_id,
        detail=f"chunks_removed={removed}", request_id=_request_id(request),
    )
    return DeleteResponse(message="Document deleted.", doc_id=doc_id, chunks_removed=removed)

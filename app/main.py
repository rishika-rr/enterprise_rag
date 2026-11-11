import logging
import os
import re
import time
import uuid
from contextlib import asynccontextmanager

from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware

from app.api.auth_routes import admin_router, auth_router
from app.api.errors import error_response, install_error_handlers
from app.api.routes import router
from app.config import settings
from app.logging_config import configure_logging, request_id_var

configure_logging()
logger = logging.getLogger("rag.http")

_SAFE_REQUEST_ID = re.compile(r"^[A-Za-z0-9_-]{1,64}$")

DESCRIPTION = """
Retrieval-augmented question answering over an access-controlled document collection.

**Authentication.** `POST /v1/auth/login` (form fields `username`, `password`) returns a
bearer token. Send it as `Authorization: Bearer <token>` on every other `/v1` request.
Tokens expire after `TOKEN_TTL_MINUTES`; deactivating a user or changing a password revokes
them immediately.

**Roles.** `reader` (ask, list documents) · `document_manager` (+ upload, delete, ingestion
status) · `administrator` (+ users, audit log, server-side ingestion).

**Errors.** Every error is `{"detail", "code", "request_id"}`; see `ErrorResponse`.

**Rate limits.** `/v1/ask`: `RATE_LIMIT_PER_MINUTE` requests per user per minute.
`/v1/auth/login`: `LOGIN_RATE_LIMIT_PER_MINUTE` attempts per client and username per minute.
Exceeding either returns `429` with a `Retry-After` header (seconds).

**Limits.** Question length, request-body size, upload size and files per batch are bounded
by configuration; oversized requests get `413` or `422`.
"""


@asynccontextmanager
async def lifespan(app: FastAPI):
    from app.database.state import get_state_store

    get_state_store()  # creates the schema and the bootstrap administrator

    # Optional: load indexes (and local models) at boot so the first query is not slow.
    if os.getenv("WARMUP_ON_START", "false").lower() == "true":
        from app.api.routes import get_rag
        from app.services.embedding_service import embedding_service

        try:
            get_rag()
            embedding_service._get_model()
            if settings.RERANKER == "cross_encoder":
                from app.services.cross_encoder_service import cross_encoder_service

                cross_encoder_service._get_model()
            logger.info("warmup_complete")
        except Exception:
            logger.exception("warmup_failed")
    yield


app = FastAPI(
    title="Enterprise RAG API",
    description=DESCRIPTION,
    version="1.1.0",
    lifespan=lifespan,
)

install_error_handlers(app)

app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.CORS_ORIGINS,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
    expose_headers=["X-Request-ID", "Retry-After"],
)


def _body_limit(path: str) -> int:
    upload = settings.MAX_UPLOAD_MB * 1024 * 1024
    if path == "/v1/upload/batch":
        return upload * settings.MAX_BATCH_FILES + 1024 * 1024
    if path == "/v1/upload":
        return upload + 1024 * 1024  # multipart framing overhead
    return settings.MAX_REQUEST_BYTES


@app.middleware("http")
async def request_context(request: Request, call_next):
    """
    Attach a non-secret request id, bound the request size and log the
    outcome (EVAL-005, NFR-MNT-002, PERF-006).
    """
    supplied = request.headers.get("X-Request-ID", "")
    request_id = supplied if _SAFE_REQUEST_ID.match(supplied) else uuid.uuid4().hex[:12]
    request.state.request_id = request_id
    token = request_id_var.set(request_id)
    start = time.perf_counter()
    status = 500
    try:
        declared = request.headers.get("content-length", "")
        if declared.isdigit() and int(declared) > _body_limit(request.url.path):
            response = error_response(request, 413, "The request body is too large.")
        else:
            response = await call_next(request)
        status = response.status_code
        response.headers["X-Request-ID"] = request_id
        response.headers["X-Content-Type-Options"] = "nosniff"
        if request.url.path.startswith("/v1/"):
            response.headers["Cache-Control"] = "no-store"
        return response
    finally:
        user = getattr(request.state, "user", None)
        logger.info(
            "request method=%s path=%s status=%s latency_ms=%.1f user=%s",
            request.method, request.url.path, status,
            (time.perf_counter() - start) * 1000, user.id if user else "-",
        )
        request_id_var.reset(token)


app.include_router(auth_router)
app.include_router(router)
app.include_router(admin_router)


@app.get("/", tags=["Service"])
async def root():
    """Basic liveness check."""
    return {"message": "Enterprise RAG API is running."}


@app.get("/health", tags=["Service"])
def health():
    """Readiness: reports state without exposing secrets or internals (NFR-MNT-003)."""
    status = {"status": "ok", "llm_configured": bool(settings.OPENROUTER_API_KEY)}
    try:
        from app.database.chroma import get_vector_store

        status["indexed_chunks"] = get_vector_store().count()
    except Exception:
        logger.exception("health_vector_store_error")
        status["status"] = "degraded"
        status["indexed_chunks"] = None
    try:
        from app.database.state import get_state_store

        get_state_store().count_users()
        status["identity_store"] = "ok"
    except Exception:
        logger.exception("health_identity_store_error")
        status["status"] = "degraded"
        status["identity_store"] = "unavailable"
    return status

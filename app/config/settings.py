"""
Runtime configuration. Every value comes from the environment (or `.env`),
so models, limits and endpoints change without editing source code
(NFR-MNT-001, NFR-PORT-002).
"""
import os

from dotenv import load_dotenv

# Tests set RAG_SKIP_DOTENV so a developer's real `.env` (and real API key)
# can never leak into the test run.
if os.getenv("RAG_SKIP_DOTENV", "").lower() not in {"1", "true"}:
    load_dotenv()


def _int(name: str, default: int, minimum: int | None = None, maximum: int | None = None) -> int:
    raw = os.getenv(name)
    try:
        value = int(raw) if raw not in (None, "") else default
    except ValueError:
        raise ValueError(f"{name} must be an integer.") from None
    if minimum is not None and value < minimum:
        raise ValueError(f"{name} must be >= {minimum}.")
    if maximum is not None and value > maximum:
        raise ValueError(f"{name} must be <= {maximum}.")
    return value


def _float(name: str, default: float, minimum: float | None = None) -> float:
    raw = os.getenv(name)
    try:
        value = float(raw) if raw not in (None, "") else default
    except ValueError:
        raise ValueError(f"{name} must be a number.") from None
    if minimum is not None and value < minimum:
        raise ValueError(f"{name} must be >= {minimum}.")
    return value


def _choice(name: str, default: str, allowed: set[str]) -> str:
    value = os.getenv(name, default).strip().lower() or default
    if value not in allowed:
        raise ValueError(f"{name} must be one of: {', '.join(sorted(allowed))}")
    return value


def _list(name: str, default: str) -> list[str]:
    return [item.strip() for item in os.getenv(name, default).split(",") if item.strip()]


# ==========================
# LLM provider (OpenAI-compatible endpoint; OpenRouter by default)
# ==========================
OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY") or None
LLM_BASE_URL = os.getenv("LLM_BASE_URL", "https://openrouter.ai/api/v1")

# Any model id the provider serves, e.g. a DeepSeek or a Gemini model.
LLM_MODEL = os.getenv("LLM_MODEL", "deepseek/deepseek-chat")

LLM_TIMEOUT_SECONDS = _float("LLM_TIMEOUT_SECONDS", 30, minimum=1)  # ERR-002
LLM_MAX_RETRIES = _int("LLM_MAX_RETRIES", 3, minimum=1, maximum=10)  # ERR-003
LLM_MAX_TOKENS = _int("LLM_MAX_TOKENS", 1024, minimum=64, maximum=8192)

# ==========================
# Storage
# ==========================
CHROMA_DB_PATH = os.getenv("CHROMA_DB_PATH", "data/chroma_db")
# Users, sessions, document registry and audit log (SQLite).
APP_DB_PATH = os.getenv("APP_DB_PATH", "data/app.db")

# ==========================
# Embeddings / reranker
# ==========================
EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "sentence-transformers/all-MiniLM-L6-v2")
CROSS_ENCODER_MODEL = os.getenv("CROSS_ENCODER_MODEL", "cross-encoder/ms-marco-MiniLM-L-6-v2")

# "cross_encoder" - local cross-encoder model
# "none"          - reranking disabled; fused order is used as-is
RERANKER = _choice("RERANKER", "cross_encoder", {"cross_encoder", "none"})

# ==========================
# Chunking
# ==========================
CHUNK_SIZE = _int("CHUNK_SIZE", 500, minimum=50, maximum=8000)
CHUNK_OVERLAP = _int("CHUNK_OVERLAP", 100, minimum=0)
if CHUNK_OVERLAP >= CHUNK_SIZE:
    raise ValueError("CHUNK_OVERLAP must be smaller than CHUNK_SIZE.")
CHUNK_STRATEGY = _choice("CHUNK_STRATEGY", "recursive", {"fixed", "recursive"})

# ==========================
# Retrieval (bounded: PERF-006)
# ==========================
TOP_K = _int("TOP_K", 10, minimum=1, maximum=100)  # candidates per retriever
RERANK_TOP_K = _int("RERANK_TOP_K", 5, minimum=1, maximum=50)  # passages sent to the LLM
# Upper bound on the retrieved text placed in one generation prompt.
MAX_CONTEXT_CHARS = _int("MAX_CONTEXT_CHARS", 12000, minimum=500)

# Minimum normalised (0-1) reranker relevance of the best passage. Below it the
# system answers "no evidence" instead of calling the LLM (RET-008).
MIN_RELEVANCE = _float("MIN_RELEVANCE", 0.01, minimum=0)

# "hybrid" (dense + BM25 + fusion), "dense" or "bm25"   (RET-004 / RET-005)
RETRIEVAL_MODE = _choice("RETRIEVAL_MODE", "hybrid", {"hybrid", "dense", "bm25"})

# ==========================
# Citation verification
# ==========================
MAX_VERIFICATIONS = _int("MAX_VERIFICATIONS", 8, minimum=0, maximum=50)
VERIFICATION_WORKERS = _int("VERIFICATION_WORKERS", 8, minimum=1, maximum=16)  # one wave

# ==========================
# API / uploads
# ==========================
UPLOAD_DIR = os.getenv("UPLOAD_DIR", "uploads")
# Server-side directories administrators may ingest from. Never include
# UPLOAD_DIR: it holds every tenant's uploaded files.
INGEST_ALLOWED_DIRS = _list("INGEST_ALLOWED_DIRS", "data/raw")
MAX_UPLOAD_MB = _int("MAX_UPLOAD_MB", 20, minimum=1)  # IF-002
MAX_BATCH_FILES = _int("MAX_BATCH_FILES", 20, minimum=1, maximum=200)
MAX_QUESTION_CHARS = _int("MAX_QUESTION_CHARS", 2000, minimum=1)  # RET-002
# Largest accepted non-upload request body (PERF-006).
MAX_REQUEST_BYTES = _int("MAX_REQUEST_BYTES", 65536, minimum=1024)

CORS_ORIGINS = _list(
    "CORS_ORIGINS", "http://localhost:5173,https://northbridge-ai-liart.vercel.app"
)

# Requests per user per minute on /v1/ask, and login attempts per client per
# minute. 0 disables the limit. Exceeding it returns 429 with Retry-After.
RATE_LIMIT_PER_MINUTE = _int("RATE_LIMIT_PER_MINUTE", 120, minimum=0)
LOGIN_RATE_LIMIT_PER_MINUTE = _int("LOGIN_RATE_LIMIT_PER_MINUTE", 10, minimum=0)

# ==========================
# Identity and access (SRS 3.2.1)
# ==========================
DEFAULT_TENANT = os.getenv("DEFAULT_TENANT", "default").strip() or "default"
# First administrator, created only while the user table is empty.
ADMIN_USERNAME = (os.getenv("ADMIN_USERNAME") or "").strip() or None
ADMIN_PASSWORD = os.getenv("ADMIN_PASSWORD") or None
TOKEN_TTL_MINUTES = _int("TOKEN_TTL_MINUTES", 480, minimum=1)
MIN_PASSWORD_LENGTH = _int("MIN_PASSWORD_LENGTH", 10, minimum=8)

# ==========================
# Retention (DATA-005, NFR-SEC-006). 0 = keep until explicitly deleted.
# ==========================
AUDIT_RETENTION_DAYS = _int("AUDIT_RETENTION_DAYS", 365, minimum=0)
DOCUMENT_RETENTION_DAYS = _int("DOCUMENT_RETENTION_DAYS", 0, minimum=0)
# Application/query logs go to stdout; the log platform enforces this period.
LOG_RETENTION_DAYS = _int("LOG_RETENTION_DAYS", 30, minimum=0)

# ==========================
# Logging
# ==========================
LOG_FORMAT = _choice("LOG_FORMAT", "text", {"text", "json"})

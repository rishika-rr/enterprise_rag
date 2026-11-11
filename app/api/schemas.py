from typing import Any, Literal

from pydantic import BaseModel, Field, field_validator

from app.config.settings import MAX_QUESTION_CHARS
from app.models.identity import DocumentStatus, Role, Visibility

# --------------------------------------------------
# Errors (ERR-001, API-002)
# --------------------------------------------------

ErrorCode = Literal[
    "validation_error",
    "payload_too_large",
    "unsupported_media_type",
    "unauthenticated",
    "forbidden",
    "not_found",
    "conflict",
    "rate_limited",
    "dependency_unavailable",
    "internal_error",
]


class ErrorResponse(BaseModel):
    """Body of every non-2xx response."""

    detail: Any = Field(description="Human-readable message, or a list of field errors for 422.")
    code: ErrorCode = Field(description="Stable machine-readable error category.")
    request_id: str | None = Field(None, description="Also sent as the X-Request-ID header.")


# --------------------------------------------------
# Ask
# --------------------------------------------------


class AskRequest(BaseModel):
    """Request model for asking questions."""

    question: str = Field(
        ...,
        min_length=1,
        max_length=MAX_QUESTION_CHARS,
        description="User question.",
    )

    @field_validator("question")
    @classmethod
    def not_blank(cls, value: str) -> str:
        value = value.strip()
        if not value:
            raise ValueError("question must not be blank")
        return value


class SourceResponse(BaseModel):
    title: str
    score: float | None = None  # cross-encoder relevance, 0-1 (None when reranking is off)
    citation: int | None = None  # matches the [n] markers in the answer
    page: int | None = None
    snippet: str | None = None
    doc_id: str | None = None


class CitationDetail(BaseModel):
    claim: str
    citation: int
    supported: bool
    reason: str


class AskResponse(BaseModel):
    answer: str
    confidence: float | None = Field(
        description=(
            "Heuristic in [0, 1]: 0.5 x mean reranker relevance of the passages used "
            "+ 0.5 x share of (claim, citation) pairs the verifier judged supported. "
            "Not a probability of correctness and not proof of verification."
        )
    )
    model: str | None
    latency_ms: float | None
    retrieved_chunks: int
    citations: int  # number of claim-citation pairs checked
    sources: list[SourceResponse]

    verified_citations: int = 0
    citation_details: list[CitationDetail] = []
    no_evidence: bool = False
    request_id: str | None = None


# --------------------------------------------------
# Documents
# --------------------------------------------------


class IngestRequest(BaseModel):
    """Request model for ingestion of a server-side file."""

    document_path: str = Field(..., min_length=1, max_length=1024)
    strategy: Literal["fixed", "recursive"] | None = None
    visibility: Visibility = Visibility.TENANT


class IngestResponse(BaseModel):
    message: str
    doc_id: str
    filename: str
    status: Literal["indexed", "duplicate"]
    chunks: int


class BatchItem(BaseModel):
    filename: str
    status: Literal["indexed", "duplicate", "failed", "rejected"]
    doc_id: str | None = None
    chunks: int = 0
    reason: str | None = None


class BatchIngestResponse(BaseModel):
    """Outcome of one multi-file ingestion operation (ING-006)."""

    received: int
    accepted: int  # passed type/size/content validation
    processed: int  # indexed or recognised as duplicate
    failed: int  # rejected, or accepted but not indexed
    results: list[BatchItem]


class DocumentOut(BaseModel):
    doc_id: str
    name: str
    chunks: int
    size: int | None
    status: DocumentStatus
    visibility: Visibility
    owner_id: str | None = None
    error: str | None = None
    created_at: str
    updated_at: str


class DocumentList(BaseModel):
    documents: list[DocumentOut]


class DocumentUpdate(BaseModel):
    visibility: Visibility


class DeleteResponse(BaseModel):
    message: str
    doc_id: str
    chunks_removed: int


# --------------------------------------------------
# Identity
# --------------------------------------------------


class UserOut(BaseModel):
    id: str
    username: str
    role: Role
    tenant_id: str
    is_active: bool
    created_at: str
    updated_at: str


class TokenResponse(BaseModel):
    access_token: str
    token_type: Literal["bearer"] = "bearer"
    expires_in: int = Field(description="Token lifetime in seconds.")
    user: UserOut


class UserCreate(BaseModel):
    username: str = Field(..., min_length=3, max_length=64, pattern=r"^[A-Za-z0-9._@-]+$")
    password: str = Field(..., min_length=1, max_length=256)
    role: Role = Role.READER


class UserUpdate(BaseModel):
    role: Role | None = None
    is_active: bool | None = None
    password: str | None = Field(None, min_length=1, max_length=256)


class PasswordChange(BaseModel):
    current_password: str = Field(..., min_length=1, max_length=256)
    new_password: str = Field(..., min_length=1, max_length=256)


class MessageResponse(BaseModel):
    message: str


class AuditEvent(BaseModel):
    id: int
    ts: str
    actor_id: str | None
    actor_name: str | None
    action: str
    target: str | None
    outcome: str
    detail: str | None
    request_id: str | None


class AuditList(BaseModel):
    events: list[AuditEvent]

"""
The error contract (ERR-001, API-002, API-005).

Every error response is  {"detail": ..., "code": ..., "request_id": ...}.

| status        | code                    | category                         |
| ------------- | ----------------------- | -------------------------------- |
| 422           | validation_error        | client validation                |
| 413           | payload_too_large       | client validation                |
| 415           | unsupported_media_type  | client validation                |
| 401           | unauthenticated         | authentication                   |
| 403           | forbidden               | authorization                    |
| 404           | not_found               | client                           |
| 409           | conflict                | client                           |
| 429           | rate_limited            | client (see Retry-After header)  |
| 502 / 503     | dependency_unavailable  | dependency failure               |
| 500           | internal_error          | internal processing error        |

Internal details are logged with the request id and never returned.
"""
import logging

from fastapi import FastAPI, HTTPException, Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse
from starlette.exceptions import HTTPException as StarletteHTTPException

from app.api.schemas import ErrorResponse
from app.services.llm_service import LLMNotConfiguredError, LLMUnavailableError

logger = logging.getLogger("rag.api")

_CODES = {
    400: "validation_error",
    401: "unauthenticated",
    403: "forbidden",
    404: "not_found",
    405: "validation_error",
    409: "conflict",
    413: "payload_too_large",
    415: "unsupported_media_type",
    422: "validation_error",
    429: "rate_limited",
    502: "dependency_unavailable",
    503: "dependency_unavailable",
}


def error_code(status: int) -> str:
    if status in _CODES:
        return _CODES[status]
    return "internal_error" if status >= 500 else "validation_error"


def error_response(
    request: Request, status: int, detail, headers: dict | None = None
) -> JSONResponse:
    request_id = getattr(request.state, "request_id", None)
    response = JSONResponse(
        status_code=status,
        content={"detail": detail, "code": error_code(status), "request_id": request_id},
        headers=headers,
    )
    if request_id:
        response.headers["X-Request-ID"] = request_id
    return response


def to_http_error(e: Exception) -> HTTPException:
    """Map internal exceptions to the contract. Details are logged, never returned."""
    if isinstance(e, HTTPException):
        return e
    if isinstance(e, LLMNotConfiguredError):
        logger.error("llm_not_configured")
        return HTTPException(503, "The language model is not configured.")
    if isinstance(e, LLMUnavailableError):
        logger.warning("dependency_failure dependency=llm")
        return HTTPException(502, "The language-model provider is unavailable. Please try again.")
    if isinstance(e, FileNotFoundError):
        return HTTPException(404, "File not found.")
    if isinstance(e, ValueError):
        return HTTPException(422, str(e))
    logger.exception("unhandled_error")
    return HTTPException(500, "Internal server error.")


def install_error_handlers(app: FastAPI) -> None:
    @app.exception_handler(StarletteHTTPException)
    async def http_error(request: Request, exc: StarletteHTTPException):
        return error_response(request, exc.status_code, exc.detail, getattr(exc, "headers", None))

    @app.exception_handler(RequestValidationError)
    async def validation_error(request: Request, exc: RequestValidationError):
        # Never echo the submitted values back: they can contain passwords.
        detail = [
            {"type": err.get("type"), "loc": list(err.get("loc", ())), "msg": err.get("msg")}
            for err in exc.errors()
        ]
        return error_response(request, 422, detail)

    @app.exception_handler(Exception)
    async def unhandled(request: Request, exc: Exception):
        logger.error("unhandled_error type=%s", type(exc).__name__, exc_info=exc)
        return error_response(request, 500, "Internal server error.")


def _doc(description: str) -> dict:
    return {"model": ErrorResponse, "description": description}


# Attached to the routers so the error contract appears in the OpenAPI document.
AUTH_ERRORS = {
    401: _doc("Missing, invalid or expired access token."),
    403: _doc("The caller lacks the required permission."),
    422: _doc("The request failed validation."),
    500: _doc("Internal processing error. Details are only in the server logs."),
}
RATE_LIMITED = {429: _doc("Rate limit exceeded. Retry after the seconds in `Retry-After`.")}
DEPENDENCY_ERRORS = {
    502: _doc("The language-model provider failed or timed out."),
    503: _doc("A required dependency is not configured or unavailable."),
}
UPLOAD_ERRORS = {
    413: _doc("The file or request exceeds the configured size limit."),
    415: _doc("The file type is not supported."),
}
NOT_FOUND = {404: _doc("The resource does not exist or is not visible to the caller.")}

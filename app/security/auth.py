"""
Request authentication and authorization.

Every protected endpoint depends on `require(<permission>)`. Access is denied
unless a valid bearer token resolves to an active user holding the permission
(SEC-001, SEC-003). Anything missing, invalid or unavailable denies (SEC-005).
The user record is re-read on every request, so revocations and role changes
apply to the next request - the propagation interval is zero (SEC-006).
"""
import logging

from fastapi import Depends, HTTPException, Request
from fastapi.security import OAuth2PasswordBearer

from app.database.state import StateStore, get_state_store
from app.models.identity import Permission, User

logger = logging.getLogger("rag.auth")

oauth2_scheme = OAuth2PasswordBearer(tokenUrl="/v1/auth/login", auto_error=False)

_UNAUTHENTICATED = {"WWW-Authenticate": "Bearer"}


def get_state() -> StateStore:
    try:
        return get_state_store()
    except Exception:
        logger.exception("identity_store_unavailable")
        raise HTTPException(503, "The identity service is unavailable.") from None


def get_current_user(
    request: Request,
    token: str | None = Depends(oauth2_scheme),
    state: StateStore = Depends(get_state),
) -> User:
    if not token:
        raise HTTPException(401, "Authentication is required.", headers=_UNAUTHENTICATED)
    try:
        user = state.user_for_token(token)
    except Exception:
        logger.exception("identity_lookup_failed")
        raise HTTPException(503, "The identity service is unavailable.") from None
    if user is None:
        raise HTTPException(
            401, "The access token is invalid or has expired.", headers=_UNAUTHENTICATED
        )
    request.state.user = user
    return user


def require(permission: Permission):
    """Dependency factory: the caller must hold `permission`."""

    def dependency(
        request: Request,
        user: User = Depends(get_current_user),
        state: StateStore = Depends(get_state),
    ) -> User:
        if not user.has(permission):
            state.audit(
                "access.denied", "denied", actor=user,
                target=f"{request.method} {request.url.path}",
                detail=f"missing permission {permission.value}",
                request_id=getattr(request.state, "request_id", None),
            )
            raise HTTPException(403, "You do not have permission to perform this action.")
        return user

    return dependency

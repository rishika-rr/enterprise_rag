from fastapi import APIRouter, Depends, HTTPException, Request
from fastapi.security import OAuth2PasswordRequestForm

from app.api.errors import AUTH_ERRORS, NOT_FOUND, RATE_LIMITED
from app.api.schemas import (
    AuditEvent,
    AuditList,
    MessageResponse,
    PasswordChange,
    TokenResponse,
    UserCreate,
    UserOut,
    UserUpdate,
)
from app.config import settings
from app.database.state import StateStore
from app.models.identity import Permission, Role, User
from app.security.auth import get_current_user, get_state, oauth2_scheme, require
from app.security.rate_limit import login_limiter

auth_router = APIRouter(prefix="/v1/auth", tags=["Authentication"], responses=AUTH_ERRORS)
admin_router = APIRouter(prefix="/v1/admin", tags=["Administration"], responses=AUTH_ERRORS)


def _request_id(request: Request) -> str | None:
    return getattr(request.state, "request_id", None)


def _user_out(user: User) -> UserOut:
    return UserOut(
        id=user.id, username=user.username, role=user.role, tenant_id=user.tenant_id,
        is_active=user.is_active, created_at=user.created_at, updated_at=user.updated_at,
    )


# --------------------------------------------------
# Authentication
# --------------------------------------------------


@auth_router.post(
    "/login",
    response_model=TokenResponse,
    responses=RATE_LIMITED,
    summary="Exchange username and password for a bearer token",
)
def login(
    request: Request,
    form: OAuth2PasswordRequestForm = Depends(),
    state: StateStore = Depends(get_state),
):
    """
    Form-encoded `username` and `password`. Send the returned token as
    `Authorization: Bearer <access_token>` on every other request.
    """
    client = request.client.host if request.client else "unknown"
    retry_after = login_limiter.check(
        f"{client}|{form.username.strip().lower()}", settings.LOGIN_RATE_LIMIT_PER_MINUTE
    )
    if retry_after:
        raise HTTPException(
            429, "Too many login attempts. Please retry later.",
            headers={"Retry-After": str(retry_after)},
        )

    user = state.authenticate(form.username, form.password)
    if user is None:
        state.audit(
            "auth.login", "failure", actor_name=form.username.strip()[:64],
            tenant_id=settings.DEFAULT_TENANT, request_id=_request_id(request),
        )
        # One message for unknown user, wrong password and disabled account.
        raise HTTPException(
            401, "Incorrect username or password.", headers={"WWW-Authenticate": "Bearer"}
        )

    token, expires_in = state.create_session(user.id)
    state.audit("auth.login", "success", actor=user, request_id=_request_id(request))
    return TokenResponse(access_token=token, expires_in=expires_in, user=_user_out(user))


@auth_router.post("/logout", response_model=MessageResponse, summary="Revoke the current token")
def logout(
    token: str | None = Depends(oauth2_scheme),
    user: User = Depends(get_current_user),
    state: StateStore = Depends(get_state),
):
    state.revoke_token(token or "")
    return MessageResponse(message="Signed out.")


@auth_router.get("/me", response_model=UserOut, summary="The authenticated user")
def me(user: User = Depends(get_current_user)):
    return _user_out(user)


@auth_router.post(
    "/password",
    response_model=MessageResponse,
    summary="Change your own password (revokes all your tokens)",
)
def change_password(
    payload: PasswordChange,
    request: Request,
    user: User = Depends(get_current_user),
    state: StateStore = Depends(get_state),
):
    if state.authenticate(user.username, payload.current_password) is None:
        state.audit("auth.password_change", "failure", actor=user, request_id=_request_id(request))
        raise HTTPException(403, "The current password is incorrect.")
    try:
        state.update_user(user.id, password=payload.new_password)
    except ValueError as e:
        raise HTTPException(422, str(e)) from None
    state.audit("auth.password_change", "success", actor=user, request_id=_request_id(request))
    return MessageResponse(message="Password changed. Please sign in again.")


# --------------------------------------------------
# Administration (users of the administrator's own tenant)
# --------------------------------------------------


@admin_router.get("/users", response_model=list[UserOut], summary="List users")
def list_users(
    admin: User = Depends(require(Permission.USERS_MANAGE)),
    state: StateStore = Depends(get_state),
):
    return [_user_out(u) for u in state.list_users(admin.tenant_id)]


@admin_router.post(
    "/users", response_model=UserOut, status_code=201, summary="Create a user in your tenant"
)
def create_user(
    payload: UserCreate,
    request: Request,
    admin: User = Depends(require(Permission.USERS_MANAGE)),
    state: StateStore = Depends(get_state),
):
    try:
        user = state.create_user(payload.username, payload.password, payload.role, admin.tenant_id)
    except ValueError as e:
        state.audit(
            "user.create", "failure", actor=admin, detail=str(e), request_id=_request_id(request)
        )
        status = 409 if "already exists" in str(e) else 422
        raise HTTPException(status, str(e)) from None
    state.audit(
        "user.create", "success", actor=admin, target=user.id,
        detail=f"role={user.role.value}", request_id=_request_id(request),
    )
    return _user_out(user)


@admin_router.patch(
    "/users/{user_id}",
    response_model=UserOut,
    responses=NOT_FOUND,
    summary="Change a user's role, reset the password, or deactivate / reactivate the account",
)
def update_user(
    user_id: str,
    payload: UserUpdate,
    request: Request,
    admin: User = Depends(require(Permission.USERS_MANAGE)),
    state: StateStore = Depends(get_state),
):
    """Changes apply to the user's next request; deactivation also revokes their tokens."""
    target = state.get_user(user_id)
    if target is None or target.tenant_id != admin.tenant_id:
        raise HTTPException(404, "User not found.")

    loses_admin = target.role == Role.ADMINISTRATOR and target.is_active and (
        payload.is_active is False
        or (payload.role is not None and payload.role != Role.ADMINISTRATOR)
    )
    if loses_admin and state.count_active_admins(admin.tenant_id) <= 1:
        state.audit(
            "user.update", "denied", actor=admin, target=user_id,
            detail="would remove the last administrator", request_id=_request_id(request),
        )
        raise HTTPException(409, "The last active administrator cannot be removed.")

    try:
        updated = state.update_user(
            user_id, role=payload.role, is_active=payload.is_active, password=payload.password
        )
    except ValueError as e:
        raise HTTPException(422, str(e)) from None

    changes = []
    if payload.role is not None and payload.role != target.role:
        changes.append(f"role {target.role.value} -> {payload.role.value}")
    if payload.is_active is not None and payload.is_active != target.is_active:
        changes.append("reactivated" if payload.is_active else "deactivated")
    if payload.password is not None:
        changes.append("password reset")
    state.audit(
        "user.update", "success", actor=admin, target=user_id,
        detail="; ".join(changes) or "no change", request_id=_request_id(request),
    )
    return _user_out(updated)


@admin_router.get("/audit", response_model=AuditList, summary="Audit events of your tenant")
def audit_events(
    limit: int = 100,
    offset: int = 0,
    action: str | None = None,
    admin: User = Depends(require(Permission.AUDIT_READ)),
    state: StateStore = Depends(get_state),
):
    limit = min(max(limit, 1), 500)
    rows = state.list_audit(admin.tenant_id, limit=limit, offset=max(offset, 0), action=action)
    return AuditList(events=[AuditEvent(**{k: row[k] for k in AuditEvent.model_fields}) for row in rows])

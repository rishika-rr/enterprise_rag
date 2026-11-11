"""
Persistent application state in SQLite: users, sessions, the document
registry and the audit log.

The document registry is the single source of truth for who may read which
document; retrieval filters are derived from it on every request, so a
permission change applies to the very next request (SEC-006).
"""
import hashlib
import logging
import secrets
import sqlite3
import threading
import uuid
from datetime import datetime, timedelta, timezone
from pathlib import Path

from app.config import settings
from app.models.identity import (
    DocumentRecord,
    DocumentStatus,
    Role,
    User,
    Visibility,
    can_read,
)
from app.security.passwords import DUMMY_HASH, hash_password, verify_password

logger = logging.getLogger("rag.state")

_SCHEMA = """
CREATE TABLE IF NOT EXISTS users (
    id            TEXT PRIMARY KEY,
    username      TEXT NOT NULL UNIQUE COLLATE NOCASE,
    password_hash TEXT NOT NULL,
    role          TEXT NOT NULL,
    tenant_id     TEXT NOT NULL,
    is_active     INTEGER NOT NULL DEFAULT 1,
    created_at    TEXT NOT NULL,
    updated_at    TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS sessions (
    token_hash TEXT PRIMARY KEY,
    user_id    TEXT NOT NULL REFERENCES users(id) ON DELETE CASCADE,
    created_at TEXT NOT NULL,
    expires_at TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS documents (
    doc_id      TEXT PRIMARY KEY,
    tenant_id   TEXT NOT NULL,
    owner_id    TEXT,
    name        TEXT NOT NULL,
    visibility  TEXT NOT NULL,
    status      TEXT NOT NULL,
    chunks      INTEGER NOT NULL DEFAULT 0,
    size_bytes  INTEGER,
    stored_path TEXT,
    error       TEXT,
    created_at  TEXT NOT NULL,
    updated_at  TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_documents_tenant ON documents(tenant_id);
CREATE TABLE IF NOT EXISTS audit_events (
    id         INTEGER PRIMARY KEY AUTOINCREMENT,
    ts         TEXT NOT NULL,
    actor_id   TEXT,
    actor_name TEXT,
    tenant_id  TEXT,
    action     TEXT NOT NULL,
    target     TEXT,
    outcome    TEXT NOT NULL,
    detail     TEXT,
    request_id TEXT
);
CREATE INDEX IF NOT EXISTS idx_audit_tenant_ts ON audit_events(tenant_id, ts);
"""


def utcnow() -> datetime:
    return datetime.now(timezone.utc)


def _iso(moment: datetime) -> str:
    return moment.isoformat(timespec="seconds")


def _hash_token(token: str) -> str:
    return hashlib.sha256(token.encode("utf-8")).hexdigest()


def validate_password(password: str) -> None:
    if len(password) < settings.MIN_PASSWORD_LENGTH:
        raise ValueError(
            f"Password must be at least {settings.MIN_PASSWORD_LENGTH} characters long."
        )
    if len(password) > 256:
        raise ValueError("Password must be at most 256 characters long.")


class StateStore:
    def __init__(self, path: str):
        if path != ":memory:":
            Path(path).parent.mkdir(parents=True, exist_ok=True)
        self._lock = threading.RLock()
        self._db = sqlite3.connect(path, check_same_thread=False, isolation_level=None)
        self._db.row_factory = sqlite3.Row
        with self._lock:
            self._db.execute("PRAGMA foreign_keys = ON")
            if path != ":memory:":
                self._db.execute("PRAGMA journal_mode = WAL")
            self._db.executescript(_SCHEMA)

    def _execute(self, sql: str, params: tuple = ()) -> sqlite3.Cursor:
        with self._lock:
            return self._db.execute(sql, params)

    def _all(self, sql: str, params: tuple = ()) -> list[sqlite3.Row]:
        with self._lock:
            return self._db.execute(sql, params).fetchall()

    def _one(self, sql: str, params: tuple = ()) -> sqlite3.Row | None:
        with self._lock:
            return self._db.execute(sql, params).fetchone()

    def close(self) -> None:
        with self._lock:
            self._db.close()

    # ------------------------------------------------------------------
    # Users
    # ------------------------------------------------------------------

    @staticmethod
    def _user(row: sqlite3.Row) -> User:
        return User(
            id=row["id"],
            username=row["username"],
            role=Role(row["role"]),
            tenant_id=row["tenant_id"],
            is_active=bool(row["is_active"]),
            created_at=row["created_at"],
            updated_at=row["updated_at"],
        )

    def count_users(self) -> int:
        return self._one("SELECT COUNT(*) AS n FROM users")["n"]

    def create_user(self, username: str, password: str, role: Role, tenant_id: str) -> User:
        username = username.strip()
        if not username:
            raise ValueError("Username must not be blank.")
        validate_password(password)
        now = _iso(utcnow())
        user_id = uuid.uuid4().hex
        try:
            self._execute(
                "INSERT INTO users (id, username, password_hash, role, tenant_id, is_active,"
                " created_at, updated_at) VALUES (?, ?, ?, ?, ?, 1, ?, ?)",
                (user_id, username, hash_password(password), Role(role).value, tenant_id, now, now),
            )
        except sqlite3.IntegrityError:
            raise ValueError("A user with this username already exists.") from None
        return self.get_user(user_id)

    def get_user(self, user_id: str) -> User | None:
        row = self._one("SELECT * FROM users WHERE id = ?", (user_id,))
        return self._user(row) if row else None

    def get_user_by_username(self, username: str) -> User | None:
        row = self._one("SELECT * FROM users WHERE username = ?", (username.strip(),))
        return self._user(row) if row else None

    def list_users(self, tenant_id: str) -> list[User]:
        rows = self._all(
            "SELECT * FROM users WHERE tenant_id = ? ORDER BY username COLLATE NOCASE", (tenant_id,)
        )
        return [self._user(r) for r in rows]

    def count_active_admins(self, tenant_id: str) -> int:
        return self._one(
            "SELECT COUNT(*) AS n FROM users WHERE tenant_id = ? AND role = ? AND is_active = 1",
            (tenant_id, Role.ADMINISTRATOR.value),
        )["n"]

    def update_user(
        self,
        user_id: str,
        role: Role | None = None,
        is_active: bool | None = None,
        password: str | None = None,
    ) -> User | None:
        """Apply the given changes. Deactivation and password change revoke all sessions."""
        if password is not None:
            validate_password(password)
        with self._lock:
            if self.get_user(user_id) is None:
                return None
            now = _iso(utcnow())
            if role is not None:
                self._execute(
                    "UPDATE users SET role = ?, updated_at = ? WHERE id = ?",
                    (Role(role).value, now, user_id),
                )
            if is_active is not None:
                self._execute(
                    "UPDATE users SET is_active = ?, updated_at = ? WHERE id = ?",
                    (1 if is_active else 0, now, user_id),
                )
            if password is not None:
                self._execute(
                    "UPDATE users SET password_hash = ?, updated_at = ? WHERE id = ?",
                    (hash_password(password), now, user_id),
                )
            if is_active is False or password is not None:
                self.revoke_sessions(user_id)
            return self.get_user(user_id)

    # ------------------------------------------------------------------
    # Authentication / sessions
    # ------------------------------------------------------------------

    def authenticate(self, username: str, password: str) -> User | None:
        row = self._one("SELECT * FROM users WHERE username = ?", (username.strip(),))
        if row is None:
            verify_password(password, DUMMY_HASH)  # constant-ish time for unknown users
            return None
        if not verify_password(password, row["password_hash"]) or not row["is_active"]:
            return None
        return self._user(row)

    def create_session(self, user_id: str) -> tuple[str, int]:
        """Returns (opaque bearer token, lifetime in seconds). Only its hash is stored."""
        token = secrets.token_urlsafe(32)
        now = utcnow()
        ttl = timedelta(minutes=settings.TOKEN_TTL_MINUTES)
        self._execute("DELETE FROM sessions WHERE expires_at < ?", (_iso(now),))
        self._execute(
            "INSERT INTO sessions (token_hash, user_id, created_at, expires_at) VALUES (?, ?, ?, ?)",
            (_hash_token(token), user_id, _iso(now), _iso(now + ttl)),
        )
        return token, int(ttl.total_seconds())

    def user_for_token(self, token: str) -> User | None:
        """Resolves a token to its *current* user record; None unless valid and active."""
        if not token:
            return None
        row = self._one(
            "SELECT u.* FROM sessions s JOIN users u ON u.id = s.user_id"
            " WHERE s.token_hash = ? AND s.expires_at > ?",
            (_hash_token(token), _iso(utcnow())),
        )
        if row is None or not row["is_active"]:
            return None
        return self._user(row)

    def revoke_token(self, token: str) -> None:
        self._execute("DELETE FROM sessions WHERE token_hash = ?", (_hash_token(token),))

    def revoke_sessions(self, user_id: str) -> None:
        self._execute("DELETE FROM sessions WHERE user_id = ?", (user_id,))

    # ------------------------------------------------------------------
    # Document registry
    # ------------------------------------------------------------------

    @staticmethod
    def _document(row: sqlite3.Row) -> DocumentRecord:
        return DocumentRecord(
            doc_id=row["doc_id"],
            tenant_id=row["tenant_id"],
            owner_id=row["owner_id"],
            name=row["name"],
            visibility=Visibility(row["visibility"]),
            status=DocumentStatus(row["status"]),
            chunks=row["chunks"],
            size_bytes=row["size_bytes"],
            stored_path=row["stored_path"],
            error=row["error"],
            created_at=row["created_at"],
            updated_at=row["updated_at"],
        )

    def get_document(self, doc_id: str) -> DocumentRecord | None:
        row = self._one("SELECT * FROM documents WHERE doc_id = ?", (doc_id,))
        return self._document(row) if row else None

    def begin_document(
        self,
        doc_id: str,
        tenant_id: str,
        owner_id: str | None,
        name: str,
        visibility: Visibility,
        size_bytes: int | None,
        stored_path: str | None,
    ) -> None:
        """Create (or reset) the record of an ingestion job as `processing`."""
        now = _iso(utcnow())
        self._execute(
            "INSERT INTO documents (doc_id, tenant_id, owner_id, name, visibility, status, chunks,"
            " size_bytes, stored_path, error, created_at, updated_at)"
            " VALUES (?, ?, ?, ?, ?, ?, 0, ?, ?, NULL, ?, ?)"
            " ON CONFLICT(doc_id) DO UPDATE SET owner_id = excluded.owner_id,"
            " visibility = excluded.visibility, name = excluded.name, status = excluded.status,"
            " chunks = 0, size_bytes = excluded.size_bytes, stored_path = excluded.stored_path,"
            " error = NULL, updated_at = excluded.updated_at",
            (doc_id, tenant_id, owner_id, name, Visibility(visibility).value,
             DocumentStatus.PROCESSING.value, size_bytes, stored_path, now, now),
        )

    def mark_indexed(self, doc_id: str, chunks: int) -> None:
        self._execute(
            "UPDATE documents SET status = ?, chunks = ?, error = NULL, updated_at = ?"
            " WHERE doc_id = ?",
            (DocumentStatus.INDEXED.value, chunks, _iso(utcnow()), doc_id),
        )

    def mark_failed(self, doc_id: str, reason: str) -> None:
        self._execute(
            "UPDATE documents SET status = ?, chunks = 0, error = ?, stored_path = NULL,"
            " updated_at = ? WHERE doc_id = ?",
            (DocumentStatus.FAILED.value, reason[:500], _iso(utcnow()), doc_id),
        )

    def set_visibility(self, doc_id: str, visibility: Visibility) -> None:
        self._execute(
            "UPDATE documents SET visibility = ?, updated_at = ? WHERE doc_id = ?",
            (Visibility(visibility).value, _iso(utcnow()), doc_id),
        )

    def delete_document(self, doc_id: str) -> None:
        self._execute("DELETE FROM documents WHERE doc_id = ?", (doc_id,))

    def all_documents(self) -> list[DocumentRecord]:
        return [self._document(r) for r in self._all("SELECT * FROM documents")]

    def tenant_documents(self, tenant_id: str) -> list[DocumentRecord]:
        rows = self._all(
            "SELECT * FROM documents WHERE tenant_id = ? ORDER BY name COLLATE NOCASE, doc_id",
            (tenant_id,),
        )
        return [self._document(r) for r in rows]

    def readable_doc_ids(self, user: User) -> list[str]:
        """Ids of the indexed documents `user` may retrieve from. Empty = nothing (fail closed)."""
        return [
            doc.doc_id
            for doc in self.tenant_documents(user.tenant_id)
            if doc.status == DocumentStatus.INDEXED and can_read(user, doc)
        ]

    def documents_older_than(self, days: int) -> list[DocumentRecord]:
        cutoff = _iso(utcnow() - timedelta(days=days))
        rows = self._all("SELECT * FROM documents WHERE created_at < ?", (cutoff,))
        return [self._document(r) for r in rows]

    # ------------------------------------------------------------------
    # Audit log (NFR-SEC-005)
    # ------------------------------------------------------------------

    def audit(
        self,
        action: str,
        outcome: str,
        actor: User | None = None,
        target: str | None = None,
        detail: str | None = None,
        tenant_id: str | None = None,
        actor_name: str | None = None,
        request_id: str | None = None,
    ) -> None:
        """Never pass secrets, tokens or document content in `detail`."""
        self._execute(
            "INSERT INTO audit_events (ts, actor_id, actor_name, tenant_id, action, target,"
            " outcome, detail, request_id) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)",
            (
                _iso(utcnow()),
                actor.id if actor else None,
                actor.username if actor else actor_name,
                actor.tenant_id if actor else tenant_id,
                action,
                target,
                outcome,
                detail[:500] if detail else None,
                request_id,
            ),
        )
        logger.info(
            "audit action=%s outcome=%s actor=%s target=%s",
            action, outcome, actor.id if actor else "-", target or "-",
        )

    def list_audit(
        self, tenant_id: str, limit: int = 100, offset: int = 0, action: str | None = None
    ) -> list[dict]:
        sql = "SELECT * FROM audit_events WHERE tenant_id = ?"
        params: list = [tenant_id]
        if action:
            sql += " AND action = ?"
            params.append(action)
        sql += " ORDER BY id DESC LIMIT ? OFFSET ?"
        params += [limit, offset]
        return [dict(r) for r in self._all(sql, tuple(params))]

    def purge_audit_older_than(self, days: int) -> int:
        cutoff = _iso(utcnow() - timedelta(days=days))
        return self._execute("DELETE FROM audit_events WHERE ts < ?", (cutoff,)).rowcount


# ----------------------------------------------------------------------

_state: StateStore | None = None
_state_lock = threading.Lock()


def _bootstrap_admin(store: StateStore) -> None:
    if store.count_users() > 0:
        return
    if not (settings.ADMIN_USERNAME and settings.ADMIN_PASSWORD):
        logger.warning(
            "no_users_configured: set ADMIN_USERNAME and ADMIN_PASSWORD (or run "
            "scripts/create_user.py) - every protected endpoint denies access until then"
        )
        return
    try:
        user = store.create_user(
            settings.ADMIN_USERNAME, settings.ADMIN_PASSWORD,
            Role.ADMINISTRATOR, settings.DEFAULT_TENANT,
        )
    except ValueError as e:
        # Never start half-configured with an account nobody asked for.
        raise RuntimeError(f"Cannot create the bootstrap administrator: {e}") from None
    store.audit("user.bootstrap", "success", actor=user, target=user.id)
    logger.info("bootstrap_admin_created user_id=%s", user.id)


def get_state_store() -> StateStore:
    """Process-wide shared store."""
    global _state
    if _state is None:
        with _state_lock:
            if _state is None:
                store = StateStore(settings.APP_DB_PATH)
                _bootstrap_admin(store)
                _state = store
    return _state


def reset_state_store() -> None:
    """Close and forget the shared store (tests / reconfiguration)."""
    global _state
    with _state_lock:
        if _state is not None:
            _state.close()
            _state = None

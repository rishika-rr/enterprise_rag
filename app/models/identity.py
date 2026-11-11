from dataclasses import dataclass
from enum import Enum


class Role(str, Enum):
    READER = "reader"
    DOCUMENT_MANAGER = "document_manager"
    ADMINISTRATOR = "administrator"


class Permission(str, Enum):
    QUERY = "query"
    DOCUMENTS_READ = "documents:read"
    DOCUMENTS_WRITE = "documents:write"  # upload / ingest / change visibility
    DOCUMENTS_DELETE = "documents:delete"
    SERVER_INGEST = "documents:ingest_path"  # index a file already on the server
    USERS_MANAGE = "users:manage"
    AUDIT_READ = "audit:read"


# Least privilege (NFR-SEC-003): each role gets only what its job needs.
ROLE_PERMISSIONS: dict[Role, frozenset[Permission]] = {
    Role.READER: frozenset({Permission.QUERY, Permission.DOCUMENTS_READ}),
    Role.DOCUMENT_MANAGER: frozenset(
        {
            Permission.QUERY,
            Permission.DOCUMENTS_READ,
            Permission.DOCUMENTS_WRITE,
            Permission.DOCUMENTS_DELETE,
        }
    ),
    Role.ADMINISTRATOR: frozenset(Permission),
}


@dataclass(frozen=True)
class User:
    id: str
    username: str
    role: Role
    tenant_id: str
    is_active: bool
    created_at: str
    updated_at: str

    def has(self, permission: Permission) -> bool:
        return self.is_active and permission in ROLE_PERMISSIONS.get(self.role, frozenset())


class Visibility(str, Enum):
    TENANT = "tenant"  # every user of the owning tenant
    PRIVATE = "private"  # the owner only


class DocumentStatus(str, Enum):
    PROCESSING = "processing"
    INDEXED = "indexed"
    FAILED = "failed"


@dataclass(frozen=True)
class DocumentRecord:
    doc_id: str
    tenant_id: str
    owner_id: str | None
    name: str
    visibility: Visibility
    status: DocumentStatus
    chunks: int
    size_bytes: int | None
    stored_path: str | None
    error: str | None
    created_at: str
    updated_at: str


def can_read(user: User, doc: DocumentRecord) -> bool:
    """Document-level read rule used for listing AND retrieval (SEC-004)."""
    if not user.is_active or doc.tenant_id != user.tenant_id:
        return False
    return doc.visibility == Visibility.TENANT or doc.owner_id == user.id


def can_manage(user: User, doc: DocumentRecord) -> bool:
    """Who may delete a document or change its visibility."""
    if not user.is_active or doc.tenant_id != user.tenant_id:
        return False
    if user.role == Role.ADMINISTRATOR:
        return True
    return user.has(Permission.DOCUMENTS_WRITE) and can_read(user, doc)

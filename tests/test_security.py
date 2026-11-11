"""Identity, roles, document-level authorization, audit (SRS 3.2.1, 3.6.1)."""
import pytest

from tests.conftest import ADMIN, PASSWORD, POLICY, login

HR = b"Employees receive 24 days of paid leave per year."
FIN = b"Expense claims must be submitted within 30 days of travel."


def upload(client, name, content, visibility="tenant"):
    r = client.post(
        "/v1/upload", files={"file": (name, content, "text/plain")}, data={"visibility": visibility}
    )
    assert r.status_code == 200, r.text
    return r.json()


def listed(client):
    return {d["name"] for d in client.get("/v1/documents").json()["documents"]}


# ---------- authentication (SEC-001, SEC-005, API-004) ----------

PROTECTED = [
    ("post", "/v1/ask", {"json": {"question": "hi"}}),
    ("post", "/v1/upload", {"files": {"file": ("a.txt", b"x", "text/plain")}}),
    ("post", "/v1/upload/batch", {"files": {"files": ("a.txt", b"x", "text/plain")}}),
    ("post", "/v1/ingest", {"json": {"document_path": "x.txt"}}),
    ("get", "/v1/documents", {}),
    ("get", "/v1/documents/abc", {}),
    ("patch", "/v1/documents/abc", {"json": {"visibility": "private"}}),
    ("delete", "/v1/documents/abc", {}),
    ("get", "/v1/auth/me", {}),
    ("post", "/v1/auth/logout", {}),
    ("post", "/v1/auth/password", {"json": {"current_password": "a", "new_password": "b"}}),
    ("get", "/v1/admin/users", {}),
    ("post", "/v1/admin/users", {"json": {"username": "abc", "password": "x"}}),
    ("patch", "/v1/admin/users/abc", {"json": {"is_active": False}}),
    ("get", "/v1/admin/audit", {}),
]


@pytest.mark.parametrize("method,path,kwargs", PROTECTED)
def test_every_protected_endpoint_denies_without_or_with_bad_token(anonymous, method, path, kwargs):
    r = getattr(anonymous, method)(path, **kwargs)
    assert r.status_code == 401, (path, r.text)
    assert r.json()["code"] == "unauthenticated"
    r = getattr(anonymous, method)(path, headers={"Authorization": "Bearer not-a-token"}, **kwargs)
    assert r.status_code == 401, (path, r.text)


def test_login_failures_are_uniform_and_do_not_echo_secrets(anonymous):
    wrong = anonymous.post("/v1/auth/login", data={"username": ADMIN[0], "password": "nope-nope-nope"})
    unknown = anonymous.post("/v1/auth/login", data={"username": "ghost", "password": "nope-nope-nope"})
    assert wrong.status_code == unknown.status_code == 401
    assert wrong.json()["detail"] == unknown.json()["detail"]
    # validation errors never echo submitted values
    r = anonymous.post("/v1/auth/login", data={"password": "super-secret-value"})
    assert r.status_code == 422 and "super-secret-value" not in r.text


def test_passwords_and_tokens_are_not_stored_in_plaintext(env, client):
    import sqlite3

    token = client.headers["Authorization"].split()[1]
    raw = sqlite3.connect(env.tmp / "state" / "app.db")
    dump = repr(raw.execute("SELECT * FROM users").fetchall()) + repr(
        raw.execute("SELECT * FROM sessions").fetchall()
    )
    raw.close()
    assert ADMIN[1] not in dump and token not in dump
    assert "scrypt$" in dump


def test_expired_token_is_rejected(env, client):
    env.store._execute("UPDATE sessions SET expires_at = '2000-01-01T00:00:00+00:00'")
    assert client.get("/v1/auth/me").status_code == 401


def test_logout_revokes_the_token(client):
    assert client.get("/v1/auth/me").status_code == 200
    assert client.post("/v1/auth/logout").status_code == 200
    assert client.get("/v1/auth/me").status_code == 401


def test_login_is_rate_limited(env, anonymous, monkeypatch):
    from app.config import settings

    monkeypatch.setattr(settings, "LOGIN_RATE_LIMIT_PER_MINUTE", 3)
    codes = [
        anonymous.post("/v1/auth/login", data={"username": "ghost", "password": "x"}).status_code
        for _ in range(4)
    ]
    assert codes == [401, 401, 401, 429]


# ---------- role matrix (SEC-002, SEC-003, UI-004) ----------

def test_role_matrix(client, user_client, policy_file):
    reader = user_client("rita", "reader")
    manager = user_client("mona", "document_manager")
    doc = upload(client, "policy.txt", POLICY.encode())

    def allowed(who, method, path, **kw):
        return getattr(who, method)(path, **kw).status_code not in (401, 403)

    file = {"files": {"file": ("n.txt", b"new note text", "text/plain")}}
    ingest = {"json": {"document_path": str(policy_file)}}
    matrix = {
        # (method, path, kwargs): (reader, manager, admin)
        ("post", "/v1/ask", "ask"): (True, True, True),
        ("get", "/v1/documents", "list"): (True, True, True),
        ("post", "/v1/upload", "upload"): (False, True, True),
        ("post", "/v1/ingest", "ingest"): (False, False, True),
        ("get", "/v1/admin/users", "users"): (False, False, True),
        ("get", "/v1/admin/audit", "audit"): (False, False, True),
        ("delete", f"/v1/documents/{doc['doc_id']}", "delete"): (False, True, True),
    }
    kwargs = {"ask": {"json": {"question": "paid leave days"}}, "upload": file, "ingest": ingest}
    for (method, path, name), expected in matrix.items():
        for who, expect in zip((reader, manager), expected):
            assert allowed(who, method, path, **kwargs.get(name, {})) is expect, (name, expect)
    assert client.get("/v1/admin/users").status_code == 200

    denied = reader.get("/v1/admin/users")
    assert denied.status_code == 403 and denied.json()["code"] == "forbidden"


def test_reader_sees_no_ingestion_diagnostics(client, user_client, env):
    reader = user_client("rita", "reader")
    upload(client, "ok.txt", HR)
    bad = client.post("/v1/upload", files={"file": ("blank.txt", b"   \n ", "text/plain")})
    assert bad.status_code == 422

    admin_view = {d["name"]: d for d in client.get("/v1/documents").json()["documents"]}
    assert admin_view["blank.txt"]["status"] == "failed" and admin_view["blank.txt"]["error"]
    assert admin_view["ok.txt"]["status"] == "indexed" and admin_view["ok.txt"]["created_at"]

    reader_view = reader.get("/v1/documents").json()["documents"]
    assert [d["name"] for d in reader_view] == ["ok.txt"]
    assert reader_view[0]["error"] is None and reader_view[0]["owner_id"] is None
    failed_id = admin_view["blank.txt"]["doc_id"]
    assert reader.get(f"/v1/documents/{failed_id}").status_code == 404
    assert client.get(f"/v1/documents/{failed_id}").json()["status"] == "failed"


# ---------- document-level authorization (SEC-004, RET-003, DATA-006, NFR-SEC-004) ----------

def test_private_documents_are_excluded_from_other_users_retrieval(env, client, user_client):
    alice = user_client("alice", "document_manager")
    bob = user_client("bob", "document_manager")
    secret = upload(alice, "alice-notes.txt", b"Project Falcon budget is 4 million dollars.", "private")
    upload(alice, "shared.txt", HR)

    assert listed(alice) == {"alice-notes.txt", "shared.txt"}
    assert listed(bob) == {"shared.txt"}

    env.state["answer"] = "The budget is 4 million dollars [1]."
    mine = alice.post("/v1/ask", json={"question": "Project Falcon budget dollars"}).json()
    assert mine["no_evidence"] is False
    assert mine["sources"][0]["doc_id"] == secret["doc_id"]

    before = env.calls["generation"]
    theirs = bob.post("/v1/ask", json={"question": "Project Falcon budget dollars"}).json()
    assert secret["doc_id"] not in {s["doc_id"] for s in theirs["sources"]}
    assert theirs["no_evidence"] is True and env.calls["generation"] == before
    # the passage never reached the model for bob
    assert "Falcon" not in str(env.state["messages"][before:]) or theirs["no_evidence"]

    # bob cannot address it directly either, and cannot learn that it exists
    for call in (bob.get, bob.delete):
        assert call(f"/v1/documents/{secret['doc_id']}").status_code == 404
    assert bob.patch(f"/v1/documents/{secret['doc_id']}", json={"visibility": "tenant"}).status_code == 404


def test_tenants_are_isolated(env, client, user_client):
    other_manager = user_client("olga", "document_manager", tenant="acme")
    other_admin = user_client("oscar", "administrator", tenant="acme")
    ours = upload(client, "hr.txt", HR)
    theirs = upload(other_manager, "fin.txt", FIN)

    assert listed(client) == {"hr.txt"} and listed(other_manager) == {"fin.txt"}

    env.state["answer"] = "Claims are due in 30 days [1]."
    cross = client.post("/v1/ask", json={"question": "expense claims submitted days travel"}).json()
    assert theirs["doc_id"] not in {s["doc_id"] for s in cross["sources"]}

    # even an administrator of another tenant cannot read, change or delete it
    assert other_admin.get(f"/v1/documents/{ours['doc_id']}").status_code == 404
    assert other_admin.delete(f"/v1/documents/{ours['doc_id']}").status_code == 404
    assert client.get(f"/v1/documents/{ours['doc_id']}").status_code == 200

    # identical content in two tenants: two documents, no cross-tenant duplicate signal
    same = upload(other_manager, "hr-copy.txt", HR)
    assert same["status"] == "indexed" and same["doc_id"] != ours["doc_id"]

    # user administration is tenant-scoped too
    assert {u["username"] for u in other_admin.get("/v1/admin/users").json()} == {"olga", "oscar"}
    admin_id = client.get("/v1/auth/me").json()["id"]
    assert other_admin.patch(f"/v1/admin/users/{admin_id}", json={"is_active": False}).status_code == 404


def test_visibility_change_applies_to_next_request_and_is_audited(env, client, user_client):
    alice = user_client("alice", "document_manager")
    reader = user_client("rita", "reader")
    doc = upload(alice, "notes.txt", HR, "private")
    assert listed(reader) == set()

    r = alice.patch(f"/v1/documents/{doc['doc_id']}", json={"visibility": "tenant"})
    assert r.status_code == 200 and r.json()["visibility"] == "tenant"
    assert listed(reader) == {"notes.txt"}
    answer = reader.post("/v1/ask", json={"question": "paid leave days per year"}).json()
    assert answer["sources"][0]["doc_id"] == doc["doc_id"]

    events = client.get("/v1/admin/audit", params={"action": "document.visibility_change"}).json()["events"]
    assert events[0]["target"] == doc["doc_id"] and events[0]["detail"] == "private -> tenant"


def test_no_readable_documents_means_no_evidence_and_no_model_call(env, client, user_client):
    newcomer = user_client("nina", "reader", tenant="empty-tenant")
    upload(client, "hr.txt", HR)
    body = newcomer.post("/v1/ask", json={"question": "paid leave days per year"}).json()
    assert body["no_evidence"] is True and body["sources"] == []
    assert env.calls == {"generation": 0, "verification": 0}


# ---------- revocation and permission changes (SEC-006) ----------

def test_revocation_and_role_change_apply_to_the_next_request(client, user_client):
    mona = user_client("mona", "document_manager")
    mona_id = mona.get("/v1/auth/me").json()["id"]
    assert mona.post("/v1/upload", files={"file": ("a.txt", HR, "text/plain")}).status_code == 200

    assert client.patch(f"/v1/admin/users/{mona_id}", json={"role": "reader"}).status_code == 200
    assert mona.post("/v1/upload", files={"file": ("b.txt", FIN, "text/plain")}).status_code == 403
    assert mona.get("/v1/documents").status_code == 200

    assert client.patch(f"/v1/admin/users/{mona_id}", json={"is_active": False}).status_code == 200
    assert mona.get("/v1/documents").status_code == 401
    assert mona.post("/v1/auth/login", data={"username": "mona", "password": PASSWORD}).status_code == 401

    assert client.patch(f"/v1/admin/users/{mona_id}", json={"is_active": True}).status_code == 200
    assert login("mona", PASSWORD).get("/v1/auth/me").json()["role"] == "reader"


def test_password_change_revokes_tokens(user_client):
    rita = user_client("rita")
    wrong = rita.post("/v1/auth/password", json={"current_password": "x", "new_password": "another-pass-9"})
    assert wrong.status_code == 403
    weak = rita.post("/v1/auth/password", json={"current_password": PASSWORD, "new_password": "short"})
    assert weak.status_code == 422
    ok = rita.post("/v1/auth/password", json={"current_password": PASSWORD, "new_password": "another-pass-9"})
    assert ok.status_code == 200
    assert rita.get("/v1/auth/me").status_code == 401
    assert login("rita", "another-pass-9").get("/v1/auth/me").status_code == 200


def test_last_administrator_cannot_be_removed(client):
    me = client.get("/v1/auth/me").json()["id"]
    assert client.patch(f"/v1/admin/users/{me}", json={"is_active": False}).status_code == 409
    assert client.patch(f"/v1/admin/users/{me}", json={"role": "reader"}).status_code == 409
    assert client.get("/v1/auth/me").status_code == 200


def test_user_creation_rules(client):
    ok = client.post("/v1/admin/users", json={"username": "newbie", "password": "long-enough-pw", "role": "reader"})
    assert ok.status_code == 201 and "password" not in ok.text.replace("long-enough-pw", "")
    assert "long-enough-pw" not in ok.text
    dup = client.post("/v1/admin/users", json={"username": "NEWBIE", "password": "long-enough-pw"})
    assert dup.status_code == 409 and dup.json()["code"] == "conflict"
    weak = client.post("/v1/admin/users", json={"username": "weak", "password": "short"})
    assert weak.status_code == 422
    bad_role = client.post("/v1/admin/users", json={"username": "x-user", "password": "long-enough-pw", "role": "root"})
    assert bad_role.status_code == 422


# ---------- audit (NFR-SEC-005) ----------

def test_security_relevant_actions_are_audited(env, client, user_client, anonymous):
    reader = user_client("rita", "reader")
    doc = upload(client, "hr.txt", HR)
    client.delete(f"/v1/documents/{doc['doc_id']}")
    created = client.post("/v1/admin/users", json={"username": "temp-user", "password": "long-enough-pw"}).json()
    client.patch(f"/v1/admin/users/{created['id']}", json={"role": "document_manager", "password": "new-long-password"})
    reader.get("/v1/admin/users")  # denied
    anonymous.post("/v1/auth/login", data={"username": ADMIN[0], "password": "wrong-password"})

    events = client.get("/v1/admin/audit", params={"limit": 500}).json()["events"]
    by_action = {}
    for e in events:
        by_action.setdefault(e["action"], []).append(e)

    deletion = by_action["document.delete"][0]
    assert deletion["target"] == doc["doc_id"] and deletion["outcome"] == "success"
    assert deletion["actor_name"] == ADMIN[0] and deletion["ts"] and deletion["request_id"]

    assert by_action["user.create"][0]["target"] == created["id"]
    update = by_action["user.update"][0]
    assert "reader -> document_manager" in update["detail"] and "password reset" in update["detail"]
    assert by_action["access.denied"][0]["actor_name"] == "rita"
    assert any(e["outcome"] == "failure" for e in by_action["auth.login"])
    # audit entries never contain secrets
    assert "new-long-password" not in str(events) and "wrong-password" not in str(events)


# ---------- limits (PERF-006, IF-002, NFR-SEC-008, API-006) ----------

def test_ask_is_rate_limited_per_user(env, client, user_client, monkeypatch):
    from app.config import settings

    monkeypatch.setattr(settings, "RATE_LIMIT_PER_MINUTE", 2)
    other = user_client("rita")
    codes = [client.post("/v1/ask", json={"question": "hello there"}).status_code for _ in range(3)]
    assert codes == [200, 200, 429]
    limited = client.post("/v1/ask", json={"question": "hello there"})
    assert limited.json()["code"] == "rate_limited" and int(limited.headers["Retry-After"]) >= 1
    assert other.post("/v1/ask", json={"question": "hello there"}).status_code == 200  # per user


def test_oversized_json_body_is_rejected(client):
    r = client.post("/v1/ingest", json={"document_path": "x" * 100_000})
    assert r.status_code == 413 and r.json()["code"] == "payload_too_large"


def test_file_content_must_match_its_type(client, env):
    fake_pdf = client.post("/v1/upload", files={"file": ("report.pdf", b"this is not a pdf", "application/pdf")})
    assert fake_pdf.status_code == 422
    binary = client.post("/v1/upload", files={"file": ("notes.txt", b"MZ\x90\x00\x03\x00\x00\x00", "text/plain")})
    assert binary.status_code == 422
    # rejected files are neither stored nor registered
    assert list((env.tmp / "uploads").iterdir()) == []
    assert client.get("/v1/documents").json()["documents"] == []


def test_batch_upload_reports_accepted_processed_failed(client):
    files = [
        ("files", ("a.txt", HR, "text/plain")),
        ("files", ("a-again.txt", HR, "text/plain")),
        ("files", ("virus.exe", b"MZ", "application/octet-stream")),
        ("files", ("blank.txt", b"  \n ", "text/plain")),
        ("files", ("empty.txt", b"", "text/plain")),
    ]
    r = client.post("/v1/upload/batch", files=files)
    assert r.status_code == 200, r.text
    body = r.json()
    assert (body["received"], body["accepted"], body["processed"], body["failed"]) == (5, 3, 2, 3)
    statuses = {item["filename"]: item["status"] for item in body["results"]}
    assert statuses == {
        "a.txt": "indexed", "a-again.txt": "duplicate", "virus.exe": "rejected",
        "blank.txt": "failed", "empty.txt": "rejected",
    }
    assert all(item["reason"] for item in body["results"] if item["status"] in ("failed", "rejected"))


def test_error_contract_distinguishes_categories(client, user_client, anonymous):
    reader = user_client("rita")
    cases = [
        (client.post("/v1/ask", json={"question": ""}), 422, "validation_error"),
        (anonymous.get("/v1/documents"), 401, "unauthenticated"),
        (reader.get("/v1/admin/users"), 403, "forbidden"),
        (client.get("/v1/documents/nope"), 404, "not_found"),
        (client.get("/v1/does-not-exist"), 404, "not_found"),
        (client.post("/v1/upload", files={"file": ("x.exe", b"MZ", "application/octet-stream")}), 415,
         "unsupported_media_type"),
    ]
    for response, status, code in cases:
        body = response.json()
        assert (response.status_code, body["code"]) == (status, code), response.text
        assert body["request_id"] == response.headers["X-Request-ID"]
        assert "Traceback" not in response.text

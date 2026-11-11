import io

from tests.conftest import POLICY


def test_question_validation(client):
    assert client.post("/v1/ask", json={"question": ""}).status_code == 422
    assert client.post("/v1/ask", json={"question": "   "}).status_code == 422
    assert client.post("/v1/ask", json={"question": "x" * 5000}).status_code == 422
    assert client.post("/v1/ask", json={}).status_code == 422


def test_upload_index_list_delete_roundtrip(client, env):
    up = client.post("/v1/upload", files={"file": ("policy.txt", POLICY.encode(), "text/plain")})
    assert up.status_code == 200, up.text
    doc = up.json()
    assert doc["status"] == "indexed" and doc["chunks"] >= 1 and doc["doc_id"]

    again = client.post("/v1/upload", files={"file": ("copy.txt", POLICY.encode(), "text/plain")})
    assert again.json()["status"] == "duplicate"

    listing = client.get("/v1/documents").json()["documents"]
    assert [d["doc_id"] for d in listing] == [doc["doc_id"]]
    assert listing[0]["size"] is not None

    assert client.delete(f"/v1/documents/{doc['doc_id']}").status_code == 200
    assert client.delete(f"/v1/documents/{doc['doc_id']}").status_code == 404
    assert client.get("/v1/documents").json()["documents"] == []


def test_upload_rejects_bad_type_empty_and_oversize(client):
    r = client.post("/v1/upload", files={"file": ("evil.exe", b"MZ", "application/octet-stream")})
    assert r.status_code == 415
    r = client.post("/v1/upload", files={"file": ("a.txt", b"", "text/plain")})
    assert r.status_code == 422
    big = b"a " * (1024 * 1024)  # 2 MB > MAX_UPLOAD_MB=1
    r = client.post("/v1/upload", files={"file": ("big.txt", big, "text/plain")})
    assert r.status_code == 413


def test_upload_blocks_path_traversal(client, env):
    r = client.post("/v1/upload", files={"file": ("../../escape.txt", POLICY.encode(), "text/plain")})
    assert r.status_code == 200
    assert not (env.tmp.parent / "escape.txt").exists()
    # stored inside the upload directory, under the document id
    stored = list((env.tmp / "uploads").iterdir())
    assert [p.name for p in stored] == [r.json()["doc_id"] + ".txt"]
    assert r.json()["filename"] == "escape.txt"


def test_ingest_path_is_restricted(client, tmp_path):
    outside = tmp_path / "secret.txt"
    outside.write_text("top secret salary data")
    r = client.post("/v1/ingest", json={"document_path": str(outside)})
    assert r.status_code == 403
    r = client.post("/v1/ingest", json={"document_path": "/etc/passwd"})
    assert r.status_code == 403
    r = client.post("/v1/ingest", json={"document_path": "../../etc/passwd"})
    assert r.status_code == 403


def test_ingest_rejects_semantic_and_missing_file(client, env, policy_file):
    r = client.post("/v1/ingest", json={"document_path": str(policy_file), "strategy": "semantic"})
    assert r.status_code == 422
    r = client.post("/v1/ingest", json={"document_path": str(env.tmp / "raw" / "nope.txt")})
    assert r.status_code == 404


def test_failed_upload_leaves_no_file_behind(client, env):
    r = client.post("/v1/upload", files={"file": ("blank.txt", b"   \n  ", "text/plain")})
    assert r.status_code == 422
    assert list((env.tmp / "uploads").iterdir()) == []


def test_llm_outage_returns_controlled_502_without_internals(client, env, policy_file):
    client.post("/v1/ingest", json={"document_path": str(policy_file)})
    from app.services import llm_service as ls

    def boom(**kw):
        raise ValueError("secret-internal-detail sk-test")

    ls.get_llm_service().client.chat.completions.create = boom
    r = client.post("/v1/ask", json={"question": "paid leave days employees"})
    assert r.status_code == 502
    assert "secret-internal-detail" not in r.text and "sk-test" not in r.text


def test_llm_transient_failures_are_retried_then_surface(client, env, policy_file, monkeypatch):
    from openai import APIConnectionError
    import httpx
    from app.services import llm_service as ls

    monkeypatch.setattr(ls.time, "sleep", lambda s: None)
    client.post("/v1/ingest", json={"document_path": str(policy_file)})
    attempts = {"n": 0}

    def flaky(**kw):
        attempts["n"] += 1
        raise APIConnectionError(request=httpx.Request("POST", "https://x"))

    ls.get_llm_service().client.chat.completions.create = flaky
    r = client.post("/v1/ask", json={"question": "paid leave days employees"})
    assert r.status_code == 502 and attempts["n"] == 3  # bounded retry policy

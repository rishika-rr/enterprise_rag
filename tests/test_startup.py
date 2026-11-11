from tests.conftest import ADMIN, login, reset_app


def test_import_without_api_key_does_not_crash(monkeypatch, tmp_path):
    """Regression: llm_service was instantiated at import time."""
    reset_app(tmp_path, monkeypatch, api_key=None)
    import app.main  # noqa: F401  (must import cleanly)


def test_missing_key_gives_503_not_crash(monkeypatch, tmp_path):
    """Without an LLM key: asking is a controlled 503, document management still works."""
    reset_app(tmp_path, monkeypatch, api_key=None)
    client = login(*ADMIN)

    r = client.post("/v1/ask", json={"question": "hi"})
    assert r.status_code == 503
    assert r.json()["code"] == "dependency_unavailable"
    assert "OPENROUTER" not in r.text

    assert client.get("/v1/documents").status_code == 200
    assert client.get("/health").json()["llm_configured"] is False

    from app.database import state
    state.reset_state_store()


def test_dotenv_is_not_loaded_in_tests():
    """Regression: the developer's real .env key leaked into the test run."""
    import os

    assert os.environ.get("RAG_SKIP_DOTENV") == "1"


def test_invalid_configuration_fails_loudly(monkeypatch, tmp_path):
    import importlib

    import pytest

    import app.config.settings as settings

    for name, value in [("TOP_K", "0"), ("RETRIEVAL_MODE", "semantic"), ("CHUNK_SIZE", "abc"),
                        ("RERANKER", "llm")]:
        monkeypatch.setenv(name, value)
        with pytest.raises(ValueError):
            importlib.reload(settings)
        monkeypatch.delenv(name)
    importlib.reload(settings)


def test_health_and_request_id(client, anonymous):
    r = anonymous.get("/health")  # readiness needs no credentials
    assert r.status_code == 200
    body = r.json()
    assert body["status"] == "ok" and body["indexed_chunks"] == 0
    assert body["identity_store"] == "ok"
    assert "sk-test" not in r.text and "admin-password" not in r.text
    assert r.headers["X-Request-ID"]

    # a well-formed caller id is echoed; anything else is replaced (log-injection guard)
    assert anonymous.get("/health", headers={"X-Request-ID": "abc-123"}).headers["X-Request-ID"] == "abc-123"
    bad = anonymous.get("/health", headers={"X-Request-ID": "x y\tz=1"}).headers["X-Request-ID"]
    assert bad != "x y\tz=1"


def test_openapi_documents_the_contract(anonymous):
    spec = anonymous.get("/openapi.json").json()
    paths = spec["paths"]
    for path in ["/v1/ask", "/v1/upload", "/v1/upload/batch", "/v1/ingest", "/v1/documents",
                 "/v1/documents/{doc_id}", "/v1/auth/login", "/v1/admin/users", "/v1/admin/audit"]:
        assert path in paths, path
    ask = paths["/v1/ask"]["post"]
    assert {"200", "401", "403", "422", "429", "502", "503"} <= set(ask["responses"])
    assert ask["security"]  # authentication requirement is published
    assert "ErrorResponse" in spec["components"]["schemas"]
    assert "Rate limits" in spec["info"]["description"]

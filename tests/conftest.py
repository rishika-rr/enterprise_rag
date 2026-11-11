"""
Test fixtures. Everything external is faked: no models are downloaded,
no network is used, no API key is needed.
"""
import hashlib
import importlib
import os
import types

# Must be set before the first `app.config.settings` import: keeps the
# developer's real .env (and real API key) out of the test run.
os.environ["RAG_SKIP_DOTENV"] = "1"

import numpy as np  # noqa: E402
import pytest  # noqa: E402

ADMIN = ("admin", "admin-password-1")
PASSWORD = "user-password-1"


def _fake_embed(text: str):
    v = np.zeros(64)
    for w in text.lower().split():
        v[int(hashlib.md5(w.encode()).hexdigest(), 16) % 64] += 1
    n = np.linalg.norm(v) or 1.0
    return (v / n).tolist()


class FakeChoice:
    def __init__(self, text):
        self.message = types.SimpleNamespace(content=text)


class FakeCompletion:
    def __init__(self, text):
        self.choices = [FakeChoice(text)]


def reset_app(tmp_path, monkeypatch, api_key: str | None = "sk-test", **extra_env):
    """Point the app at fresh temp storage and drop every process-wide singleton."""
    if api_key:
        monkeypatch.setenv("OPENROUTER_API_KEY", api_key)
    else:
        monkeypatch.delenv("OPENROUTER_API_KEY", raising=False)
    monkeypatch.setenv("CHROMA_DB_PATH", str(tmp_path / "chroma"))
    monkeypatch.setenv("APP_DB_PATH", str(tmp_path / "state" / "app.db"))
    monkeypatch.setenv("UPLOAD_DIR", str(tmp_path / "uploads"))
    monkeypatch.setenv("INGEST_ALLOWED_DIRS", str(tmp_path / "raw"))
    monkeypatch.setenv("MAX_UPLOAD_MB", "1")
    monkeypatch.setenv("RERANKER", "cross_encoder")
    monkeypatch.setenv("ADMIN_USERNAME", ADMIN[0])
    monkeypatch.setenv("ADMIN_PASSWORD", ADMIN[1])
    for key, value in extra_env.items():
        monkeypatch.setenv(key, str(value))
    (tmp_path / "raw").mkdir(exist_ok=True)

    import app.config.settings as settings
    importlib.reload(settings)  # modules read `settings.X` at call time

    import app.main  # noqa: F401  (must import cleanly)
    from app.api import routes
    from app.database import chroma, state
    from app.retrieval import sparse_index
    from app.security import rate_limit
    from app.services import llm_service

    state.reset_state_store()
    monkeypatch.setattr(chroma, "_store", None)
    monkeypatch.setattr(sparse_index, "_sparse_index", None)
    monkeypatch.setattr(llm_service, "_llm_service", None)
    monkeypatch.setattr(routes, "rag", None)
    rate_limit.ask_limiter.reset()
    rate_limit.login_limiter.reset()
    return settings


@pytest.fixture
def env(tmp_path, monkeypatch):
    """Fresh app wired to temp storage with fake models and a fake LLM."""
    reset_app(tmp_path, monkeypatch)

    from app.services import cross_encoder_service as ce
    from app.services import embedding_service as es
    from app.services import llm_service as ls

    monkeypatch.setattr(
        es.embedding_service, "embed_documents", lambda texts: [_fake_embed(t) for t in texts]
    )
    monkeypatch.setattr(es.embedding_service, "embed_query", _fake_embed)

    def fake_score(query, docs):
        q = set(query.lower().replace("?", "").split())
        # mimic ms-marco logits: no shared term -> very negative, overlap -> positive
        out = []
        for d in docs:
            overlap = len(q & set(d.lower().split()))
            out.append(-10.0 if overlap == 0 else overlap * 2.0 - 1.0)
        return out

    monkeypatch.setattr(ce.cross_encoder_service, "score", fake_score)

    calls = {"generation": 0, "verification": 0}
    state = {
        "answer": "Employees receive 24 days of paid leave per year [1].",
        "verdict": '{"supported": true, "reason": "stated in passage"}',
        "messages": [],  # every message list sent to the provider
    }

    def fake_create(**kw):
        state["messages"].append(kw["messages"])
        first = kw["messages"][0]["content"]
        if first.startswith("You are verifying citations"):
            calls["verification"] += 1
            return FakeCompletion(state["verdict"])
        calls["generation"] += 1
        return FakeCompletion(state["answer"])

    ls.get_llm_service().client.chat.completions.create = fake_create

    from app.api import routes
    from app.database.state import get_state_store

    yield types.SimpleNamespace(
        tmp=tmp_path, calls=calls, state=state, routes=routes,
        sparse=importlib.import_module("app.retrieval.sparse_index"),
        chroma=importlib.import_module("app.database.chroma"),
        store=get_state_store(),
    )

    from app.database import state as state_module
    state_module.reset_state_store()  # release the SQLite file before tmp cleanup


def login(username: str, password: str):
    """A TestClient that sends the user's bearer token on every request."""
    from fastapi.testclient import TestClient

    import app.main as main

    client = TestClient(main.app)
    response = client.post("/v1/auth/login", data={"username": username, "password": password})
    assert response.status_code == 200, response.text
    client.headers["Authorization"] = f"Bearer {response.json()['access_token']}"
    return client


@pytest.fixture
def client(env):
    """Authenticated as the bootstrap administrator (holds every permission)."""
    return login(*ADMIN)


@pytest.fixture
def anonymous(env):
    from fastapi.testclient import TestClient

    import app.main as main

    return TestClient(main.app)


@pytest.fixture
def user_client(env):
    """Factory: create a user (optionally in another tenant) and log in as them."""
    from app.models.identity import Role

    def make(username: str, role: str = "reader", tenant: str = "default"):
        env.store.create_user(username, PASSWORD, Role(role), tenant)
        return login(username, PASSWORD)

    return make


POLICY = (
    "Employees receive 24 days of paid leave per year. Unused leave can be carried forward up to 5 days.\n"
    "Remote work is allowed up to three days per week with manager approval.\n"
    "Expense claims must be submitted within 30 days with original receipts attached.\n"
)


@pytest.fixture
def policy_file(env):
    p = env.tmp / "raw" / "policy.txt"
    p.write_text(POLICY, encoding="utf-8")
    return p

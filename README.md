# Enterprise RAG

FastAPI backend for an access-controlled retrieval-augmented question-answering
platform: dense (ChromaDB) + BM25 retrieval → weighted reciprocal rank fusion →
reranking → LLM answer with verified citations and a confidence score.

It implements the backend side of the project SRS (v1.0). The React web client
lives in a separate repository.

* [docs/operations.md](docs/operations.md): configuration, access model, retention, backup and restore, TLS
* [docs/traceability.md](docs/traceability.md): every SRS requirement → implementation → test → status
* [eval/README.md](eval/README.md): evaluation dataset, metrics and rubric

## Run

```bash
cp .env.example .env        # set OPENROUTER_API_KEY, ADMIN_USERNAME, ADMIN_PASSWORD
docker compose up --build   # API on http://localhost:8000  (interactive docs: /docs)
```

Without Docker (Python 3.11 or 3.12):

```bash
python -m venv .venv && . .venv/bin/activate      # Windows: .venv\Scripts\activate
pip install torch --index-url https://download.pytorch.org/whl/cpu
pip install -r requirements.txt
uvicorn app.main:app
```

The first start creates the administrator named in `.env`. Without
`ADMIN_USERNAME` / `ADMIN_PASSWORD` (and no existing users) the service starts
but denies every protected request. More users: `POST /v1/admin/users`, or
`python -m scripts.create_user <name> --role reader`.

Run a single worker: the BM25 index and the rate limiter live in process memory.

## Authentication and roles

```bash
TOKEN=$(curl -s -X POST localhost:8000/v1/auth/login \
        -d "username=admin&password=..." | python -c "import sys,json;print(json.load(sys.stdin)['access_token'])")
curl -s localhost:8000/v1/ask -H "Authorization: Bearer $TOKEN" \
     -H "Content-Type: application/json" -d '{"question": "How many days of leave do I get?"}'
```

| Role | May |
| --- | --- |
| `reader` | ask questions, list the documents they can read |
| `document_manager` | + upload, delete, change visibility, see ingestion status and failure reasons |
| `administrator` | + manage users, read the audit log, ingest server-side files |

Tokens are opaque, stored only as hashes, and expire after `TOKEN_TTL_MINUTES`.
The user record is checked on every request, so a role change, deactivation or
password change applies to the very next request.

Every document belongs to a tenant and is either `tenant` (readable by all
users of that tenant) or `private` (readable by its owner). Retrieval is
filtered by that rule inside both retrievers, before fusion, reranking and
generation. Users never see, retrieve or learn about another tenant's content.

## API (`/v1`)

The authoritative contract is the OpenAPI document at `/openapi.json` (`/docs`).

| Method | Path | Permission | Purpose |
| --- | --- | --- | --- |
| POST | `/auth/login` | none | form `username`, `password` → bearer token |
| POST | `/auth/logout` · GET `/auth/me` · POST `/auth/password` | any user | session and own password |
| POST | `/ask` | reader | `{"question"}` → answer, sources, citation checks, confidence, latency |
| GET | `/documents` · `/documents/{doc_id}` | reader | visible documents / processing status |
| POST | `/upload` | document manager | one file (`file`, optional `visibility`) |
| POST | `/upload/batch` | document manager | several `files`; returns accepted / processed / failed |
| PATCH | `/documents/{doc_id}` | document manager | change `visibility` |
| DELETE | `/documents/{doc_id}` | document manager | remove the document, its chunks and its stored file |
| POST | `/ingest` | administrator | index a file inside `INGEST_ALLOWED_DIRS` |
| GET/POST | `/admin/users` · PATCH `/admin/users/{id}` | administrator | users of the own tenant |
| GET | `/admin/audit` | administrator | audit events of the own tenant |
| GET | `/health` (no prefix) | none | readiness, no secrets |

Supported files: `.pdf .txt .md .html .htm`, up to `MAX_UPLOAD_MB`, validated
by extension, size and content before processing.

### Errors

Every error is `{"detail": ..., "code": ..., "request_id": ...}`; the request id
is also the `X-Request-ID` header and appears on every related log line.

| Status | `code` | Meaning |
| --- | --- | --- |
| 422 / 413 / 415 | `validation_error` / `payload_too_large` / `unsupported_media_type` | client validation |
| 401 / 403 | `unauthenticated` / `forbidden` | authentication / authorization |
| 404 / 409 | `not_found` / `conflict` | client |
| 429 | `rate_limited` | see `Retry-After` (seconds) |
| 502 / 503 | `dependency_unavailable` | LLM provider failed, timed out or is not configured |
| 500 | `internal_error` | details only in the server log |

### Rate limits

`/v1/ask`: `RATE_LIMIT_PER_MINUTE` (default 120) per user. `/v1/auth/login`:
`LOGIN_RATE_LIMIT_PER_MINUTE` (default 10) per client and username.

## How answers are produced

1. Dense + BM25 candidates (`TOP_K` each) restricted to the caller's readable
   documents, fused with WRRF. `RETRIEVAL_MODE=hybrid|dense|bm25`.
2. A local cross-encoder reranks the candidates and keeps `RERANK_TOP_K`
   passages. `RERANKER=none` switches reranking off.
3. Passages below `MIN_RELEVANCE` are dropped. If none remain the API returns a
   fixed **no-evidence** answer (`no_evidence: true`) without calling the LLM.
4. The LLM receives its instructions in the system role and the passages as
   delimited, explicitly untrusted text, and must cite with `[n]` markers that
   match `sources[].citation`.
5. Each `(claim, [n])` pair is checked by an LLM verifier. A citation that does
   not resolve to a supplied passage is reported as unsupported.

LLM calls have a bounded timeout (`LLM_TIMEOUT_SECONDS`) and are retried only
for transient failures, at most `LLM_MAX_RETRIES` attempts with 1 s, 2 s, 4 s
back-off. Ingestion and deletion are never retried automatically.

### Confidence (definition)

`confidence = 0.5 × mean reranker relevance of the passages used + 0.5 × (supported citation pairs / checked pairs)`

Relevance is 0–1: `sigmoid(cross-encoder logit)`. With `RERANKER=none` confidence is the citation share alone.
It is a heuristic indicator in [0, 1], **not** a probability of correctness and
not proof that an answer is verified.

## Ingestion semantics

* `doc_id` is a SHA-256 prefix of the access scope and the file bytes: stable,
  and identical content in the same scope is indexed once (`status: "duplicate"`).
* Every accepted job has a registry record: `processing` → `indexed` or `failed`
  with a diagnostic reason, plus creation and update timestamps.
* Ingestion is all-or-nothing. A failure rolls back the new chunks and leaves
  existing documents untouched; an ingestion interrupted by a crash is rolled
  back at the next start.
* Chunks indexed before document ids existed are adopted at start-up as
  `legacy-<file name>` documents of the default tenant.

## Tests and evaluation

```bash
pytest -q                                   # no network, models or API keys needed
python -m scripts.evaluate --reranker cross_encoder --label my-run   # Recall@K, MRR, latency
python -m scripts.evaluate --compare eval/reports/a.json eval/reports/b.json
```

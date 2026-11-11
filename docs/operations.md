# Operations and configuration

This is the deployment configuration specification referred to by SRS
Appendix C.1. All settings are environment variables (see
[.env.example](../.env.example)); none require a source-code change.

## Configuration items (SRS C.1)

| Item | Value in this deployment | Where it is set |
| --- | --- | --- |
| Identity provider and login | Built-in username/password store (scrypt hashes), opaque bearer tokens | `ADMIN_USERNAME`, `ADMIN_PASSWORD`, `TOKEN_TTL_MINUTES`, `MIN_PASSWORD_LENGTH` |
| Tenant model and ownership | See "Access model" below | `DEFAULT_TENANT` |
| Supported formats and upload size | `.pdf .txt .md .html .htm`; 20 MB | `DocumentLoader.SUPPORTED_EXTENSIONS`, `MAX_UPLOAD_MB`, `MAX_BATCH_FILES` |
| API schemas | Generated OpenAPI document | `GET /openapi.json` |
| Embedding model | `sentence-transformers/all-MiniLM-L6-v2` | `EMBEDDING_MODEL` |
| Reranker | Local cross-encoder `cross-encoder/ms-marco-MiniLM-L-6-v2` (or none). Deviation from SRS 2.1.2, see below | `RERANKER`, `CROSS_ENCODER_MODEL` |
| Language model | Any model of an OpenAI-compatible provider (OpenRouter by default), e.g. a DeepSeek or Gemini model | `LLM_BASE_URL`, `LLM_MODEL`, `OPENROUTER_API_KEY` |
| Evaluation dataset and rubric | `eval/datasets/`, [eval/README.md](../eval/README.md) | `--dataset` |
| Retention, deletion, backup | See below | `AUDIT_RETENTION_DAYS`, `DOCUMENT_RETENTION_DAYS`, `LOG_RETENTION_DAYS` |
| Availability and recovery objectives | 99.5 % monthly, RTO 4 h, RPO 24 h (SRS targets; not yet measured) | hosting plan |

**Deviation from the SRS.** SRS 2.1.2 names "LLM-based relevance judging" as
the reranker technology. This deployment uses a local cross-encoder instead:
on the sample evaluation set it reached Recall@5 1.000 and MRR 1.000, it
needs no provider call, and an LLM judge added about 5 s per question against
the 6 s target of PERF-001. The two were not compared on retrieval quality. RET-006 is unaffected. The SRS table
should be updated through change control.

Changing `EMBEDDING_MODEL` requires re-ingesting all documents: vectors of
different models are not comparable.

## Access model

* A **user** has one role and belongs to one **tenant**.
* A **document** belongs to the tenant of the user who ingested it and has an
  owner and a visibility:
  * `tenant`: readable by every active user of that tenant;
  * `private`: readable by its owner only.
* Read rule (listing and retrieval): same tenant **and** (`tenant` visibility
  **or** owner).
* Manage rule (delete, change visibility): document managers for documents they
  can read; administrators for every document of their tenant. Administrators
  can manage but not query other users' private documents.
* Administrators manage users of their own tenant only. The first user of a
  new tenant is created by an operator with `scripts/create_user.py --tenant`.
* Documents the caller may not see answer `404`, never `403`, so their
  existence is not revealed.
* **Propagation interval: zero.** Permissions are evaluated from the database
  on every request. Deactivating a user or changing a password also deletes
  that user's tokens.
* Fail closed: a missing, malformed, expired or unknown token gives `401`; an
  unavailable identity store gives `503`; a user with no readable documents
  gets the no-evidence answer without any model call.

Documents indexed before this model existed are registered as `tenant`-visible
documents of `DEFAULT_TENANT`. Review them after upgrading and set sensitive
ones to `private` (`PATCH /v1/documents/{doc_id}`) or delete them.

## Secrets

Provider keys and the bootstrap password are read from the environment only
(a `.env` file locally, the platform's secret store in production). They are
never written to logs, API responses, audit events or evaluation reports.
`.env` is git-ignored. Passwords are stored as salted scrypt hashes and tokens
as SHA-256 hashes.

## TLS

The application speaks plain HTTP and must be published behind a TLS-terminating
proxy or platform load balancer (API-003, NFR-SEC-001). Do not expose port 8000
directly. Set `CORS_ORIGINS` to the exact `https://` origin of the web client.

## Logging

One line per event on stdout, `event key=value ...` (`LOG_FORMAT=text`) or JSON
(`LOG_FORMAT=json`), each carrying `request_id`. Per request: method, path,
status, latency, user id. Per answer: retrieval, generation and verification
time. Dependency failures are logged with the error class only.

Never logged: questions, answers, document text, passwords, tokens, API keys.

## Retention and deletion policy

| Data | Kept for | Mechanism |
| --- | --- | --- |
| Application / query logs | `LOG_RETENTION_DAYS` (30) | stdout; enforced by the log platform. Logs contain no question or document text. |
| Audit events | `AUDIT_RETENTION_DAYS` (365) | purged at service start |
| Documents, chunks, stored uploads | `DOCUMENT_RETENTION_DAYS` (0 = until deleted) | purged at service start; each purge is audited |
| Sessions | `TOKEN_TTL_MINUTES` | expired rows removed at login |
| Failed-ingestion records | until deleted by a manager | `DELETE /v1/documents/{doc_id}` |

Questions and answers are not stored. Deleting a document removes its vectors,
BM25 entries, stored file and registry record in the same request; there is no
result cache, so the deletion interval is immediate.

## Backup and restore

Persistent state is three locations:

| What | Path (default / Docker volume) |
| --- | --- |
| Vector index and chunk text | `CHROMA_DB_PATH` (`data/chroma_db` / `chroma_data`) |
| Users, document registry, audit log | `APP_DB_PATH` (`data/app.db` / `state_data`) |
| Uploaded source files | `UPLOAD_DIR` (`uploads` / `uploads_data`) |

The BM25 index is rebuilt from the vector index at start-up and needs no backup.

**Backup (daily, to meet the 24 h RPO).** Stop the service (or stop writes),
copy the three locations together, restart. They must be captured at the same
point in time.

**Restore.** Stop the service, replace the three locations with one backup
set, start the service. Start-up reconciliation then repairs any small
mismatch: an interrupted ingestion is rolled back and marked `failed`, and a
registered document whose chunks are missing is marked `failed` so it can be
ingested again. Verify with `GET /health` (`indexed_chunks`) and
`GET /v1/documents`.

A timed restore exercise against the 4 h RTO has not been carried out yet.

## Capacity notes

* Single process: run one worker and one replica. Scaling out needs a shared
  lexical index and rate limiter first.
* The BM25 index holds all chunk text in memory.
* `WARMUP_ON_START=true` loads the indexes and local models at boot.

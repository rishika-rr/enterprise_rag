# Requirement traceability (SRS v1.0, Appendix B)

Status of every SRS requirement for the backend in this repository, as of
2026-11-11. A requirement is **Verified** only when an automated test or a
recorded measurement demonstrates it, never because code exists.

| Status | Meaning |
| --- | --- |
| Verified | An automated test in `tests/` demonstrates it |
| Implemented | Built and documented; the SRS verification method (review, drill, deployment scan) has not been carried out |
| Open | Not demonstrated or not met |
| Frontend | Belongs to the React client (separate repository) |

Tests run with `pytest -q` (100 tests; models and the LLM provider are faked).

## 3.1 External interfaces

| ID | Implementation | Verification | Status |
| --- | --- | --- | --- |
| UI-001 … UI-005 | Backend supplies what the client needs: auth, status and error codes, `sources` separate from `answer`, status hidden from readers | — | Frontend |
| API-001 | OpenAPI at `/openapi.json` | `test_startup::test_openapi_documents_the_contract` | Verified |
| API-002 | Pydantic schemas, `ErrorResponse` (`app/api/errors.py`) | `test_api::test_question_validation`, `test_security::test_error_contract_distinguishes_categories` | Verified |
| API-003 | TLS terminated by the platform proxy (`docs/operations.md`) | Deployment scan not run | Implemented |
| API-004 | `app/security/auth.py` `require(permission)` on every `/v1` route | `test_security::test_every_protected_endpoint_denies…`, `test_role_matrix` | Verified |
| API-005 | Generic messages; no stack traces; validation errors never echo input | `test_api::test_llm_outage_returns_controlled_502_without_internals`, `test_security::test_login_failures_are_uniform…` | Verified |
| API-006 | Schemas, security scheme, status codes and rate limits in OpenAPI | `test_startup::test_openapi_documents_the_contract` | Verified |
| IF-001, ING-001 | `DocumentLoader.SUPPORTED_EXTENSIONS` | `test_api::test_upload_rejects_bad_type_empty_and_oversize`, `test_ingestion::test_html_and_non_utf8_files` | Verified |
| IF-002 | Size cap while streaming; content checks | `test_api::test_upload_rejects…`, `test_security::test_file_content_must_match_its_type` | Verified |
| IF-003 | Failed job recorded as `failed`, file removed | `test_pipeline::test_failed_ingestion_is_recorded_and_can_be_retried`, `test_api::test_failed_upload_leaves_no_file_behind` | Verified |
| IF-004 | Content-hash `doc_id`; `GET /v1/documents/{doc_id}` | `test_pipeline::test_document_record_has_identifier_status_and_timestamps` | Verified |

## 3.2.1 Identity, roles and access control

| ID | Implementation | Verification | Status |
| --- | --- | --- | --- |
| SEC-001 | Bearer-token login (`app/api/auth_routes.py`) | `test_security::test_every_protected_endpoint_denies…`, `test_expired_token_is_rejected`, `test_logout_revokes_the_token` | Verified |
| SEC-002 | `Role`, `ROLE_PERMISSIONS` (`app/models/identity.py`) | `test_security::test_role_matrix` | Verified |
| SEC-003 | Permission check per operation, `can_read` / `can_manage` per document | `test_security::test_role_matrix`, `test_private_documents…`, `test_tenants_are_isolated` | Verified |
| SEC-004 | `EnterpriseRAG.ask(user=…)` filters both retrievers by readable `doc_id`s | `test_security::test_private_documents_are_excluded_from_other_users_retrieval` | Verified |
| SEC-005 | 401 / 403 / 503; empty permission set → no retrieval | `test_security::test_every_protected_endpoint_denies…`, `test_no_readable_documents_means_no_evidence_and_no_model_call` | Verified |
| SEC-006 | User re-read per request; propagation interval zero | `test_security::test_revocation_and_role_change_apply_to_the_next_request`, `test_password_change_revokes_tokens` | Verified |
| SEC-007 | scrypt password hashes, hashed tokens, secrets from environment | `test_security::test_passwords_and_tokens_are_not_stored_in_plaintext`; configuration review not held | Verified (storage) / Implemented (secret management) |

## 3.2.2 Ingestion and indexing

| ID | Implementation | Verification | Status |
| --- | --- | --- | --- |
| ING-002 | `DocumentLoader.load` (PDF, text, Markdown, HTML) | `test_ingestion::test_html_and_non_utf8_files`, `test_pdf_text_and_page_numbers_are_extracted`. No representative-corpus test | Verified |
| ING-003 | `Chunker` fixed and recursive; `CHUNK_STRATEGY`, size, overlap | `test_ingestion::test_fixed_and_recursive_chunking_respect_configured_limits`, `test_unsupported_strategy_rejected` | Verified |
| ING-004, DATA-002 | `doc_id` in every chunk's metadata | `test_pipeline::test_document_record_has_identifier_status_and_timestamps` | Verified |
| ING-005 | Dense and BM25 indexes updated together | `test_retrieval::test_bm25_is_shared_between_indexer_and_retriever`, `test_bm25_survives_restart` | Verified |
| ING-006 | `POST /v1/upload/batch`, `ingest_directory` | `test_security::test_batch_upload_reports_accepted_processed_failed`, `test_ingestion::test_directory_summary_counts` | Verified |
| ING-007 | Same scope + same bytes → `duplicate` | `test_ingestion::test_ingest_duplicate_and_delete`, `test_api::test_upload_index_list_delete_roundtrip` | Verified |
| ING-008 | `DELETE /v1/documents/{doc_id}` | `test_pipeline::test_deletion_removes_chunks_file_and_record` | Verified |
| ING-009 | All-or-nothing with rollback | `test_ingestion::test_failed_ingestion_rolls_back` | Verified |
| ING-010 | Registry `status` + sanitised `error` | `test_pipeline::test_failed_ingestion_is_recorded_and_can_be_retried` | Verified |

## 3.2.3 Query processing and retrieval

| ID | Implementation | Verification | Status |
| --- | --- | --- | --- |
| RET-001, RET-002 | `AskRequest`; body-size limit | `test_api::test_question_validation`, `test_security::test_oversized_json_body_is_rejected` | Verified |
| RET-003 | Access filter in both retrievers | `test_security::test_private_documents…`, `test_tenants_are_isolated` | Verified |
| RET-004 | `RETRIEVAL_MODE` | `test_retrieval::test_retrieval_modes`, `test_invalid_mode_rejected` | Verified |
| RET-005 | Weighted reciprocal rank fusion | `test_retrieval::test_fusion_does_not_mutate_inputs_and_weights_work` | Verified |
| RET-006 | `RERANKER`, `RERANK_TOP_K` | `test_pipeline::test_cross_encoder_reranker_orders_by_relevance_and_caps_top_k`, `test_reranker_is_selected_by_configuration` | Verified |
| RET-007 | Metadata carried unchanged to `sources` | `test_pipeline::test_metadata_survives_retrieval_fusion_rerank_and_response` | Verified |
| RET-008 | Fixed no-evidence response, no LLM call | `test_generation::test_no_evidence_skips_llm_and_returns_no_sources`, `test_empty_corpus_is_no_evidence` | Verified |

## 3.2.4 Generation, citations, confidence

| ID | Implementation | Verification | Status |
| --- | --- | --- | --- |
| GEN-001 | `Generator.generate` | `test_generation::test_answer_has_verified_citations` | Verified |
| GEN-002 | `app/prompts/system.txt` (uncertainty section) | Evaluation set with the real model not yet run in answer mode | Implemented |
| GEN-003 | `sources` with citation numbers | `test_generation::test_answer_has_verified_citations` | Verified |
| GEN-004 | Each `[n]` resolved against the supplied passages | `test_generation::test_fabricated_citation_is_flagged` | Verified |
| GEN-005 | No "verified" label; per-pair `supported` flags only | `test_generation::test_fabricated_citation_is_flagged`; adversarial review not held | Implemented |
| GEN-006 | Definition in README and the OpenAPI field description | `test_generation::test_confidence_is_bounded_even_with_extreme_logits` (bounds); documentation review not held | Implemented |
| GEN-007 | `latency_ms`, `confidence` in the response | `test_generation::test_answer_has_verified_citations` | Verified |
| GEN-008, NFR-REL-001 | 502 / 503 `dependency_unavailable` | `test_api::test_llm_outage_returns_controlled_502…`, `test_startup::test_missing_key_gives_503_not_crash` | Verified |
| GEN-009 | No sources without evidence; invalid citations flagged | `test_generation::test_no_evidence_skips_llm…`, `test_fabricated_citation_is_flagged` | Verified |

## 3.2.5 Evaluation and observability

| ID | Implementation | Verification | Status |
| --- | --- | --- | --- |
| EVAL-001, DATA-007 | `eval/datasets/sample-v1.json`, version + SHA-256 in every report | `test_evaluation::test_dataset_is_versioned_and_well_formed`. Sample data only; the approved dataset is outstanding | Implemented |
| EVAL-002 | `app/evaluation/metrics.py` | `test_evaluation::test_recall_and_mrr`, `test_benchmark_run_report_and_comparison` | Verified |
| EVAL-003 | Groundedness, citation correctness, answer relevance (`eval/README.md`) | `test_evaluation::test_answer_evaluation_reports_quality_metrics` (fake judge). Evaluator agreement review not held | Implemented |
| EVAL-004 | p50 / p90 / p95 and error rate per run | `test_evaluation::test_benchmark_run_report_and_comparison`, `test_failed_requests_are_counted_not_dropped` | Verified |
| EVAL-005 | `request_id` on every log line and in every response | `test_pipeline::test_logs_carry_request_id_and_no_secrets`, `test_startup::test_health_and_request_id` | Verified |
| EVAL-006 | No questions, content or secrets in logs | `test_pipeline::test_logs_carry_request_id_and_no_secrets` | Verified |
| EVAL-007 | `scripts/evaluate.py --compare` | `test_evaluation::test_benchmark_run_report_and_comparison` | Verified |

## 3.2.6 Error handling and recovery

| ID | Implementation | Verification | Status |
| --- | --- | --- | --- |
| ERR-001 | Status + `code` per category | `test_security::test_error_contract_distinguishes_categories` | Verified |
| ERR-002 | `LLM_TIMEOUT_SECONDS` on the provider client | Configuration only; no timeout fault test | Implemented |
| ERR-003 | Bounded retries for transient errors only | `test_api::test_llm_transient_failures_are_retried_then_surface` | Verified |
| ERR-004 | No automatic retry of ingest / delete; deterministic ids + upsert | `test_retrieval::test_reingest_after_partial_failure_is_idempotent`, `test_api::test_upload_index_list_delete_roundtrip` | Verified |
| ERR-005 | Rollback leaves existing documents intact | `test_ingestion::test_failed_ingestion_keeps_existing_documents`, `test_pipeline::test_interrupted_ingestion_is_rolled_back_on_restart` | Verified |

## 3.3 Performance

| ID | Evidence so far | Status |
| --- | --- | --- |
| PERF-001 (90 % ≤ 6 s) | Not benchmarked with 200 queries. Four spot checks on a developer laptop, 6,445 chunks, cross-encoder reranker, DeepSeek via OpenRouter: answered questions took 4.5 s, 11.3 s and 12.9 s; a no-evidence question 0.5 s. Local retrieval is ≈ 0.4 s; the rest is provider time (generation 3–5 s, citation verification 1–8 s). | **Open – not met in spot checks** |
| PERF-002 (10 concurrent, 5 q/s) | No load test | Open |
| PERF-003 (≥ 10,000 chunks) | Runs correctly with 6,445 chunks; not tested at 10,000 | Open |
| PERF-004 (100 documents ≤ 10 min) | No benchmark | Open |
| PERF-005 (validation ≤ 500 ms p95) | 400 malformed requests in-process: p50 3.4 ms, p95 4.8 ms. Not measured on the production tier | Implemented |
| PERF-006 | Question, body, upload, batch, top-K and context limits | `test_security::test_oversized_json_body_is_rejected`, `test_pipeline::test_context_is_bounded…`, `test_startup::test_invalid_configuration_fails_loudly` — Verified |

## 3.4 Data

| ID | Implementation | Verification | Status |
| --- | --- | --- | --- |
| DATA-001 | `documents` table | `test_pipeline::test_document_record_has_identifier_status_and_timestamps` | Verified |
| DATA-003 | Start-up reconciliation of registry and indexes | `test_pipeline::test_interrupted_ingestion…`, `test_legacy_chunks_are_adopted_and_unregistered_chunks_stay_unreadable` | Verified |
| DATA-004 | Immediate deletion everywhere; no cache | `test_pipeline::test_deletion_removes_chunks_file_and_record` | Verified |
| DATA-005, NFR-SEC-006 | Retention settings and policy (`docs/operations.md`) | `test_pipeline::test_retention_periods_are_enforced`; policy review not held | Verified (mechanism) / Implemented (policy) |
| DATA-006, NFR-SEC-004 | Tenant in the access rule and in `doc_id` | `test_security::test_tenants_are_isolated` | Verified |
| DATA-008 | Backup and restore procedure (`docs/operations.md`) | Recovery test not run | Implemented |

## 3.6 Software system attributes

| ID | Implementation | Verification | Status |
| --- | --- | --- | --- |
| NFR-SEC-001 | TLS at the proxy | TLS scan not run | Implemented |
| NFR-SEC-002 | `.env` git-ignored; nothing secret logged | `test_pipeline::test_logs_carry_request_id_and_no_secrets`; secret scan of history not run | Implemented |
| NFR-SEC-003 | Role → permission map | `test_security::test_role_matrix` | Verified |
| NFR-SEC-005 | `audit_events`, `GET /v1/admin/audit` | `test_security::test_security_relevant_actions_are_audited` | Verified |
| NFR-SEC-007 | System role for instructions, delimited untrusted context | `test_pipeline::test_document_text_never_enters_the_system_role` checks prompt construction. Model behaviour under attack is not tested | Implemented |
| NFR-SEC-008 | Type, size and content checks before processing | `test_security::test_file_content_must_match_its_type` | Verified |
| NFR-REL-002 | Roll back interrupted ingestion at start | `test_pipeline::test_interrupted_ingestion_is_rolled_back_on_restart` | Verified |
| NFR-REL-003, NFR-REL-004 (P2) | Targets and procedure documented | No availability measurement or timed restore | Open |
| NFR-MNT-001, NFR-PORT-002 | Everything in environment variables | `test_pipeline::test_reranker_is_selected_by_configuration`, `test_retrieval::test_retrieval_modes`, `test_startup::test_invalid_configuration_fails_loudly` | Verified |
| NFR-MNT-002 | Request, answer-stage and dependency-failure log lines | `test_pipeline::test_logs_carry_request_id_and_no_secrets` | Verified |
| NFR-MNT-003 | `GET /health` | `test_startup::test_health_and_request_id` | Verified |
| NFR-MNT-004, NFR-PORT-001 | README, `requirements.txt`, Dockerfile, compose file | Clean-environment and container deployment test not run | Implemented |
| NFR-USE-001, NFR-USE-002 | Actionable `detail` messages; `sources[].title` / `page` / `snippet` | UI test | Frontend |

## Release acceptance (C.2): outstanding

* Change-control entry for the reranker: a cross-encoder replaces the LLM-based judging named in SRS 2.1.2 (`docs/operations.md`).

* Performance benchmark on the production tier (PERF-001 … PERF-005).
* Organisation-approved evaluation dataset and an answer-mode run with reviewer agreement.
* TLS scan, clean-environment and container deployment test, timed restore.
* Web-client requirements (UI-*, NFR-USE-*).
* An owner for each P1 requirement.

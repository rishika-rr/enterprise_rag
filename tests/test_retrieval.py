from app.retrieval.sparse_index import SparseIndex, tokenize
from app.retrieval.fusion import WeightedReciprocalRankFusion
from app.models.retrieval import RetrievalResult


def test_tokenizer_strips_punctuation():
    assert tokenize("Leave, per-year (24).") == ["leave", "per", "year", "24"]


def test_bm25_is_shared_between_indexer_and_retriever(env, policy_file):
    """Regression: indexer and retriever each had their own empty index."""
    from app.rag import EnterpriseRAG
    rag = EnterpriseRAG()
    rag.ingest(str(policy_file))
    bm25 = rag.generator.retrieval_pipeline.retriever.bm25
    assert bm25.sparse_index is rag.indexer.sparse_index
    hits = bm25.search("paid leave")
    assert hits and hits[0].retriever == "bm25"


def test_bm25_survives_restart(env, policy_file):
    """Regression: BM25 lived only in memory and vanished on restart."""
    from app.rag import EnterpriseRAG
    EnterpriseRAG().ingest(str(policy_file))
    env.sparse._sparse_index = None  # simulate a new process
    assert len(env.sparse.get_sparse_index().ids) == 1


def test_bm25_ignores_zero_score_docs():
    idx = SparseIndex()
    idx.add_documents(["a", "b"], ["alpha beta", "gamma delta"], [{}, {}])
    assert [r.id for r in idx.search("alpha")] == ["a"]
    assert idx.search("zzz") == []


def test_fusion_does_not_mutate_inputs_and_weights_work():
    a = [RetrievalResult("x", "d", {}, 0.1, "dense"), RetrievalResult("y", "d", {}, 0.2, "dense")]
    b = [RetrievalResult("y", "d", {}, 5.0, "bm25")]
    fused = WeightedReciprocalRankFusion().fuse(a, b)
    assert fused[0].id == "y" and fused[0].retriever == "hybrid"
    assert a[0].retriever == "dense" and a[0].score == 0.1  # untouched
    heavy = WeightedReciprocalRankFusion(weights=[1000, 1]).fuse(a, b)
    assert heavy[0].id == "x"


# ---------- RETRIEVAL_MODE / metadata filters ----------

def _two_docs(env):
    from app.rag import EnterpriseRAG
    a = env.tmp / "raw" / "hr.txt"
    a.write_text("Employees receive 24 days of paid leave per year.")
    b = env.tmp / "raw" / "fin.txt"
    b.write_text("Expense claims must be submitted within 30 days of travel.")
    rag = EnterpriseRAG()
    ra, rb = rag.ingest(str(a)), rag.ingest(str(b))
    return rag, ra, rb


def test_retrieval_modes(env):
    from app.retrieval.hybrid import HybridRetriever
    rag, _, _ = _two_docs(env)
    h = rag.generator.retrieval_pipeline.retriever
    assert {r.retriever for r in HybridRetriever(h.dense, h.bm25, mode="dense").search("paid leave")} == {"dense"}
    assert {r.retriever for r in HybridRetriever(h.dense, h.bm25, mode="bm25").search("paid leave")} == {"bm25"}
    assert {r.retriever for r in HybridRetriever(h.dense, h.bm25, mode="hybrid").search("paid leave")} == {"hybrid"}


def test_invalid_mode_rejected(env):
    import pytest
    from app.retrieval.hybrid import HybridRetriever
    with pytest.raises(ValueError):
        HybridRetriever(mode="semantic")


def test_where_filter_applies_to_dense_and_sparse(env):
    rag, hr, fin = _two_docs(env)
    h = rag.generator.retrieval_pipeline.retriever
    for retriever in (h.dense, h.bm25):
        only_fin = retriever.search("days", where={"doc_id": fin.doc_id})
        assert only_fin and {r.metadata["doc_id"] for r in only_fin} == {fin.doc_id}
    # several keys -> $and (Chroma rejects a bare multi-key dict)
    both = h.dense.search("days", where={"doc_id": hr.doc_id, "source": "hr.txt"})
    assert {r.metadata["source"] for r in both} == {"hr.txt"}
    # contradictory filters -> nothing, not an error
    assert h.dense.search("days", where={"doc_id": hr.doc_id, "source": "fin.txt"}) == []
    # empty dict == no filter
    assert len(h.dense.search("days", where={})) == 2


def test_sparse_rejects_operator_filters(env):
    import pytest
    from app.retrieval.sparse_index import SparseIndex
    idx = SparseIndex()
    idx.add_documents(["a"], ["alpha"], [{"doc_id": "x"}])
    with pytest.raises(ValueError):
        idx.search("alpha", where={"doc_id": {"$gt": "x"}})
    # $in is supported: it carries the list of documents a user may read
    assert [r.id for r in idx.search("alpha", where={"doc_id": {"$in": ["x"]}})] == ["a"]
    assert idx.search("alpha", where={"doc_id": {"$in": []}}) == []


def test_ask_where_restricts_sources(env):
    rag, hr, fin = _two_docs(env)
    env.state["answer"] = "Claims are due in 30 days [1]."
    result = rag.ask("expense claims days", where={"doc_id": fin.doc_id})
    assert {c.metadata["doc_id"] for c in result.retrieved_chunks} == {fin.doc_id}


def test_reingest_after_partial_failure_is_idempotent(env, policy_file):
    """upsert: re-adding the same deterministic ids must not error or duplicate."""
    from app.rag import EnterpriseRAG
    rag = EnterpriseRAG()
    r = rag.ingest(str(policy_file))
    ids, docs, metas = rag.indexer.vector_store.get_all()
    rag.indexer.vector_store.add_documents(ids, docs, [[0.1] * 64] * len(ids), metas)
    assert rag.stats()["total_chunks"] == r.chunk_count

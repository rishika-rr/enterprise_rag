import pytest

from tests.conftest import POLICY


def test_ingest_duplicate_and_delete(env, policy_file):
    from app.rag import EnterpriseRAG
    rag = EnterpriseRAG()
    first = rag.ingest(str(policy_file))
    assert first.status == "indexed" and first.chunk_count >= 1

    again = rag.ingest(str(policy_file))
    assert again.status == "duplicate" and again.doc_id == first.doc_id
    assert rag.stats()["total_chunks"] == first.chunk_count  # not doubled

    docs = rag.list_documents()
    assert docs == [{"doc_id": first.doc_id, "name": "policy.txt", "chunks": first.chunk_count}]

    assert rag.delete_document(first.doc_id) == first.chunk_count
    assert rag.stats()["total_chunks"] == 0
    assert rag.indexer.sparse_index.search("leave") == []


def test_failed_ingestion_rolls_back(env, policy_file, monkeypatch):
    from app.rag import EnterpriseRAG
    rag = EnterpriseRAG()

    def boom(**kw):
        raise RuntimeError("sparse failure")

    monkeypatch.setattr(rag.indexer.sparse_index, "add_documents", boom)
    with pytest.raises(RuntimeError):
        rag.ingest(str(policy_file))
    assert rag.stats()["total_chunks"] == 0  # no half-indexed document


def test_failed_ingestion_keeps_existing_documents(env, policy_file):
    from app.rag import EnterpriseRAG
    rag = EnterpriseRAG()
    rag.ingest(str(policy_file))
    before = rag.stats()["total_chunks"]
    empty = env.tmp / "raw" / "empty.txt"
    empty.write_text("   \n  ")
    with pytest.raises(ValueError):
        rag.ingest(str(empty))
    assert rag.stats()["total_chunks"] == before


def test_html_and_non_utf8_files(env):
    from app.rag import EnterpriseRAG
    rag = EnterpriseRAG()
    h = env.tmp / "raw" / "page.html"
    h.write_text("<html><script>x=1</script><body><p>Quarterly revenue grew.</p></body></html>")
    t = env.tmp / "raw" / "latin.txt"
    t.write_bytes("Café policy: refunds within 14 days.".encode("cp1252"))
    assert rag.ingest(str(h)).status == "indexed"
    assert rag.ingest(str(t)).status == "indexed"
    texts = " ".join(rag.indexer.sparse_index.documents)
    assert "Quarterly revenue" in texts and "x=1" not in texts and "Café" in texts


def test_unsupported_strategy_rejected(env, policy_file):
    from app.rag import EnterpriseRAG
    with pytest.raises(ValueError):
        EnterpriseRAG().ingest(str(policy_file), strategy="semantic")


def test_directory_summary_counts(env, policy_file):
    from app.rag import EnterpriseRAG
    (env.tmp / "raw" / "notes.xyz").write_text("ignored")
    (env.tmp / "raw" / "bad.txt").write_text("")
    s = EnterpriseRAG().ingest_directory(str(env.tmp / "raw"))
    assert (s["accepted"], s["indexed"], s["failed"], s["skipped_unsupported"]) == (2, 1, 1, 1)


def test_pdf_text_and_page_numbers_are_extracted(env, client):
    """ING-002 / NFR-USE-002: PDF text is indexed and sources carry 1-based pages."""
    import fitz  # PyMuPDF

    pdf = fitz.open()
    for text in ("Introduction and scope of the travel policy.",
                 "Employees receive 24 days of paid leave per year."):
        pdf.new_page().insert_text((72, 72), text)
    path = env.tmp / "raw" / "handbook.pdf"
    pdf.save(str(path))
    pdf.close()

    doc = client.post("/v1/ingest", json={"document_path": str(path)}).json()
    assert doc["status"] == "indexed" and doc["chunks"] == 2
    source = client.post("/v1/ask", json={"question": "paid leave days per year"}).json()["sources"][0]
    assert source["title"] == "handbook.pdf" and source["page"] == 2


def test_fixed_and_recursive_chunking_respect_configured_limits(env):
    """ING-003: both strategies honour CHUNK_SIZE; the strategy is recorded per chunk."""
    from app.rag import EnterpriseRAG

    text = "\n".join(f"Policy clause {i} describes the handling of case number {i} in detail." for i in range(60))
    rag = EnterpriseRAG()
    counts = {}
    for strategy in ("fixed", "recursive"):
        path = env.tmp / "raw" / f"{strategy}.txt"
        path.write_text(text + f"\n{strategy}")  # distinct content -> distinct documents
        result = rag.ingest(str(path), strategy=strategy)
        counts[strategy] = result.chunk_count
        _, documents, metadatas = rag.indexer.vector_store.get_all()
        mine = [(d, m) for d, m in zip(documents, metadatas) if m["doc_id"] == result.doc_id]
        assert len(mine) > 1 and {m["strategy"] for _, m in mine} == {strategy}
        assert all(len(d) <= 500 for d, _ in mine)
        assert sorted(m["chunk_index"] for _, m in mine) == list(range(len(mine)))
    assert counts["fixed"] > 1 and counts["recursive"] > 1

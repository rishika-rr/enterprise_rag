from app.generation.citation_verifier import parse_verdict
from app.generation.confidence_scorer import ConfidenceScorer
from app.models.citation import CitationResult
from app.models.retrieval import RetrievalResult


def test_parse_verdict_handles_fences_and_prose():
    assert parse_verdict('```json\n{"supported": true, "reason": "x"}\n```')["supported"] is True
    assert parse_verdict('Sure! {"supported": false, "reason": "no"} hope that helps')["supported"] is False


def test_confidence_is_bounded_even_with_extreme_logits():
    """Regression: raw cross-encoder logits made confidence >1 or negative."""
    hi = [RetrievalResult("a", "d", {}, 0.0, "hybrid", rerank_score=9.0)]
    lo = [RetrievalResult("a", "d", {}, 0.0, "hybrid", rerank_score=-11.0)]
    ok = [CitationResult("c", 1, True, "")]
    s = ConfidenceScorer()
    assert 0.0 <= s.score(lo, []) < 0.01
    assert 0.9 < s.score(hi, ok) <= 1.0
    assert s.score([], ok) == 0.0


def test_answer_has_verified_citations(env, policy_file, client):
    """Regression: the prompt forbade [n] markers so nothing was ever verified."""
    from app.rag import EnterpriseRAG  # noqa
    client.post("/v1/ingest", json={"document_path": str(policy_file)})
    r = client.post("/v1/ask", json={"question": "How many days of paid leave do employees get?"})
    assert r.status_code == 200
    body = r.json()
    assert body["no_evidence"] is False
    assert body["citations"] == 1 and body["verified_citations"] == 1
    assert body["sources"][0]["citation"] == 1 and body["sources"][0]["page"] is None
    assert body["sources"][0]["snippet"]
    assert 0 < body["confidence"] <= 1
    assert env.calls["generation"] == 1 and env.calls["verification"] == 1


def test_fabricated_citation_is_flagged(env, policy_file, client):
    env.state["answer"] = "Employees get 24 days [1]. They also get a car [9]."
    client.post("/v1/ingest", json={"document_path": str(policy_file)})
    body = client.post("/v1/ask", json={"question": "paid leave days employees"}).json()
    bad = [d for d in body["citation_details"] if d["citation"] == 9][0]
    assert bad["supported"] is False and "match" in bad["reason"].lower()
    assert body["verified_citations"] == 1 and body["citations"] == 2


def test_unparseable_or_failing_verifier_does_not_break_answer(env, policy_file, client):
    env.state["verdict"] = "I think so, yes."
    client.post("/v1/ingest", json={"document_path": str(policy_file)})
    body = client.post("/v1/ask", json={"question": "paid leave days employees"}).json()
    assert body["citation_details"][0]["supported"] is False
    assert body["answer"]


def test_no_evidence_skips_llm_and_returns_no_sources(env, policy_file, client):
    """RET-008: unrelated question must not produce a fabricated answer."""
    client.post("/v1/ingest", json={"document_path": str(policy_file)})
    body = client.post("/v1/ask", json={"question": "quantum chromodynamics lagrangian"}).json()
    assert body["no_evidence"] is True
    assert body["sources"] == [] and body["confidence"] == 0.0
    assert env.calls["generation"] == 0


def test_empty_corpus_is_no_evidence(env, client):
    body = client.post("/v1/ask", json={"question": "anything at all"}).json()
    assert body["no_evidence"] is True and env.calls["generation"] == 0


def test_citation_after_the_full_stop_belongs_to_that_sentence():
    """Regression: 'Fact. [1]' was parsed as a claim with no citation and never verified."""
    from app.generation.citation_parser import CitationParser

    parse = CitationParser().parse
    assert parse("The code word is **marmalade-42**. [1]") == [("The code word is **marmalade-42**.", [1])]
    assert parse("Leave is 24 days [1]. Remote work is allowed. [2][3]") == [
        ("Leave is 24 days.", [1]), ("Remote work is allowed.", [2, 3])]
    # bullets and lines are separate claims, each with its own citation
    assert parse("## Goals\n- Create an SSOT [4]\n- Train teams [2]\n\n[9]") == [
        ("## Goals", []), ("- Create an SSOT", [4]), ("- Train teams", [2])]


def test_irrelevant_passages_are_not_sent_to_the_model_or_listed(env, client):
    """Passages the reranker scores below MIN_RELEVANCE are dropped, not shown as sources."""
    for name, text in [("hr.txt", "Employees receive 24 days of paid leave per year."),
                       ("misc.txt", "Leave the parking garage through gate B.")]:
        p = env.tmp / "raw" / name
        p.write_text(text)
        client.post("/v1/ingest", json={"document_path": str(p)})

    from app.services import cross_encoder_service as ce
    ce.cross_encoder_service.score = lambda q, docs: [5.0 if "paid" in d else -12.0 for d in docs]

    body = client.post("/v1/ask", json={"question": "paid leave days"}).json()
    assert [s["title"] for s in body["sources"]] == ["hr.txt"] and body["retrieved_chunks"] == 1
    generation = [m for m in env.state["messages"] if m[0]["role"] == "system"][-1]
    assert "parking garage" not in generation[1]["content"]

"""Evaluation harness (EVAL-001..004, EVAL-007, DATA-007)."""
import json
from pathlib import Path

from app.evaluation.metrics import latency_summary, percentile, recall_at_k, reciprocal_rank

DATASET = Path(__file__).parent.parent / "eval" / "datasets" / "sample-v1.json"


def test_recall_and_mrr():
    labels = [{"source": "a.md", "contains": "alpha"}, {"source": "b.md", "contains": "beta"}]
    ranked = [("c.md", "noise"), ("a.md", "the ALPHA passage"), ("b.md", "nothing"), ("b.md", "beta here")]
    assert recall_at_k(labels, ranked, 4) == 1.0
    assert recall_at_k(labels, ranked, 3) == 0.5  # the beta passage is at rank 4
    assert recall_at_k(labels, ranked, 1) == 0.0
    assert reciprocal_rank(labels, ranked, 4) == 0.5
    assert reciprocal_rank(labels, ranked, 1) == 0.0
    # same phrase in the wrong document is not a hit
    assert recall_at_k([{"source": "a.md", "contains": "alpha"}], [("z.md", "alpha")], 5) == 0.0
    assert recall_at_k([], ranked, 4) is None and reciprocal_rank([], ranked, 4) is None


def test_percentiles():
    values = list(range(1, 101))
    assert (percentile(values, 50), percentile(values, 90), percentile(values, 95)) == (50, 90, 95)
    assert percentile([], 50) is None
    assert latency_summary([10.0, 20.0, 30.0])["max_ms"] == 30.0


def test_dataset_is_versioned_and_well_formed():
    from app.evaluation.runner import load_dataset

    data = load_dataset(DATASET)
    assert data["version"] == "1.0.0" and len(data["_sha256"]) == 64
    corpus = (DATASET.parent / data["corpus"]).resolve()
    for q in data["questions"]:
        assert q.get("answerable", True) is False or q["relevant"], q["id"]
        for label in q["relevant"]:
            text = (corpus / label["source"]).read_text(encoding="utf-8")
            assert label["contains"].lower() in text.lower(), (q["id"], label)


def test_benchmark_run_report_and_comparison(env, tmp_path):
    from app.evaluation.runner import compare_reports, run_evaluation, save_report

    report = run_evaluation(DATASET, label="baseline", reranker="none")
    m = report["metrics"]
    assert report["dataset"]["version"] == "1.0.0" and report["mode"] == "retrieval"
    assert m["questions"] == 16 and m["failed_requests"] == 0 and m["error_rate"] == 0.0
    assert 0.0 <= m["recall_at_k"] <= 1.0 and 0.0 <= m["mrr"] <= 1.0
    assert m["latency"]["p50_ms"] <= m["latency"]["p90_ms"] <= m["latency"]["p95_ms"]
    assert report["corpus"] == {"documents": 4, "chunks": report["corpus"]["chunks"]}
    assert report["config"]["reranker"] == "none" and "api_key" not in json.dumps(report).lower()
    # the production index is untouched by an evaluation run
    assert env.chroma.get_vector_store().count() == 0

    path = save_report(report, tmp_path)
    assert json.loads(path.read_text(encoding="utf-8"))["metrics"] == m

    candidate = run_evaluation(DATASET, label="candidate", reranker="cross_encoder", k=3)
    comparison = compare_reports(report, candidate)
    assert comparison["config_changes"]["reranker"] == ["none", "cross_encoder"]
    assert any("different K" in w for w in comparison["warnings"])
    assert {row["metric"] for row in comparison["metrics"]} >= {"recall_at_k", "mrr", "error_rate"}


def test_failed_requests_are_counted_not_dropped(env):
    from app.evaluation.runner import run_evaluation

    class Flaky:
        calls = 0

        def retrieve(self, question, where=None):
            Flaky.calls += 1
            if Flaky.calls % 2 == 0:
                raise RuntimeError("retriever down")
            return []

    report = run_evaluation(DATASET, pipeline=Flaky(), reranker="none")
    m = report["metrics"]
    assert m["failed_requests"] == 8 and m["error_rate"] == 0.5
    assert m["recall_at_k"] == 0.0  # failures are misses, not omissions
    assert sum(1 for r in report["per_question"] if r["error"] == "RuntimeError") == 8


def test_answer_evaluation_reports_quality_metrics(env):
    from app.evaluation.runner import run_evaluation

    env.state["answer"] = "Full-time employees receive 24 days [1]."
    report = run_evaluation(DATASET, answers=True, reranker="cross_encoder")
    m = report["metrics"]
    assert report["mode"] == "answers"
    for name in ("groundedness", "citation_correctness", "answer_relevance", "no_evidence_accuracy"):
        assert name in m
    assert m["groundedness"] == 1.0  # the fake verifier supports every pair
    answered = [r for r in report["per_question"] if not r.get("no_evidence")]
    assert answered and all("judgement" in r for r in answered)

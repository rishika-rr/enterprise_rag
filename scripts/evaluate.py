"""
Run a benchmark over a versioned evaluation dataset and write a JSON report.

    python -m scripts.evaluate                               # retrieval metrics, sample set
    python -m scripts.evaluate --answers --label release-1.1 # + answer quality (calls the LLM)
    python -m scripts.evaluate --reranker none --label no-rerank
    python -m scripts.evaluate --compare eval/reports/a.json eval/reports/b.json

Retrieval-only runs need no API key.
"""
import argparse
import json
import sys
from pathlib import Path

from app.evaluation.runner import compare_reports, run_evaluation, save_report


def _fmt(value) -> str:
    if value is None:
        return "-"
    return f"{value:.3f}" if isinstance(value, float) else str(value)


def print_summary(report: dict) -> None:
    m = report["metrics"]
    d = report["dataset"]
    print(f"dataset   {d['name']} v{d['version']}  ({d['questions']} questions, sha256 {d['sha256'][:12]})")
    print(f"config    reranker={report['config']['reranker']} mode={report['config']['retrieval_mode']}"
          f" top_k={report['config']['top_k']} rerank_top_k={report['config']['rerank_top_k']}")
    print(f"Recall@{m['k']}  {_fmt(m['recall_at_k'])}")
    print(f"MRR       {_fmt(m['mrr'])}")
    for name in ("groundedness", "citation_correctness", "answer_relevance", "no_evidence_accuracy"):
        if name in m:
            print(f"{name:<22}{_fmt(m[name])}")
    lat = m["latency"]
    print(f"latency   p50={_fmt(lat['p50_ms'])} ms  p90={_fmt(lat['p90_ms'])} ms  p95={_fmt(lat['p95_ms'])} ms")
    print(f"errors    {m['failed_requests']}/{m['questions']} (rate {_fmt(m['error_rate'])})")
    if m["unlabelled_questions"]:
        print(f"unlabelled questions (excluded from Recall/MRR): {', '.join(m['unlabelled_questions'])}")
    for row in report["per_question"]:
        if row["error"]:
            print(f"  FAILED {row['id']}: {row['error']}")


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--dataset", default="eval/datasets/sample-v1.json")
    parser.add_argument("--k", type=int, default=None, help="K for Recall@K / MRR (default RERANK_TOP_K)")
    parser.add_argument("--label", default="run", help="name of this configuration or release")
    parser.add_argument("--reranker", choices=["cross_encoder", "none"], default=None)
    parser.add_argument("--answers", action="store_true", help="also generate and assess answers")
    parser.add_argument("--out", default="eval/reports")
    parser.add_argument("--compare", nargs=2, metavar=("BASELINE", "CANDIDATE"))
    args = parser.parse_args(argv)

    if args.compare:
        baseline, candidate = (json.loads(Path(p).read_text(encoding="utf-8")) for p in args.compare)
        comparison = compare_reports(baseline, candidate)
        for warning in comparison["warnings"]:
            print(f"WARNING: {warning}")
        for key, (old, new) in comparison["config_changes"].items():
            print(f"config {key}: {old} -> {new}")
        print(f"baseline  = {comparison['baseline']}")
        print(f"candidate = {comparison['candidate']}")
        print(f"{'metric':<24}{'baseline':>14}{'candidate':>14}{'delta':>12}")
        for row in comparison["metrics"]:
            print(f"{row['metric']:<24}{_fmt(row['baseline']):>14}{_fmt(row['candidate']):>14}{_fmt(row['delta']):>12}")
        return 0

    report = run_evaluation(
        args.dataset, k=args.k, label=args.label, answers=args.answers, reranker=args.reranker
    )
    path = save_report(report, args.out)
    print_summary(report)
    print(f"report    {path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())

"""
Evaluation harness (SRS 3.2.5, UC-04).

Runs a versioned, labelled question set against the retrieval pipeline (and
optionally the full answer pipeline) and produces a JSON report that records
the dataset version and hash, the configuration, latency and error rate, so
that runs are reproducible and comparable (EVAL-007, DATA-007).

By default the dataset's own corpus is indexed into a throw-away index, so a
run never reads or modifies the production index or its documents.
"""
import hashlib
import json
import logging
import platform
import tempfile
import time
from datetime import datetime, timezone
from pathlib import Path

from app.config import settings
from app.evaluation.metrics import latency_summary, matches, mean, recall_at_k, reciprocal_rank

logger = logging.getLogger("rag.eval")

REPORT_VERSION = 1


def load_dataset(path: str | Path) -> dict:
    path = Path(path)
    data = json.loads(path.read_text(encoding="utf-8"))
    for field in ("name", "version", "questions"):
        if field not in data:
            raise ValueError(f"Dataset is missing '{field}'.")
    ids = [q.get("id") for q in data["questions"]]
    if None in ids or len(set(ids)) != len(ids):
        raise ValueError("Every question needs a unique 'id'.")
    data["_sha256"] = hashlib.sha256(path.read_bytes()).hexdigest()
    data["_path"] = str(path)
    return data


def config_snapshot(reranker: str) -> dict:
    """The settings that influence results. Never includes credentials."""
    return {
        "embedding_model": settings.EMBEDDING_MODEL,
        "reranker": reranker,
        "cross_encoder_model": settings.CROSS_ENCODER_MODEL if reranker == "cross_encoder" else None,
        "llm_model": settings.LLM_MODEL,
        "retrieval_mode": settings.RETRIEVAL_MODE,
        "top_k": settings.TOP_K,
        "rerank_top_k": settings.RERANK_TOP_K,
        "chunk_size": settings.CHUNK_SIZE,
        "chunk_overlap": settings.CHUNK_OVERLAP,
        "chunk_strategy": settings.CHUNK_STRATEGY,
        "min_relevance": settings.MIN_RELEVANCE,
        "max_context_chars": settings.MAX_CONTEXT_CHARS,
    }


def build_isolated_pipeline(corpus_dir: Path, index_dir: str, reranker: str):
    """Index `corpus_dir` into a private index and return (pipeline, corpus summary)."""
    from app.database.chroma import ChromaVectorStore
    from app.ingestion.indexer import Indexer
    from app.models.document import ChunkingStrategy
    from app.retrieval.bm25 import BM25Retriever
    from app.retrieval.dense import DenseRetriever
    from app.retrieval.hybrid import HybridRetriever
    from app.retrieval.pipeline import RetrievalPipeline
    from app.retrieval.reranker import build_reranker
    from app.retrieval.sparse_index import SparseIndex

    store = ChromaVectorStore(collection_name="evaluation", path=index_dir)
    sparse = SparseIndex()
    indexer = Indexer(vector_store=store, sparse_index=sparse)
    strategy = ChunkingStrategy(settings.CHUNK_STRATEGY)

    files = sorted(
        p for p in corpus_dir.rglob("*") if p.is_file() and indexer.loader.is_supported(p.name)
    )
    if not files:
        raise ValueError(f"No supported documents found in corpus {corpus_dir}.")
    for file in files:
        indexer.index_document(str(file), strategy)

    pipeline = RetrievalPipeline(
        retriever=HybridRetriever(DenseRetriever(store), BM25Retriever(sparse)),
        reranker=build_reranker(reranker),
    )
    return pipeline, {"documents": len(files), "chunks": store.count()}


def _judge_answer(generator, question: str, answer: str, context) -> dict | None:
    """LLM judgement of faithfulness / correctness / completeness (rubric: eval/README.md)."""
    from app.generation.citation_verifier import parse_verdict

    prompt = generator.prompt_builder.build_evaluation_prompt(question, answer, context)
    try:
        return parse_verdict(generator.llm_service.generate(prompt, temperature=0.0).text)
    except Exception as e:
        logger.warning("answer_judgement_failed error=%s", type(e).__name__)
        return None


def run_evaluation(
    dataset_path: str | Path,
    k: int | None = None,
    label: str = "run",
    answers: bool = False,
    reranker: str | None = None,
    pipeline=None,
) -> dict:
    """
    `pipeline`: evaluate an existing RetrievalPipeline (e.g. the live index)
    instead of indexing the dataset corpus.
    """
    dataset = load_dataset(dataset_path)
    k = k or settings.RERANK_TOP_K
    reranker = reranker or settings.RERANKER
    started = datetime.now(timezone.utc)

    with tempfile.TemporaryDirectory(ignore_cleanup_errors=True) as index_dir:
        corpus = None
        if pipeline is None:
            corpus_dir = Path(dataset.get("corpus", ""))
            if not corpus_dir.is_absolute():
                corpus_dir = Path(dataset["_path"]).parent / corpus_dir
            pipeline, corpus = build_isolated_pipeline(corpus_dir.resolve(), index_dir, reranker)

        generator = None
        if answers:
            from app.generation.generator import Generator

            generator = Generator(retrieval_pipeline=pipeline)

        per_question = []
        for item in dataset["questions"]:
            labels = item.get("relevant", [])
            answerable = item.get("answerable", True)
            row = {
                "id": item["id"],
                "question": item["question"],
                "answerable": answerable,
                "has_labels": bool(labels),
                "labelled": bool(labels) or not answerable,
                "error": None,
            }
            start = time.perf_counter()
            try:
                if generator is not None:
                    result = generator.generate(item["question"])
                    chunks = result.retrieved_chunks
                else:
                    result = None
                    chunks = pipeline.retrieve(item["question"])
            except Exception as e:  # reported and counted, never silently dropped
                row["error"] = type(e).__name__
                row["latency_ms"] = (time.perf_counter() - start) * 1000
                per_question.append(row)
                continue
            row["latency_ms"] = (time.perf_counter() - start) * 1000

            ranked = [(c.metadata.get("source", ""), c.document) for c in chunks]
            row["retrieved"] = [
                {"rank": i, "source": c.metadata.get("source"), "chunk_id": c.id,
                 "relevance": c.relevance}
                for i, c in enumerate(chunks, start=1)
            ]
            row["recall_at_k"] = recall_at_k(labels, ranked, k)
            row["reciprocal_rank"] = reciprocal_rank(labels, ranked, k)

            if result is not None:
                pairs = result.citation_results
                row["no_evidence"] = result.no_evidence
                row["no_evidence_correct"] = result.no_evidence == (not answerable)
                row["citations"] = len(pairs)
                # groundedness: share of (claim, citation) pairs the verifier found supported
                row["groundedness"] = (
                    sum(1 for p in pairs if p.supported) / len(pairs) if pairs else None
                )
                # citation correctness: the citation resolves to a retrieved passage
                # AND that passage is one of the labelled relevant passages
                row["citation_correctness"] = (
                    sum(
                        1 for p in pairs
                        if p.chunk is not None and any(
                            matches(lb, p.chunk.metadata.get("source", ""), p.chunk.document)
                            for lb in labels
                        )
                    ) / len(pairs)
                    if pairs and labels else None
                )
                if not result.no_evidence:
                    context = generator.context_builder.build(chunks)
                    verdict = _judge_answer(generator, item["question"], result.answer, context)
                    row["judgement"] = verdict
                    row["answer_relevant"] = (
                        verdict.get("correct") is True if verdict is not None else None
                    )
            per_question.append(row)

    ok = [r for r in per_question if r["error"] is None]

    def avg(key):
        return mean([r[key] for r in ok if r.get(key) is not None])

    # Failed requests count as misses in the retrieval aggregates.
    def retrieval_avg(key):
        return mean(
            [r[key] if r["error"] is None else 0.0 for r in per_question if r["has_labels"]]
        )

    metrics = {
        "k": k,
        "recall_at_k": retrieval_avg("recall_at_k"),
        "mrr": retrieval_avg("reciprocal_rank"),
        "questions": len(per_question),
        "failed_requests": len(per_question) - len(ok),
        "error_rate": (len(per_question) - len(ok)) / len(per_question) if per_question else None,
        "unlabelled_questions": [r["id"] for r in per_question if not r["labelled"]],
        "latency": latency_summary([r["latency_ms"] for r in ok]),
    }
    if answers:
        metrics.update(
            groundedness=avg("groundedness"),
            citation_correctness=avg("citation_correctness"),
            answer_relevance=mean(
                [1.0 if r["answer_relevant"] else 0.0 for r in ok
                 if r.get("answer_relevant") is not None]
            ),
            no_evidence_accuracy=mean(
                [1.0 if r["no_evidence_correct"] else 0.0 for r in ok if "no_evidence_correct" in r]
            ),
        )

    return {
        "report_version": REPORT_VERSION,
        "label": label,
        "started_at": started.isoformat(timespec="seconds"),
        "dataset": {
            "name": dataset["name"],
            "version": dataset["version"],
            "sha256": dataset["_sha256"],
            "questions": len(dataset["questions"]),
        },
        "corpus": corpus or {"source": "existing index"},
        "mode": "answers" if answers else "retrieval",
        "config": config_snapshot(reranker),
        "environment": {
            "python": platform.python_version(),
            "platform": platform.platform(),
            "processor": platform.processor(),
            "concurrency": 1,
            "cache_state": "cold index, models loaded on first question",
        },
        "metrics": metrics,
        "per_question": per_question,
    }


def save_report(report: dict, directory: str | Path = "eval/reports") -> Path:
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    stamp = report["started_at"].replace(":", "").replace("-", "")[:15]
    name = f"{report['dataset']['name']}-{report['dataset']['version']}__{report['label']}__{stamp}.json"
    path = directory / name
    path.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    return path


_COMPARED = [
    "recall_at_k", "mrr", "error_rate", "groundedness", "citation_correctness",
    "answer_relevance", "no_evidence_accuracy",
]


def compare_reports(baseline: dict, candidate: dict) -> dict:
    """Metric-by-metric difference between two reports (EVAL-007)."""
    warnings = []
    if baseline["dataset"]["sha256"] != candidate["dataset"]["sha256"]:
        warnings.append("The reports use different dataset files; metrics are not comparable.")
    if baseline["metrics"]["k"] != candidate["metrics"]["k"]:
        warnings.append("The reports use a different K.")

    rows = []
    for name in _COMPARED + ["latency.p50_ms", "latency.p90_ms", "latency.p95_ms"]:
        def pick(report):
            value = report["metrics"]
            for part in name.split("."):
                value = value.get(part) if isinstance(value, dict) else None
            return value

        a, b = pick(baseline), pick(candidate)
        if a is None and b is None:
            continue
        rows.append(
            {"metric": name, "baseline": a, "candidate": b,
             "delta": (b - a) if a is not None and b is not None else None}
        )

    changed = sorted(
        k for k in set(baseline["config"]) | set(candidate["config"])
        if baseline["config"].get(k) != candidate["config"].get(k)
    )
    return {
        "baseline": baseline["label"],
        "candidate": candidate["label"],
        "warnings": warnings,
        "config_changes": {
            k: [baseline["config"].get(k), candidate["config"].get(k)] for k in changed
        },
        "metrics": rows,
    }

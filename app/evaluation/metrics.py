"""Retrieval and latency metrics (EVAL-002, EVAL-004). Pure functions, no I/O."""
import math
from typing import Sequence


def matches(label: dict, source: str, text: str) -> bool:
    """A retrieved chunk satisfies a label when it comes from the labelled source
    document and contains the labelled key phrase (case-insensitive)."""
    if label.get("source") and label["source"] != source:
        return False
    phrase = label.get("contains")
    return phrase.lower() in text.lower() if phrase else True


def recall_at_k(labels: Sequence[dict], retrieved: Sequence[tuple[str, str]], k: int) -> float | None:
    """
    Fraction of the labelled relevant passages found in the top-K results.
    `retrieved` is a ranked list of (source, text). None when there are no labels.
    """
    if not labels:
        return None
    top = retrieved[:k]
    found = sum(1 for label in labels if any(matches(label, s, t) for s, t in top))
    return found / len(labels)


def reciprocal_rank(labels: Sequence[dict], retrieved: Sequence[tuple[str, str]], k: int) -> float | None:
    """1 / rank of the first relevant result in the top-K (0 when none). None without labels."""
    if not labels:
        return None
    for rank, (source, text) in enumerate(retrieved[:k], start=1):
        if any(matches(label, source, text) for label in labels):
            return 1.0 / rank
    return 0.0


def mean(values: Sequence[float]) -> float | None:
    return sum(values) / len(values) if values else None


def percentile(values: Sequence[float], pct: float) -> float | None:
    """Nearest-rank percentile (pct in 0-100)."""
    if not values:
        return None
    ordered = sorted(values)
    rank = max(1, math.ceil(pct / 100 * len(ordered)))
    return ordered[rank - 1]


def latency_summary(latencies_ms: Sequence[float]) -> dict:
    return {
        "p50_ms": percentile(latencies_ms, 50),
        "p90_ms": percentile(latencies_ms, 90),
        "p95_ms": percentile(latencies_ms, 95),
        "max_ms": max(latencies_ms) if latencies_ms else None,
    }

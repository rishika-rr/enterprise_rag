# Evaluation

## Dataset format

A dataset is one versioned JSON file in `eval/datasets/`:

```json
{
  "name": "sample",
  "version": "1.0.0",
  "corpus": "../corpus",
  "questions": [
    {"id": "leave-01", "question": "...",
     "relevant": [{"source": "leave-policy.md", "contains": "24 days of paid annual leave"}]},
    {"id": "none-01", "question": "...", "answerable": false, "relevant": []}
  ]
}
```

* `corpus`: directory (relative to the dataset file) that is indexed into a
  throw-away index for the run. The production index is never read or changed.
* `relevant`: the labelled relevant passages. A retrieved chunk satisfies a
  label when it comes from `source` and contains the `contains` phrase
  (case-insensitive). Labelling by phrase keeps labels valid when chunking
  parameters change.
* `answerable: false`: the corpus holds no answer; the expected outcome is the
  no-evidence response.

Change the file → bump `version`. Each report stores the version and the
SHA-256 of the dataset file, and `--compare` warns when two reports used
different files.

`sample-v1.json` is a synthetic set over four fictional policy documents. It
exercises the harness and catches regressions. It is **not** the organisation's
approved acceptance dataset, which still has to be supplied (SRS 2.5).

## Running

```bash
python -m scripts.evaluate --label baseline                    # retrieval metrics
python -m scripts.evaluate --answers --label release-1.1       # + answer quality (LLM calls)
python -m scripts.evaluate --reranker cross_encoder --k 3
python -m scripts.evaluate --compare eval/reports/A.json eval/reports/B.json
```

Reports are written to `eval/reports/<dataset>-<version>__<label>__<time>.json`
with the configuration, environment, per-question results, latency p50/p90/p95
and error rate. Failed requests are listed and counted as misses; questions
without labels are listed and excluded from Recall/MRR.

## Metrics and criteria

| Metric | Definition |
| --- | --- |
| Recall@K | Share of a question's labelled passages found in the top K results; mean over labelled questions. |
| MRR | 1 / rank of the first relevant result in the top K (0 if none); mean over labelled questions. |
| Groundedness | Share of (claim, citation) pairs in the answer that the citation verifier judged supported by the cited passage. |
| Citation correctness | Share of (claim, citation) pairs whose citation resolves to a supplied passage **and** that passage is a labelled relevant passage. |
| Answer relevance | Share of answers an LLM judge marked `correct` for the question given the supplied context (prompt: `app/prompts/answer_evaluation.txt`; the full `faithful / correct / complete` verdict is stored per question). |
| No-evidence accuracy | Share of questions where the no-evidence response was returned exactly when `answerable` is false. |
| Latency | End-to-end per question, one at a time, nearest-rank p50 / p90 / p95. |
| Error rate | Failed requests / all questions. |

Groundedness and answer relevance are LLM judgements. They are indicators for
comparing configurations, not ground truth; a human review of a sample (the
"evaluator agreement review" of EVAL-003) is still required for acceptance.

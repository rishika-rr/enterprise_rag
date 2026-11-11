import contextvars
import json
import logging
import re
from concurrent.futures import ThreadPoolExecutor

from app.config import settings
from app.generation.prompt_builder import PromptBuilder
from app.models.citation import CitationResult
from app.models.context import PromptContext
from app.services.llm_service import LLMService, get_llm_service

logger = logging.getLogger("rag.citations")

_JSON_OBJECT = re.compile(r"\{.*\}", re.DOTALL)


def parse_verdict(text: str) -> dict:
    """
    Parse the verifier's JSON, tolerating ```json fences and surrounding prose.
    Raises ValueError when no JSON object can be recovered.
    """
    cleaned = re.sub(r"^```(?:json)?|```$", "", text.strip(), flags=re.MULTILINE).strip()
    try:
        data = json.loads(cleaned)
    except json.JSONDecodeError:
        match = _JSON_OBJECT.search(cleaned)
        if not match:
            raise ValueError("no JSON object in verifier output")
        data = json.loads(match.group(0))
    if not isinstance(data, dict):
        raise ValueError("verifier output is not a JSON object")
    return data


class CitationVerifier:
    """
    Verifies each claim against the passage it cites.

    * Citations that do not resolve to a retrieved passage are reported as
      unsupported (never silently dropped) - GEN-004 / GEN-009.
    * Verification calls run concurrently and are capped so latency stays
      bounded (PERF-001).
    """

    def __init__(
        self,
        prompt_builder: PromptBuilder | None = None,
        llm_service_instance: LLMService | None = None,
        max_verifications: int | None = None,
        workers: int | None = None,
    ):
        self.prompt_builder = prompt_builder if prompt_builder else PromptBuilder()
        self._llm = llm_service_instance
        self.max_verifications = (
            max_verifications if max_verifications is not None else settings.MAX_VERIFICATIONS
        )
        self.workers = max(1, workers if workers is not None else settings.VERIFICATION_WORKERS)

    @property
    def llm_service(self) -> LLMService:
        return self._llm if self._llm is not None else get_llm_service()

    def _verify_pair(self, claim: str, citation: int, chunk) -> CitationResult:
        prompt = self.prompt_builder.build_verification_prompt(
            claim=claim, evidence=chunk.document
        )
        try:
            verdict = parse_verdict(
                self.llm_service.generate(prompt, temperature=0.0, max_tokens=200).text
            )
            supported = verdict.get("supported", False) is True
            reason = str(verdict.get("reason", ""))
        except ValueError:
            supported, reason = False, "Could not parse verifier response."
        except Exception as e:  # provider outage must not fail the whole answer
            logger.warning("verification_unavailable error=%s", type(e).__name__)
            supported, reason = False, "Verification unavailable."

        return CitationResult(
            claim=claim, citation=citation, supported=supported, reason=reason, chunk=chunk
        )

    def verify(
        self,
        claims: list[tuple[str, list[int]]],
        context: PromptContext,
    ) -> list[CitationResult]:
        results: list[CitationResult | None] = []
        jobs: list[tuple[int, str, int, object]] = []

        for claim, citations in claims:
            for citation in dict.fromkeys(citations):  # de-duplicate, keep order
                chunk = context.citations.get(citation)
                if chunk is None:
                    results.append(
                        CitationResult(
                            claim=claim,
                            citation=citation,
                            supported=False,
                            reason="Citation does not match any retrieved passage.",
                            chunk=None,
                        )
                    )
                elif len(jobs) < self.max_verifications:
                    jobs.append((len(results), claim, citation, chunk))
                    results.append(None)  # placeholder, filled below
                else:
                    results.append(
                        CitationResult(
                            claim=claim,
                            citation=citation,
                            supported=False,
                            reason="Not verified (verification limit reached).",
                            chunk=chunk,
                        )
                    )

        if jobs:
            with ThreadPoolExecutor(max_workers=self.workers) as pool:
                futures = [
                    pool.submit(contextvars.copy_context().run, self._verify_pair, j[1], j[2], j[3])
                    for j in jobs
                ]
                verified = [f.result() for f in futures]
            for (slot, *_), outcome in zip(jobs, verified):
                results[slot] = outcome

        return [r for r in results if r is not None]

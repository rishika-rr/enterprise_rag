import logging
import time

from app.config import settings
from app.generation.citation_verifier import CitationVerifier
from app.generation.context_builder import ContextBuilder
from app.generation.prompt_builder import PromptBuilder
from app.generation.quality_pipeline import QualityPipeline
from app.models.generation import GenerationResult
from app.retrieval.pipeline import RetrievalPipeline
from app.services.llm_service import LLMService, get_llm_service

logger = logging.getLogger("rag.generator")

NO_EVIDENCE_ANSWER = (
    "I couldn't find information in the indexed documents that answers "
    "this question."
)


class Generator:
    """
    Complete Retrieval-Augmented Generation pipeline.
    """

    def __init__(
        self,
        retrieval_pipeline: RetrievalPipeline | None = None,
        context_builder: ContextBuilder | None = None,
        prompt_builder: PromptBuilder | None = None,
        quality_pipeline: QualityPipeline | None = None,
        llm_service_instance: LLMService | None = None,
    ):
        self.llm_service = (
            llm_service_instance if llm_service_instance is not None else get_llm_service()
        )
        self.retrieval_pipeline = (
            retrieval_pipeline
            if retrieval_pipeline is not None
            else RetrievalPipeline()
        )
        self.context_builder = context_builder if context_builder is not None else ContextBuilder()
        self.prompt_builder = prompt_builder if prompt_builder is not None else PromptBuilder()
        self.quality_pipeline = (
            quality_pipeline
            if quality_pipeline is not None
            else QualityPipeline(
                citation_verifier=CitationVerifier(llm_service_instance=self.llm_service)
            )
        )

    @staticmethod
    def _relevant(chunks) -> list:
        """
        Drop passages the reranker judged irrelevant (below MIN_RELEVANCE), so
        they are neither sent to the model nor presented as sources. When
        reranking is off there is no relevance
        signal: every passage is kept and the model, which is instructed to
        say when the context is insufficient, decides.
        """
        return [
            c for c in chunks
            if c.relevance is None or c.relevance >= settings.MIN_RELEVANCE
        ]

    def generate(self, question: str, where: dict | None = None) -> GenerationResult:
        start = time.perf_counter()

        retrieved_chunks = self._relevant(self.retrieval_pipeline.retrieve(question, where=where))
        retrieval_ms = (time.perf_counter() - start) * 1000

        # RET-008 / GEN-009: no relevant evidence -> say so, do not call the LLM
        # and do not present irrelevant passages as sources.
        if not retrieved_chunks:
            logger.info("answer outcome=no_evidence retrieval_ms=%.0f", retrieval_ms)
            return GenerationResult(
                answer=NO_EVIDENCE_ANSWER,
                retrieved_chunks=[],
                confidence=0.0,
                citation_results=[],
                model=None,
                latency_ms=(time.perf_counter() - start) * 1000,
                no_evidence=True,
            )

        context = self.context_builder.build(retrieved_chunks)
        used_chunks = list(context.citations.values())  # exactly what the model saw
        system, prompt = self.prompt_builder.build_generation_messages(
            question=question, context=context
        )
        llm_response = self.llm_service.generate(prompt, system=system)
        generation_ms = llm_response.latency_ms

        quality_start = time.perf_counter()
        citation_results, confidence = self.quality_pipeline.evaluate(
            answer=llm_response.text,
            context=context,
            retrieved_chunks=used_chunks,
        )
        verification_ms = (time.perf_counter() - quality_start) * 1000
        total_ms = (time.perf_counter() - start) * 1000

        logger.info(
            "answer outcome=ok passages=%d citations=%d retrieval_ms=%.0f generation_ms=%.0f"
            " verification_ms=%.0f total_ms=%.0f",
            len(used_chunks), len(citation_results), retrieval_ms, generation_ms,
            verification_ms, total_ms,
        )

        return GenerationResult(
            answer=llm_response.text,
            retrieved_chunks=used_chunks,
            confidence=confidence,
            citation_results=citation_results,
            model=llm_response.model,
            latency_ms=total_ms,  # end-to-end (GEN-007)
        )

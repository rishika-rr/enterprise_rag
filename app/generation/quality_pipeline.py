from app.generation.citation_parser import CitationParser
from app.generation.citation_verifier import CitationVerifier
from app.generation.confidence_scorer import ConfidenceScorer
from app.models.context import PromptContext
from app.models.citation import CitationResult
from app.models.retrieval import RetrievalResult


class QualityPipeline:
    """
    Executes the complete answer quality pipeline.

    Steps:
    1. Parse citations from the generated answer.
    2. Verify each claim against its cited evidence.
    3. Compute an overall confidence score.
    """

    def __init__(
        self,
        citation_parser: CitationParser | None = None,
        citation_verifier: CitationVerifier | None = None,
        confidence_scorer: ConfidenceScorer | None = None,
    ):

        self.citation_parser = (
            citation_parser
            if citation_parser is not None
            else CitationParser()
        )

        self.citation_verifier = (
            citation_verifier
            if citation_verifier is not None
            else CitationVerifier()
        )

        self.confidence_scorer = (
            confidence_scorer
            if confidence_scorer is not None
            else ConfidenceScorer()
        )

    def evaluate(
        self,
        answer: str,
        context: PromptContext,
        retrieved_chunks: list[RetrievalResult],
    ) -> tuple[list[CitationResult], float]:
        """
        Executes the complete quality evaluation pipeline.
        """

        claims = self.citation_parser.parse(
            answer
        )

        citation_results = (
            self.citation_verifier.verify(
                claims,
                context,
            )
        )

        confidence = (
            self.confidence_scorer.score(
                retrieved_chunks,
                citation_results,
            )
        )

        return (
            citation_results,
            confidence,
        )
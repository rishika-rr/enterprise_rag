from typing import List

from app.config import settings
from app.models.context import PromptContext
from app.models.retrieval import RetrievalResult


class ContextBuilder:
    """
    Builds the formatted context supplied to the LLM from retrieved chunks.

    The total passage text is capped at MAX_CONTEXT_CHARS (PERF-006): passages
    are taken in rank order and the rest are dropped, so `citations` contains
    exactly the passages the model saw.
    """

    def __init__(self, max_chars: int | None = None):
        self.max_chars = max_chars if max_chars is not None else settings.MAX_CONTEXT_CHARS

    def build(
        self,
        results: List[RetrievalResult],
    ) -> PromptContext:

        formatted_chunks = []
        citations = {}
        used = 0

        for result in results:
            text = result.document
            remaining = self.max_chars - used
            if remaining <= 0:
                break
            if len(text) > remaining:
                if citations:  # keep whole passages; never drop the best one
                    break
                text = text[:remaining]
            used += len(text)

            index = len(citations) + 1
            metadata = result.metadata

            source = metadata.get("source", "Unknown")
            section = metadata.get("section", "N/A")
            page = metadata.get("page", "N/A")

            chunk = (
                f"### Context [{index}]\n\n"
                f"Source : {source}\n"
                f"Section: {section}\n"
                f"Page   : {page}\n\n"
                f"{text}"
            )

            formatted_chunks.append(chunk)
            citations[index] = result

        prompt = "\n\n".join(formatted_chunks)

        return PromptContext(
            formatted_context=prompt,
            formatted_chunks=formatted_chunks,
            citations=citations,
        )

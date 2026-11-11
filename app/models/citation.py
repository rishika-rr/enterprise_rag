from dataclasses import dataclass
from typing import Optional

from app.models.retrieval import RetrievalResult


@dataclass
class CitationResult:
    """
    Verification result for a single claim-citation pair.

    `chunk` is None when the citation number does not resolve to any
    retrieved passage (a fabricated / invalid citation) - GEN-004.
    """

    claim: str

    citation: int

    supported: bool

    reason: str

    chunk: Optional[RetrievalResult] = None

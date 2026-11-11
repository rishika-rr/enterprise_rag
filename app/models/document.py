from dataclasses import dataclass
from enum import Enum
from typing import Optional


class ChunkingStrategy(str, Enum):
    FIXED = "fixed"
    RECURSIVE = "recursive"
    SEMANTIC = "semantic"  # reserved - not implemented yet


@dataclass
class DocumentChunk:
    """
    Represents one chunk of a document along with all metadata
    required for retrieval, reranking, and citation.
    """

    # Unique identifier ("<doc_id>-<chunk_index>", deterministic)
    id: str

    # Actual chunk text
    text: str

    # Original document filename
    source: str

    # Page number, 1-based (None for txt/markdown/html)
    page: Optional[int]

    # Section heading if available
    section: Optional[str]

    # Position of this chunk inside the document
    chunk_index: int

    # Which chunking strategy created it
    strategy: ChunkingStrategy

    # Number of characters in the chunk
    char_count: int

    # Stable parent-document identifier (content hash)
    doc_id: str = ""


@dataclass
class IngestResult:
    """Outcome of ingesting one document (IF-004, ING-006)."""

    doc_id: str
    source: str
    status: str  # "indexed" | "duplicate"
    chunk_count: int

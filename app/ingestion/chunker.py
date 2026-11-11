from typing import List

from langchain_core.documents import Document

from langchain_text_splitters import (
    CharacterTextSplitter,
    RecursiveCharacterTextSplitter,
)

from app.config import settings

from app.models.document import ChunkingStrategy


class Chunker:
    """
    Handles different document chunking strategies.
    """

    def __init__(
        self,
        chunk_size: int | None = None,
        chunk_overlap: int | None = None,
    ):
        self.chunk_size = chunk_size if chunk_size is not None else settings.CHUNK_SIZE
        self.chunk_overlap = (
            chunk_overlap if chunk_overlap is not None else settings.CHUNK_OVERLAP
        )

    def fixed_chunk(
        self,
        documents: List[Document]
    ) -> List[Document]:
        """
        Splits documents into fixed-size chunks.
        """

        splitter = CharacterTextSplitter(
            separator="\n",
            chunk_size=self.chunk_size,
            chunk_overlap=self.chunk_overlap,
        )

        return splitter.split_documents(documents)

    def recursive_chunk(
        self,
        documents: List[Document]
    ) -> List[Document]:
        """
        Splits documents recursively while preserving
        semantic structure as much as possible.
        """

        splitter = RecursiveCharacterTextSplitter(
            chunk_size=self.chunk_size,
            chunk_overlap=self.chunk_overlap,
            separators=[
                "\n\n",
                "\n",
                ". ",
                " ",
                "",
            ],
        )

        return splitter.split_documents(documents)

    def chunk(
        self,
        documents: List[Document],
        strategy: ChunkingStrategy,
    ) -> List[Document]:
        """
        Dispatches to the selected chunking strategy.
        """

        if strategy == ChunkingStrategy.FIXED:
            return self.fixed_chunk(documents)

        elif strategy == ChunkingStrategy.RECURSIVE:
            return self.recursive_chunk(documents)

        raise ValueError(
            f"Unsupported chunking strategy: {strategy}"
        )
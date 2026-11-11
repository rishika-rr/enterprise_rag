import logging
import threading
from typing import List

from app.config import settings

logger = logging.getLogger("rag.embeddings")


class EmbeddingService:
    """
    Generates embeddings. The model (and the heavy torch import)
    is loaded lazily on the first request.
    """

    def __init__(self):
        self.model = None
        self._lock = threading.Lock()

    def _get_model(self):
        if self.model is None:
            with self._lock:
                if self.model is None:
                    from sentence_transformers import SentenceTransformer

                    logger.info("loading_embedding_model name=%s", settings.EMBEDDING_MODEL)
                    self.model = SentenceTransformer(settings.EMBEDDING_MODEL, device="cpu")
        return self.model

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        embeddings = self._get_model().encode(
            texts,
            batch_size=8,
            convert_to_numpy=True,
            normalize_embeddings=True,
            show_progress_bar=False,
        )
        return embeddings.tolist()

    def embed_query(self, query: str) -> List[float]:
        embedding = self._get_model().encode(
            query,
            convert_to_numpy=True,
            normalize_embeddings=True,
            show_progress_bar=False,
        )
        return embedding.tolist()


embedding_service = EmbeddingService()

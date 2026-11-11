import logging
import math
import threading
from typing import List

from app.config import settings

logger = logging.getLogger("rag.crossencoder")


def sigmoid(x: float) -> float:
    """Map a raw cross-encoder logit to a 0-1 relevance value."""
    if x >= 0:
        return 1.0 / (1.0 + math.exp(-x))
    e = math.exp(x)
    return e / (1.0 + e)


class CrossEncoderService:
    """
    Scores (query, passage) pairs. Loaded lazily on first use.
    Raw scores are logits (unbounded); use `sigmoid` for a 0-1 value.
    """

    def __init__(self):
        self.model = None
        self._lock = threading.Lock()

    def _get_model(self):
        if self.model is None:
            with self._lock:
                if self.model is None:
                    from sentence_transformers import CrossEncoder

                    logger.info("loading_cross_encoder name=%s", settings.CROSS_ENCODER_MODEL)
                    self.model = CrossEncoder(settings.CROSS_ENCODER_MODEL, device="cpu")
        return self.model

    def score(self, query: str, documents: List[str]) -> List[float]:
        pairs = [(query, document) for document in documents]
        return [float(s) for s in self._get_model().predict(pairs, show_progress_bar=False)]


cross_encoder_service = CrossEncoderService()

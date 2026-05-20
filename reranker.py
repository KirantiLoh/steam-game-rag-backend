"""Cross-encoder reranker for re-scoring retrieval results."""
import logging
import torch
from sentence_transformers import CrossEncoder
from typing import List, Tuple, Optional

logger = logging.getLogger("steamrec")


class Reranker:
    def __init__(
        self,
        model_name: str = "cross-encoder/ms-marco-MiniLM-L-6-v2",
        device: Optional[str] = None,
        default_k: int = 10,
        batch_size: int = 32,
        max_length: int = 512,
    ):
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        logger.info("Loading reranker %s on %s (batch_size=%d)", model_name, self.device, batch_size)
        self.model = CrossEncoder(
            model_name,
            device=self.device,
            max_length=max_length,
        )
        self.default_k = default_k
        self.batch_size = batch_size

    def rerank(
        self,
        query: str,
        candidates: List[Tuple[str, str]],
        k: Optional[int] = None,
    ) -> List[Tuple[str, float]]:
        k = k or self.default_k
        if not candidates:
            return []

        pairs = [(query, text) for _, text in candidates]
        scores = self.model.predict(pairs, show_progress_bar=False, batch_size=self.batch_size)

        scored = []
        for (app_id, text), score in zip(candidates, scores):
            scored.append((app_id, float(score)))

        scored.sort(key=lambda x: x[1], reverse=True)
        return scored[:k]

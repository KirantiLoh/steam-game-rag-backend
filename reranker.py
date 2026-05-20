"""Cross-encoder reranker for re-scoring retrieval results."""
import os
import torch
from sentence_transformers import CrossEncoder
from typing import List, Tuple, Optional


class Reranker:
    def __init__(
        self,
        model_name: str = "cross-encoder/ms-marco-MiniLM-L-6-v2",
        device: Optional[str] = None,
        default_k: int = 10,
    ):
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.model = CrossEncoder(model_name, device=self.device)
        self.default_k = default_k

    def rerank(
        self,
        query: str,
        candidates: List[Tuple[str, str]],
        k: Optional[int] = None,
    ) -> List[Tuple[str, float]]:
        k = k or self.default_k
        if not candidates:
            return []

        pairs = [(query, desc) for _, desc in candidates]
        scores = self.model.predict(pairs, show_progress_bar=False)

        scored = []
        for (app_id, desc), score in zip(candidates, scores):
            scored.append((app_id, float(score)))

        scored.sort(key=lambda x: x[1], reverse=True)
        return scored[:k]

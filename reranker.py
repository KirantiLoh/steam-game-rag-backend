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
        self.device = device or (
            "cuda" if torch.cuda.is_available() else "cpu")
        logger.info("Loading reranker %s on %s (batch_size=%d)",
                    model_name, self.device, batch_size)
        self.model = CrossEncoder(
            model_name,
            device=self.device,
            max_length=max_length,
        )
        self.default_k = default_k
        self.batch_size = batch_size
        self.max_length = max_length

    def prepare_text(self, name: str, short_desc: str, desc: str) -> str:
        """
        Intelligently prepare game text for cross-encoder reranking.

        Prioritizes high-signal fields within the 512 token budget:
        - Name: ~50 tokens (~200 chars) - Always included
        - Short description: ~150 tokens (~600 chars) - Highest signal-to-noise
        - Detailed description: ~250 tokens (~1000 chars) - Often verbose/marketing fluff

        Truncates at sentence boundaries to avoid cutting mid-sentence.

        Args:
            name: Game name
            short_desc: Short description (high relevance)
            desc: Detailed description (lower relevance per token)

        Returns:
            Optimally formatted text string for cross-encoder input
        """
        parts = []

        # Name: Always include, truncate only if absurdly long
        if name:
            if len(name) <= 200:
                parts.append(name)
            else:
                parts.append(name[:197] + "...")

        # Short description: Prioritize this (most informative per token)
        if short_desc:
            if len(short_desc) <= 600:
                parts.append(short_desc)
            else:
                # Truncate at sentence boundary
                truncated = short_desc[:600]
                last_period = truncated.rfind('.')
                if last_period > 400:  # Keep at least 400 chars
                    parts.append(truncated[:last_period + 1])
                else:
                    parts.append(truncated + "...")

        # Detailed description: Use remaining budget
        if desc:
            # Allocate less budget if we already have short_desc
            budget = 800 if not short_desc else 1000
            if len(desc) <= budget:
                parts.append(desc)
            else:
                # Truncate at sentence boundary
                truncated = desc[:budget]
                last_period = truncated.rfind('.')
                if last_period > budget * 0.7:  # Keep at least 70% if period found
                    parts.append(truncated[:last_period + 1])
                else:
                    parts.append(truncated + "...")

        return "\n\n".join(parts)

    def rerank(
        self,
        query: str,
        candidates: List[Tuple[str, str]],
        k: Optional[int] = None,
    ) -> List[Tuple[str, float]]:
        """
        Rerank candidates using cross-encoder model.

        Args:
            query: Search query string
            candidates: List of (app_id, text) tuples where text should be pre-formatted
                       using prepare_text() for optimal results
            k: Number of top results to return (default: self.default_k)

        Returns:
            List of (app_id, score) tuples sorted by score descending, length ≤ k
        """
        k = k or self.default_k
        if not candidates:
            return []

        # Truncate query if extremely long (rare but possible)
        query_truncated = query[:200] if len(query) > 200 else query

        # Ensure candidate texts fit within model's token budget
        # Each pair (query + document) should be < max_length tokens
        # Estimate: ~4 chars per token, so ~2048 chars for 512 tokens
        pairs = []
        for _, text in candidates:
            # Final safety truncation (text should already be optimized via prepare_text)
            if len(text) > 2048:
                text = text[:2048]
            pairs.append((query_truncated, text))

        # Batch inference with optimized batch size
        scores = self.model.predict(
            pairs, show_progress_bar=False, batch_size=self.batch_size)

        # Combine app_ids with scores
        scored = []
        for (app_id, _), score in zip(candidates, scores):
            scored.append((app_id, float(score)))

        # Sort by score descending and return top-k
        scored.sort(key=lambda x: x[1], reverse=True)
        return scored[:k]

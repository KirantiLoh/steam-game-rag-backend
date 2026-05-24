"""Optimized RRF fusion for production retrieval backends."""
import heapq
from collections import defaultdict
from typing import List, Tuple, Union

DocID = Union[str, int]


def reciprocal_rank_fusion(
    results_list: List[List[DocID]],
    k: int = 60,
    final_k: int = 10
) -> List[Tuple[DocID, float]]:
    """
    Optimized RRF fusion with heap-based top-K selection.

    Args:
        results_list: List of ranked doc ID lists from different retrievers
        k: Damping constant for RRF formula (default: 60)
        final_k: Number of final results to return (avoids full sort)

    Returns:
        List of (doc_id, rrf_score) tuples, sorted descending by score, length ≤ final_k

    Complexity: O(N * M + N * log(final_k)) where N=total unique docs, M=retriever count
    """
    # Use defaultdict for faster accumulation (avoids .get() overhead)
    scores: defaultdict[DocID, float] = defaultdict(float)

    # Pre-compute inverse ranks for common range (micro-optimization)
    # Only beneficial if you call RRF thousands of times/sec
    # inverse_ranks = [1.0 / (k + r) for r in range(1, max_len + 1)]

    # Accumulate scores: O(total candidates across all retrievers)
    for rank_list in results_list:
        for rank, doc_id in enumerate(rank_list, start=1):
            scores[doc_id] += 1.0 / (k + rank)

    # Use heap to get top-K without full sort: O(N * log(final_k)) vs O(N log N)
    # Only beneficial when final_k << total candidates (typical: 10 vs 200)
    if final_k < len(scores) * 0.3:  # Heuristic: heap wins when selecting <30% of candidates
        top_k = heapq.nlargest(final_k, scores.items(), key=lambda x: x[1])
    else:
        # Full sort is faster when retrieving most candidates
        top_k = sorted(scores.items(), key=lambda x: x[1], reverse=True)[
            :final_k]

    return top_k

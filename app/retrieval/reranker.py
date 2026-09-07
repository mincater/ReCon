"""Cross-Encoder reranking module for retrieved candidates.

Measures reranker latency and keeps the model cached as a singleton.
"""

from functools import lru_cache
import logging
import time
from typing import Any, Dict, List

from app.config import RERANKER_MODEL

logger = logging.getLogger(__name__)


@lru_cache()
def get_reranker():
    """Initializes and caches the CrossEncoder reranker model."""
    try:
        from sentence_transformers import CrossEncoder

        logger.info(f"Loading CrossEncoder model: {RERANKER_MODEL}")
        return CrossEncoder(RERANKER_MODEL)
    except ImportError as e:
        raise ImportError(
            "sentence-transformers is required for reranking. Install via requirements.txt."
        ) from e


def rerank(
    query: str,
    candidates: List[Dict[str, Any]],
    top_k: int = 5,
) -> List[Dict[str, Any]]:
    """Reranks candidates using the CrossEncoder model.

    Args:
        query: User input search query.
        candidates: Retrieved candidate documents [{"text": str, "score": float, "metadata": dict}].
        top_k: Number of top reranked items to return.

    Returns:
        List of reranked candidates: {"text": str, "score": float, "metadata": dict}
    """
    if not candidates or not query.strip():
        return candidates[:top_k]

    model = get_reranker()
    pairs = [[query, c.get("text", "")] for c in candidates]
    scores = model.predict(pairs)

    reranked = []
    for candidate, score in zip(candidates, scores):
        reranked.append(
            {
                "text": candidate.get("text", ""),
                "score": float(score),
                "metadata": candidate.get("metadata", {}),
            }
        )

    # Sort descending by cross-encoder relevance score
    reranked.sort(key=lambda x: x["score"], reverse=True)
    return reranked[:top_k]


def benchmark_reranker(query: str, candidates: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Measures CrossEncoder scoring latency on candidates.

    Args:
        query: Query string.
        candidates: List of candidate dicts.

    Returns:
        Dict: {"num_candidates": int, "elapsed_sec": float, "ms_per_candidate": float}
    """
    if not candidates:
        return {"num_candidates": 0, "elapsed_sec": 0.0, "ms_per_candidate": 0.0}

    start = time.perf_counter()
    _ = rerank(query=query, candidates=candidates, top_k=len(candidates))
    elapsed = time.perf_counter() - start

    ms_per_cand = (elapsed * 1000) / len(candidates)
    return {
        "num_candidates": len(candidates),
        "elapsed_sec": round(elapsed, 4),
        "ms_per_candidate": round(ms_per_cand, 2),
    }

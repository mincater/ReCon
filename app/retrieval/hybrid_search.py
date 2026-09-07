"""Hybrid retrieval combining dense vector search and BM25 keyword search using Reciprocal Rank Fusion (RRF)."""

from typing import Any, Dict, List, Optional

from app.retrieval.keyword_search import search_keywords
from app.retrieval.vector_search import search_vectors
from app.vectorstore.qdrant_store import DEFAULT_COLLECTION_NAME

RRF_K = 60


def hybrid_search(
    query: str,
    top_k: int = 5,
    collection_name: str = DEFAULT_COLLECTION_NAME,
) -> List[Dict[str, Any]]:
    """Performs hybrid retrieval fusing dense vector and BM25 keyword search results.

    Uses Reciprocal Rank Fusion (RRF): RRF_score = 1 / (60 + rank).

    Args:
        query: User input search query.
        top_k: Number of combined candidates to return.
        collection_name: Target Qdrant collection name.

    Returns:
        List of dicts: {"text": str, "score": float, "metadata": dict}
    """
    if not query or not query.strip():
        return []

    # Fetch top candidates from both channels
    vector_results = search_vectors(
        query_text=query,
        top_k=top_k * 2,
        collection_name=collection_name,
    )
    keyword_results = search_keywords(
        query=query,
        top_k=top_k * 2,
    )

    # Reciprocal Rank Fusion (RRF)
    doc_scores: Dict[str, float] = {}
    doc_map: Dict[str, Dict[str, Any]] = {}

    for rank, doc in enumerate(vector_results):
        text = doc.get("text", "")
        if text:
            doc_scores[text] = doc_scores.get(text, 0.0) + (1.0 / (RRF_K + rank + 1))
            doc_map[text] = doc

    for rank, doc in enumerate(keyword_results):
        text = doc.get("text", "")
        if text:
            doc_scores[text] = doc_scores.get(text, 0.0) + (1.0 / (RRF_K + rank + 1))
            if text not in doc_map:
                doc_map[text] = doc

    # Sort combined candidate list by descending RRF score
    sorted_texts = sorted(doc_scores.keys(), key=lambda t: doc_scores[t], reverse=True)

    fused_results: List[Dict[str, Any]] = []
    for text in sorted_texts[:top_k]:
        original_doc = doc_map[text]
        fused_results.append(
            {
                "text": text,
                "score": float(doc_scores[text]),
                "metadata": original_doc.get("metadata", {}),
            }
        )

    return fused_results

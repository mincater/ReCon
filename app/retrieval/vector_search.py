"""Vector search module querying vectorstore via public interfaces."""

from typing import Any, Dict, List, Optional

from app.embeddings.model import embed_query
from app.vectorstore.qdrant_store import DEFAULT_COLLECTION_NAME, query as qdrant_query


def search_vectors(
    query_text: str,
    top_k: int = 5,
    collection_name: str = DEFAULT_COLLECTION_NAME,
) -> List[Dict[str, Any]]:
    """Performs dense vector retrieval using Sentence Transformers and Qdrant.

    Args:
        query_text: User input query.
        top_k: Number of nearest chunks to retrieve.
        collection_name: Name of the target Qdrant collection.

    Returns:
        List of dicts: {"text": str, "score": float, "metadata": dict}
    """
    if not query_text or not query_text.strip():
        return []

    embedding = embed_query(query_text)
    return qdrant_query(
        embedding=embedding,
        top_k=top_k,
        collection_name=collection_name,
    )

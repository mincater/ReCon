"""Qdrant vector store implementation supporting both local memory and hosted Qdrant Cloud."""

from functools import lru_cache
import logging
from typing import Any, Dict, List, Optional
import uuid

from app.config import QDRANT_API_KEY, QDRANT_URL

logger = logging.getLogger(__name__)

DEFAULT_COLLECTION_NAME = "research_rag_collection"
DEFAULT_VECTOR_SIZE = 384  # Matches all-MiniLM-L6-v2 dimension


@lru_cache()
def get_qdrant_client():
    """Initializes and caches the QdrantClient instance.

    Connects to hosted Qdrant Cloud if QDRANT_URL is set; otherwise falls back
    to an in-memory client for local development and testing.
    """
    try:
        from qdrant_client import QdrantClient

        if QDRANT_URL and QDRANT_URL.strip():
            logger.info(f"Connecting to hosted Qdrant instance at {QDRANT_URL}")
            return QdrantClient(url=QDRANT_URL.strip(), api_key=QDRANT_API_KEY or None)
        else:
            logger.info("Connecting to in-memory local Qdrant instance (:memory:)")
            return QdrantClient(location=":memory:")
    except ImportError as e:
        raise ImportError(
            "qdrant-client is required for vector storage. Install via requirements.txt."
        ) from e


def reset_client_cache() -> None:
    """Clears the cached Qdrant client singleton (useful for testing or switching environments)."""
    get_qdrant_client.cache_clear()


def init_collection(
    collection_name: str = DEFAULT_COLLECTION_NAME,
    vector_size: int = DEFAULT_VECTOR_SIZE,
    client: Optional[Any] = None,
) -> None:
    """Ensures that the target Qdrant collection exists with the specified dimension and Cosine distance."""
    from qdrant_client.http import models

    q_client = client or get_qdrant_client()
    collections_response = q_client.get_collections()
    existing_names = [c.name for c in collections_response.collections]

    if collection_name not in existing_names:
        logger.info(f"Creating Qdrant collection '{collection_name}' (dim: {vector_size}, distance: Cosine)")
        q_client.create_collection(
            collection_name=collection_name,
            vectors_config=models.VectorParams(
                size=vector_size,
                distance=models.Distance.COSINE,
            ),
        )


def upsert_chunks(
    chunks: List[Dict[str, Any]],
    embeddings: List[List[float]],
    collection_name: str = DEFAULT_COLLECTION_NAME,
    client: Optional[Any] = None,
) -> None:
    """Upserts document chunks and their embedding vectors into Qdrant.

    Args:
        chunks: List of chunk dictionaries containing text, source, chunk_id, and metadata.
        embeddings: Corresponding embedding vectors (matching length with chunks).
        collection_name: Target collection name.
        client: Optional custom QdrantClient instance.
    """
    if not chunks or not embeddings:
        return

    if len(chunks) != len(embeddings):
        raise ValueError(
            f"Chunks count ({len(chunks)}) does not match embeddings count ({len(embeddings)})"
        )

    vector_size = len(embeddings[0])
    q_client = client or get_qdrant_client()
    init_collection(collection_name=collection_name, vector_size=vector_size, client=q_client)

    from qdrant_client.http import models

    points = []
    for chunk, emb in zip(chunks, embeddings):
        raw_id = chunk.get("chunk_id", "")
        # Derive deterministic UUID from chunk_id
        try:
            point_id = str(uuid.UUID(str(raw_id)))
        except (ValueError, AttributeError):
            point_id = str(uuid.uuid5(uuid.NAMESPACE_DNS, str(raw_id) or str(uuid.uuid4())))

        payload = {
            "text": chunk.get("text", ""),
            "source": chunk.get("source", ""),
            "chunk_id": raw_id,
            "metadata": chunk.get("metadata", {}),
        }

        points.append(
            models.PointStruct(
                id=point_id,
                vector=emb,
                payload=payload,
            )
        )

    q_client.upsert(collection_name=collection_name, points=points)
    logger.info(f"Successfully upserted {len(points)} chunks into collection '{collection_name}'")


def query(
    embedding: List[float],
    top_k: int = 5,
    collection_name: str = DEFAULT_COLLECTION_NAME,
    client: Optional[Any] = None,
) -> List[Dict[str, Any]]:
    """Queries Qdrant with an embedding vector and returns nearest chunks.

    Args:
        embedding: Dense query vector.
        top_k: Number of nearest neighbors to retrieve.
        collection_name: Target collection name.
        client: Optional custom QdrantClient instance.

    Returns:
        List of dicts: {"text": str, "score": float, "metadata": dict}
    """
    if not embedding:
        return []

    q_client = client or get_qdrant_client()
    init_collection(collection_name=collection_name, vector_size=len(embedding), client=q_client)

    if hasattr(q_client, "query_points"):
        response = q_client.query_points(
            collection_name=collection_name,
            query=embedding,
            limit=top_k,
        )
        hits = response.points
    else:
        hits = q_client.search(
            collection_name=collection_name,
            query_vector=embedding,
            limit=top_k,
        )

    formatted: List[Dict[str, Any]] = []
    for hit in hits:
        payload = hit.payload or {}
        # Ensure metadata contains source and page if present in payload
        meta = dict(payload.get("metadata", {}))
        if "source" not in meta and "source" in payload:
            meta["source"] = payload["source"]
        if "chunk_id" not in meta and "chunk_id" in payload:
            meta["chunk_id"] = payload["chunk_id"]

        formatted.append(
            {
                "text": payload.get("text", ""),
                "score": float(hit.score),
                "metadata": meta,
            }
        )

    return formatted

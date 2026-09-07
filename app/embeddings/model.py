"""Local Sentence Transformers embedding model wrapper.

Adheres strictly to the memory budget (~1GB RAM production limit) and avoids
re-instantiating the model on every function call via cached singleton loading.
"""

from functools import lru_cache
import logging
import time
import tracemalloc
from typing import Any, Dict, List

from app.config import EMBEDDING_MODEL

logger = logging.getLogger(__name__)


@lru_cache()
def get_model():
    """Loads and caches the SentenceTransformer embedding model once.

    Avoids re-instantiating the model on every function call.
    """
    try:
        from sentence_transformers import SentenceTransformer

        logger.info(f"Loading embedding model: {EMBEDDING_MODEL}")
        return SentenceTransformer(EMBEDDING_MODEL)
    except ImportError as e:
        raise ImportError(
            "sentence-transformers is required for embeddings. Install via requirements.txt."
        ) from e


def get_embedding_dimension() -> int:
    """Returns the vector dimensionality of the loaded model.

    For 'all-MiniLM-L6-v2', this is 384.
    """
    model = get_model()
    if hasattr(model, "get_embedding_dimension"):
        dim = model.get_embedding_dimension()
    else:
        dim = model.get_sentence_embedding_dimension()
    return int(dim) if dim is not None else 384


def embed_texts(texts: List[str], batch_size: int = 32) -> List[List[float]]:
    """Generates embedding vectors for a batch of text chunks.

    Args:
        texts: List of text strings to embed.
        batch_size: Batch size for model inference.

    Returns:
        List of embedding vectors (list of floats).
    """
    if not texts:
        return []

    model = get_model()
    embeddings = model.encode(
        texts,
        batch_size=batch_size,
        show_progress_bar=False,
        convert_to_numpy=True,
        normalize_embeddings=True,
    )
    return [vec.tolist() for vec in embeddings]


def embed_query(query: str) -> List[float]:
    """Generates an embedding vector for a single search query.

    Args:
        query: Query string to embed.

    Returns:
        Normalized embedding vector as list of floats.
    """
    model = get_model()
    vec = model.encode(
        query,
        convert_to_numpy=True,
        normalize_embeddings=True,
    )
    return vec.tolist()


def benchmark_embeddings(texts: List[str], batch_size: int = 32) -> Dict[str, Any]:
    """Benchmarks embedding generation measuring latency, throughput, and memory footprint.

    Args:
        texts: Sample chunks to encode.
        batch_size: Batch size for encoding.

    Returns:
        Dictionary with benchmark statistics (elapsed_sec, throughput_chunks_per_sec,
        ms_per_chunk, peak_memory_mb).
    """
    if not texts:
        return {"count": 0, "elapsed_sec": 0.0, "peak_memory_mb": 0.0}

    tracemalloc.start()
    start_time = time.perf_counter()

    vectors = embed_texts(texts, batch_size=batch_size)

    elapsed = time.perf_counter() - start_time
    current, peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()

    num_chunks = len(texts)
    throughput = num_chunks / elapsed if elapsed > 0 else 0.0
    ms_per_chunk = (elapsed * 1000) / num_chunks if num_chunks > 0 else 0.0
    peak_mb = peak / (1024 * 1024)

    return {
        "num_chunks": num_chunks,
        "vector_dimension": len(vectors[0]) if vectors else 0,
        "elapsed_sec": round(elapsed, 4),
        "ms_per_chunk": round(ms_per_chunk, 2),
        "throughput_chunks_per_sec": round(throughput, 2),
        "peak_memory_mb": round(peak_mb, 2),
    }

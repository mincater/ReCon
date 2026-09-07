"""BM25 keyword search module using rank-bm25."""

import logging
import re
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)


def tokenize(text: str) -> List[str]:
    """Simple alphanumeric tokenizer for BM25 indexing and querying."""
    if not text:
        return []
    return re.findall(r"\w+", text.lower())


class BM25Index:
    """Manages an in-memory BM25 index over documents."""

    def __init__(self, corpus: Optional[List[Dict[str, Any]]] = None) -> None:
        self.corpus: List[Dict[str, Any]] = corpus or []
        self._bm25 = None
        if self.corpus:
            self._build_index()

    def _build_index(self) -> None:
        try:
            from rank_bm25 import BM25Okapi

            tokenized_corpus = [tokenize(doc.get("text", "")) for doc in self.corpus]
            self._bm25 = BM25Okapi(tokenized_corpus)
            logger.info(f"Built BM25 index over {len(self.corpus)} document chunks.")
        except ImportError as e:
            raise ImportError(
                "rank-bm25 is required for keyword search. Install via requirements.txt."
            ) from e

    def update_corpus(self, corpus: List[Dict[str, Any]]) -> None:
        """Updates the corpus and rebuilds the BM25 index."""
        self.corpus = corpus
        self._build_index()

    def search(self, query: str, top_k: int = 5) -> List[Dict[str, Any]]:
        """Queries the BM25 index for keyword matches.

        Args:
            query: Query string.
            top_k: Number of matching documents to return.

        Returns:
            List of dicts: {"text": str, "score": float, "metadata": dict}
        """
        if not self.corpus or self._bm25 is None or not query.strip():
            return []

        tokens = tokenize(query)
        if not tokens:
            return []

        scores = self._bm25.get_scores(tokens)

        # Pair scores with docs
        scored = []
        for idx, score in enumerate(scores):
            if score > 0.0:  # Only return items with non-zero match score
                doc = self.corpus[idx]
                scored.append(
                    {
                        "text": doc.get("text", ""),
                        "score": float(score),
                        "metadata": doc.get("metadata", {}),
                    }
                )

        scored.sort(key=lambda x: x["score"], reverse=True)
        return scored[:top_k]


# Global singleton index for session/app state
_global_bm25_index: Optional[BM25Index] = None


def get_bm25_index() -> BM25Index:
    """Returns the global BM25Index instance."""
    global _global_bm25_index
    if _global_bm25_index is None:
        _global_bm25_index = BM25Index()
    return _global_bm25_index


def index_chunks(chunks: List[Dict[str, Any]]) -> None:
    """Indexes a list of chunks into the global BM25 index."""
    get_bm25_index().update_corpus(chunks)


def reset_bm25_index() -> None:
    """Clears the global BM25 index."""
    global _global_bm25_index
    _global_bm25_index = BM25Index()


def search_keywords(query: str, top_k: int = 5) -> List[Dict[str, Any]]:
    """Performs keyword search using the default BM25 index."""
    return get_bm25_index().search(query=query, top_k=top_k)

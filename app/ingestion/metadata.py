"""Metadata enrichment and validation for ingested chunks."""

from collections import Counter
import datetime
from typing import Any, Dict, List


def enrich_metadata(chunks: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Enriches chunk metadata with statistics, timestamps, and sequence indices.

    Args:
        chunks: List of chunk dictionaries.

    Returns:
        List of chunks with comprehensive metadata.
    """
    now_iso = datetime.datetime.now(datetime.timezone.utc).isoformat()

    # Count total chunks per source
    source_counts = Counter(chunk.get("source", "") for chunk in chunks)

    enriched: List[Dict[str, Any]] = []
    for chunk in chunks:
        item = dict(chunk)
        text = item.get("text", "")
        source = item.get("source", "")
        meta = dict(item.get("metadata", {}))

        words = text.split()
        word_count = len(words)
        char_count = len(text)
        # Approximate tokens: ~1.3 tokens per word in English
        token_estimate = max(1, int(round(word_count * 1.3)))

        meta["char_count"] = char_count
        meta["word_count"] = word_count
        meta["token_estimate"] = token_estimate
        meta["total_chunks_in_source"] = source_counts.get(source, 1)
        meta["ingested_at"] = now_iso

        item["metadata"] = meta
        enriched.append(item)

    return enriched

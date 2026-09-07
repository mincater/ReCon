"""Web search integration via Tavily with session caching and typed errors."""

import logging
from typing import Any, Dict, List

from app.config import TAVILY_API_KEY

logger = logging.getLogger(__name__)

# Session cache to avoid burning free Tavily credits on repeated queries (~1000/month limit)
_SEARCH_CACHE: Dict[str, List[Dict[str, str]]] = {}


class SearchProviderError(Exception):
    """Exception raised when an external search provider fails."""


def clear_search_cache() -> None:
    """Clears the in-memory web search session cache."""
    global _SEARCH_CACHE
    _SEARCH_CACHE.clear()


def get_search_cache_size() -> int:
    """Returns number of cached search queries."""
    return len(_SEARCH_CACHE)


def search(query: str, max_results: int = 5) -> List[Dict[str, str]]:
    """Performs web search using Tavily.

    Returns structured results directly without re-parsing HTML.
    Caches identical queries within the session.

    Args:
        query: Search query string.
        max_results: Maximum number of search results to return.

    Returns:
        List of dicts: {"title": str, "url": str, "snippet": str}

    Raises:
        SearchProviderError: If the Tavily API fails or key is missing.
    """
    if not query or not query.strip():
        return []

    cache_key = f"{query.strip().lower()}::{max_results}"
    if cache_key in _SEARCH_CACHE:
        logger.info(f"Returning cached search results for query: '{query}'")
        return _SEARCH_CACHE[cache_key]

    import os
    from app.config import get_config_val

    key = os.environ.get("TAVILY_API_KEY") or TAVILY_API_KEY or get_config_val("TAVILY_API_KEY")
    if not key:
        raise SearchProviderError("TAVILY_API_KEY is not configured.")

    try:
        from tavily import TavilyClient

        client = TavilyClient(api_key=key)
        response = client.search(
            query=query,
            max_results=max_results,
            include_answer=False,
            include_raw_content=False,
        )
        raw_results = response.get("results", [])

        formatted_results: List[Dict[str, str]] = []
        for r in raw_results:
            formatted_results.append(
                {
                    "title": r.get("title", ""),
                    "url": r.get("url", ""),
                    "snippet": r.get("content", ""),
                }
            )

        _SEARCH_CACHE[cache_key] = formatted_results
        return formatted_results
    except ImportError as e:
        raise SearchProviderError(
            "tavily-python is required for web search. Install via requirements.txt."
        ) from e
    except Exception as e:
        logger.error(f"Tavily search invocation failed: {e}")
        raise SearchProviderError(f"Tavily search failed: {e}") from e

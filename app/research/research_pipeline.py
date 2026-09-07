"""Research pipeline orchestrating query routing, local retrieval, and live web search."""

import logging
from typing import Any, Dict, List

from app.query.query_router import route
from app.research.web_search import search as tavily_search
from app.retrieval.hybrid_search import hybrid_search

logger = logging.getLogger(__name__)


def run_research(query: str, top_k: int = 5) -> Dict[str, Any]:
    """Runs a full research step depending on router classification.

    Args:
        query: User input query.
        top_k: Number of results to retrieve per branch.

    Returns:
        Dict: {
            "route": str ('local', 'web', or 'both'),
            "chunks": List[Dict[str, Any]],  # Unified list for generator
            "local_chunks": List[Dict[str, Any]],
            "web_results": List[Dict[str, Any]],
        }
    """
    decision = route(query)
    local_chunks: List[Dict[str, Any]] = []
    web_results: List[Dict[str, Any]] = []
    combined_chunks: List[Dict[str, Any]] = []

    if decision in ("local", "both"):
        try:
            local_chunks = hybrid_search(query=query, top_k=top_k)
            combined_chunks.extend(local_chunks)
        except Exception as e:
            logger.error(f"Local retrieval error during research: {e}")

    if decision in ("web", "both"):
        try:
            web_results = tavily_search(query=query, max_results=top_k)
            # Format web results into chunk structure for answer generator
            for r in web_results:
                combined_chunks.append(
                    {
                        "text": f"{r.get('title', '')}\n{r.get('snippet', '')}",
                        "source": r.get("url", "Web Search"),
                        "chunk_id": f"web_{hash(r.get('url', ''))}",
                        "metadata": {
                            "source": r.get("url", "Web Search"),
                            "title": r.get("title", ""),
                            "url": r.get("url", ""),
                            "is_web": True,
                            "page": None,
                        },
                    }
                )
        except Exception as e:
            logger.warning(f"Web search skipped or failed: {e}")

    return {
        "route": decision,
        "chunks": combined_chunks,
        "local_chunks": local_chunks,
        "web_results": web_results,
    }

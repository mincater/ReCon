"""Query rewriter for clarifying and expanding user queries for retrieval."""

from typing import List


def rewrite_query(query: str) -> List[str]:
    """Expands or reformulates a user query into search variations.

    Args:
        query: The raw query from the user.

    Returns:
        List of reformulated query strings.
    """
    stripped = query.strip()
    if not stripped:
        return []
    # Base version returns stripped original query; LLM-based multi-query expansion
    # can be activated in Phase 7.
    return [stripped]

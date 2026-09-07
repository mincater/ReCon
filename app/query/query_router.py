"""Query router determining whether a query requires local docs, live web search, or both."""

import re

# Patterns indicating need for real-time / current / external web information
WEB_KEYWORDS = [
    r"\blatest\b",
    r"\brecent\b",
    r"\btoday\b",
    r"\bcurrent\b",
    r"\bnews\b",
    r"\bweather\b",
    r"\bstock\b",
    r"\bmarket trends?\b",
    r"\bprice\b",
    r"\b202[5-9]\b",
    r"\bwho is currently\b",
    r"\bupcoming\b",
]

# Patterns indicating specific document or internal corpus reference
LOCAL_KEYWORDS = [
    r"\bdocument\b",
    r"\bfile\b",
    r"\buploaded\b",
    r"\bpdf\b",
    r"\bdocx\b",
    r"\bcsv\b",
    r"\bxlsx\b",
    r"\bsection\b",
    r"\bpage\b",
    r"\bcontract\b",
    r"\breport\b",
    r"\btable\b",
    r"\binternal\b",
    r"\bthis text\b",
    r"\babove context\b",
]


def route(query: str) -> str:
    """Routes a query to 'local', 'web', or 'both'.

    Args:
        query: User input query string.

    Returns:
        One of 'local', 'web', or 'both'.
    """
    if not query or not query.strip():
        return "local"

    lower_query = query.lower()

    has_web_intent = any(re.search(pat, lower_query) is not None for pat in WEB_KEYWORDS)
    has_local_intent = any(re.search(pat, lower_query) is not None for pat in LOCAL_KEYWORDS)

    if has_web_intent and has_local_intent:
        return "both"
    elif has_web_intent:
        return "web"
    else:
        # Default to local documents
        return "local"

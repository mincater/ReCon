"""Answer generator combining retrieved chunks with prompt engineering and LLM invocation."""

from typing import Any, Dict, List

from app.generation.llm import generate
from app.generation.prompts import RAG_SYSTEM_PROMPT, build_rag_prompt


def generate_answer(query: str, chunks: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Generates an answer with citations from retrieved chunks.

    Args:
        query: User input query.
        chunks: List of retrieved context chunks.

    Returns:
        Dict: {"answer": str, "citations": list[{"source": str, "page": int | None}]}
    """
    if not chunks:
        return {
            "answer": "No relevant context documents were found to answer your question.",
            "citations": [],
        }

    # Deduplicate and extract citations
    citations: List[Dict[str, Any]] = []
    seen = set()
    for chunk in chunks:
        meta = chunk.get("metadata", {})
        source = chunk.get("source") or meta.get("source", "Unknown")
        page = meta.get("page")
        cite_key = (source, page)
        if cite_key not in seen:
            seen.add(cite_key)
            citations.append({"source": source, "page": page})

    prompt = build_rag_prompt(query=query, context_chunks=chunks)
    answer_text = generate(prompt=prompt, system=RAG_SYSTEM_PROMPT)

    return {
        "answer": answer_text,
        "citations": citations,
    }

"""Prompt templates and formatting utilities for RAG generation."""

RAG_SYSTEM_PROMPT = """You are an expert research assistant.
Answer the user query accurately and comprehensively, basing your answer ONLY on the provided context chunks.
For each statement or fact you derive from the context, include in-text citations referencing the source document and page number where available, formatted like [Source: <source_name>, Page: <page_number>].
If the context does not contain enough information to answer the question, clearly state what is missing instead of hallucinating."""


def build_rag_prompt(query: str, context_chunks: list[dict]) -> str:
    """Constructs prompt containing query and formatted context chunks."""
    formatted_context_blocks = []
    for i, chunk in enumerate(context_chunks):
        meta = chunk.get("metadata", {})
        source = chunk.get("source", meta.get("source", "Unknown"))
        page = meta.get("page")
        page_str = f", Page: {page}" if page is not None else ""
        text = chunk.get("text", "")
        formatted_context_blocks.append(
            f"--- Document [{i+1}] (Source: {source}{page_str}) ---\n{text}"
        )

    context_str = "\n\n".join(formatted_context_blocks)
    return (
        f"Context Information:\n"
        f"{context_str}\n\n"
        f"User Query: {query}\n\n"
        f"Please provide a well-cited answer based on the context above."
    )

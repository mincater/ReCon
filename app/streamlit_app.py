"""research-rag Streamlit Application Entrypoint.

Modular RAG with Hybrid Search (Qdrant + BM25), Local Embeddings,
Cross-Encoder Reranking, Fallback Generation (Groq -> Gemini), and Tavily Web Search.
"""

import os
import tempfile
from typing import Any, Dict, List
import streamlit as st

from app.config import get_settings
from app.embeddings.model import embed_texts
from app.generation.answer_generator import generate_answer
from app.ingestion.pipeline import ingest_document
from app.research.research_pipeline import run_research
from app.retrieval.hybrid_search import hybrid_search
from app.retrieval.keyword_search import index_chunks
from app.retrieval.reranker import rerank
from app.ui.chat import render_chat_history
from app.ui.citations import render_citations
from app.ui.sidebar import render_sidebar
from app.ui.upload import render_upload_widget
from app.vectorstore.qdrant_store import upsert_chunks


def initialize_session_state() -> None:
    """Sets up persistent conversation and document session state."""
    if "messages" not in st.session_state:
        st.session_state.messages = []
    if "all_chunks" not in st.session_state:
        st.session_state.all_chunks = []
    if "processed_files" not in st.session_state:
        st.session_state.processed_files = set()


def handle_file_uploads(uploaded_files: List[Any], settings: Any) -> None:
    """Processes uploaded files, chunks them, computes embeddings, and indexes them."""
    if not uploaded_files:
        return

    new_files = [f for f in uploaded_files if f.name not in st.session_state.processed_files]
    if not new_files:
        return

    with st.status("Processing and indexing documents...", expanded=True) as status:
        for file_obj in new_files:
            st.write(f"Ingesting `{file_obj.name}`...")
            suffix = os.path.splitext(file_obj.name)[1].lower()

            # Write file to temporary path for processing
            with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp_file:
                tmp_file.write(file_obj.getvalue())
                tmp_path = tmp_file.name

            try:
                # 1. Ingest, clean, and chunk
                chunks = ingest_document(
                    tmp_path,
                    chunk_size=settings.chunk_size,
                    chunk_overlap=settings.chunk_overlap,
                )
                # Override temp filename with original uploaded filename in metadata
                for c in chunks:
                    c["source"] = file_obj.name
                    if "metadata" in c:
                        c["metadata"]["source"] = file_obj.name

                st.session_state.all_chunks.extend(chunks)

                # 2. Embed locally
                chunk_texts = [c["text"] for c in chunks]
                embeddings = embed_texts(chunk_texts)

                # 3. Index in Qdrant and BM25
                upsert_chunks(chunks=chunks, embeddings=embeddings)
                index_chunks(st.session_state.all_chunks)

                st.session_state.processed_files.add(file_obj.name)
                st.write(f"✅ `{file_obj.name}`: {len(chunks)} chunks embedded & stored.")
            finally:
                if os.path.exists(tmp_path):
                    os.remove(tmp_path)

        status.update(
            label=f"Indexed {len(st.session_state.processed_files)} document(s) ({len(st.session_state.all_chunks)} total chunks)!",
            state="complete",
            expanded=False,
        )


def main() -> None:
    """Main application loop."""
    settings = get_settings()

    st.set_page_config(
        page_title="ReCon",
        page_icon="🔍",
        layout="wide",
        initial_sidebar_state="expanded",
    )

    initialize_session_state()

    # Render Sidebar Controls
    ui_controls = render_sidebar(settings)
    top_k = ui_controls.get("top_k", 5)
    use_reranker = ui_controls.get("use_reranker", False)
    route_mode = ui_controls.get("route_mode", "Auto (Router)")

    # Main Application Header
    st.title("ReCon")
    st.caption("Your Intelligent Research Assistant.")

    # Document Upload Section
    uploaded_files = render_upload_widget()
    if uploaded_files:
        handle_file_uploads(uploaded_files, settings)

    # Display Chat History
    render_chat_history(st.session_state.messages)

    # Chat Input
    if user_prompt := st.chat_input("Ask a question about your documents or research topic..."):
        # Record user query
        st.session_state.messages.append({"role": "user", "content": user_prompt})
        with st.chat_message("user"):
            st.markdown(user_prompt)

        # Assistant Processing
        with st.chat_message("assistant"):
            with st.spinner("Searching and synthesizing answer..."):
                # Determine routing mode
                if route_mode == "Local Docs Only":
                    route_decision = "local"
                    candidate_chunks = hybrid_search(user_prompt, top_k=top_k)
                elif route_mode == "Web Search Only":
                    route_decision = "web"
                    research_res = run_research(user_prompt, top_k=top_k)
                    candidate_chunks = research_res.get("chunks", [])
                elif route_mode == "Hybrid (Both)":
                    route_decision = "both"
                    research_res = run_research(user_prompt, top_k=top_k)
                    candidate_chunks = research_res.get("chunks", [])
                else:
                    # Auto (Router)
                    research_res = run_research(user_prompt, top_k=top_k)
                    route_decision = research_res.get("route", "local")
                    candidate_chunks = research_res.get("chunks", [])

                # Optional Reranker step
                if use_reranker and candidate_chunks:
                    st.caption(f"⚡ Reranking {len(candidate_chunks)} candidates with Cross-Encoder...")
                    candidate_chunks = rerank(user_prompt, candidate_chunks, top_k=top_k)

                from app.generation.llm import LLMProviderError

                try:
                    # Generate Answer with Citations
                    answer_result = generate_answer(query=user_prompt, chunks=candidate_chunks)
                    answer_text = answer_result.get("answer", "")
                    citations = answer_result.get("citations", [])

                    # Render Answer
                    st.markdown(answer_text)
                    if citations:
                        render_citations(citations)

                    # Save assistant response in history
                    st.session_state.messages.append(
                        {
                            "role": "assistant",
                            "content": answer_text,
                            "citations": citations,
                        }
                    )
                except LLMProviderError as e:
                    st.warning(
                        "⚠️ **LLM API Key Required**: Groq and Gemini calls failed (API keys not configured). "
                        "Please open **⚙️ Settings & Keys** in the sidebar and enter your **Groq** or **Gemini** API key."
                    )
                    if candidate_chunks:
                        st.info(f"Retrieved {len(candidate_chunks)} matching document chunks for your question:")
                        for idx, c in enumerate(candidate_chunks[:top_k]):
                            st.markdown(f"> **Chunk {idx + 1}** (`{c.get('source', 'Doc')}`):\n> {c.get('text', '')}")


if __name__ == "__main__":
    main()

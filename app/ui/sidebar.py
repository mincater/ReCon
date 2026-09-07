"""Sidebar UI component for status, controls, and configuration display."""

import os
from typing import Any, Dict


def render_sidebar(settings: Any) -> Dict[str, Any]:
    """Renders Streamlit sidebar controls and returns user selections.

    Args:
        settings: Application settings instance.

    Returns:
        Dictionary of user-adjusted UI controls:
            - top_k: int
            - use_reranker: bool
            - route_mode: str ('auto', 'local', 'web', 'both')
    """
    try:
        import streamlit as st

        with st.sidebar:
            st.header("⚙️ Configuration")

            # Model Selection
            groq_model_options = [
                "openai/gpt-oss-20b",
                "llama-3.3-70b-versatile",
                "llama-3.1-8b-instant",
            ]
            active_model = os.environ.get("LLM_MODEL_PRIMARY") or settings.llm_model_primary or "openai/gpt-oss-20b"
            if active_model not in groq_model_options:
                groq_model_options.insert(0, active_model)
            chosen_model = st.selectbox(
                "Primary LLM Model",
                options=groq_model_options,
                index=groq_model_options.index(active_model) if active_model in groq_model_options else 0,
                help="Model used for primary generation",
            )
            if chosen_model:
                os.environ["LLM_MODEL_PRIMARY"] = chosen_model

            display_primary = os.environ.get("LLM_MODEL_PRIMARY") or settings.llm_model_primary
            display_fallback = os.environ.get("LLM_MODEL_FALLBACK") or settings.llm_model_fallback
            st.markdown(f"**Primary LLM**: `{settings.llm_provider_primary}` ({display_primary})")
            st.markdown(f"**Fallback LLM**: `{settings.llm_provider_fallback}` ({display_fallback})")
            st.markdown(f"**Embeddings**: `{settings.embedding_model}`")

            st.divider()
            st.subheader("🔍 Retrieval Options")
            top_k = st.slider("Top-K Candidates", min_value=1, max_value=15, value=5)
            use_reranker = st.checkbox("Enable Cross-Encoder Reranker", value=False)

            route_mode = st.selectbox(
                "Routing Mode",
                options=["Auto (Router)", "Local Docs Only", "Web Search Only", "Hybrid (Both)"],
                index=0,
            )

            st.divider()
            if settings.qdrant_url:
                st.success("🟢 Qdrant Cloud Connected")
            else:
                st.info("💻 Local In-Memory Qdrant Active")

            return {
                "top_k": top_k,
                "use_reranker": use_reranker,
                "route_mode": route_mode,
            }
    except Exception:
        return {"top_k": 5, "use_reranker": False, "route_mode": "Auto (Router)"}

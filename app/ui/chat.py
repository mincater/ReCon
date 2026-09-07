"""Chat interaction interface component."""

from typing import Any, Dict, List


def render_chat_history(messages: List[Dict[str, Any]]) -> None:
    """Renders conversation history in Streamlit with citations.

    Args:
        messages: List of message dictionaries with 'role', 'content', and optional 'citations'.
    """
    try:
        import streamlit as st

        from app.ui.citations import render_citations

        for msg in messages:
            role = msg.get("role", "assistant")
            content = msg.get("content", "")
            citations = msg.get("citations", [])

            with st.chat_message(role):
                st.markdown(content)
                if citations:
                    render_citations(citations)
    except Exception:
        pass

"""Citations and sources presentation UI component."""

from typing import Any, Dict, List


def render_citations(citations: List[Dict[str, Any]]) -> None:
    """Renders formatted citation cards and source previews in Streamlit.

    Args:
        citations: List of citation dictionaries {"source": str, "page": int | None}.
    """
    if not citations:
        return

    try:
        import streamlit as st

        with st.expander(f"📚 Verified Sources & Citations ({len(citations)})", expanded=False):
            for idx, cite in enumerate(citations):
                source = cite.get("source", "Unknown")
                page = cite.get("page")
                page_info = f" (Page {page})" if page is not None else ""

                # Distinguish web URLs from local files
                if str(source).startswith("http://") or str(source).startswith("https://"):
                    st.markdown(f"**[{idx + 1}]** 🌐 [{source}]({source})")
                else:
                    import os
                    basename = os.path.basename(str(source))
                    st.markdown(f"**[{idx + 1}]** 📄 `{basename}`{page_info}")
    except Exception:
        pass

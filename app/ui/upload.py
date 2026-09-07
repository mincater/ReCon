"""Document upload UI component for file ingestion."""

from typing import Any, List, Optional


def render_upload_widget() -> Optional[List[Any]]:
    """Renders the file uploader widget in Streamlit.

    Returns:
        List of uploaded file objects or None.
    """
    try:
        import streamlit as st

        uploaded_files = st.file_uploader(
            "Upload Documents (PDF, DOCX, TXT, CSV, XLSX)",
            type=["pdf", "docx", "txt", "csv", "xlsx"],
            accept_multiple_files=True,
            help="Uploaded documents are automatically chunked, embedded locally, and stored in Qdrant.",
        )
        return uploaded_files
    except Exception:
        return None

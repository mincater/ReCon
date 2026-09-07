"""PDF document loader using PyMuPDF (fitz)."""

import os
from typing import Any, Dict, List


def load(file_path: str) -> List[Dict[str, Any]]:
    """Loads a PDF document and extracts text per page.

    Args:
        file_path: Path to the PDF file.

    Returns:
        List of dicts with {"text": str, "source": str, "page": int}.

    Raises:
        FileNotFoundError: If the file does not exist.
        ImportError: If pymupdf is not installed.
    """
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"PDF file not found: {file_path}")

    try:
        try:
            import pymupdf as fitz
        except ImportError:
            import fitz
    except ImportError as e:
        raise ImportError(
            "pymupdf is required for loading PDF documents. Install via requirements.txt."
        ) from e

    documents: List[Dict[str, Any]] = []
    doc = fitz.open(file_path)
    try:
        total_pages = len(doc)
        for page_num in range(total_pages):
            page = doc[page_num]
            text = page.get_text().strip()
            # Retain page even if empty to keep page index alignment, or record text
            documents.append(
                {
                    "text": text,
                    "source": file_path,
                    "page": page_num + 1,
                }
            )
    finally:
        doc.close()

    return documents

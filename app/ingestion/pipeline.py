"""Ingestion pipeline coordinating loaders, cleaner, chunker, and metadata enrichment."""

import os
from typing import Any, Dict, List

from app.config import CHUNK_OVERLAP, CHUNK_SIZE
from app.ingestion.cleaner import clean_text
from app.ingestion.chunker import create_chunks
from app.ingestion.metadata import enrich_metadata
from app.ingestion.loaders import (
    csv_loader,
    docx_loader,
    pdf_loader,
    txt_loader,
    web_loader,
    xlsx_loader,
)

LOADER_MAPPING = {
    ".pdf": pdf_loader.load,
    ".docx": docx_loader.load,
    ".txt": txt_loader.load,
    ".csv": csv_loader.load,
    ".xlsx": xlsx_loader.load,
    ".html": web_loader.load,
    ".htm": web_loader.load,
}


def ingest_document(
    file_path: str,
    chunk_size: int = CHUNK_SIZE,
    chunk_overlap: int = CHUNK_OVERLAP,
) -> List[Dict[str, Any]]:
    """Ingests a document or URL, cleans text, chunks it, and enriches metadata.

    Args:
        file_path: Path to the target file or URL.
        chunk_size: Target chunk size (default: config.CHUNK_SIZE).
        chunk_overlap: Overlap between consecutive chunks (default: config.CHUNK_OVERLAP).

    Returns:
        List of enriched chunk dictionaries.

    Raises:
        ValueError: If the file extension is unsupported.
        FileNotFoundError: If the file path does not exist.
    """
    if file_path.startswith("http://") or file_path.startswith("https://"):
        raw_docs = web_loader.load(file_path)
    else:
        ext = os.path.splitext(file_path)[1].lower()
        loader_fn = LOADER_MAPPING.get(ext)
        if loader_fn is None:
            supported = ", ".join(LOADER_MAPPING.keys())
            raise ValueError(f"Unsupported file format '{ext}'. Supported: {supported}")

        if not os.path.exists(file_path):
            raise FileNotFoundError(f"File not found: {file_path}")
        raw_docs = loader_fn(file_path)

    # Clean raw document texts
    cleaned_docs: List[Dict[str, Any]] = []
    for doc in raw_docs:
        text = clean_text(doc.get("text", ""))
        if text:
            cleaned_docs.append(
                {
                    "text": text,
                    "source": doc.get("source", file_path),
                    "page": doc.get("page"),
                }
            )

    if not cleaned_docs:
        return []

    # Chunk cleaned documents with explicit named constants
    chunks = create_chunks(
        documents=cleaned_docs,
        chunk_size=chunk_size,
        overlap=chunk_overlap,
    )

    # Enrich metadata
    return enrich_metadata(chunks)


def ingest_directory(
    dir_path: str,
    chunk_size: int = CHUNK_SIZE,
    chunk_overlap: int = CHUNK_OVERLAP,
) -> List[Dict[str, Any]]:
    """Recursively ingests all supported files in a directory.

    Args:
        dir_path: Directory path to scan.
        chunk_size: Target chunk size.
        chunk_overlap: Target chunk overlap.

    Returns:
        Combined list of enriched chunk dictionaries.
    """
    all_chunks: List[Dict[str, Any]] = []
    if not os.path.exists(dir_path):
        raise FileNotFoundError(f"Directory not found: {dir_path}")

    for root, _, files in os.walk(dir_path):
        for f in files:
            ext = os.path.splitext(f)[1].lower()
            if ext in LOADER_MAPPING:
                file_full = os.path.join(root, f)
                try:
                    chunks = ingest_document(
                        file_full,
                        chunk_size=chunk_size,
                        chunk_overlap=chunk_overlap,
                    )
                    all_chunks.extend(chunks)
                except Exception as e:
                    # Continue scanning directory even if one file fails
                    continue

    return all_chunks

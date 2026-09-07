"""Text chunking logic using recursive character splitting and explicit named constants."""

import hashlib
from typing import Any, Dict, List, Optional

from app.config import CHUNK_OVERLAP, CHUNK_SIZE

DEFAULT_SEPARATORS = ["\n\n", "\n", ". ", "? ", "! ", " ", ""]


def _split_text_recursively(
    text: str,
    chunk_size: int,
    overlap: int,
    separators: Optional[List[str]] = None,
) -> List[str]:
    """Recursively splits text into chunks of target size with specified overlap.

    Tries higher-priority separators first (paragraphs, then lines, sentences, words, chars).
    """
    if separators is None:
        separators = DEFAULT_SEPARATORS

    if len(text) <= chunk_size:
        return [text] if text.strip() else []

    # Find the first separator that appears in text
    chosen_sep = ""
    next_separators: List[str] = []
    for idx, sep in enumerate(separators):
        if sep == "":
            chosen_sep = ""
            next_separators = []
            break
        if sep in text:
            chosen_sep = sep
            next_separators = separators[idx + 1 :]
            break

    # Split by chosen separator
    if chosen_sep:
        splits = text.split(chosen_sep)
    else:
        # Fallback to character splitting
        splits = list(text)

    # Merge splits up to chunk_size with overlap
    chunks: List[str] = []
    current_chunk: List[str] = []
    current_len = 0

    for piece in splits:
        piece_len = len(piece) + (len(chosen_sep) if current_chunk else 0)

        # If a single piece is too large, recursively split it
        if len(piece) > chunk_size and next_separators:
            if current_chunk:
                merged = chosen_sep.join(current_chunk).strip()
                if merged:
                    chunks.append(merged)
                current_chunk = []
                current_len = 0
            sub_chunks = _split_text_recursively(piece, chunk_size, overlap, next_separators)
            chunks.extend(sub_chunks)
            continue

        if current_len + piece_len <= chunk_size:
            current_chunk.append(piece)
            current_len += piece_len
        else:
            if current_chunk:
                merged = chosen_sep.join(current_chunk).strip()
                if merged:
                    chunks.append(merged)

                # Keep overlap pieces from the end of current_chunk
                overlap_pieces: List[str] = []
                overlap_len = 0
                for p in reversed(current_chunk):
                    p_addition = len(p) + (len(chosen_sep) if overlap_pieces else 0)
                    if overlap_len + p_addition <= overlap:
                        overlap_pieces.insert(0, p)
                        overlap_len += p_addition
                    else:
                        break
                current_chunk = overlap_pieces
                current_len = overlap_len

            current_chunk.append(piece)
            current_len += len(piece) + (len(chosen_sep) if len(current_chunk) > 1 else 0)

    if current_chunk:
        merged = chosen_sep.join(current_chunk).strip()
        if merged:
            chunks.append(merged)

    return chunks


def create_chunks(
    documents: List[Dict[str, Any]],
    chunk_size: int = CHUNK_SIZE,
    overlap: int = CHUNK_OVERLAP,
) -> List[Dict[str, Any]]:
    """Splits documents into overlapping chunks using explicit named constants.

    Args:
        documents: List of dicts with {"text": str, "source": str, "page": int | None}.
        chunk_size: Target size per chunk (default from app.config.CHUNK_SIZE).
        overlap: Overlap between adjacent chunks (default from app.config.CHUNK_OVERLAP).

    Returns:
        List of dicts containing:
            - text: Chunk content
            - source: Source file or URL
            - chunk_id: Unique deterministic chunk identifier
            - metadata: Retained page number, chunk index, char offsets
    """
    all_chunks: List[Dict[str, Any]] = []

    for doc_idx, doc in enumerate(documents):
        raw_text = doc.get("text", "")
        source = doc.get("source", f"document_{doc_idx}")
        page = doc.get("page")
        inherited_meta = doc.get("metadata", {})

        if not raw_text.strip():
            continue

        raw_chunks = _split_text_recursively(raw_text, chunk_size=chunk_size, overlap=overlap)

        char_offset = 0
        for chunk_idx, chunk_text in enumerate(raw_chunks):
            # Compute start/end offset if possible
            start_pos = raw_text.find(chunk_text[:50], char_offset)
            if start_pos != -1:
                end_pos = start_pos + len(chunk_text)
                char_offset = max(char_offset, start_pos)
            else:
                start_pos = char_offset
                end_pos = start_pos + len(chunk_text)

            # Generate deterministic chunk ID based on source, page, and chunk index
            hasher = hashlib.sha256()
            hasher.update(f"{source}_{page}_{chunk_idx}_{chunk_text[:40]}".encode("utf-8"))
            chunk_id = hasher.hexdigest()[:16]

            meta = dict(inherited_meta)
            meta.update(
                {
                    "source": source,
                    "page": page,
                    "chunk_index": chunk_idx,
                    "start_char": start_pos,
                    "end_char": end_pos,
                }
            )

            all_chunks.append(
                {
                    "text": chunk_text,
                    "source": source,
                    "chunk_id": chunk_id,
                    "metadata": meta,
                }
            )

    return all_chunks

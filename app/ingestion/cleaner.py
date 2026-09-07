"""Text cleaning and normalization utilities for ingested documents."""

import re
import unicodedata


def clean_text(text: str) -> str:
    """Cleans and normalizes extracted text.

    - Removes null bytes and non-printable control characters.
    - Normalizes line endings to \n.
    - Normalizes unicode characters (e.g. smart quotes, non-breaking spaces).
    - Collapses consecutive whitespace while preserving paragraph breaks.

    Args:
        text: Raw text string to clean.

    Returns:
        Cleaned, normalized text string.
    """
    if not text:
        return ""

    # Remove null bytes
    text = text.replace("\x00", "")

    # Normalize unicode to NFKC
    text = unicodedata.normalize("NFKC", text)

    # Normalize line breaks
    text = re.sub(r"\r\n|\r", "\n", text)

    # Replace non-breaking spaces and tabs with standard space
    text = text.replace("\xa0", " ")
    text = re.sub(r"[^\S\n]+", " ", text)

    # Replace fancy curly quotes and apostrophes
    text = text.replace("“", '"').replace("”", '"')
    text = text.replace("‘", "'").replace("’", "'")

    # Remove non-printable control characters except newline and carriage return
    text = "".join(ch for ch in text if ch == "\n" or unicodedata.category(ch)[0] != "C")

    # Clean lines and trim excessive blank lines
    lines = [line.strip() for line in text.split("\n")]
    cleaned_text = "\n".join(lines)
    cleaned_text = re.sub(r"\n{3,}", "\n\n", cleaned_text)

    return cleaned_text.strip()

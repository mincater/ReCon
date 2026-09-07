"""Text file loader with multi-encoding fallback."""

import os
from typing import Any, Dict, List


def load(file_path: str) -> List[Dict[str, Any]]:
    """Loads a plain text file using multi-encoding detection.

    Args:
        file_path: Path to the text file.

    Returns:
        List containing a single dict with {"text": str, "source": str, "page": None}.

    Raises:
        FileNotFoundError: If the file does not exist.
    """
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"Text file not found: {file_path}")

    encodings = ["utf-8", "utf-8-sig", "latin-1", "cp1252"]
    content = ""
    for enc in encodings:
        try:
            with open(file_path, "r", encoding=enc) as f:
                content = f.read()
            break
        except UnicodeDecodeError:
            continue
    else:
        # Ultimate fallback with replace
        with open(file_path, "r", encoding="utf-8", errors="replace") as f:
            content = f.read()

    return [{"text": content, "source": file_path, "page": None}]

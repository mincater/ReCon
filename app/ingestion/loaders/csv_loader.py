"""CSV document loader converting structured tabular rows into text."""

import csv
import os
from typing import Any, Dict, List


def load(file_path: str) -> List[Dict[str, Any]]:
    """Loads a CSV file and converts each row into a structured textual line.

    Args:
        file_path: Path to the CSV file.

    Returns:
        List containing {"text": str, "source": str, "page": None}.

    Raises:
        FileNotFoundError: If the file does not exist.
    """
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"CSV file not found: {file_path}")

    rows_text: List[str] = []
    with open(file_path, "r", encoding="utf-8", errors="replace") as f:
        # Detect delimiter if possible
        sample = f.read(2048)
        f.seek(0)
        try:
            dialect = csv.Sniffer().sniff(sample)
            delimiter = dialect.delimiter
        except Exception:
            delimiter = ","

        reader = csv.DictReader(f, delimiter=delimiter)
        for idx, row in enumerate(reader):
            items = [f"{k.strip()}: {v.strip()}" for k, v in row.items() if k and v and v.strip()]
            if items:
                rows_text.append(f"Row {idx + 1}: " + "; ".join(items))

    full_text = "\n".join(rows_text)
    return [{"text": full_text, "source": file_path, "page": None}]

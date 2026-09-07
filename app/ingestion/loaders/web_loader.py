"""Web and HTML document loader."""

import os
from typing import Any, Dict, List


def load(file_path: str) -> List[Dict[str, Any]]:
    """Loads a web page from a URL or local HTML file, stripping markup tags.

    Args:
        file_path: URL (http/https) or local file path to an HTML file.

    Returns:
        List containing {"text": str, "source": str, "page": None}.

    Raises:
        FileNotFoundError: If a local HTML file is specified but does not exist.
        ImportError: If beautifulsoup4 or requests are missing.
    """
    try:
        from bs4 import BeautifulSoup
        import requests
    except ImportError as e:
        raise ImportError(
            "beautifulsoup4 and requests are required for web page loading."
        ) from e

    if file_path.startswith("http://") or file_path.startswith("https://"):
        headers = {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64)"}
        resp = requests.get(file_path, headers=headers, timeout=10)
        resp.raise_for_status()
        html_content = resp.text
    else:
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"HTML file not found: {file_path}")
        with open(file_path, "r", encoding="utf-8", errors="replace") as f:
            html_content = f.read()

    soup = BeautifulSoup(html_content, "html.parser")

    # Extract title if present
    title_text = ""
    if soup.title and soup.title.string:
        title_text = f"Title: {soup.title.string.strip()}\n\n"

    # Remove non-content tags
    for tag in soup(["script", "style", "nav", "footer", "header", "noscript", "svg", "form"]):
        tag.decompose()

    text = soup.get_text(separator="\n")
    cleaned_lines = [line.strip() for line in text.splitlines() if line.strip()]
    full_text = title_text + "\n".join(cleaned_lines)

    return [{"text": full_text.strip(), "source": file_path, "page": None}]

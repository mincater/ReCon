"""Manages document sources and web search citation origins."""

from typing import Any, Dict, List


class SourceManager:
    """Tracks sources, URLs, and local files referenced during research."""

    def __init__(self) -> None:
        self.sources: Dict[str, Dict[str, Any]] = {}

    def add_source(self, identifier: str, metadata: Dict[str, Any]) -> None:
        """Registers a source with its metadata."""
        self.sources[identifier] = metadata

    def get_source(self, identifier: str) -> Dict[str, Any]:
        """Retrieves metadata for a given source."""
        return self.sources.get(identifier, {})

    def list_sources(self) -> List[Dict[str, Any]]:
        """Returns all registered sources."""
        return list(self.sources.values())

"""Configuration and settings module for research-rag.

Supports seamless loading from Streamlit Secrets (in production)
or environment variables (.env / os.environ in Colab and local development).
"""

from dataclasses import dataclass, field
from functools import lru_cache
import os
from typing import Optional

from dotenv import load_dotenv

# Automatically load environment variables from .env if present
load_dotenv()


def get_config_val(key: str, default: Optional[str] = None) -> Optional[str]:
    """Retrieve configuration value from Streamlit secrets or environment variables.

    Args:
        key: The configuration key to look up.
        default: Fallback value if key is not found.

    Returns:
        The configuration string or default.
    """
    # 1. Attempt lookup from Streamlit secrets (production deployment)
    try:
        import streamlit as st

        if hasattr(st, "secrets") and key in st.secrets:
            val = st.secrets[key]
            if val is not None:
                return str(val)
    except Exception:
        # Streamlit might not be running in a Streamlit context
        pass

    # 2. Fallback to os.environ / .env (local / Colab)
    val = os.environ.get(key)
    if val is not None and val != "":
        return val

    return default


@dataclass(frozen=True)
class Settings:
    """Application settings and configuration parameters."""

    # LLM Settings
    llm_provider_primary: str = field(
        default_factory=lambda: get_config_val("LLM_PROVIDER_PRIMARY", "groq") or "groq"
    )
    llm_model_primary: str = field(
        default_factory=lambda: get_config_val("LLM_MODEL_PRIMARY", "openai/gpt-oss-20b")
        or "openai/gpt-oss-20b"
    )
    llm_provider_fallback: str = field(
        default_factory=lambda: get_config_val("LLM_PROVIDER_FALLBACK", "gemini") or "gemini"
    )
    llm_model_fallback: str = field(
        default_factory=lambda: get_config_val("LLM_MODEL_FALLBACK", "gemini-3.1-flash-lite")
        or "gemini-3.1-flash-lite"
    )
    groq_api_key: Optional[str] = field(
        default_factory=lambda: get_config_val("GROQ_API_KEY", None)
    )
    gemini_api_key: Optional[str] = field(
        default_factory=lambda: get_config_val("GEMINI_API_KEY", None)
    )

    # Search Settings
    search_provider: str = field(
        default_factory=lambda: get_config_val("SEARCH_PROVIDER", "tavily") or "tavily"
    )
    tavily_api_key: Optional[str] = field(
        default_factory=lambda: get_config_val("TAVILY_API_KEY", None)
    )

    # Vector Database Settings
    qdrant_url: Optional[str] = field(
        default_factory=lambda: get_config_val("QDRANT_URL", None)
    )
    qdrant_api_key: Optional[str] = field(
        default_factory=lambda: get_config_val("QDRANT_API_KEY", None)
    )

    # Embedding & Reranking Models
    embedding_model: str = field(
        default_factory=lambda: get_config_val("EMBEDDING_MODEL", "all-MiniLM-L6-v2")
        or "all-MiniLM-L6-v2"
    )
    reranker_model: str = field(
        default_factory=lambda: get_config_val(
            "RERANKER_MODEL", "cross-encoder/ms-marco-MiniLM-L-6-v2"
        )
        or "cross-encoder/ms-marco-MiniLM-L-6-v2"
    )

    # Chunking Defaults
    chunk_size: int = field(
        default_factory=lambda: int(get_config_val("CHUNK_SIZE", "500") or 500)
    )
    chunk_overlap: int = field(
        default_factory=lambda: int(get_config_val("CHUNK_OVERLAP", "50") or 50)
    )


@lru_cache()
def get_settings() -> Settings:
    """Returns a cached Settings instance."""
    return Settings()


# Named constants for direct access
settings = get_settings()
LLM_PROVIDER_PRIMARY = settings.llm_provider_primary
LLM_MODEL_PRIMARY = settings.llm_model_primary
LLM_PROVIDER_FALLBACK = settings.llm_provider_fallback
LLM_MODEL_FALLBACK = settings.llm_model_fallback
GROQ_API_KEY = settings.groq_api_key
GEMINI_API_KEY = settings.gemini_api_key
SEARCH_PROVIDER = settings.search_provider
TAVILY_API_KEY = settings.tavily_api_key
QDRANT_URL = settings.qdrant_url
QDRANT_API_KEY = settings.qdrant_api_key
EMBEDDING_MODEL = settings.embedding_model
RERANKER_MODEL = settings.reranker_model
CHUNK_SIZE = settings.chunk_size
CHUNK_OVERLAP = settings.chunk_overlap

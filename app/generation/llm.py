"""LLM provider integration with graceful fallback (Groq -> Gemini Flash-Lite)."""

import logging
from typing import Optional

from app.config import (
    GEMINI_API_KEY,
    GROQ_API_KEY,
    LLM_MODEL_FALLBACK,
    LLM_MODEL_PRIMARY,
    LLM_PROVIDER_FALLBACK,
    LLM_PROVIDER_PRIMARY,
)

logger = logging.getLogger(__name__)


class LLMProviderError(Exception):
    """Base exception for all LLM provider errors."""


class ProviderError(LLMProviderError):
    """General error communicating with an LLM provider."""


class RateLimitError(ProviderError):
    """Rate limit (HTTP 429 / quota exceeded) encountered on LLM provider."""


def _call_groq(prompt: str, system: Optional[str] = None) -> str:
    """Invokes Groq API, normalizing SDK exceptions into typed errors."""
    import os
    from app.config import get_config_val

    key = os.environ.get("GROQ_API_KEY") or GROQ_API_KEY or get_config_val("GROQ_API_KEY")
    if not key:
        raise ProviderError("GROQ_API_KEY is not configured.")

    model_name = (
        os.environ.get("LLM_MODEL_PRIMARY")
        or get_config_val("LLM_MODEL_PRIMARY")
        or LLM_MODEL_PRIMARY
        or "openai/gpt-oss-20b"
    )
    # 'llama-4-maverick' is not an active Groq model; alias to 'openai/gpt-oss-20b'
    if model_name in ("llama-4-maverick", "gpt-oss-120b"):
        model_name = "openai/gpt-oss-20b"

    try:
        from groq import (
            APIError as GroqAPIError,
            Groq,
            RateLimitError as GroqRateLimitError,
        )

        client = Groq(api_key=key)
        messages = []
        if system:
            messages.append({"role": "system", "content": system})
        messages.append({"role": "user", "content": prompt})

        try:
            response = client.chat.completions.create(
                model=model_name,
                messages=messages,
            )
        except GroqAPIError as api_err:
            # If the configured model does not exist (404), retry with llama-3.3-70b-versatile or llama-3.1-8b-instant
            if getattr(api_err, "status_code", None) == 404 or "model_not_found" in str(api_err):
                logger.warning(f"Groq model '{model_name}' not found. Retrying with 'llama-3.3-70b-versatile'...")
                response = client.chat.completions.create(
                    model="llama-3.3-70b-versatile",
                    messages=messages,
                )
            else:
                raise
        content = response.choices[0].message.content
        return content or ""
    except ImportError as e:
        raise ProviderError("groq package is required. Install via requirements.txt.") from e
    except GroqRateLimitError as e:
        logger.warning(f"Groq rate limit encountered: {e}")
        raise RateLimitError(f"Groq rate limit: {e}") from e
    except GroqAPIError as e:
        logger.error(f"Groq API error: {e}")
        if getattr(e, "status_code", None) == 429:
            raise RateLimitError(f"Groq rate limit: {e}") from e
        raise ProviderError(f"Groq API failure: {e}") from e
    except Exception as e:
        err_str = str(e).lower()
        if "429" in err_str or "rate limit" in err_str:
            raise RateLimitError(f"Groq rate limit: {e}") from e
        raise ProviderError(f"Unexpected Groq error: {e}") from e


def _call_gemini(prompt: str, system: Optional[str] = None) -> str:
    """Invokes Google Gemini API, normalizing SDK exceptions into typed errors."""
    import os
    from app.config import get_config_val

    key = os.environ.get("GEMINI_API_KEY") or GEMINI_API_KEY or get_config_val("GEMINI_API_KEY")
    if not key:
        raise ProviderError("GEMINI_API_KEY is not configured.")

    model_name = (
        os.environ.get("LLM_MODEL_FALLBACK")
        or get_config_val("LLM_MODEL_FALLBACK")
        or LLM_MODEL_FALLBACK
        or "gemini-3.1-flash-lite"
    )
    if model_name in ("gemini-2.5-flash-lite",):
        model_name = "gemini-3.1-flash-lite"

    try:
        import google.generativeai as genai

        genai.configure(api_key=key)
        try:
            model = genai.GenerativeModel(
                model_name=model_name,
                system_instruction=system if system else None,
            )
            response = model.generate_content(prompt)
        except Exception as model_err:
            if "not found" in str(model_err).lower() or "404" in str(model_err):
                logger.warning(f"Gemini model '{model_name}' not found. Retrying with 'gemini-1.5-flash'...")
                model = genai.GenerativeModel(
                    model_name="gemini-1.5-flash",
                    system_instruction=system if system else None,
                )
                response = model.generate_content(prompt)
            else:
                raise
        return response.text or ""
    except ImportError as e:
        raise ProviderError(
            "google-generativeai is required for Gemini fallback. Install via requirements.txt."
        ) from e
    except Exception as e:
        err_str = str(e)
        logger.warning(f"Gemini API error: {err_str}")
        # Detect quota / rate limit errors across various Gemini SDK exception types
        if any(term in err_str for term in ["429", "ResourceExhausted", "QuotaExceeded", "RATE_LIMIT_EXCEEDED"]):
            raise RateLimitError(f"Gemini rate limit: {err_str}") from e
        raise ProviderError(f"Gemini API failure: {err_str}") from e


def generate(prompt: str, system: Optional[str] = None) -> str:
    """Generates an answer trying primary LLM provider, falling back on rate limits or failures.

    Args:
        prompt: The user query or formatted prompt.
        system: Optional system instruction.

    Returns:
        Generated string response.

    Raises:
        LLMProviderError: If both primary and fallback providers fail.
    """
    # Note: Free-tier limits for Groq and Gemini are volatile as of writing (Google cut free limits
    # 50-80% in Dec 2025). We never hardcode rate-limits; we handle exceptions dynamically via fallback.
    logger.info(f"Attempting primary provider: {LLM_PROVIDER_PRIMARY} ({LLM_MODEL_PRIMARY})")
    try:
        if LLM_PROVIDER_PRIMARY == "groq":
            return _call_groq(prompt, system)
        elif LLM_PROVIDER_PRIMARY == "gemini":
            return _call_gemini(prompt, system)
        else:
            raise ProviderError(f"Unknown primary provider: {LLM_PROVIDER_PRIMARY}")
    except (RateLimitError, ProviderError) as primary_exc:
        logger.warning(
            f"Primary LLM provider ({LLM_PROVIDER_PRIMARY}) failed ({primary_exc}). "
            f"Failing over to fallback provider ({LLM_PROVIDER_FALLBACK}: {LLM_MODEL_FALLBACK})."
        )
        try:
            if LLM_PROVIDER_FALLBACK == "gemini":
                return _call_gemini(prompt, system)
            elif LLM_PROVIDER_FALLBACK == "groq":
                return _call_groq(prompt, system)
            else:
                raise ProviderError(f"Unknown fallback provider: {LLM_PROVIDER_FALLBACK}")
        except Exception as fallback_exc:
            logger.error(
                f"Fallback LLM provider ({LLM_PROVIDER_FALLBACK}) also failed: {fallback_exc}"
            )
            raise LLMProviderError(
                f"Both providers failed. Primary: {primary_exc}; Fallback: {fallback_exc}"
            ) from fallback_exc

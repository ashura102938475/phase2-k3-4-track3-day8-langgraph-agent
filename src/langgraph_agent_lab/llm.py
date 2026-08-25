"""LLM factory helper.

Provides a simple interface to create LLM clients for use in nodes.
Students should use this helper so the lab works with any supported provider.
"""

from __future__ import annotations

import os
from typing import Any
from urllib.parse import urlsplit

from dotenv import load_dotenv
from langchain_core.language_models.chat_models import BaseChatModel
from pydantic import SecretStr

# Load the project-local configuration once, when this module is first imported.
# Existing process environment values retain precedence over values in ``.env``.
load_dotenv()

NVIDIA_BASE_URL = "https://integrate.api.nvidia.com/v1"
NVIDIA_HOST = "integrate.api.nvidia.com"
NVIDIA_MODEL = "nvidia/nemotron-3.5-lightning-30b-a3b"


def is_nvidia_endpoint(base_url: str | None) -> bool:
    """Return whether ``base_url`` is exactly NVIDIA's hosted OpenAI endpoint."""
    if not base_url:
        return False
    try:
        parsed = urlsplit(base_url)
        port = parsed.port
    except ValueError:
        return False
    return (
        parsed.scheme == "https"
        and parsed.hostname == NVIDIA_HOST
        and port in (None, 443)
        and parsed.username is None
        and parsed.password is None
        and parsed.path.rstrip("/") == "/v1"
        and not parsed.query
        and not parsed.fragment
    )


def is_nvidia_chat_model(llm: BaseChatModel) -> bool:
    """Identify a NVIDIA-configured OpenAI-compatible chat model.

    ``ChatOpenAI`` exposes ``openai_api_base``. Checking its exact parsed endpoint
    keeps NVIDIA-only request settings away from Gemini, Anthropic, and standard
    OpenAI clients.
    """
    base_url = getattr(llm, "openai_api_base", None)
    return isinstance(base_url, str) and is_nvidia_endpoint(base_url)


def _openai_llm(
    *,
    api_key: str,
    model: str,
    temperature: float,
    base_url: str | None = None,
    timeout: float | None = None,
    max_retries: int | None = None,
    max_tokens: int | None = None,
) -> BaseChatModel:
    """Construct a standard OpenAI-compatible chat model."""
    try:
        from langchain_openai import ChatOpenAI
    except ImportError as exc:
        raise RuntimeError("Install: pip install langchain-openai") from exc

    options: dict[str, Any] = {
        "model": model,
        "api_key": SecretStr(api_key),
        "temperature": temperature,
    }
    if base_url is not None:
        options["base_url"] = base_url
    if timeout is not None:
        options["timeout"] = timeout
    if max_retries is not None:
        options["max_retries"] = max_retries
    if max_tokens is not None:
        options["max_completion_tokens"] = max_tokens
    return ChatOpenAI(**options)


def _nvidia_llm(
    *,
    api_key: str,
    model: str,
    temperature: float,
    timeout: float | None = None,
    max_retries: int | None = None,
    max_tokens: int | None = None,
) -> BaseChatModel:
    """Construct the NVIDIA client with its canonical hosted endpoint."""
    try:
        from langchain_openai import ChatOpenAI
    except ImportError as exc:
        raise RuntimeError("Install: pip install langchain-openai") from exc

    options: dict[str, Any] = {
        "model": model,
        "api_key": SecretStr(api_key),
        "temperature": temperature,
        "base_url": NVIDIA_BASE_URL,
        "extra_body": {"chat_template_kwargs": {"enable_thinking": False}},
    }
    if timeout is not None:
        options["timeout"] = timeout
    if max_retries is not None:
        options["max_retries"] = max_retries
    if max_tokens is not None:
        options["max_completion_tokens"] = max_tokens
    return ChatOpenAI(**options)


def get_llm(
    model: str | None = None,
    temperature: float = 0.0,
    *,
    timeout: float | None = None,
    max_retries: int | None = None,
    max_tokens: int | None = None,
) -> BaseChatModel:
    """Create an LLM client from environment configuration.

    ``NVIDIA_API_KEY`` wins over all other provider credentials and is always
    sent to NVIDIA's canonical hosted endpoint. ``OPENAI_API_KEY`` is treated
    as NVIDIA credentials only when ``OPENAI_BASE_URL`` is that exact endpoint.
    An explicit model argument has precedence over ``LLM_MODEL`` for every
    provider. Optional runtime limits are translated to each adapter's native
    constructor names.
    """
    if timeout is not None and timeout <= 0:
        raise ValueError("timeout must be positive")
    if max_retries is not None and max_retries < 0:
        raise ValueError("max_retries must be non-negative")
    if max_tokens is not None and max_tokens <= 0:
        raise ValueError("max_tokens must be positive")

    nvidia_key = os.getenv("NVIDIA_API_KEY")
    configured_model = model or os.getenv("LLM_MODEL")
    if nvidia_key:
        return _nvidia_llm(
            api_key=nvidia_key,
            model=configured_model or NVIDIA_MODEL,
            temperature=temperature,
            timeout=timeout,
            max_retries=max_retries,
            max_tokens=max_tokens,
        )

    if os.getenv("GEMINI_API_KEY"):
        try:
            from langchain_google_genai import (  # type: ignore[import-not-found]
                ChatGoogleGenerativeAI,
            )
        except ImportError as exc:
            raise RuntimeError("Install: pip install langchain-google-genai") from exc
        options: dict[str, Any] = {
            "model": configured_model or "gemini-2.5-flash",
            "google_api_key": os.getenv("GEMINI_API_KEY"),
            "temperature": temperature,
        }
        if timeout is not None:
            options["request_timeout"] = timeout
        if max_retries is not None:
            options["retries"] = max_retries
        if max_tokens is not None:
            options["max_tokens"] = max_tokens
        return ChatGoogleGenerativeAI(**options)

    openai_key = os.getenv("OPENAI_API_KEY")
    base_url = os.getenv("OPENAI_BASE_URL")
    if openai_key and is_nvidia_endpoint(base_url):
        return _nvidia_llm(
            api_key=openai_key,
            model=configured_model or NVIDIA_MODEL,
            temperature=temperature,
            timeout=timeout,
            max_retries=max_retries,
            max_tokens=max_tokens,
        )

    if openai_key:
        return _openai_llm(
            api_key=openai_key,
            model=configured_model or "gpt-4o-mini",
            temperature=temperature,
            base_url=base_url,
            timeout=timeout,
            max_retries=max_retries,
            max_tokens=max_tokens,
        )

    if os.getenv("ANTHROPIC_API_KEY"):
        try:
            from langchain_anthropic import ChatAnthropic  # type: ignore[import-not-found]
        except ImportError as exc:
            raise RuntimeError("Install: pip install langchain-anthropic") from exc
        options = {
            "model": configured_model or "claude-sonnet-4-20250514",
            "temperature": temperature,
        }
        if timeout is not None:
            options["timeout"] = timeout
        if max_retries is not None:
            options["max_retries"] = max_retries
        if max_tokens is not None:
            options["max_tokens"] = max_tokens
        return ChatAnthropic(**options)

    raise RuntimeError(
        "No LLM API key found. Set NVIDIA_API_KEY, GEMINI_API_KEY, OPENAI_API_KEY, or "
        "ANTHROPIC_API_KEY in .env\nSee .env.example for configuration."
    )

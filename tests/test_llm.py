"""Behavioral tests for provider-aware LLM configuration."""

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

import pytest

from langgraph_agent_lab.llm import get_llm


def _clear_provider_environment(monkeypatch: pytest.MonkeyPatch) -> None:
    for name in (
        "GEMINI_API_KEY",
        "GOOGLE_API_KEY",
        "NVIDIA_API_KEY",
        "OPENAI_API_KEY",
        "ANTHROPIC_API_KEY",
        "OPENAI_BASE_URL",
        "LLM_MODEL",
    ):
        monkeypatch.delenv(name, raising=False)


def test_get_llm_uses_nvidia_openai_compatible_configuration_without_leaking_key(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    secret = "nvidia-test-secret"
    _clear_provider_environment(monkeypatch)
    monkeypatch.setenv("NVIDIA_API_KEY", secret)
    monkeypatch.setenv("OPENAI_BASE_URL", "https://integrate.api.nvidia.com/v1")
    monkeypatch.setenv("LLM_MODEL", "meta/llama-3.1-8b-instruct")

    llm = get_llm()

    assert llm.__class__.__name__ == "ChatOpenAI"
    assert llm.model_name == "meta/llama-3.1-8b-instruct"
    assert str(llm.root_client.base_url).rstrip("/") == "https://integrate.api.nvidia.com/v1"
    assert bool(llm.openai_api_key)
    assert secret not in repr(llm)


def test_get_llm_keeps_openai_provider_compatible_when_nvidia_key_is_absent(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _clear_provider_environment(monkeypatch)
    monkeypatch.setenv("OPENAI_API_KEY", "openai-test-secret")
    monkeypatch.setenv("LLM_MODEL", "existing-openai-model")

    llm = get_llm(model="explicit-model")

    assert llm.__class__.__name__ == "ChatOpenAI"
    assert llm.model_name == "explicit-model"
    assert bool(llm.openai_api_key)


def test_nvidia_key_takes_precedence_over_unrelated_provider_keys(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _clear_provider_environment(monkeypatch)
    monkeypatch.setenv("GEMINI_API_KEY", "unused-gemini-key")
    monkeypatch.setenv("NVIDIA_API_KEY", "nvidia-test-secret")

    llm = get_llm()

    assert llm.__class__.__name__ == "ChatOpenAI"
    assert llm.model_name == "nvidia/nemotron-3.5-lightning-30b-a3b"


def test_nvidia_key_is_only_sent_to_nvidia_host(monkeypatch: pytest.MonkeyPatch) -> None:
    _clear_provider_environment(monkeypatch)
    monkeypatch.setenv("NVIDIA_API_KEY", "nvidia-test-secret")
    monkeypatch.setenv("OPENAI_BASE_URL", "https://integrate.api.nvidia.com.attacker.example/v1")

    llm = get_llm()

    assert str(llm.root_client.base_url).rstrip("/") == "https://integrate.api.nvidia.com/v1"


def test_openai_key_and_nvidia_model_without_nvidia_endpoint_stay_on_openai(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _clear_provider_environment(monkeypatch)
    monkeypatch.setenv("OPENAI_API_KEY", "openai-test-secret")
    monkeypatch.setenv("LLM_MODEL", "nvidia/nemotron-3.5-lightning-30b-a3b")

    llm = get_llm()

    assert llm.openai_api_base is None
    assert llm.extra_body is None


def test_llm_factory_loads_dotenv_once_for_cli_and_graph_imports(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    env_file = tmp_path / ".env"
    env_file.write_text(
        "NVIDIA_API_KEY=dotenv-test-secret\n"
        "LLM_MODEL=nvidia/nemotron-3.5-lightning-30b-a3b\n",
        encoding="utf-8",
    )
    clean_env = {
        key: value
        for key, value in os.environ.items()
        if key
        not in {
            "GEMINI_API_KEY",
            "GOOGLE_API_KEY",
            "NVIDIA_API_KEY",
            "OPENAI_API_KEY",
            "ANTHROPIC_API_KEY",
            "OPENAI_BASE_URL",
            "LLM_MODEL",
        }
    }

    completed = subprocess.run(
        [
            sys.executable,
            "-c",
            "from langgraph_agent_lab.llm import get_llm; print(get_llm().model_name)",
        ],
        cwd=tmp_path,
        env=clean_env,
        check=False,
        capture_output=True,
        text=True,
    )

    assert completed.returncode == 0, completed.stderr
    assert completed.stdout.strip() == "nvidia/nemotron-3.5-lightning-30b-a3b"

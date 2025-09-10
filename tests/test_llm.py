import os

from dotenv import dotenv_values

from driftwatch.llm import _ensure_env, chat_completion
from openai import APIConnectionError
import pytest


def test_ensure_env_loads_dotenv(monkeypatch):
    """_ensure_env loads credentials from .env when missing."""
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    monkeypatch.delenv("OPENAI_BASE_URL", raising=False)
    _ensure_env()
    values = dotenv_values(".env")
    assert os.environ["OPENAI_API_KEY"] == values["OPENAI_API_KEY"]
    assert os.environ["OPENAI_BASE_URL"] == values["OPENAI_BASE_URL"]


def test_chat_completion_integration():
    """chat_completion returns a message from the LLM."""
    try:
        result = chat_completion(
            "Say hello in one word",
            model="openai/gpt-5-nano",
            max_tokens=16,  # OpenRouter enforces a minimum of 16 tokens
        )
    except APIConnectionError as exc:  # pragma: no cover - network issues
        pytest.skip(f"API connection failed: {exc}")
    assert isinstance(result["message"], str) and result["message"].strip()
    assert result["usage"] is not None

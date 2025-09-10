import os

import pytest
from dotenv import load_dotenv

from driftwatch.llm import chat_completion
from openai import APIConnectionError


@pytest.mark.integration
def test_openai_hello_world():
    if not os.getenv("OPENAI_API_KEY") or not os.getenv("OPENAI_BASE_URL"):
        load_dotenv()
    if not os.getenv("OPENAI_API_KEY"):
        pytest.fail("OPENAI_API_KEY must be set for integration test")
    try:
        result = chat_completion(
            "Say hello world",
            model="openai/gpt-5-nano",
            max_tokens=16,
        )
    except APIConnectionError as exc:  # pragma: no cover - network issues
        pytest.skip(f"API connection failed: {exc}")
    normalized = result["message"].lower().replace(",", "")
    assert "hello world" in normalized

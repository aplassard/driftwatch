import os

import pytest
from dotenv import load_dotenv
from openai import APIConnectionError

from driftwatch import evaluate
from driftwatch.datasets import NATALIA_CLIPS


@pytest.mark.integration
def test_gsm8k_evaluate_format_and_correctness():
    if not os.getenv("OPENAI_API_KEY") or not os.getenv("OPENAI_BASE_URL"):
        load_dotenv()
    if not os.getenv("OPENAI_API_KEY"):
        pytest.fail("OPENAI_API_KEY must be set for integration test")
    try:
        result = evaluate(NATALIA_CLIPS, model="openai/gpt-5-nano")
    except APIConnectionError as exc:  # pragma: no cover - network issues
        pytest.skip(f"API connection failed: {exc}")
    assert "####" in result["response"]
    reasoning, _, answer = result["response"].partition("####")
    assert reasoning.strip()
    assert result["correct"] is True
    assert answer.strip().startswith("72")

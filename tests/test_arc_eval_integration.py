import os

import pytest
from dotenv import load_dotenv
from openai import APIConnectionError

from driftwatch import evaluate
from driftwatch.datasets import JUAN_LAKEISHA_RAMP


@pytest.mark.integration
def test_arc_evaluate_format_and_correctness():
    if not os.getenv("OPENAI_API_KEY") or not os.getenv("OPENAI_BASE_URL"):
        load_dotenv()
    if not os.getenv("OPENAI_API_KEY"):
        pytest.fail("OPENAI_API_KEY must be set for integration test")
    try:
        result = evaluate(JUAN_LAKEISHA_RAMP, model="openai/gpt-5-nano")
    except APIConnectionError as exc:  # pragma: no cover - network issues
        pytest.skip(f"API connection failed: {exc}")
    assert "####" in result["response"]
    reasoning, _, answer = result["response"].partition("####")
    assert reasoning.strip()
    assert result["correct"] is True
    assert answer.strip().upper().startswith("D")


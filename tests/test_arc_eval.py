from driftwatch.datasets import JUAN_LAKEISHA_RAMP
from driftwatch import evaluate


def test_juan_lakeisha_ramp(monkeypatch):
    prompts = {}

    def fake_chat_completion(prompt: str, model: str | None = None, max_tokens: int = 10240):
        prompts["prompt"] = prompt
        return {
            "message": (
                "Record details allows repetition.\n"
                "#### D"
            ),
            "usage": {},
        }

    monkeypatch.setattr("driftwatch.evaluator.chat_completion", fake_chat_completion)
    result = evaluate(JUAN_LAKEISHA_RAMP)
    assert result["correct"] is True
    assert "A." in prompts["prompt"]
    assert "####" in prompts["prompt"]


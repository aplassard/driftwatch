from driftwatch.datasets import NATALIA_CLIPS
from driftwatch import evaluate


def test_natalia_clips(monkeypatch):
    prompts = {}

    def fake_chat_completion(
        prompt: str,
        model: str | None = None,
        max_tokens: int = 10240,
        temperature: float = 0.7,
    ):
        prompts["prompt"] = prompt
        return {
            "message": (
                "Natalia sold 48/2 = 24 clips in May.\n"
                "Natalia sold 48+24 = 72 clips altogether in April and May.\n"
                "#### 72"
            ),
            "usage": {},
        }

    monkeypatch.setattr("driftwatch.evaluator.chat_completion", fake_chat_completion)
    result = evaluate(NATALIA_CLIPS)
    assert result["correct"] is True
    assert "Show your reasoning" in prompts["prompt"]
    assert "####" in prompts["prompt"]

"""OpenAI-compatible LLM client utilities."""

from __future__ import annotations

import os
import time
from json import JSONDecodeError

import httpx
from openai import OpenAI

MODEL_NAME = os.getenv("MODEL_NAME", "openai/gpt-5-nano")


def _ensure_env() -> None:
    """Load API credentials from ``.env`` when missing.

    ``OPENAI_API_KEY`` must be provided; ``OPENAI_BASE_URL`` is optional and
    falls back to the OpenAI default endpoint when absent.
    """
    if not os.getenv("OPENAI_API_KEY") or not os.getenv("OPENAI_BASE_URL"):
        from dotenv import load_dotenv

        load_dotenv()

    if not os.getenv("OPENAI_API_KEY"):
        raise RuntimeError("Missing required environment variable: OPENAI_API_KEY")


def chat_completion(
    prompt: str,
    model: str | None = None,
    max_tokens: int = 10_240,
) -> dict:
    """Return the assistant message and token usage details.

    The returned dictionary contains the assistant ``message`` along with ``usage``
    statistics (prompt, cache, reasoning and completion tokens) and ``cost`` for
    each token type when pricing information is available for ``model``.
    """
    _ensure_env()
    client_kwargs = {"api_key": os.environ["OPENAI_API_KEY"]}
    base_url = os.getenv("OPENAI_BASE_URL")
    if base_url:
        client_kwargs["base_url"] = base_url
    client = OpenAI(**client_kwargs)
    target_model = model or MODEL_NAME
    # ``openai`` occasionally returns malformed JSON or encounters transient
    # network issues.  These manifest as ``JSONDecodeError`` or ``httpx``
    # exceptions bubbling out of ``client.chat.completions.create``.  Instead of
    # failing immediately, attempt a few simple retries with exponential
    # backoff.  If all retries fail, surface a more helpful ``RuntimeError`` so
    # callers don't see an opaque JSON decoding stack trace.
    completion = None
    for attempt in range(3):
        try:
            completion = client.chat.completions.create(
                model=target_model,
                messages=[{"role": "user", "content": prompt}],
                max_tokens=max_tokens,
                extra_body={"max_output_tokens": max_tokens},
                extra_headers={
                    "HTTP-Referer": "https://github.com/gavellm",
                    "X-Title": "gavellm",
                },
            )
            break
        except (JSONDecodeError, httpx.HTTPError) as exc:  # pragma: no cover - network
            if attempt == 2:
                raise RuntimeError("Failed to retrieve completion") from exc
            time.sleep(2**attempt)
    assert completion is not None  # for type checkers
    usage = getattr(completion, "usage", None)
    if hasattr(usage, "model_dump"):
        usage = usage.model_dump()
    return {
        "message": completion.choices[0].message.content,
        "usage": usage,
    }

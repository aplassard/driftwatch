# AGENTS

This repository uses [uv](https://docs.astral.sh/uv/) for Python dependency management and task execution.

## Development setup
- Run `uv sync --dev` to create the `.venv` and install runtime + test dependencies.
- Execute tools inside that environment with `uv run <command>`, e.g. `uv run pytest`.

## Testing
- Unit tests run without network access.
- Integration tests are marked with `@pytest.mark.integration` and require `OPENAI_API_KEY` and `OPENAI_BASE_URL` in your environment. For local runs, place them in a `.env` file at the repo root (do not commit secrets).
- The integration test makes a real request to `openai/gpt-5-nano` and will skip automatically if the API is unreachable.

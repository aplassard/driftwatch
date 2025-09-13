# driftwatch
DriftWatch tracks LLM performance over time to identify degradation in accuracy, latency, and behavior. It runs a 100-item rotating eval plus a per-minute sentinel across multiple models to surface changes as they happen.

## Sentinel CLI

Install dependencies and expose the CLI:

```bash
uv sync --dev
```

Run a specific test problem from either the GSM8K or ARC-Challenge benchmarks
against one or more models. Select the dataset with ``--dataset`` (defaults to
``gsm8k``). Control sampling diversity with ``--temperature`` (defaults to
``0.7``). The results are written as a JSONL file in the provided output
directory. The file name is the timestamp at the start of the run.

```bash
uv run python -m driftwatch.cli --dataset arc-challenge --index 5 --models openai/gpt-5-nano other/model --threads 2 --temperature 0.7 --output-dir results/
```

Each line contains the prompt, raw OpenAI response object, token usage, latency,
and whether the model produced the correct final answer.

### Cron job

To execute one problem every minute using the current minute as the question
index:

```
* * * * * cd /path/to/driftwatch && \
    uv run python -m driftwatch.cli --index $(date +\%M) \
    --models openai/gpt-5-nano another/model \
    --threads 4 \
    --output-dir /path/to/output >> /path/to/cron.log 2>&1
```

At 11:29 this will evaluate problem 29 from the GSM8K test set. Adjust the
model list and output location as needed. Include ``--dataset arc-challenge`` to
schedule ARC-Challenge problems instead.

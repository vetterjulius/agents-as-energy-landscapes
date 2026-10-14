# Isolated LLM PoC rewrite

This folder is a minimal, standalone replacement candidate for the existing LLM PoC. It does not modify the simulated benchmark or its energy formulations, and it is not intended to implement the paper's main benchmark a second time. It reuses the existing frozen orchestration and task definitions by import/copy while keeping its runner and Gemini client isolated here.

## Fixed experiment

There is one live Gemini mode, with three pinned worker agents:

- Analyst: `gemini-3.5-flash-lite`
- Extractor: `gemma-4-26b-a4b-it`
- Synthesist: `gemini-3.1-flash-lite`
- LLM judge: `gemma-4-26b-a4b-it`

The fixed design has 3 conditions × 2 blocks × 3 episodes × 3 repetitions × 10 tasks = **540 worker calls**. Each result is scored with one separate judge call, so a full run makes **540 task judge calls**, plus one judge preflight call before any worker request on a fresh run (1,081 planned Gemini calls for a fresh full run). Resume attempts perform a new preflight before continuing. Judge requests set Gemma 4 `thinkingLevel` to `minimal` and allow 32 output tokens. Each response records its `finishReason`, thought-token count, candidate-token count, and total-token count. HTTP 5xx responses are retried up to the configured limit; other API failures stop the run.

The run writes a per-task CSV, JSON summary, and atomically updated `llm_poc_rewrite_checkpoint.json` under `results/llm_poc_rewrite` by default. Before any worker call, the runner makes a one-task judge preflight and requires the expected rubric score; a failed preflight aborts without starting worker requests. Every task judge result must have `finishReason=STOP` and a parseable in-range integer score. A malformed score or non-`STOP` finish reason aborts the run and leaves the current worker result and judge failure checkpointed for diagnosis/resume; invalid judge responses are never converted into apparent score-zero outcomes. The output is hypothesis-generating transfer evidence only, not a confirmatory benchmark.

Resume an interrupted run with the same output directory:

```bash
python -m llm_poc_rewrite.run --resume
```

A normal run refuses to overwrite an existing checkpoint; use `--resume` to continue it or choose a new output directory for a fresh run. A completed checkpoint cannot be resumed.

## Safe design check and manual execution

The design and call counts can be inspected without an API key or any API request:

```bash
python -m llm_poc_rewrite.run --plan-only
```

A live run requires `GEMINI_API_KEY` and is intentionally not run as part of local validation. Run manually only when you are ready to spend the API quota:

```bash
python -m llm_poc_rewrite.run
```

The old `llm_poc` folder and tests are deliberately left in place until the new implementation has been reviewed and manually validated; replacing them is a separate final step.

# Isolated LLM PoC rewrite

This folder is a minimal, standalone replacement candidate for the existing LLM PoC. It does not modify the simulated benchmark or its energy formulations, and it is not intended to implement the paper's main benchmark a second time. It reuses the existing frozen orchestration and task definitions by import/copy while keeping its runner and Gemini client isolated here.

## Fixed experiment

There is one live Gemini mode, with three pinned worker agents:

- Analyst: `gemini-3.5-flash-lite`
- Extractor: `gemma-4-26b-a4b-it`
- Synthesist: `gemini-3.1-flash-lite`
- LLM judge: `gemma-4-26b-a4b-it`

The fixed design has 3 conditions × 2 blocks × 3 episodes × 3 repetitions × 10 tasks = **540 worker calls**. Each result is scored with one separate judge call, so a complete run makes **540 judge calls (1,080 Gemini API calls total)**. HTTP 5xx responses are retried up to five times with exponential backoff (1, 2, 4, 8, and 16 seconds); other API/transport failures stop the run.

The run writes a per-task CSV, JSON summary, and atomically updated `llm_poc_rewrite_checkpoint.json` under `results/llm_poc_rewrite` by default. Each completed task is checkpointed with its worker and judge output. If interrupted during judge scoring, the checkpoint also saves the worker result, so resume repeats only the unfinished judge call instead of paying for the worker again. The output is hypothesis-generating transfer evidence only, not a confirmatory benchmark. If the judge returns malformed or out-of-range text, the raw text is kept in the CSV, a parse error is counted in the summary, and that item receives the conservative score 0 so the run can continue. API/transport errors still stop the run.

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

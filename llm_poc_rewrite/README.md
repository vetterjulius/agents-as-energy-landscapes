# Isolated LLM PoC rewrite

This folder is a minimal, standalone replacement candidate for the existing LLM PoC. It does not modify the simulated benchmark or its energy formulations, and it is not intended to implement the paper's main benchmark a second time. It reuses the existing frozen orchestration and task definitions by import/copy while keeping its runner and Gemini client isolated here.

## Fixed experiment

There is one live Gemini mode, with three pinned worker agents:

- Analyst: `gemini-3.5-flash-lite`
- Extractor: `gemma-4-26b-a4b-it`
- Synthesist: `gemini-3.1-flash-lite`
- LLM judge: `gemma-4-26b-a4b-it`

The fixed design has 3 conditions × 2 blocks × 3 episodes × 3 repetitions × 10 tasks = **540 worker calls**. Tasks with exact arithmetic or extraction answers are scored deterministically from episode-specific answer keys; only the two semantic synthesis tasks are sent to the LLM judge (**108 judge calls**), plus one judge preflight on a fresh run (649 Gemini calls total). Resume attempts perform a new preflight before continuing. Judge requests set Gemma 4 `thinkingLevel` to `minimal` and allow 32 output tokens. Judge responses record finish reason and token metadata. HTTP 5xx responses are retried up to the configured limit; other API failures stop the run. The score is not a general agent-quality metric; this compact task family tests role allocation on controlled information-processing tasks only.

The `shift_mid` block changes the dependency edges after episode 0, but the adaptive controller does **not** receive those ground-truth edges: its state update uses its own prior assignment/co-assignment observation. Therefore this is a test of whether assignment-history adaptation improves downstream task scores under a changed environment, not a direct test that the controller discovers the hidden dependency graph. The live pilot remains small and hypothesis-generating; the fully simulated benchmark is the central empirical evaluation.

The run writes a per-task CSV, JSON summary, and atomically updated `llm_poc_rewrite_checkpoint.json` under `results/llm_poc_rewrite` by default. Before any worker call, the runner makes a one-task judge preflight and requires the expected rubric score; a failed preflight aborts without starting worker requests. Semantic judge results must have `finishReason=STOP` and a parseable in-range integer score. Invalid judge responses are checkpointed and abort rather than becoming zero scores. The output is hypothesis-generating transfer evidence, not a confirmatory benchmark.

Resume an interrupted run with the same output directory:

```bash
python -m llm_poc_rewrite.run --resume
```

A normal run refuses to overwrite an existing checkpoint; use `--resume` to continue it or choose a new output directory for a fresh run. A completed checkpoint cannot be resumed.

## Real-issue transfer pilot

The plan is documented in [REAL_ISSUE_PILOT_PLAN.md](REAL_ISSUE_PILOT_PLAN.md). The unsupported Flask cohort is replaced with a two-node, dated dependency-path change: Gensim #3525 and Flair #3441 share a documented SciPy 1.13 import break at `G0`; Flair 0.14.0 removes Gensim from core imports at `G1`. This is a package-dependency proxy, not measured coding synergy. Since positive `Theta` rewards same-agent co-assignment, using this edge also imposes an unvalidated same-agent heuristic; it does not establish that these fixes benefit from shared ownership. The pilot-specific orchestration helpers now initialize `Theta=G0`, use `theta-only` adaptation after assignment, and expose assignment/state diagnostics; ordinary rewrite defaults are unchanged. Run `python -m llm_poc_rewrite.run --real-issue-plan-only` for the offline two-episode mechanism trace and the budget of 36 planned patch generations (3 conditions × 3 repetitions × 2 issues × 2 episodes), with no judge/preflight or API requests. This is not a live run: validated coding feature vectors and historical reproduction tests remain gates. [real_issue_design.py](real_issue_design.py) is the offline fail-closed feature-schema gate; earlier six-issue capacity-allocation scaffolding has been removed.

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

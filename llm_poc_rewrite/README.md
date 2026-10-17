# Isolated LLM PoC rewrite

This folder is a minimal, standalone replacement candidate for the existing LLM PoC. It does not modify the simulated benchmark or its energy formulations, and it is not intended to implement the paper's main benchmark a second time. It reuses the existing frozen orchestration and task definitions by import/copy while keeping its runner and Gemini client isolated here.

## Fixed experiment

There is one live Gemini mode, with three pinned worker agents:

- Analyst: `gemini-3.5-flash-lite`
- Extractor: `gemma-4-26b-a4b-it`
- Synthesist: `gemini-3.1-flash-lite`
- LLM judge: `gemma-4-26b-a4b-it`

The fixed design has 3 conditions × 2 blocks × 3 episodes × 3 repetitions × 10 tasks = **540 worker calls**. Tasks with exact arithmetic or extraction answers are scored deterministically from episode-specific answer keys; only the two semantic synthesis tasks are sent to the LLM judge (**108 judge calls**), plus one judge preflight on a fresh run (649 Gemini calls total). Resume attempts perform a new preflight before continuing. Judge requests set Gemma 4 `thinkingLevel` to `minimal` and allow 32 output tokens. Judge responses record finish reason and token metadata. HTTP 5xx responses are retried up to the configured limit; other API failures stop the run. The score is not a general agent-quality metric; this compact task family tests role allocation on controlled information-processing tasks only.

The `shift_mid` block changes synthetic dependency edges after episode 0, but the adaptive controller does **not** receive those ground-truth edges: its state update uses its own assignment/co-assignment observation. This tests assignment-history adaptation on a controlled synthetic task family, not direct discovery of hidden dependencies. The separate real-issue mode is only a mechanism smoke trace and does not introduce a task-input/dependency change between its episodes. The simulated benchmark remains the central empirical evaluation.

The run writes a per-task CSV, JSON summary, and atomically updated `llm_poc_rewrite_checkpoint.json` under `results/llm_poc_rewrite` by default. Before any worker call, the runner makes a one-task judge preflight and requires the expected rubric score; a failed preflight aborts without starting worker requests. Semantic judge results must have `finishReason=STOP` and a parseable in-range integer score. Invalid judge responses are checkpointed and abort rather than becoming zero scores. The output is hypothesis-generating transfer evidence, not a confirmatory benchmark.

Resume an interrupted run with the same output directory:

```bash
python -m llm_poc_rewrite.run --resume
```

A normal run refuses to overwrite an existing checkpoint; use `--resume` to continue it or choose a new output directory for a fresh run. A completed checkpoint cannot be resumed.

## Real-issue pilot status

[REAL_ISSUE_PILOT_PLAN.md](REAL_ISSUE_PILOT_PLAN.md) describes a **pragmatic transfer pilot design, not an implemented real-issue benchmark**. The goal is to evaluate the orchestration approach on real issue text, repository snapshots, patches, and tests—not to replay history 1:1. A fully historical graph rewire is optional for this transfer question; any tracker-declared blockers and curated task-context affinities must remain distinct and labeled by evidence quality. The current Gensim #3525 / Flair #3441 pair is only a two-node historical candidate, not formal issue blocking or evidence of patch synergy. `--real-issue-plan-only` still runs only a zero-API mechanism smoke trace with synthetic placeholder tasks and role vectors; it does not change issue inputs/dependencies, generate/apply patches, or run hidden tests. The displayed 36-call count is proposed arithmetic, not a live schedule. The proposed transfer protocol targets 4–8 real, independently actionable issues with frozen base revisions, verified tests, paired patch attempts, and per-task outcome/cost records; it does **not** require a historical graph rewire. A live run is not implemented: task capture, sandboxed patch/test execution, and patch outcome logging are still future work. A draft, empty intake package and dependency-free validator now exist at [data/real_issue_transfer/v1/](data/real_issue_transfer/v1/study.json): run `python -m llm_poc_rewrite.transfer_data` to check record structure and see readiness blockers; `--export-worker-packets` emits only included worker-visible packets. Evaluator references/results are isolated in a separate ledger. The current Pydantic issue leads remain provisional: none has passed a controlled fail-on-base/pass-on-fix check, so no cohort is selected. The feature-schema gate is needed only for any claimed skill-based routing; without validated slot capabilities, use a fixed code-capable worker for feasibility transfer or defer routing claims.


## Real-issue intake package validation

From the repository root:

```bash
python -m llm_poc_rewrite.transfer_data
python -m llm_poc_rewrite.transfer_data --export-worker-packets
```

The first command exits nonzero only for malformed/inconsistent package records; a structurally valid draft can still report `ready: false` and enumerate study blockers. The export command intentionally returns only included solver-visible fields, never evaluator-only trusted-fix/test data. No candidate issue records have been added yet because web screening is not reproduction evidence.

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

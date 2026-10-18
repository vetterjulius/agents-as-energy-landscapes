# Isolated LLM PoC rewrite

This package has two separate parts: the active synthetic orchestration benchmark (`run.py`) and the draft v2 text-only real-issue response study (`transfer_data.py`). The former is an executable Gemini experiment; the latter is currently a validated data protocol, not a runner. Neither should be confused with the legacy v1 patch-transfer proposal or the unrelated synthetic mechanism-only `--real-issue-plan-only` diagnostic.

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

## Real-issue text-transfer pilot

The text-only pilot is separate from the synthetic allocation benchmark. It reuses the three existing `AGENT_MODELS` entries and their role-specific system prompts (analyst, extractor, synthesist) from `llm_poc_rewrite/config.py`, but runs each independently on the same authentic issue packet and bounded reference context—not as an orchestration pipeline. It does **not** evaluate agent frameworks, patch generation/application, repository-wide correctness, or test-pass claims. No large repository clone is needed or included. Agents get only the frozen task/context packet and have no browsing or repository tools. The pilot records each role/model response per issue/repetition and scores anonymized responses with a two-rater human rubric; report results per issue and descriptively. Because model and system prompt vary together by role, this estimates configured role/model/prompt bundles, not isolated model superiority.

The draft v2 intake package is at [data/real_issue_transfer/v2/study.json](data/real_issue_transfer/v2/study.json), with two distinct Pydantic issue families (#10694 nested union serialization; #10477 cached-property export) in [tasks.jsonl](data/real_issue_transfer/v2/tasks.jsonl). Each packet includes the bounded task brief, acceptance criteria, short official-doc context snippets, provenance/transformation note, and hashes. Comment-derived workarounds or causal hints stay out of the worker packet and in evaluator-only disclosure notes. Both examples are public historical reports, so the draft marks eligibility pending: treat them as leakage-aware exploratory candidates, not blind discovery evidence. The study protocol copies all three current `AGENT_MODELS` IDs from config and requires an exact match at validation; it uses temperature 0.0 and three repetitions per task/agent (18 responses; 36 rubric ratings). Reviewer identities have not been invented, and the package is intentionally not ready for a live run or packet export.

Validate the v2 draft and view blockers (exit status 0 means the package is structurally valid; `ready: false` is expected):

```bash
python -m llm_poc_rewrite.transfer_data
```

After actual independent eligibility adjudication and model preregistration, export only the worker-visible packets:

```bash
python -m llm_poc_rewrite.transfer_data --export-worker-packets
```

The export is fail-closed until readiness gates pass. The v1 patch/test package remains available only for legacy compatibility via `python -m llm_poc_rewrite.transfer_data --version 1`; it is not the selected study design. The text-only pilot evaluates human-rated agent responses; it does not use the existing synthetic-task LLM judge. See [REAL_ISSUE_PILOT_PLAN.md](REAL_ISSUE_PILOT_PLAN.md) for model/role interpretation, limitations, and the separation from the historical mechanism smoke trace.

## Safe design check and manual execution

The design and call counts can be inspected without an API key or any API request:

```bash
python -m llm_poc_rewrite.run --plan-only
```

A live run requires `GEMINI_API_KEY` and is intentionally not run as part of local validation. Run manually only when you are ready to spend the API quota:

```bash
python -m llm_poc_rewrite.run
```

The pre-existing `llm_poc` package remains in the repository as a separate legacy implementation; it is not imported by `llm_poc_rewrite` and is not replaced by this package.

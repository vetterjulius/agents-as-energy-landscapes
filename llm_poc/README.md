# LLM Orchestration PoC (supplemental pilot)

**Status: offline-tested; live Gemini inference is opt-in and requires `GEMINI_API_KEY`.**

The PoC is a separate, hypothesis-generating external validation. It does not modify the controlled benchmark or its results.

## Layers and conditions

- **Layer 1 — Energy formulation:** frozen `landscape.Landscape` / `LandscapeState`, with design-time task/agent embeddings and declared dependencies.
- **Layer 2 — Solver:** frozen `controlled_benchmark.solvers` (Conventional Greedy / Energy Greedy / Simulated Annealing) under its 500-evaluation solver budget.
- **Layer 3 — Workers:** Gemma 4 31B IT through the direct Google Gemini API, or deterministic offline mocks. Worker answers do not change assignments within an episode.

| Condition | Layer 1 | Layer 2 | Adaptation |
|---|---|---|---|
| `baseline` (B0′) | none | Conventional Greedy | none |
| `static_energy` (B1′) | fixed dependency graph | Energy Greedy | none |
| `adaptive_energy` (B3′) | learned κ / Θ EMA | Simulated Annealing | η_memory=0.05, η_theta=0.10 |

The fixed task bank has 10 tasks and 3 specialization prompts. The `shift_mid` block changes dependencies from episode 2. These design elements are supplemental PoC work, not changes to the benchmark.

## Key and model

Set `GEMINI_API_KEY` in `.env` to the API key created for the AI Studio project. An `AQ.` prefix is the current Google AI Studio Authentication Key format, not evidence by itself that this particular key/project has model access. The implementation sends it via the `x-goog-api-key` header; it never prints or records the value. The runner loads `.env` and calls the official direct endpoint for `gemma-4-31b-it`.

Google documents Gemma 4 31B IT as supported on the Gemini API. Project/model quotas and free access are account-specific and can change; check the AI Studio quota page. One minimal model reachability probe is available via `python -m llm_poc.run --probe-model` and uses one tiny generation request. It is separate from a pilot run.

**Privacy:** Google states that prompts and outputs on unpaid services may be used to improve products and may be reviewed by humans. Do not send personal, confidential, or sensitive data. This PoC's inputs are designed synthetic examples.

## Running

```bash
# Deterministic offline pipeline check; makes no API requests
python -m llm_poc.run

# Optional live pilot defaults to one stationary episode × one repetition:
# 3 conditions × 1 episode × 1 repetition × 10 tasks = 30 worker calls, no LLM judge
python -m llm_poc.run --worker-mode gemini --output-dir results/llm_poc_gemma4_pilot

# Resume an interrupted run (keeps its existing global attempt ceiling)
python -m llm_poc.run --worker-mode gemini --resume --output-dir results/llm_poc_gemma4_pilot

# The explicit one-call model/key reachability check (not a benchmark run)
python -m llm_poc.run --probe-model
```

Live inference is disabled by default. The live pilot allows at most 30 planned task calls and three retries per transient task failure (hard ceiling 60 total HTTP attempts, including retries and resumed work), rejects designs above the task-call cap before sending requests, disables LLM judging, and uses a deterministic rubric proxy instead. Only network/timeouts and HTTP 408/500/502/503/504 are retried; 400/402/403/429 and model/configuration errors stop immediately. Retries use exponential delays starting at 10 seconds, capped at 60 seconds, with ±20% jitter. A valid `Retry-After` is honored; waits above the 5-minute safety bound abort rather than being ignored. Every attempt is journaled before retry. The proxy is **not** an independent or valid model-quality judge; treat its scores as a transparent plumbing aid only. For meaningful response-quality measurement, collect worker outputs and score them independently (for example, blinded human rubric review) in a separately approved step. The full 540-worker-call design and a 540-call same-model judge are not the free-tier default.

The request protocol pins the model ID, uses temperature 0, top-p 1, explicit stable per-task seeds, bounded output tokens, a 120-second timeout, and a 2.2-second minimum interval (below the screenshot's 30 RPM cap). API seed is best-effort, not a guarantee of identical hosted outputs; Google's returned `modelVersion` is checked and captured. The flushed JSONL journal stores every HTTP attempt. On an interrupted run, rerun with `--resume` and the same task design/seed/model in the same `--output-dir`; previously completed identical requests are reused rather than regenerated. The existing total-attempt cap may not be increased. An exhausted task may receive at most two additional attempts on resume (up to six attempts total for that task), but only if its last event is a retryable terminal failure. If the last journal event scheduled a retry before interruption, resume continues within the original four-attempt budget; it does not grant a fresh retry budget. Raising the timeout is allowed. Resume rejects changed task prompts/request parameters for already-started coordinates rather than silently replaying them. The journal and run manifest are checked against the requested configuration; don't edit them manually. A request with only a `request_started` event and no completion may have been processed remotely despite a local timeout; it is not assumed complete and retrying it can consume quota again. The persistent journal count enforces the original cumulative 60-attempt cap across restarts, including resumed attempts. Non-transient failures stop the run immediately. `--block`, `--episodes`, and `--repetitions` let you choose a smaller run; `--limit-calls` rejects an oversized design rather than silently truncating it. Existing output directories with unrelated artifacts are not resumable; use a new directory for a new live run.

The free quota may still be insufficient or unavailable for a project. Google quota resets and model limits are project-specific; the pilot's 30 calls are a ceiling, not a guarantee of access or free availability.

## Critical review and current evidence

See [POC_REVIEW.md](POC_REVIEW.md) for the detailed engineering/validity review, the current partial-run snapshot, and staged recommendations. In short, the current live results are an incomplete plumbing pilot, not model-quality or benchmark evidence; the rubric proxy is not a valid semantic evaluator.

## Artifacts and interpretation

Runs write `poc_episodes.csv`, `poc_summary.json`, `poc_report.md`, and `poc_judges.csv` in the chosen output directory. The summary captures requested/resolved model version, provider, API protocol, decoding parameters, seed scheme, source/dependency hashes, installed package versions, and worktree status. Per-task rows include model responses, errors, token usage, latency, assignment, and solver diagnostics. Never share artifacts containing outputs if they could contain sensitive information.

Results are hypothesis-generating only—not a confirmatory benchmark, and must not be combined with or substituted for benchmark results. Offline mock outputs validate plumbing and solver mechanism wiring only, never LLM behavior. Hosted model versions may drift and seeds do not ensure exact determinism.

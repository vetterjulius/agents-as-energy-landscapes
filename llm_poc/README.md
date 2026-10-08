# LLM Orchestration PoC (supplemental pilot)

**Status: offline-tested; live Gemini inference is opt-in and requires `GEMINI_API_KEY`.**

The PoC is a separate, hypothesis-generating external validation. It does not modify the controlled benchmark or its results.

## Layers and conditions

- **Layer 1 — Energy formulation:** frozen `landscape.Landscape` / `LandscapeState`, with design-time task/agent embeddings and declared dependencies.
- **Layer 2 — Solver:** frozen `controlled_benchmark.solvers` (Conventional Greedy / Energy Greedy / Simulated Annealing) under its 500-evaluation solver budget.
- **Layer 3 — Workers:** Heterogeneous multi-agent slots via the direct Google Gemini API, or deterministic offline mocks. Worker answers do not change assignments within an episode.

| Condition | Layer 1 | Layer 2 | Adaptation |
|---|---|---|---|
| `baseline` (B0′) | none | Conventional Greedy | none |
| `static_energy` (B1′) | fixed dependency graph | Energy Greedy | none |
| `adaptive_energy` (B3′) | learned κ / Θ EMA | Simulated Annealing | η_memory=0.05, η_theta=0.10 |

The fixed task bank has 10 tasks and 3 specialization prompts. The `shift_mid` block changes dependencies from episode 2. These design elements are supplemental PoC work, not changes to the benchmark.

## Agent Slots & Heterogeneous Models

The PoC uses three specialized agent slots, each mapped to a dedicated model optimized for its role (customizable via CLI arguments):

1. **Analyst (Agent 0):** `gemini-3.5-flash-lite` (precise reasoning and problem breakdown; override via `--analyst-model`).
2. **Extractor (Agent 1):** `gemma-4-26b-a4b-it` (meticulous data extraction; override via `--extractor-model`).
3. **Synthesist (Agent 2):** `gemini-2.5-flash` (coherent synthesis; override via `--synthesist-model`).

## Key and model

Set `GEMINI_API_KEY` in `.env` to the API key created for the AI Studio project. The implementation sends it via the `x-goog-api-key` header; it never prints or records the value. The runner loads `.env` and calls the official direct endpoint for each agent's pinned model.

Google documents these models as supported on the Gemini API. Project/model quotas and free access are account-specific and can change; check the AI Studio quota page. One minimal model reachability probe is available via `python -m llm_poc.run --probe-model` and uses one tiny generation request. It is separate from a pilot run.

**Privacy:** Google states that prompts and outputs on unpaid services may be used to improve products and may be reviewed by humans. Do not send personal, confidential, or sensitive data. This PoC's inputs are designed synthetic examples.

## Running

```bash
# Deterministic offline pipeline check; makes no API requests
python -m llm_poc.run

# Optional live pilot defaults to one stationary episode × one repetition:
# 3 conditions × 1 episode × 1 repetition × 10 tasks = 30 worker calls
python -m llm_poc.run --worker-mode gemini --output-dir results/llm_poc_pilot

# Optional LLM judge evaluation enabled
python -m llm_poc.run --worker-mode gemini --enable-judge --output-dir results/llm_poc_pilot

# Resume an interrupted run (keeps its existing global attempt ceiling)
python -m llm_poc.run --worker-mode gemini --resume --output-dir results/llm_poc_pilot

# The explicit one-call model/key reachability check (not a benchmark run)
python -m llm_poc.run --probe-model
```

Live inference is disabled by default. The live pilot allows at most 30 planned task calls and three retries per transient task failure (hard ceiling 60 total HTTP attempts, including retries and resumed work), rejects designs above the task-call cap before sending requests. Only network/timeouts and HTTP 408/500/502/503/504 are retried; 400/402/403/429 and model/configuration errors stop immediately. Retries use exponential delays starting at 10 seconds, capped at 60 seconds, with ±20% jitter. A valid `Retry-After` is honored; waits above the 5-minute safety bound abort rather than being ignored. Every attempt is journaled before retry.

The request protocol pins model IDs per agent slot, uses temperature 0, top-p 1, explicit stable per-task seeds, bounded output tokens, a 120-second timeout, and a 2.2-second minimum interval. API seed is best-effort, not a guarantee of identical hosted outputs; Google's returned `modelVersion` is checked and captured. The flushed JSONL journal stores every HTTP attempt. On an interrupted run, rerun with `--resume` and the same task design/seed in the same `--output-dir`; previously completed identical requests are reused rather than regenerated. Non-transient failures stop the run immediately. `--block`, `--episodes`, and `--repetitions` let you choose a smaller run; `--limit-calls` rejects an oversized design rather than silently truncating it.

The free quota may still be insufficient or unavailable for a project. Google quota resets and model limits are project-specific; the pilot's 30 calls are a ceiling, not a guarantee of access or free availability.

## Critical review and current evidence

See [POC_REVIEW.md](POC_REVIEW.md) for the detailed engineering/validity review, the current partial-run snapshot, and staged recommendations.

## Artifacts and interpretation

Runs write `poc_episodes.csv`, `poc_summary.json`, `poc_report.md`, and `poc_judges.csv` in the chosen output directory. The summary captures requested/resolved model versions per agent slot, provider, API protocol, decoding parameters, seed scheme, source/dependency hashes, installed package versions, and worktree status. Per-task rows include model responses, errors, token usage, latency, assignment, and solver diagnostics. Never share artifacts containing outputs if they could contain sensitive information.

Results are hypothesis-generating only—not a confirmatory benchmark, and must not be combined with or substituted for benchmark results. Offline mock outputs validate plumbing and solver mechanism wiring only, never LLM behavior. Hosted model versions may drift and seeds do not ensure exact determinism.

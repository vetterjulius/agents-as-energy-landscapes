# LLM Orchestration PoC (external validation)

**Status: implemented and dry-run-verified. Live OpenRouter execution is a separate,
explicitly-triggered step (spends API budget).**

Three-layer separation, exactly as specified in `paper_artifacts/llm_poc_design.md`:

- **Layer 1 — Energy formulation:** the frozen `landscape.Landscape` / `LandscapeState`
  from the main benchmark, fed with design-time agent/task embeddings and the declared
  dependency graph. *Unmodified.*
- **Layer 2 — Solver:** the frozen `controlled_benchmark.solvers` (Conventional Greedy /
  Energy Greedy / Simulated Annealing) under the same 500-evaluation `BudgetedLandscape`
  budget. *Unmodified.*
- **Layer 3 — LLM workers:** new code in this package — OpenRouter chat-completion workers
  (or a deterministic mock) plus a rubric-anchored judge. Worker outcomes are measurement
  channels only; they never influence the assignment within an episode.

## Conditions (B0′ / B1′ / B3′ mirror the benchmark's B0 / B1 / B3)

| Condition | Layer 1 | Layer 2 | Adaptation |
|---|---|---|---|
| `baseline` (B0′) | none | Conventional Greedy | none |
| `static_energy` (B1′) | fixed Θ₀ = declared dependencies | Energy Greedy | none |
| `adaptive_energy` (B3′) | κ_t / Θ_t EMA | Simulated Annealing | η_memory=0.05, η_theta=0.10 |

## Design-time task family (deterministic, in `tasks.py`)

- 10 tasks per episode, 3 specialization slots (analyst / extractor / synthesist),
  8-d hand-designed embeddings (dims 0–2 analysis, 3–5 extraction, 6–7 synthesis).
- 3 **declared dependency pairs** per episode (downstream prompts consume upstream
  output when both land on the same slot — co-execution is what the interaction term rewards).
- Dependency-downstream tasks are **cross-functional** (equal affinity to two slots), so
  co-location is a genuine energy trade-off rather than an affinity accident.
- 3 rubric levels per task with written anchors; every episode has fixed input payloads.
- `shift_mid` block: at episode ≥ 2 the dependency structure changes
  (1→3 and 6→7 become 0→3 and 5→7), mirroring the benchmark's Dependency Change.

## Running

```bash
# 1. Dry run (offline, deterministic, ~3 s): validates the full pipeline
python -m llm_poc.run

# 2. Live workers + LLM judging (spends API budget; ~540 calls + judging)
python -m llm_poc.run --worker-mode openrouter

# 3. Live workers, rubric-proxy scoring (workers only; no judge calls)
python -m llm_poc.run --worker-mode openrouter --no-judge

# useful flags: --repetitions N  --episodes N  --limit-calls N  --output-dir DIR  --model ID
```

`OPENROUTER_API_KEY` is read from `.env` (already present in this repo). The runner
fails fast with a clear message if it is missing in live modes.

## Artifacts

- `results/llm_poc/poc_episodes.csv` — one row per executed task: condition, block,
  episode, repetition, assignment, dependency co-execution, worker output/error,
  usage (prompt/completion tokens), latency, rubric score, judge raw level, solver
  diagnostics, and adaptation-state norms (‖Θ‖, ‖Θ − G_gt‖, ‖κ‖).
- `results/llm_poc/poc_summary.json` — per-condition aggregates, per-block splits,
  worker failure counts, token totals, and the transferability correlation
  (realized external score vs mean quality).
- `results/llm_poc/poc_report.md` — human-readable report with the three
  pre-registered expectations and their observed outcomes.

## Dry-run results (mock workers — pipeline validation only, NOT evidence about LLMs)

| Condition | mean quality | dep. satisfaction | realized score |
|---|---|---|---|
| B0′ Conventional Greedy | 0.267 | 0.06 | 0.214 |
| B1′ Static Energy | 0.317 | 0.94 | 0.258 |
| B3′ Adaptive Energy | 0.317 | 0.94 | 0.258 |

Mechanism checks reproduced in dry-run (each covered by a test in `tests/test_llm_poc.py`):

- energy conditions co-locate dependency pairs (0.94–1.00) where the baseline does not (0.06);
- the solver moves the cross-functional dependency task (id 3) onto the extractor slot —
  a pure coordination trade-off (equal affinity 1.917 vs 1.917; co-location gain 1/(N·M));
- adaptive Θ drifts when stationary (‖Θ − G‖: 0.24 → 0.65 over 3 episodes) and moves toward
  the new structure after the shift (2.506 < static 2.828 after one EMA step);
- B0′ uses 0 energy evaluations; B1′ converges early; B3′ exhausts the budget.

## Scope limits (unchanged from the design doc)

Hypothesis-generating only; no multiplicity-corrected confirmatory claims from the PoC.
The mock mode validates plumbing and mechanism wiring, never LLM behaviour. Live results
replace `scorer: rubric_proxy` with `scorer: judge:<model>` and add second-judge spot
checks (20% subsample) for rater-agreement reporting.

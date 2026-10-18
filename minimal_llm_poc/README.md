# Minimal LLM PoC – Dependency Change Transfer

This package implements the minimal empirical transfer of the **Dependency Change** scenario from the `controlled_benchmark` to real LLM agents.

## Key Design Principles

1. **Lightweight and Minimal**: No multi-seed sweeps or statistical battery required; primary empirical evidence remains `controlled_benchmark`.
2. **Transfer of Dependency Change**: Models an evolving e-commerce platform across 8 subsystem slots and 3 architectural phases:
   - **Phase 1 (Monolith Split, Episodes 0–1)**: Synergies `(0, 1)`, `(2, 3)`, `(4, 5)`.
   - **Phase 2 (Gateway & Direct Checkout, Episodes 2–3)**: Synergies `(0, 6)`, `(2, 4)`, `(3, 5)`.
   - **Phase 3 (Event-Driven Reactive Platform, Episodes 4–5)**: Synergies `(0, 2)`, `(3, 6)`, `(5, 6)`, `(4, 7)`.
3. **No Heavy Frameworks or Sandboxes**: Solves self-contained coding and configuration prompts, evaluating traces deterministically (75%) and via LLM-as-judge (25%).
4. **Disjoint Judge Model & Catalog Coverage**:
   - `gemini-3.5-flash-lite`: `backend`, `frontend`
   - `gemini-3.1-flash-lite`: `infra`
   - `gemma-4-26b-a4b-it`: `security`
   - `gemini-3.8-flash`: `judge` (strictly disjoint from all agent roles)
5. **3 Orchestrators**:
   - `baseline`: Conventional myopic capability-greedy allocation.
   - `static_energy_sa`: Simulated Annealing on fixed initial interaction belief ($\Theta_0$).
   - `dynamic_energy_sa`: Simulated Annealing with online $\Theta_t$ and $\kappa_t$ adaptation.

## Usage

### 1. Verification / Mock Mode (Fast local test without API)
```bash
python -m minimal_llm_poc.run --mock
```

### 2. Live Run (Google AI Studio Gemini API)
Ensure `GEMINI_API_KEY` is set in your environment:
```bash
python -m minimal_llm_poc.run
```

### 3. Single Condition Run
```bash
python -m minimal_llm_poc.run --condition dynamic_energy_sa
```

## Output Artifacts
Results are written to `results/minimal_llm_poc/`:
- `episodes.csv`: Ground-truth energy, internal energy, base score, synergy bonus per episode.
- `task_traces.csv`: Detailed evaluation record, assigned agent, and raw response snippet for every task.
- `summary.json`: Aggregated metrics comparing the 3 orchestrators across phases.

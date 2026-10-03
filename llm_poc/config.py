"""PoC-wide constants and default configuration.

Frozen design values come from paper_artifacts/llm_poc_design.md and mirror the
main benchmark's hyperparameters where applicable (energy weights, adaptation
rates, budget semantics).
"""
from __future__ import annotations

import os
from dataclasses import dataclass, field

# ---------------------------------------------------------------------------
# API / execution
DEFAULT_MODEL = "minimax/minimax-m3:free"  # single fixed model; specialization via system prompt
OPENROUTER_URL = "https://openrouter.ai/api/v1/chat/completions"
API_TIMEOUT_SEC = 45.0
MAX_RETRIES = 1  # design: one retry on failure, then record as worker failure

# ---------------------------------------------------------------------------
# Scale (deliberately small: validation experiment, not a benchmark)
N_AGENTS = 3               # analyst / extractor / synthesist slots
N_TASKS = 10               # tasks per episode
D = 8                      # embedding dimensionality, mirrors benchmark d=8
N_EPISODES = 3             # episodes per scenario block
N_REPETITIONS = 3          # repetitions per (condition, episode, repetition)
BLOCKS = ("stationary", "shift_mid")  # 2 scenario blocks; shift injected after episode 2

# ---------------------------------------------------------------------------
# Layer 1/2 (identical frozen values to the synthetic benchmark)
LAMBDA_ALIGN = 0.5
LAMBDA_MEMORY = 0.5
INTERACTION_WEIGHT = 1.0
COST_WEIGHT = 1.0
RISK_WEIGHT = 1.0
ETA_MEMORY = 0.05          # kappa EMA
ETA_THETA = 0.10           # theta EMA
MAX_ENERGY_EVALUATIONS = 500  # per-episode budget for the SA/greedy solvers
SA_TEMPERATURE_INIT = 1.0
SA_MIN_TEMPERATURE = 0.01
SA_COOLING_RATE = 0.95

# ---------------------------------------------------------------------------
# Scoring
JUDGE_MODEL = DEFAULT_MODEL
JUDGE_TEMPERATURE = 0.0
RUBRIC_SPOTCHECK_RATE = 0.2   # design: 20% subsample double-rated
SPOTCHECK_JUDGE_B = "meta-llama/llama-3.3-70b-instruct"  # second judge for the subsample


@dataclass
class PoCConfig:
    """Runtime configuration for one PoC execution."""
    model: str = DEFAULT_MODEL
    worker_mode: str = "mock"            # "mock" | "openrouter"
    judge_enabled: bool | None = None    # None => True iff worker_mode == "openrouter"
    judge_model: str = JUDGE_MODEL
    spotcheck_judge_model: str = SPOTCHECK_JUDGE_B
    judge_temperature: float = JUDGE_TEMPERATURE
    n_repetitions: int = N_REPETITIONS
    max_retries: int = MAX_RETRIES
    seed: int = 20260923
    output_dir: str = "results/llm_poc"
    # populated from .env at runtime; never logged
    api_key_env: str = "OPENROUTER_API_KEY"

    def __post_init__(self) -> None:
        if self.judge_enabled is None:
            self.judge_enabled = self.worker_mode == "openrouter"

    @property
    def api_key(self) -> str:
        return os.environ.get(self.api_key_env, "")

    def require_api(self) -> None:
        if self.worker_mode == "openrouter" or self.judge_enabled:
            if not self.api_key:
                raise SystemExit(
                    f"[llm_poc] missing {self.api_key_env}; add it to .env or run with "
                    "--worker-mode mock --no-judge")

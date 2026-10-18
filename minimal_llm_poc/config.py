"""Fixed configuration for the Minimal LLM PoC.

Model IDs and API endpoint are carried over from the existing LLM PoC
implementations. Retry logic parameters are also preserved.
"""
from __future__ import annotations

# API endpoint and models (from llm_poc_rewrite/config.py)
GEMINI_API_URL = (
    "https://generativelanguage.googleapis.com/v1beta/models/{model}:generateContent"
)
API_TIMEOUT_SEC = 240.0
MIN_REQUEST_INTERVAL_SEC = 2.2

# Retry logic (exponential backoff for 5xx, 429, timeouts)
API_MAX_RETRIES = 7
API_RETRY_BASE_SEC = 2.0

# All available models
AVAILABLE_MODELS = frozenset({
    "gemini-3.8-flash",
    "gemini-3.5-flash-lite",
    "gemini-3.1-flash-lite",
    "gemma-4-26b-a4b-it",
})

# Agent model assignments (each model used at least once)
AGENT_MODELS = {
    "backend":  "gemini-3.5-flash-lite",
    "infra":    "gemini-3.1-flash-lite",
    "security": "gemma-4-26b-a4b-it",
    "frontend": "gemini-3.5-flash-lite",
}

# Judge model — NOT used by any agent
JUDGE_MODEL = "gemini-3.8-flash"

# Verify constraint: judge model not in any agent model
assert JUDGE_MODEL not in set(AGENT_MODELS.values()), (
    f"Judge model {JUDGE_MODEL} must not be used by any agent"
)
# Verify constraint: all models used at least once
_all_used = set(AGENT_MODELS.values()) | {JUDGE_MODEL}
assert _all_used == AVAILABLE_MODELS, (
    f"Not all models used. Missing: {AVAILABLE_MODELS - _all_used}"
)

# LLM generation parameters
WORKER_TEMPERATURE = 0.0
WORKER_TOP_P = 1.0
WORKER_MAX_TOKENS = 1024
JUDGE_TEMPERATURE = 0.0
JUDGE_MAX_TOKENS = 256

# Orchestrator and energy landscape parameters (from controlled_benchmark)
LAMBDA_ALIGN = 0.5
LAMBDA_MEMORY = 0.5
INTERACTION_WEIGHT = 1.0
COST_WEIGHT = 1.0
RISK_WEIGHT = 1.0
ETA_MEMORY = 0.05
ETA_THETA = 0.10
SA_TEMPERATURE_INIT = 1.0
SA_MIN_TEMPERATURE = 0.01
SA_COOLING_RATE = 0.95
MAX_ENERGY_EVALUATIONS = 150

# Scenario dimensions
NUM_AGENTS = 4
NUM_TASKS_PER_EPISODE = 8
DIM = 4  # Capability embedding dimension
NUM_EPISODES = 6
NUM_PHASES = 3
EPISODES_PER_PHASE = NUM_EPISODES // NUM_PHASES  # 2 episodes per phase

# Orchestrators to compare
CONDITIONS = ("baseline", "static_energy_sa", "dynamic_energy_sa")

# Output directory
OUTPUT_DIR = "results/minimal_llm_poc"

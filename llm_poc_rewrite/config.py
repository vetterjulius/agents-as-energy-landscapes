"""Fixed protocol settings for the isolated LLM transfer pilot."""

MODEL_PROVIDER = "google-ai-studio"
GEMINI_API_URL = "https://generativelanguage.googleapis.com/v1beta/models/{model}:generateContent"
API_TIMEOUT_SEC = 240.0
MIN_REQUEST_INTERVAL_SEC = 2.2
API_MAX_RETRIES = 7
API_RETRY_BASE_SEC = 2.0

AGENT_MODELS = {
    "analyst": "gemini-3.5-flash-lite",
    "extractor": "gemma-4-26b-a4b-it",
    "synthesist": "gemini-3.1-flash-lite",
}
JUDGE_MODEL = "gemma-4-26b-a4b-it"
ALLOWED_MODELS = frozenset((*AGENT_MODELS.values(), JUDGE_MODEL))
WORKER_TEMPERATURE = 0.0
JUDGE_TEMPERATURE = 0.0
WORKER_MAX_TOKENS = 512
JUDGE_MAX_TOKENS = 32
JUDGE_THINKING_LEVEL = "minimal"
JUDGE_PROTOCOL_VERSION = "gemma4-judge-minimal-thinking-v2"
SCORING_PROTOCOL_VERSION = "episode-gold-exact-v1"

N_AGENTS = 3
N_TASKS = 10
D = 8
N_EPISODES = 3
N_REPETITIONS = 3
BLOCKS = ("stationary", "shift_mid")
CONDITIONS = ("baseline", "static_energy", "adaptive_energy")
EXPECTED_WORKER_CALLS = len(CONDITIONS) * len(BLOCKS) * N_EPISODES * N_REPETITIONS * N_TASKS

# Fixed-node historical transfer design; never used by the normal run.
# The positive G0 edge is an unvalidated co-assignment heuristic, not a finding.
REAL_ISSUE_TASK_IDS = ("gensim-3525", "flair-3441")
REAL_ISSUE_TASK_COUNT = len(REAL_ISSUE_TASK_IDS)
REAL_ISSUE_G0 = ((0.0, 1.0), (1.0, 0.0))
REAL_ISSUE_G1 = ((0.0, 0.0), (0.0, 0.0))
REAL_ISSUE_ADAPTATION_MODE = "theta-only"
REAL_ISSUE_EPISODES = 2
REAL_ISSUE_EXPECTED_PATCH_CALLS = (
    len(CONDITIONS) * N_REPETITIONS * len(REAL_ISSUE_TASK_IDS) * REAL_ISSUE_EPISODES
)
# Tasks 7 and 9 are semantic synthesis tasks; other task outputs have deterministic checks.
LLM_JUDGE_TASK_IDS = (7, 9)
EXPECTED_JUDGE_CALLS = (
    len(CONDITIONS) * len(BLOCKS) * N_EPISODES * N_REPETITIONS
    * len(LLM_JUDGE_TASK_IDS)
)

LAMBDA_ALIGN = 0.5
LAMBDA_MEMORY = 0.5
INTERACTION_WEIGHT = 1.0
COST_WEIGHT = 1.0
RISK_WEIGHT = 1.0
ETA_MEMORY = 0.05
ETA_THETA = 0.10
MAX_ENERGY_EVALUATIONS = 500
SA_TEMPERATURE_INIT = 1.0
SA_MIN_TEMPERATURE = 0.01
SA_COOLING_RATE = 0.95

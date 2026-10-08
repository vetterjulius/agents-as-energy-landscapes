"""PoC-wide constants and configuration for the fixed-model experiment."""
from __future__ import annotations

import hashlib
import os
import platform
import re
import subprocess
from dataclasses import dataclass
from importlib import metadata
from pathlib import Path


MODEL_PROVIDER = "google-ai-studio"
MODEL_CATALOG_SNAPSHOT = "Gemma 4 supported model per Google AI docs, checked 2026-10-06"
GEMINI_API_URL = "https://generativelanguage.googleapis.com/v1beta/models/{model}:generateContent"
DEFAULT_MODEL = "gemma-4-26b-a4b-it"
ALLOWED_MODELS = {
    "gemma-4-26b-a4b-it",
    "gemini-3.5-flash-lite",
    "gemini-2.5-flash",
    "gemini-1.5-flash",
}
API_TIMEOUT_SEC = 120.0
MIN_REQUEST_INTERVAL_SEC = 2.2
MAX_RETRIES = 3
MAX_RESUME_EXTRA_ATTEMPTS = 2
MAX_LIVE_REQUEST_ATTEMPTS = MAX_RETRIES + 1
MAX_RESUME_REQUEST_ATTEMPTS = MAX_LIVE_REQUEST_ATTEMPTS + MAX_RESUME_EXTRA_ATTEMPTS
RETRY_BASE_SEC = 10.0
RETRY_MAX_SEC = 60.0
RETRY_JITTER_RATIO = 0.20
MAX_RETRY_AFTER_SEC = 300.0
MAX_LIVE_TASK_CALLS = 30
MAX_LIVE_API_CALLS = MAX_LIVE_TASK_CALLS * 2
PROBE_PROMPT = "Reply with exactly: OK"
PROBE_SYSTEM_PROMPT = "Reply with the requested text only."
RNG_SCHEME = (
    "sha256(base_seed|block|episode|repetition|task_id|role) mod 2^31; "
    "shared across conditions; sent as API seed where supported"
)
EXPERIMENT_PROTOCOL_VERSION = "gemini-api-gemma4-31b-pilot-v1"
MODEL_SEED_SUPPORTED = True
MODEL_PROVIDER_ONLY = True
WORKER_TEMPERATURE = 0.0
WORKER_TOP_P = 1.0
JUDGE_TEMPERATURE = 0.0
MAX_WORKER_TOKENS = 512
MAX_JUDGE_TOKENS = 8
MAX_WORKER_OUTPUT_CHARS = 1200
MAX_JUDGE_OUTPUT_CHARS = 16
SEED_BASE = 20260923
RUBRIC_SPOTCHECK_RATE = 0.0
SPOTCHECK_JUDGE_B = ""
JUDGE_MODEL = DEFAULT_MODEL

# ---------------------------------------------------------------------------
# Scale (small validation design, not a confirmatory benchmark)
N_AGENTS = 3
N_TASKS = 10
D = 8
N_EPISODES = 3
N_REPETITIONS = 3
BLOCKS = ("stationary", "shift_mid")

# ---------------------------------------------------------------------------
# Layer 1/2 design values
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
JUDGE_MODEL = DEFAULT_MODEL


def _sha256_files(paths: list[Path], root: Path) -> str:
    digest = hashlib.sha256()
    for path in sorted(paths, key=lambda item: item.relative_to(root).as_posix()):
        if not path.is_file():
            continue
        digest.update(path.relative_to(root).as_posix().encode("utf-8"))
        digest.update(b"\0")
        digest.update(path.read_bytes())
        digest.update(b"\0")
    return digest.hexdigest()


def collect_environment(root: Path | None = None) -> dict:
    """Return a run manifest for software/source provenance without API secrets."""
    root = (root or Path(__file__).resolve().parents[1]).resolve()
    try:
        revision = subprocess.run(
            ["git", "rev-parse", "HEAD"], cwd=root, capture_output=True,
            text=True, check=True, timeout=5,
        ).stdout.strip()
        dirty = bool(subprocess.run(
            ["git", "status", "--porcelain"], cwd=root, capture_output=True,
            text=True, check=True, timeout=5,
        ).stdout.strip())
    except (OSError, subprocess.SubprocessError):
        revision, dirty = "unavailable", None

    requirements = root / "requirements.txt"
    direct_packages: set[str] = set()
    if requirements.is_file():
        for line in requirements.read_text(encoding="utf-8").splitlines():
            match = re.match(r"\s*([A-Za-z0-9_.-]+)", line)
            if match and not line.lstrip().startswith(("#", "-")):
                direct_packages.add(match.group(1))
    direct_packages.update(("torch", "numpy", "pandas", "python-dotenv"))
    versions: dict[str, str] = {}
    for package in sorted(direct_packages, key=str.lower):
        try:
            versions[package] = metadata.version(package)
        except metadata.PackageNotFoundError:
            versions[package] = "not-installed"

    source_files = list((root / "llm_poc").glob("*.py"))
    source_files += list((root / "controlled_benchmark").glob("*.py"))
    source_files += [root / "landscape.py", root / "tests" / "test_llm_poc.py"]
    lock_files = [requirements] if requirements.is_file() else []
    return {
        "git_revision": revision,
        "git_worktree_dirty": dirty,
        "source_sha256": _sha256_files(source_files, root),
        "source_files": sorted(path.relative_to(root).as_posix() for path in source_files
                                if path.is_file()),
        "requirements_sha256": _sha256_files(lock_files, root) if lock_files else None,
        "python_version": platform.python_version(),
        "python_implementation": platform.python_implementation(),
        "platform": platform.platform(),
        "package_versions": versions,
        "rng_scheme": RNG_SCHEME,
    }


@dataclass
class PoCConfig:
    """Runtime configuration for one PoC execution."""
    model: str = DEFAULT_MODEL
    worker_mode: str = "mock"
    judge_enabled: bool | None = None
    judge_model: str = JUDGE_MODEL
    spotcheck_judge_model: str = SPOTCHECK_JUDGE_B
    provider: str = MODEL_PROVIDER
    pinned_provider: bool = MODEL_PROVIDER_ONLY
    protocol_version: str = EXPERIMENT_PROTOCOL_VERSION
    model_catalog_snapshot: str = MODEL_CATALOG_SNAPSHOT
    worker_temperature: float = WORKER_TEMPERATURE
    worker_top_p: float = WORKER_TOP_P
    judge_temperature: float = JUDGE_TEMPERATURE
    n_repetitions: int = N_REPETITIONS
    n_episodes: int = N_EPISODES
    worker_seed: bool = MODEL_SEED_SUPPORTED
    judge_seed: bool = MODEL_SEED_SUPPORTED
    max_worker_tokens: int = MAX_WORKER_TOKENS
    max_judge_tokens: int = MAX_JUDGE_TOKENS
    max_worker_output_chars: int = MAX_WORKER_OUTPUT_CHARS
    max_judge_output_chars: int = MAX_JUDGE_OUTPUT_CHARS
    max_retries: int = MAX_RETRIES
    seed: int = SEED_BASE
    output_dir: str = "results/llm_poc"
    api_key_env: str = "GEMINI_API_KEY"

    def __post_init__(self) -> None:
        if self.judge_enabled is None:
            self.judge_enabled = False
        if self.judge_model == JUDGE_MODEL:
            self.judge_model = self.model
        if self.worker_mode not in {"mock", "gemini"}:
            raise ValueError("worker_mode must be 'mock' or 'gemini'")
        if self.worker_mode == "gemini" and self.model not in ALLOWED_MODELS:
            raise ValueError(
                f"Model {self.model!r} is not in allowed models: {sorted(ALLOWED_MODELS)}"
            )
        if self.worker_mode == "gemini" and self.protocol_version != EXPERIMENT_PROTOCOL_VERSION:
            raise ValueError("Gemini pilot protocol version mismatch")
        if self.worker_mode == "gemini" and self.provider != MODEL_PROVIDER:
            raise ValueError(f"Gemini pilot requires provider {MODEL_PROVIDER!r}")
        if self.worker_mode == "gemini" and not self.pinned_provider:
            raise ValueError("Gemini pilot requires a fixed provider")
        if self.worker_mode == "gemini" and self.judge_enabled:
            raise ValueError("the free-tier pilot disables LLM judging; use rubric_proxy")
        if self.max_retries > MAX_RETRIES and self.worker_mode == "gemini":
            raise ValueError(f"Gemini pilot allows at most {MAX_RETRIES} retries")
        if self.model_catalog_snapshot != MODEL_CATALOG_SNAPSHOT and self.worker_mode == "gemini":
            raise ValueError("Gemini pilot model snapshot mismatch")
        if self.judge_model != self.model:
            raise ValueError("workers and judge must use the same pinned model in protocol v1")
        if self.max_retries < 0 or self.n_repetitions < 1 or self.n_episodes < 1:
            raise ValueError("retries must be nonnegative; episodes and repetitions must be positive")
        if self.max_worker_tokens < 1 or self.max_judge_tokens < 1:
            raise ValueError("token limits must be positive")

    @property
    def api_key(self) -> str:
        return os.environ.get(self.api_key_env, "")

    def require_api(self) -> None:
        if self.worker_mode == "gemini" or self.judge_enabled:
            if not self.api_key:
                raise SystemExit(
                    f"[llm_poc] missing {self.api_key_env}; add it to .env or run with "
                    "--worker-mode mock"
                )

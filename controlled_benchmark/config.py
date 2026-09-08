from __future__ import annotations

import json
import os
import subprocess
from dataclasses import asdict, dataclass, field
from typing import Any, List, Optional


def get_git_revision() -> str:
    """Return current git revision hash if available."""
    try:
        rev = subprocess.check_output(
            ["git", "rev-parse", "--short", "HEAD"],
            stderr=subprocess.DEVNULL,
            text=True,
        ).strip()
        return rev
    except Exception:
        return "unknown"


@dataclass
class BenchmarkConfig:
    """Explicit, fully serializable configuration for the controlled benchmark."""

    mode: str = "quick"  # 'quick' or 'research'
    benchmark_version: str = "1.0.0"
    git_commit: str = field(default_factory=get_git_revision)

    # Seeds: research mode defaults to 20 seeds (42..61), configurable to 30
    seeds: List[int] = field(default_factory=lambda: [42, 43])

    # Problem Dimensions
    num_agents: int = 3
    num_tasks: int = 6
    dim: int = 4

    # Horizon and Perturbation
    num_episodes: int = 10
    perturb_episode: int = 5
    pre_window: int = 3
    post_window: int = 3

    # Budgets
    max_energy_evaluations: int = 150  # Hard common budget for stochastic/search solvers
    ilp_time_limit_sec: float = 5.0    # Exact reference wall-clock limit
    run_ilp_dynamic: bool = True       # Whether to run ILP on dynamic episodes (small instances only)

    # Experimental Factors
    scenarios: List[str] = field(
        default_factory=lambda: [
            "Stationary",
            "Capability Drift",
            "Task Shift",
            "Dependency Change",
        ]
    )
    solvers: List[str] = field(
        default_factory=lambda: ["Simulated Annealing", "Energy Greedy"]
    )
    landscapes: List[str] = field(
        default_factory=lambda: ["Static", "Adaptive Full"]
    )
    ablations: List[str] = field(
        default_factory=lambda: ["Static", "kappa-only", "theta-only", "Full"]
    )

    # Initial Assignment Policy (Fairness: identical starting point for all solvers)
    initial_x_mode: str = "deterministic_round_robin"

    # Solver Hyperparameters (strictly frozen across compared methods)
    sa_temperature_init: float = 1.0
    sa_min_temperature: float = 0.01
    sa_cooling_rate: float = 0.95

    # Adaptation Hyperparameters (strictly frozen across compared methods)
    eta_memory: float = 0.05
    eta_theta: float = 0.10

    # Energy Weights (strictly frozen across compared methods)
    lambda_align: float = 0.5
    lambda_memory: float = 0.5
    interaction_weight: float = 1.0
    cost_weight: float = 1.0
    risk_weight: float = 1.0

    # Output directory
    output_dir: str = "results/controlled_landscape_solver"

    @classmethod
    def quick_mode(cls, **overrides: Any) -> "BenchmarkConfig":
        """Fast configuration for rapid testing and CI."""
        cfg = cls(
            mode="quick",
            seeds=[42, 43],
            num_agents=3,
            num_tasks=6,
            dim=4,
            num_episodes=10,
            perturb_episode=5,
            pre_window=3,
            post_window=3,
            max_energy_evaluations=150,
            ilp_time_limit_sec=5.0,
            run_ilp_dynamic=True,
        )
        for k, v in overrides.items():
            setattr(cfg, k, v)
        return cfg

    @classmethod
    def research_mode(cls, num_seeds: int = 20, **overrides: Any) -> "BenchmarkConfig":
        """Standard 20-seed (or 30-seed) research configuration."""
        if num_seeds not in (20, 30):
            seeds = list(range(42, 42 + num_seeds))
        elif num_seeds == 30:
            seeds = list(range(42, 72))
        else:
            seeds = list(range(42, 62))

        cfg = cls(
            mode="research",
            seeds=seeds,
            num_agents=5,
            num_tasks=10,
            dim=8,
            num_episodes=50,
            perturb_episode=25,
            pre_window=10,
            post_window=10,
            max_energy_evaluations=500,
            ilp_time_limit_sec=10.0,
            run_ilp_dynamic=False,  # For N=5, M=10, exact ILP dynamic sweep is 5^10 ~ 9.7M space; used as static reference
        )
        for k, v in overrides.items():
            setattr(cfg, k, v)
        return cfg

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    def save_json(self, path: str) -> None:
        os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
        with open(path, "w", encoding="utf-8") as f:
            json.dump(self.to_dict(), f, indent=2)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "BenchmarkConfig":
        return cls(**data)


from __future__ import annotations

import json
import argparse
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
DEFAULT_RESULTS = ROOT / "results" / "controlled_benchmark"
DEFAULT_ARTIFACTS = ROOT / "paper_artifacts"
RESULTS = DEFAULT_RESULTS
ARTIFACTS = DEFAULT_ARTIFACTS


def add_path_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--results-dir",
        type=Path,
        required=True,
        help="Benchmark output directory from the selected fresh reproduction run.",
    )
    parser.add_argument(
        "--artifacts-dir",
        type=Path,
        required=True,
        help="Analysis artifact directory from the selected fresh reproduction run.",
    )


def configure_paths(results_dir: Path, artifacts_dir: Path) -> None:
    """Configure process-local input/output roots for an analysis command."""
    global RESULTS, ARTIFACTS
    RESULTS = results_dir.resolve()
    ARTIFACTS = artifacts_dir.resolve()


def require_file(path: Path) -> Path:
    if not path.is_file():
        raise FileNotFoundError(f"Required reproduction input is missing: {path}")
    return path

CONDITION_MAP = {
    "conventional_greedy": "B0",
    "static_energy_greedy": "B1",
    "static_energy_sa": "B2",
    "adaptive_energy_greedy": "B3",
    "adaptive_energy_sa": "B4",
}
CONDITION_ORDER = ["B0", "B1", "B2", "B3", "B4"]
CONDITION_LABEL = {
    "B0": "B0 Conventional Greedy",
    "B1": "B1 Static Energy Greedy",
    "B2": "B2 Static Energy SA",
    "B3": "B3 Adaptive Energy Greedy",
    "B4": "B4 Adaptive Energy SA",
}
SOLVER_OF = {"B0": "Conventional Greedy", "B1": "Energy Greedy", "B2": "Simulated Annealing",
             "B3": "Energy Greedy", "B4": "Simulated Annealing"}
SCENARIOS = ["Stationary", "Capability Drift", "Task Shift", "Dependency Change"]
SEEDS = list(range(42, 62))
PERTURB_EPISODE = 25
POST_WINDOW_EPISODES = list(range(25, 35))
PPEE_WINDOW = 10

COMPARISONS = [
    ("B0-B1", "B0", "B1", "formulation"),
    ("B0-B2", "B0", "B2", "formulation"),
    ("B1-B3", "B1", "B3", "adaptation"),
    ("B2-B4", "B2", "B4", "adaptation"),
    ("B1-B2", "B1", "B2", "solver"),
    ("B3-B4", "B3", "B4", "solver"),
]

DEFINITION_SUFFIX = ("D>0 means the first-listed condition has HIGHER PPEE_10 than the "
                     "second-listed condition (first minus second); negative = improvement "
                     "of the first-listed condition over the second.")


def load_runs() -> pd.DataFrame:
    df = pd.read_csv(RESULTS / "runs.csv")
    df["condition"] = df["condition_id"].map(CONDITION_MAP)
    return df


def load_episodes() -> pd.DataFrame:
    df = pd.read_csv(RESULTS / "episodes.csv")
    df["condition"] = df["condition_id"].map(CONDITION_MAP)
    return df


def load_config() -> dict:
    with open(RESULTS / "config.json", encoding="utf-8") as fh:
        return json.load(fh)


def load_manifest() -> dict:
    with open(RESULTS / "mechanism_diagnostics.json", encoding="utf-8") as fh:
        return json.load(fh)


def paired_differences(runs: pd.DataFrame, metric: str, cond_a: str, cond_b: str,
                       scenario: str | None = None) -> pd.DataFrame:
    """Return per-seed paired differences (A minus B) keyed by (scenario, seed)."""
    sub = runs[runs.condition.isin([cond_a, cond_b])]
    if scenario is not None:
        sub = sub[sub.scenario_id == scenario]
    cols = ["scenario_id", "seed", "condition", metric]
    wide = sub[cols].pivot_table(index=["scenario_id", "seed"], columns="condition",
                                 values=metric, aggfunc="first")
    wide = wide.dropna()
    diff = (wide[cond_a] - wide[cond_b]).rename("diff").reset_index()
    return diff


def summarize_metric(values: np.ndarray) -> dict:
    values = np.asarray(values, dtype=float)
    finite = np.isfinite(values)
    v = values[finite]
    n = int(v.size)
    if n == 0:
        return {"n": 0, "mean": np.nan, "median": np.nan, "sd": np.nan,
                "iqr_low": np.nan, "iqr_high": np.nan, "zero_fraction": np.nan}
    q1, q3 = np.percentile(v, [25, 75])
    return {
        "n": n,
        "mean": float(np.mean(v)),
        "median": float(np.median(v)),
        "sd": float(np.std(v, ddof=1)) if n > 1 else 0.0,
        "iqr_low": float(q1),
        "iqr_high": float(q3),
        "zero_fraction": float(np.mean(v == 0.0)),
    }

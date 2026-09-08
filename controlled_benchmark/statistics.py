from __future__ import annotations

import itertools
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple
import numpy as np
import scipy.stats as stats


@dataclass
class PairedAnalysisResult:
    """Rigorous paired statistical comparison record."""

    metric: str
    comparison: str
    n_pairs: int
    mean_adaptive: float
    mean_static: float
    mean_difference: float  # (Adaptive - Static)
    median_difference: float
    ci_95_lower: float
    ci_95_upper: float
    permutation_p_val: float
    wilcoxon_p_val: float
    cohens_d: float
    p_val_adjusted: Optional[float] = None


@dataclass
class SolverInteractionResult:
    """Statistical test of whether Adaptive advantage differs between solvers."""

    metric: str
    n_pairs: int
    mean_sa_diff: float      # Mean(Adaptive_SA - Static_SA)
    mean_greedy_diff: float  # Mean(Adaptive_Greedy - Static_Greedy)
    mean_interaction: float  # Mean(Delta_SA - Delta_Greedy)
    median_interaction: float
    ci_95_lower: float
    ci_95_upper: float
    permutation_p_val: float
    cohens_d: float


def paired_confidence_interval(
    diff: np.ndarray, confidence: float = 0.95
) -> Tuple[float, float]:
    """Compute 95% Studentized confidence interval for paired differences."""
    n = len(diff)
    if n < 2:
        val = float(np.mean(diff)) if n == 1 else 0.0
        return val, val
    mean = float(np.mean(diff))
    sem = float(stats.sem(diff))
    if sem < 1e-12:
        return mean, mean
    h = sem * stats.t.ppf((1.0 + confidence) / 2.0, df=n - 1)
    return mean - h, mean + h


def paired_permutation_test(
    diff: np.ndarray,
    num_permutations: int = 10000,
    seed: int = 42,
) -> float:
    """
    Two-sided paired permutation test on matched differences.

    Under H0: E[Adaptive - Static] = 0, signs of differences are exchangeable.
    Uses exact enumeration for n <= 16, and Monte Carlo sampling for n > 16.
    """
    n = len(diff)
    if n == 0:
        return 1.0
    obs_t = abs(float(np.mean(diff)))
    if obs_t < 1e-12:
        return 1.0

    if n <= 16:
        # Exact permutation test: 2^n sign flips
        all_means = []
        for signs in itertools.product([-1.0, 1.0], repeat=n):
            s = np.array(signs, dtype=np.float64)
            all_means.append(abs(float(np.mean(diff * s))))
        all_means = np.array(all_means)
        p_val = float(np.mean(all_means >= obs_t - 1e-12))
        return max(1.0 / len(all_means), min(1.0, p_val))
    else:
        # Monte Carlo sampling
        rng = np.random.default_rng(seed)
        signs = rng.choice([-1.0, 1.0], size=(num_permutations, n))
        perm_means = np.abs(np.mean(diff * signs, axis=1))
        p_val = float((1 + np.sum(perm_means >= obs_t - 1e-12)) / (num_permutations + 1))
        return max(1.0 / (num_permutations + 1), min(1.0, p_val))


def wilcoxon_signed_rank_test(diff: np.ndarray) -> float:
    """Wilcoxon signed-rank test on matched differences."""
    n = len(diff)
    if n < 5 or np.all(np.abs(diff) < 1e-12):
        return 1.0
    try:
        res = stats.wilcoxon(diff, alternative="two-sided")
        return float(res.pvalue)
    except Exception:
        return 1.0


def cohens_d_paired(diff: np.ndarray) -> float:
    """Calculate Cohen's d_z for paired samples."""
    n = len(diff)
    if n < 2:
        return 0.0
    std_d = float(np.std(diff, ddof=1))
    if std_d < 1e-12:
        return 0.0
    return float(np.mean(diff) / std_d)


def holm_bonferroni_correction(p_values: List[float]) -> List[float]:
    """Apply standard step-down Holm-Bonferroni correction for multiple hypothesis tests."""
    m = len(p_values)
    if m <= 1:
        return list(p_values)

    sorted_indices = np.argsort(p_values)
    sorted_p = np.array(p_values)[sorted_indices]

    adjusted = np.zeros(m)
    for i in range(m):
        rank = i + 1
        adj = (m - rank + 1) * sorted_p[i]
        adjusted[i] = adj

    # Monotonicity adjustment: adjusted[i] = max(adjusted[i-1], adjusted[i])
    for i in range(1, m):
        adjusted[i] = max(adjusted[i], adjusted[i - 1])

    adjusted = np.clip(adjusted, 0.0, 1.0)

    # Revert to original order
    result = np.zeros(m)
    result[sorted_indices] = adjusted
    return result.tolist()


def analyze_paired_comparison(
    adaptive_values: List[float],
    static_values: List[float],
    metric_name: str,
    comparison_name: str = "Adaptive - Static",
) -> PairedAnalysisResult:
    """Run full paired statistical analysis for a single metric."""
    arr_a = np.array(adaptive_values, dtype=np.float64)
    arr_s = np.array(static_values, dtype=np.float64)
    diff = arr_a - arr_s

    ci_l, ci_u = paired_confidence_interval(diff)
    perm_p = paired_permutation_test(diff)
    wilc_p = wilcoxon_signed_rank_test(diff)
    d = cohens_d_paired(diff)

    return PairedAnalysisResult(
        metric=metric_name,
        comparison=comparison_name,
        n_pairs=len(diff),
        mean_adaptive=float(np.mean(arr_a)),
        mean_static=float(np.mean(arr_s)),
        mean_difference=float(np.mean(diff)),
        median_difference=float(np.median(diff)),
        ci_95_lower=ci_l,
        ci_95_upper=ci_u,
        permutation_p_val=perm_p,
        wilcoxon_p_val=wilc_p,
        cohens_d=d,
    )


def analyze_solver_interaction(
    sa_adaptive: List[float],
    sa_static: List[float],
    greedy_adaptive: List[float],
    greedy_static: List[float],
    metric_name: str,
) -> SolverInteractionResult:
    """
    Test whether the Adaptive-vs-Static effect differs between SA and Greedy.

    Interaction difference for each matched seed i:
      I_i = (Adaptive_SA - Static_SA)_i - (Adaptive_Greedy - Static_Greedy)_i
    """
    diff_sa = np.array(sa_adaptive, dtype=np.float64) - np.array(sa_static, dtype=np.float64)
    diff_greedy = np.array(greedy_adaptive, dtype=np.float64) - np.array(greedy_static, dtype=np.float64)
    interaction_diff = diff_sa - diff_greedy

    ci_l, ci_u = paired_confidence_interval(interaction_diff)
    perm_p = paired_permutation_test(interaction_diff)
    d = cohens_d_paired(interaction_diff)

    return SolverInteractionResult(
        metric=metric_name,
        n_pairs=len(interaction_diff),
        mean_sa_diff=float(np.mean(diff_sa)),
        mean_greedy_diff=float(np.mean(diff_greedy)),
        mean_interaction=float(np.mean(interaction_diff)),
        median_interaction=float(np.median(interaction_diff)),
        ci_95_lower=ci_l,
        ci_95_upper=ci_u,
        permutation_p_val=perm_p,
        cohens_d=d,
    )

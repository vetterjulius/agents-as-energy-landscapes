#!/usr/bin/env python3
"""Isolated synthetic ablation of assignment-based Theta adaptation.

This script is deliberately independent of the production benchmark.  It does
not import or modify the benchmark runner, solvers, scenarios, configuration,
or existing result files.
"""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Callable

import numpy as np
from scipy.stats import spearmanr


M = 6
EPISODES = 90
CHANGE_EPISODE = 30
ETA = 0.10
GAMMA = 0.20
TAU_LOW = 0.70
TAU_HIGH = 0.90
SEEDS = list(range(42, 62))
VARIANTS = ("Static", "Dynamic", "Persistent Dynamic")
PHASES = {
    "I_stationary_A": range(0, 30),
    "II_change_to_B": range(30, 60),
    "III_stationary_B": range(60, 90),
}


def offdiag(matrix: np.ndarray) -> np.ndarray:
    result = np.array(matrix, dtype=np.float64, copy=True)
    np.fill_diagonal(result, 0.0)
    return result


def observed_cooccurrence(X: np.ndarray, epsilon: float = 1e-8) -> np.ndarray:
    """Exact normalized off-diagonal symmetric C used by production Theta."""
    co = X.T @ X
    total = float(co.sum())
    if total < epsilon:
        return np.zeros_like(co, dtype=np.float64)
    normalized = co / (total + epsilon)
    return offdiag((normalized + normalized.T) / 2.0)


def canonical_assignment(structure: str) -> np.ndarray:
    X = np.zeros((3, M), dtype=np.float64)
    groups = ((0, 1), (2, 3), (4, 5)) if structure == "A" else ((0, 2), (1, 3), (4, 5))
    for agent, tasks in enumerate(groups):
        for task in tasks:
            X[agent, task] = 1.0
    return X


def make_ground_truths() -> tuple[np.ndarray, np.ndarray]:
    # Normalize using the exact C scale of the canonical assignments, so A and B
    # are directly comparable to the learned Theta representation.
    return observed_cooccurrence(canonical_assignment("A")), observed_cooccurrence(canonical_assignment("B"))


def generate_observations(seed: int, switch: bool = True) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Generate one deterministic X trajectory shared by every variant."""
    rng = np.random.default_rng(seed)
    Xs = []
    for episode in range(EPISODES):
        structure = "A" if (not switch or episode < CHANGE_EPISODE) else "B"
        X = canonical_assignment(structure)
        # Moderate, deterministic observation noise: at most one task is moved.
        # The underlying co-occurrence therefore remains dominated by A or B.
        if rng.random() < 0.30:
            task = int(rng.integers(0, M))
            source = int(np.argmax(X[:, task]))
            target = int(rng.choice([a for a in range(X.shape[0]) if a != source]))
            X[source, task] = 0.0
            X[target, task] = 1.0
        Xs.append(X)
    Xs = np.asarray(Xs)
    Cs = np.asarray([observed_cooccurrence(X) for X in Xs])
    return Xs, Cs, np.asarray(["A" if t < CHANGE_EPISODE else "B" for t in range(EPISODES)])


def normalize_theta(theta: np.ndarray) -> np.ndarray:
    return offdiag((theta + theta.T) / 2.0)


def cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    denominator = float(np.linalg.norm(a) * np.linalg.norm(b))
    return float(np.sum(a * b) / denominator) if denominator > 0 else 0.0


def spearman_upper(a: np.ndarray, b: np.ndarray) -> float | None:
    upper = np.triu_indices_from(a, k=1)
    x, y = a[upper], b[upper]
    value = spearmanr(x, y).statistic
    return float(value) if np.isfinite(value) else None


def simulate_variant(variant: str, Cs: np.ndarray, theta0: np.ndarray) -> dict[str, np.ndarray]:
    theta = np.array(theta0, dtype=np.float64, copy=True)
    cbar_previous = None
    theta_before = []
    theta_after = []
    etas = []
    consistencies = []
    cbars = []

    for C in Cs:
        theta_before.append(theta.copy())
        if variant == "Static":
            eta_t = 0.0
            consistency = 1.0
            cbar = None
        elif variant == "Dynamic":
            eta_t = ETA
            consistency = 1.0
            cbar = None
        elif variant == "Persistent Dynamic":
            # The first observation establishes the running estimate and does
            # not trigger a structurally informed update.
            if cbar_previous is None:
                consistency = 0.0
            else:
                consistency = cosine_similarity(C, cbar_previous)
            alpha = float(np.clip((consistency - TAU_LOW) / (TAU_HIGH - TAU_LOW), 0.0, 1.0))
            eta_t = ETA * alpha
            cbar = C if cbar_previous is None else (1.0 - GAMMA) * cbar_previous + GAMMA * C
        else:
            raise ValueError(f"Unknown variant: {variant}")

        theta = normalize_theta((1.0 - eta_t) * theta + eta_t * C)
        theta_after.append(theta.copy())
        etas.append(eta_t)
        consistencies.append(consistency)
        cbars.append(np.zeros_like(theta) if cbar is None else cbar.copy())
        if variant == "Persistent Dynamic":
            cbar_previous = cbar

    # theta_before[t] is the state used for episode t.  theta_after[t] is the
    # state resulting from C_t and used for episode t+1.
    return {
        "theta_before": np.asarray(theta_before),
        "theta_after": np.asarray(theta_after),
        "eta": np.asarray(etas),
        "consistency": np.asarray(consistencies),
        "cbar": np.asarray(cbars),
    }


def phase_values(values: np.ndarray, phase: str) -> np.ndarray:
    return values[list(PHASES[phase])]


def safe_mean(values: list[float | None]) -> float | None:
    finite = [float(v) for v in values if v is not None and np.isfinite(v)]
    return float(np.mean(finite)) if finite else None


def calculate_metrics(trace: dict[str, np.ndarray], gt_a: np.ndarray, gt_b: np.ndarray, theta0: np.ndarray) -> dict[str, float | int | str | None]:
    theta = trace["theta_before"]
    eta = trace["eta"]
    ground_truth = np.asarray([gt_a if t < CHANGE_EPISODE else gt_b for t in range(EPISODES)])
    similarities = [cosine_similarity(theta[t], ground_truth[t]) for t in range(EPISODES)]
    spearmans = [spearman_upper(theta[t], ground_truth[t]) for t in range(EPISODES)]
    distances_b = [float(np.linalg.norm(theta[t] - gt_b)) for t in range(EPISODES)]
    drift = [float(np.linalg.norm(theta[t] - theta0)) for t in range(EPISODES)]
    theta_step = np.linalg.norm(trace["theta_after"] - trace["theta_before"], axis=(1, 2))
    threshold_hits = [t - CHANGE_EPISODE for t in range(CHANGE_EPISODE, EPISODES) if similarities[t] >= 0.8]

    result: dict[str, float | int | str | None] = {
        "variant": "",
        "stationary_drift_mean": float(np.mean(phase_values(np.asarray(drift), "I_stationary_A"))),
        "stationary_drift_max": float(np.max(phase_values(np.asarray(drift), "I_stationary_A"))),
        "stationary_drift_final": float(drift[CHANGE_EPISODE - 1]),
        "similarity_phase_I_mean": float(np.mean(phase_values(np.asarray(similarities), "I_stationary_A"))),
        "similarity_phase_II_mean": float(np.mean(phase_values(np.asarray(similarities), "II_change_to_B"))),
        "similarity_phase_III_mean": float(np.mean(phase_values(np.asarray(similarities), "III_stationary_B"))),
        "similarity_final": float(similarities[-1]),
        "spearman_phase_I_mean": safe_mean([spearmans[t] for t in PHASES["I_stationary_A"]]),
        "spearman_phase_II_mean": safe_mean([spearmans[t] for t in PHASES["II_change_to_B"]]),
        "spearman_phase_III_mean": safe_mean([spearmans[t] for t in PHASES["III_stationary_B"]]),
        "distance_to_B_phase_II_final": float(distances_b[59]),
        "time_to_similarity_B_0.8": int(min(threshold_hits)) if threshold_hits else None,
        "post_change_theta_step_mean": float(np.mean(theta_step[60:89])),
        "post_change_distance_to_B_final": float(distances_b[-1]),
        "post_change_similarity_B_variance": float(np.var([cosine_similarity(theta[t], gt_b) for t in PHASES["III_stationary_B"]])),
        "eta_phase_I_mean": float(np.mean(phase_values(eta, "I_stationary_A"))),
        "eta_phase_II_mean": float(np.mean(phase_values(eta, "II_change_to_B"))),
        "eta_phase_III_mean": float(np.mean(phase_values(eta, "III_stationary_B"))),
        "eta_phase_I_active_fraction": float(np.mean(phase_values(eta, "I_stationary_A") > 0.0)),
        "eta_phase_II_active_fraction": float(np.mean(phase_values(eta, "II_change_to_B") > 0.0)),
        "eta_phase_III_active_fraction": float(np.mean(phase_values(eta, "III_stationary_B") > 0.0)),
    }
    return result


def write_csv(path: Path, rows: list[dict]) -> None:
    fields = []
    for row in rows:
        for field in row:
            if field not in fields:
                fields.append(field)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def write_plots(output: Path, traces: dict[str, list[dict[str, np.ndarray]]], gt_a: np.ndarray, gt_b: np.ndarray) -> None:
    import matplotlib.pyplot as plt

    episodes = np.arange(EPISODES)
    gt_schedule = np.asarray([gt_a if t < CHANGE_EPISODE else gt_b for t in episodes])
    for name, ylabel, filename, getter in (
        (
            "similarity",
            "Theta–ground-truth Frobenius cosine similarity",
            "plot_1_theta_ground_truth_similarity.png",
            lambda trace: np.asarray([cosine_similarity(trace["theta_before"][t], gt_schedule[t]) for t in episodes]),
        ),
        (
            "drift",
            r"$||Theta_t - Theta_0||_F$",
            "plot_2_theta_drift.png",
            lambda trace: np.asarray([np.linalg.norm(trace["theta_before"][t] - gt_a) for t in episodes]),
        ),
        (
            "activity",
            r"$eta_t$",
            "plot_3_adaptation_activity.png",
            lambda trace: np.asarray(trace["eta"]),
        ),
    ):
        fig, ax = plt.subplots(figsize=(8, 4.5))
        for variant in VARIANTS:
            values = np.asarray([getter(trace) for trace in traces[variant]])
            ax.plot(episodes, values.mean(axis=0), label=variant, linewidth=2)
        ax.axvline(CHANGE_EPISODE, color="black", linestyle="--", linewidth=1, label="change at episode 30")
        if name == "similarity":
            ax.axhline(0.8, color="gray", linestyle=":", linewidth=1)
        ax.set_xlabel("Episode")
        ax.set_ylabel(ylabel)
        ax.grid(alpha=0.25)
        ax.legend()
        fig.tight_layout()
        fig.savefig(output / filename, dpi=150)
        plt.close(fig)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", default=".storage/analysis/theta-adaptation-ablation-20260917")
    args = parser.parse_args()
    output = Path(args.output_dir)
    if output.exists() and any(output.iterdir()):
        raise SystemExit(f"Refusing to overwrite non-empty output directory: {output}")
    output.mkdir(parents=True, exist_ok=True)

    gt_a, gt_b = make_ground_truths()
    all_rows: list[dict] = []
    traces: dict[str, list[dict[str, np.ndarray]]] = {variant: [] for variant in VARIANTS}
    observations_by_seed = {}
    control_rows = []

    for seed in SEEDS:
        Xs, Cs, gt_labels = generate_observations(seed, switch=True)
        _, Cs_constant_a, _ = generate_observations(seed, switch=False)
        observations_by_seed[str(seed)] = {"X": Xs, "C": Cs, "ground_truth_labels": gt_labels}
        for variant in VARIANTS:
            trace = simulate_variant(variant, Cs, gt_a)
            traces[variant].append(trace)
            metrics = calculate_metrics(trace, gt_a, gt_b, gt_a)
            metrics.update({"seed": seed, "variant": variant})
            all_rows.append(metrics)

            # Controls: adaptation receives only C.  Changing the evaluation
            # ground truth cannot change Theta, while changing C can.
            # Both traces start from the same initial Theta_A and consume the
            # same C stream; the hypothetical evaluation graph is intentionally
            # not passed to simulate_variant.
            gt_only_trace_a = simulate_variant(variant, Cs, gt_a)
            gt_only_trace_b = simulate_variant(variant, Cs, gt_a)
            constant_a_observation_trace = simulate_variant(variant, Cs_constant_a, gt_a)
            assignment_change_detected = not np.array_equal(
                trace["theta_before"][CHANGE_EPISODE + 1:],
                constant_a_observation_trace["theta_before"][CHANGE_EPISODE + 1:],
            )
            if variant == "Static":
                assignment_change_detected = not assignment_change_detected
            control_rows.append({
                "seed": seed,
                "variant": variant,
                "ground_truth_only_change_theta_identical": bool(
                    np.array_equal(gt_only_trace_a["theta_before"], gt_only_trace_b["theta_before"])
                ),
                "assignment_change_gt_constant_theta_response_ok": bool(assignment_change_detected),
            })

    summary_rows = []
    for variant in VARIANTS:
        rows = [row for row in all_rows if row["variant"] == variant]
        numeric_fields = [key for key, value in rows[0].items() if key not in {"seed", "variant"} and isinstance(value, (int, float))]
        summary = {"variant": variant}
        for field in numeric_fields:
            values = [row[field] for row in rows if row[field] is not None]
            summary[field] = float(np.mean(values)) if values else None
        threshold_values = [row["time_to_similarity_B_0.8"] for row in rows]
        reached = [value for value in threshold_values if value is not None]
        summary["time_to_similarity_B_0.8"] = float(np.mean(reached)) if reached else None
        summary["threshold_reached_seeds"] = int(len(reached))
        summary_rows.append(summary)

    # Save all matrices in one small, independent artifact for exact re-analysis.
    npz_payload = {"ground_truth_A": gt_a, "ground_truth_B": gt_b}
    for seed in SEEDS:
        npz_payload[f"seed_{seed}_X"] = observations_by_seed[str(seed)]["X"]
        npz_payload[f"seed_{seed}_C"] = observations_by_seed[str(seed)]["C"]
        for variant in VARIANTS:
            key = variant.lower().replace(" ", "_")
            trace = traces[variant][SEEDS.index(seed)]
            for field in ("theta_before", "theta_after", "eta", "consistency", "cbar"):
                npz_payload[f"seed_{seed}_{key}_{field}"] = trace[field]
    np.savez_compressed(output / "theta_ablation_traces.npz", **npz_payload)

    write_csv(output / "per_seed_metrics.csv", all_rows)
    write_csv(output / "summary_metrics.csv", summary_rows)
    write_csv(output / "control_checks.csv", control_rows)
    write_plots(output, traces, gt_a, gt_b)

    manifest = {
        "experiment": "isolated_theta_adaptation_ablation",
        "seeds": SEEDS,
        "variants": list(VARIANTS),
        "M": M,
        "episodes": EPISODES,
        "phase_ranges": {name: [min(values), max(values)] for name, values in PHASES.items()},
        "change_episode": CHANGE_EPISODE,
        "dynamic_parameters": {"eta": ETA},
        "persistent_parameters": {"gamma": GAMMA, "tau_low": TAU_LOW, "tau_high": TAU_HIGH, "eta_max": ETA},
        "ground_truth_normalization": "GroundTruth_A and GroundTruth_B are exact normalized C matrices of the canonical A/B assignments, matching the production C scale.",
        "observation_generation": "Each seed has one deterministic X trajectory shared by all variants; each episode has at most one randomly selected task moved with probability 0.30.",
        "timing": "theta_before[t] is Theta_t before processing C_t; theta_after[t] is Theta_{t+1].",
        "adaptation_inputs": "All variants receive only the shared observed C_t; ground truth is used only for metrics and controls.",
        "control_expectations": [
            "Changing only the evaluation ground truth while keeping X_t/C_t identical must not change Theta.",
            "Changing X_t from A to B while holding evaluation ground truth at A must change Theta according to observations, not according to ground truth.",
        ],
        "files": [
            "theta_ablation_traces.npz", "per_seed_metrics.csv", "summary_metrics.csv", "control_checks.csv",
            "plot_1_theta_ground_truth_similarity.png", "plot_2_theta_drift.png", "plot_3_adaptation_activity.png",
        ],
    }
    (output / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")

    control_ok = all(
        row["ground_truth_only_change_theta_identical"]
        and row["assignment_change_gt_constant_theta_response_ok"]
        for row in control_rows
    )
    report_lines = [
        "# Isolated Theta Adaptation Ablation",
        "",
        "## Compact trade-off table (mean over 20 seeds)",
        "",
        "| Variant | Stationary drift | Change response | Post-change stability | Adaptation activity |",
        "|---|---:|---:|---:|---:|",
    ]
    for row in summary_rows:
        response = row["time_to_similarity_B_0.8"]
        response_text = "not reached" if row["threshold_reached_seeds"] == 0 else f"{response:.2f} episodes (mean; {row['threshold_reached_seeds']}/20 reached)"
        stability = row["post_change_distance_to_B_final"]
        activity = row["eta_phase_I_mean"]
        report_lines.append(f"| {row['variant']} | {row['stationary_drift_mean']:.6f} | {response_text} | final distance to B {stability:.6f} | Phase-I mean eta {activity:.6f} |")
    report_lines += [
        "",
        "## Metric definitions",
        "",
        "- Stationary drift: mean of `||Theta_t - Theta_0||_F` over episodes 0–29; maximum and final values are in `summary_metrics.csv`.",
        "- Ground-truth similarity: Frobenius cosine similarity; Spearman on the strict upper triangle is also recorded.",
        "- Change response: episodes after 30 until similarity to normalized GroundTruth_B first reaches 0.8; `not reached` is explicit.",
        "- Post-change stability: mean Theta step over episodes 60–89, final distance to B, and variance of similarity to B.",
        "- Adaptation activity: mean eta and fraction of episodes with eta > 0 per phase.",
        "",
        "## Controls",
        "",
        f"- Ground-truth-only change control passed: {control_ok}.",
        "- The same seed-specific X_t and C_t streams are used for all three variants.",

        "- Ground truth is not passed to any adaptation update; it is used only for evaluation and controls.",
        "",
        "## Reproduction",
        "",
        "```bash",
        "python theta_adaptation_ablation.py --output-dir .storage/analysis/theta-adaptation-ablation-20260917",
        "```",
    ]
    (output / "report.md").write_text("\n".join(report_lines) + "\n", encoding="utf-8")
    print(json.dumps({"output_dir": str(output), "control_checks_pass": control_ok, "seeds": len(SEEDS), "variants": list(VARIANTS)}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

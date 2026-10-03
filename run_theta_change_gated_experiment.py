#!/usr/bin/env python3
"""Isolated Change-Gated Theta Adaptation Mechanism Experiment.

Builds directly on .storage/analysis/theta-adaptation-ablation-20260917/theta_ablation_traces.npz
Without modifying production code, benchmark runners, or running new benchmark iterations.
"""
from __future__ import annotations

import csv
import json
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# -----------------------------------------------------------------------------
# Configuration & Constants
# -----------------------------------------------------------------------------
INPUT_NPZ = Path(r"D:\Julius Vetter\Code\energy-landscape\.storage\analysis\theta-adaptation-ablation-20260917\theta_ablation_traces.npz")
OUTPUT_DIR = Path(r"D:\Julius Vetter\Code\energy-landscape\.storage\analysis\theta-change-gated-20260917")
PLOTS_DIR = OUTPUT_DIR / "plots"

M = 6
EPISODES = 90
CHANGE_EPISODE = 30
ETA_HIGH = 0.10
ETA_LOW = 0.01
GAMMA = 0.20
TAU_LOW = 0.70
TAU_HIGH = 0.90
SEEDS = list(range(42, 62))

PHASES = {
    "Phase_I": range(0, 30),
    "Phase_II": range(30, 60),
    "Phase_III": range(60, 90),
}


# -----------------------------------------------------------------------------
# Utility Functions
# -----------------------------------------------------------------------------
def offdiag(matrix: np.ndarray) -> np.ndarray:
    result = np.array(matrix, dtype=np.float64, copy=True)
    np.fill_diagonal(result, 0.0)
    return result


def normalize_theta(theta: np.ndarray) -> np.ndarray:
    return offdiag((theta + theta.T) / 2.0)


def cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    denominator = float(np.linalg.norm(a) * np.linalg.norm(b))
    return float(np.sum(a * b) / denominator) if denominator > 0 else 0.0


def canonical_assignment(structure: str) -> np.ndarray:
    X = np.zeros((3, M), dtype=np.float64)
    groups = ((0, 1), (2, 3), (4, 5)) if structure == "A" else ((0, 2), (1, 3), (4, 5))
    for agent, tasks in enumerate(groups):
        for task in tasks:
            X[agent, task] = 1.0
    return X


def observed_cooccurrence(X: np.ndarray, epsilon: float = 1e-8) -> np.ndarray:
    co = X.T @ X
    total = float(co.sum())
    if total < epsilon:
        return np.zeros_like(co, dtype=np.float64)
    normalized = co / (total + epsilon)
    return offdiag((normalized + normalized.T) / 2.0)


# -----------------------------------------------------------------------------
# Simulation Logic
# -----------------------------------------------------------------------------
def simulate_variant(
    variant: str,
    Cs: np.ndarray,
    theta0: np.ndarray,
    k_thresh: int = 2,
    eta_low: float = ETA_LOW,
    eta_high: float = ETA_HIGH,
    tau_low: float = TAU_LOW,
    decay: int = 1,
) -> dict[str, np.ndarray]:
    theta = np.array(theta0, dtype=np.float64, copy=True)
    cbar_prev = None
    change_evidence = 0
    active = False

    theta_before = []
    theta_after = []
    etas = []
    consistencies = []
    evidences = []

    for t, C in enumerate(Cs):
        theta_before.append(theta.copy())

        if variant == "Static":
            eta_t = 0.0
            consistency = 1.0
            cbar = None
            evidence = 0

        elif variant == "Dynamic":
            eta_t = ETA_HIGH
            consistency = 1.0
            cbar = None
            evidence = 0

        elif variant == "Persistent Dynamic":
            if cbar_prev is None:
                consistency = 0.0
            else:
                consistency = cosine_similarity(C, cbar_prev)
            alpha = float(np.clip((consistency - TAU_LOW) / (TAU_HIGH - TAU_LOW), 0.0, 1.0))
            eta_t = ETA_HIGH * alpha
            cbar = C if cbar_prev is None else (1.0 - GAMMA) * cbar_prev + GAMMA * C
            evidence = 0

        elif variant.startswith("Change-Gated"):
            if cbar_prev is None:
                consistency = 1.0
                cbar = C.copy()
            else:
                consistency = cosine_similarity(C, cbar_prev)
                cbar = (1.0 - GAMMA) * cbar_prev + GAMMA * C

            if consistency < tau_low:
                change_evidence += 1
            else:
                change_evidence = max(0, change_evidence - decay)

            evidence = change_evidence
            if change_evidence >= k_thresh:
                eta_t = eta_high
            else:
                eta_t = eta_low

        elif variant == "Latching Change-Gated":
            if cbar_prev is None:
                consistency = 1.0
                cbar = C.copy()
            else:
                consistency = cosine_similarity(C, cbar_prev)
                cbar = (1.0 - GAMMA) * cbar_prev + GAMMA * C

            if consistency < tau_low:
                change_evidence += 1
            else:
                change_evidence = max(0, change_evidence - decay)

            evidence = change_evidence
            if change_evidence >= k_thresh:
                active = True
            if active and cosine_similarity(theta, cbar) >= 0.85 and change_evidence == 0:
                active = False

            eta_t = eta_high if active else eta_low

        else:
            raise ValueError(f"Unknown variant: {variant}")

        theta = normalize_theta((1.0 - eta_t) * theta + eta_t * C)
        theta_after.append(theta.copy())
        etas.append(eta_t)
        consistencies.append(consistency)
        evidences.append(evidence)

        if cbar is not None:
            cbar_prev = cbar.copy()

    return {
        "theta_before": np.asarray(theta_before),
        "theta_after": np.asarray(theta_after),
        "eta": np.asarray(etas),
        "consistency": np.asarray(consistencies),
        "evidence": np.asarray(evidences),
    }


def calculate_per_seed_metrics(
    variant_name: str,
    seed: int,
    trace: dict[str, np.ndarray],
    gt_a: np.ndarray,
    gt_b: np.ndarray,
    theta0: np.ndarray,
    k_thresh: int = 2,
    eta_low: float = ETA_LOW,
) -> dict:
    tb = trace["theta_before"]
    ta = trace["theta_after"]
    eta = trace["eta"]

    drift = [float(np.linalg.norm(tb[t] - theta0)) for t in range(EPISODES)]
    dist_b = [float(np.linalg.norm(tb[t] - gt_b)) for t in range(EPISODES)]
    sim_b = [cosine_similarity(tb[t], gt_b) for t in range(EPISODES)]
    theta_step = [float(np.linalg.norm(ta[t] - tb[t])) for t in range(EPISODES)]

    # Episode hit for sim_b >= 0.8 in Phase II/III (t >= 30)
    b80_hits = [t for t in range(CHANGE_EPISODE, EPISODES) if sim_b[t] >= 0.8]
    episodes_to_b80 = (b80_hits[0] - CHANGE_EPISODE) if b80_hits else (EPISODES - CHANGE_EPISODE)

    # Gate behavior
    if variant_name == "Persistent Dynamic":
        active_mask = eta > 0.0
    elif variant_name in ("Static", "Dynamic"):
        active_mask = np.zeros(EPISODES, dtype=bool) if variant_name == "Static" else np.ones(EPISODES, dtype=bool)
    else:
        active_mask = eta > eta_low

    phase_1_false_acts = int(np.sum(active_mask[:30]))

    # First activation in Phase II
    phase_2_act_eps = [t for t in range(30, 60) if active_mask[t]]
    first_act_p2 = phase_2_act_eps[0] if phase_2_act_eps else None
    act_delay = (first_act_p2 - 30) if first_act_p2 is not None else 30

    # Consecutive adaptation runs count
    runs = 0
    in_run = False
    for a in active_mask:
        if a and not in_run:
            runs += 1
            in_run = True
        elif not a and in_run:
            in_run = False

    return {
        "variant": variant_name,
        "seed": seed,
        # Stationary Stability (Phase I)
        "phase_1_mean_drift": float(np.mean(drift[:30])),
        "phase_1_max_drift": float(np.max(drift[:30])),
        "phase_1_final_drift": float(drift[29]),
        "phase_1_mean_theta_step": float(np.mean(theta_step[:30])),
        # Change Response (Phase II)
        "phase_2_episodes_to_sim_b_0.8": episodes_to_b80,
        "phase_2_final_dist_to_b": float(dist_b[59]),
        "phase_2_mean_dist_to_b": float(np.mean(dist_b[30:60])),
        "phase_2_mean_eta": float(np.mean(eta[30:60])),
        "phase_2_active_fraction": float(np.mean(active_mask[30:60])),
        # Post-Change Stability (Phase III)
        "phase_3_final_dist_to_b": float(dist_b[89]),
        "phase_3_mean_dist_to_b": float(np.mean(dist_b[60:90])),
        "phase_3_sim_b_variance": float(np.var(sim_b[60:90])),
        "phase_3_mean_theta_step": float(np.mean(theta_step[60:90])),
        # Change-Gate Behavior
        "gate_phase_1_false_activations": phase_1_false_acts,
        "gate_activation_delay": act_delay,
        "gate_consecutive_active_runs": runs,
    }


def generate_stationary_observations(seed: int, noise_prob: float, num_episodes: int = 30) -> list[np.ndarray]:
    rng = np.random.default_rng(seed)
    Cs = []
    for _ in range(num_episodes):
        X = canonical_assignment("A")
        if rng.random() < noise_prob:
            task = int(rng.integers(0, M))
            source = int(np.argmax(X[:, task]))
            target = int(rng.choice([a for a in range(X.shape[0]) if a != source]))
            X[source, task] = 0.0
            X[target, task] = 1.0
        Cs.append(observed_cooccurrence(X))
    return Cs


# -----------------------------------------------------------------------------
# Main Execution
# -----------------------------------------------------------------------------
def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    PLOTS_DIR.mkdir(parents=True, exist_ok=True)

    data = np.load(INPUT_NPZ)
    gt_A = data["ground_truth_A"]
    gt_B = data["ground_truth_B"]
    theta0 = gt_A.copy()

    variant_names = [
        "Static",
        "Dynamic",
        "Persistent Dynamic",
        "Change-Gated (K=2)",
        "Change-Gated (K=3)",
        "Latching Change-Gated",
    ]

    all_traces: dict[str, dict[int, dict[str, np.ndarray]]] = {v: {} for v in variant_names}
    per_seed_rows = []

    for s in SEEDS:
        Cs = data[f"seed_{s}_C"]
        for v in variant_names:
            k_thresh = 3 if "K=3" in v else 2
            sim_v_name = "Change-Gated" if v in ("Change-Gated (K=2)", "Change-Gated (K=3)") else v
            trace = simulate_variant(sim_v_name, Cs, theta0, k_thresh=k_thresh)
            all_traces[v][s] = trace
            metrics = calculate_per_seed_metrics(v, s, trace, gt_A, gt_B, theta0, k_thresh=k_thresh)
            per_seed_rows.append(metrics)

    df_per_seed = pd.DataFrame(per_seed_rows)
    df_per_seed.to_csv(OUTPUT_DIR / "per_seed_metrics.csv", index=False)

    # Summary Metrics across seeds
    summary_rows = []
    numeric_cols = [c for c in df_per_seed.columns if c not in ("variant", "seed")]
    for v in variant_names:
        sub = df_per_seed[df_per_seed["variant"] == v]
        row = {"variant": v}
        for col in numeric_cols:
            row[f"{col}_mean"] = float(sub[col].mean())
            row[f"{col}_std"] = float(sub[col].std())
            row[f"{col}_median"] = float(sub[col].median())
        summary_rows.append(row)

    df_summary = pd.DataFrame(summary_rows)
    df_summary.to_csv(OUTPUT_DIR / "summary_metrics.csv", index=False)

    # Gate Diagnostics CSV
    gate_diag_rows = []
    for v in variant_names:
        sub = df_per_seed[df_per_seed["variant"] == v]
        gate_diag_rows.append({
            "variant": v,
            "false_activations_phase_1_mean": float(sub["gate_phase_1_false_activations"].mean()),
            "activation_delay_phase_2_mean": float(sub["gate_activation_delay"].mean()),
            "consecutive_active_runs_mean": float(sub["gate_consecutive_active_runs"].mean()),
            "phase_2_active_fraction_mean": float(sub["phase_2_active_fraction"].mean()),
            "phase_2_mean_eta": float(sub["phase_2_mean_eta"].mean()),
        })
    df_gate_diag = pd.DataFrame(gate_diag_rows)
    df_gate_diag.to_csv(OUTPUT_DIR / "gate_diagnostics.csv", index=False)

    # -------------------------------------------------------------------------
    # Additional Noise-Sensitivity Control Test
    # -------------------------------------------------------------------------
    noise_levels = [0.00, 0.10, 0.20, 0.30, 0.40]
    noise_seeds = list(range(42, 47))
    noise_rows = []

    for p in noise_levels:
        for v in variant_names:
            false_acts = []
            drifts = []
            for s in noise_seeds:
                Cs_noise = generate_stationary_observations(s, p, num_episodes=30)
                k_thresh = 3 if "K=3" in v else 2
                sim_v_name = "Change-Gated" if v in ("Change-Gated (K=2)", "Change-Gated (K=3)") else v
                tr = simulate_variant(sim_v_name, np.asarray(Cs_noise), theta0, k_thresh=k_thresh)
                tb_noise = tr["theta_before"]
                eta_noise = tr["eta"]

                if v == "Persistent Dynamic":
                    fa = int(np.sum(eta_noise > 0.0))
                elif v in ("Static", "Dynamic"):
                    fa = 0
                else:
                    fa = int(np.sum(eta_noise > ETA_LOW))

                drift = float(np.mean([np.linalg.norm(tb_noise[t] - gt_A) for t in range(30)]))
                false_acts.append(fa)
                drifts.append(drift)

            noise_rows.append({
                "noise_level": p,
                "variant": v,
                "mean_false_activations": float(np.mean(false_acts)),
                "std_false_activations": float(np.std(false_acts)),
                "mean_stationary_drift": float(np.mean(drifts)),
                "std_stationary_drift": float(np.std(drifts)),
            })

    df_noise = pd.DataFrame(noise_rows)
    df_noise.to_csv(OUTPUT_DIR / "noise_sensitivity.csv", index=False)

    # -------------------------------------------------------------------------
    # Visualizations (Max 5 plots)
    # -------------------------------------------------------------------------
    episodes = np.arange(EPISODES)
    main_variants = ["Static", "Dynamic", "Persistent Dynamic", "Change-Gated (K=2)"]

    # Plot 1: Theta similarity to GT
    fig, ax = plt.subplots(figsize=(8, 4.5))
    for v in main_variants:
        sims = np.asarray([
            [cosine_similarity(all_traces[v][s]["theta_before"][t], gt_A if t < 30 else gt_B) for t in episodes]
            for s in SEEDS
        ])
        ax.plot(episodes, sims.mean(axis=0), label=v, linewidth=2)
    ax.axvline(CHANGE_EPISODE, color="black", linestyle="--", linewidth=1, label="A -> B change (Ep 30)")
    ax.axhline(0.8, color="gray", linestyle=":", linewidth=1, label="Target Similarity 0.8")
    ax.set_xlabel("Episode")
    ax.set_ylabel("Theta–Ground Truth Cosine Similarity")
    ax.set_title("Figure 1: Theta Similarity to Ground Truth Across Phases")
    ax.grid(alpha=0.25)
    ax.legend()
    fig.tight_layout()
    fig.savefig(PLOTS_DIR / "plot_1_theta_gt_similarity.png", dpi=300)
    plt.close(fig)

    # Plot 2: Adaptation rate eta (esp episode 20-45)
    fig, ax = plt.subplots(figsize=(8, 4.5))
    ep_range = np.arange(20, 46)
    for v in main_variants:
        etas = np.asarray([[all_traces[v][s]["eta"][t] for t in ep_range] for s in SEEDS])
        ax.plot(ep_range, etas.mean(axis=0), label=v, linewidth=2)
    ax.axvline(CHANGE_EPISODE, color="black", linestyle="--", linewidth=1, label="A -> B change (Ep 30)")
    ax.set_xlabel("Episode")
    ax.set_ylabel("Adaptation Rate eta_t")
    ax.set_title("Figure 2: Adaptation Rate eta_t Around Structural Switch (Episodes 20–45)")
    ax.grid(alpha=0.25)
    ax.legend()
    fig.tight_layout()
    fig.savefig(PLOTS_DIR / "plot_2_adaptation_rate_eta.png", dpi=300)
    plt.close(fig)

    # Plot 3: Change evidence / consistency
    fig, ax = plt.subplots(figsize=(8, 4.5))
    c_pers = np.asarray([[all_traces["Persistent Dynamic"][s]["consistency"][t] for t in episodes] for s in SEEDS]).mean(axis=0)
    ev_cg = np.asarray([[all_traces["Change-Gated (K=2)"][s]["evidence"][t] for t in episodes] for s in SEEDS]).mean(axis=0)
    
    ax.plot(episodes, c_pers, label="Persistent Dynamic: Consistency c_t", color="tab:green", linewidth=2)
    ax.set_ylabel("Consistency c_t", color="tab:green")
    ax.tick_params(axis="y", labelcolor="tab:green")
    
    ax2 = ax.twinx()
    ax2.plot(episodes, ev_cg, label="Change-Gated (K=2): Change Evidence", color="tab:red", linewidth=2, linestyle="--")
    ax2.set_ylabel("Change Evidence Counter", color="tab:red")
    ax2.tick_params(axis="y", labelcolor="tab:red")
    
    ax.axvline(CHANGE_EPISODE, color="black", linestyle="--", linewidth=1, label="A -> B change (Ep 30)")
    ax.set_xlabel("Episode")
    ax.set_title("Figure 3: Consistency Metric vs. Change Evidence Accumulation")
    ax.grid(alpha=0.25)
    fig.tight_layout()
    fig.savefig(PLOTS_DIR / "plot_3_evidence_and_consistency.png", dpi=300)
    plt.close(fig)

    # Plot 4: Stationary drift vs. change response
    fig, ax = plt.subplots(figsize=(8, 4.5))
    colors = {
        "Static": "purple",
        "Dynamic": "orange",
        "Persistent Dynamic": "green",
        "Change-Gated (K=2)": "red",
        "Change-Gated (K=3)": "darkred",
        "Latching Change-Gated": "brown",
    }
    for v in variant_names:
        sub = df_per_seed[df_per_seed["variant"] == v]
        x = sub["phase_1_mean_drift"]
        y = sub["phase_2_episodes_to_sim_b_0.8"]
        ax.scatter(x, y, label=v, color=colors[v], alpha=0.7, s=40)
        ax.scatter(x.mean(), y.mean(), color=colors[v], marker="X", s=120, edgecolors="black")

    ax.set_xlabel("Phase I Mean Stationary Drift ||Theta_t - Theta_0||_F")
    ax.set_ylabel("Phase II Episodes to GT_B Sim >= 0.8")
    ax.set_title("Figure 4: Trade-off — Stationary Stability vs. Change Response Speed")
    ax.grid(alpha=0.25)
    ax.legend()
    fig.tight_layout()
    fig.savefig(PLOTS_DIR / "plot_4_stationary_vs_change_tradeoff.png", dpi=300)
    plt.close(fig)

    # Plot 5: Noise sensitivity
    fig, ax = plt.subplots(figsize=(8, 4.5))
    for v in main_variants + ["Change-Gated (K=3)", "Latching Change-Gated"]:
        sub = df_noise[df_noise["variant"] == v]
        ax.plot(sub["noise_level"], sub["mean_stationary_drift"], label=v, marker="o", linewidth=2)
    ax.set_xlabel("Observation Noise Probability (Stationary Phase I)")
    ax.set_ylabel("Mean Stationary Drift ||Theta_t - Theta_0||_F")
    ax.set_title("Figure 5: Noise Sensitivity — Stationary Drift Across Observation Noise Levels")
    ax.grid(alpha=0.25)
    ax.legend()
    fig.tight_layout()
    fig.savefig(PLOTS_DIR / "plot_5_noise_sensitivity.png", dpi=300)
    plt.close(fig)

    # Manifest JSON
    manifest = {
        "experiment": "theta-change-gated-20260917",
        "seeds": SEEDS,
        "episodes": EPISODES,
        "change_episode": CHANGE_EPISODE,
        "variants": variant_names,
        "noise_sensitivity_levels": noise_levels,
        "output_files": [
            "report.md",
            "summary_metrics.csv",
            "per_seed_metrics.csv",
            "gate_diagnostics.csv",
            "noise_sensitivity.csv",
            "manifest.json",
            "plots/plot_1_theta_gt_similarity.png",
            "plots/plot_2_adaptation_rate_eta.png",
            "plots/plot_3_evidence_and_consistency.png",
            "plots/plot_4_stationary_vs_change_tradeoff.png",
            "plots/plot_5_noise_sensitivity.png",
        ],
    }
    with open(OUTPUT_DIR / "manifest.json", "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2)

    print("Experiment execution and artifact generation completed successfully.")


if __name__ == "__main__":
    main()

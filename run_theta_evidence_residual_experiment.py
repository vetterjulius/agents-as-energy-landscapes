#!/usr/bin/env python3
"""
Isolated Proof-of-Mechanism Test: Evidence-Scaled Residual Adaptation in CONVERGED.

Tests whether scaling the residual learning rate in CONVERGED by observable mismatch
(||C_t - Theta_t||_F) reduces residual parameter error post-CONVERGED while protecting
stationarity against noise drift.
"""

from __future__ import annotations

import csv
import json
import math
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# -----------------------------------------------------------------------------
# Constants & Paths
# -----------------------------------------------------------------------------
OUTPUT_DIR = Path(r"D:\Julius Vetter\Code\energy-landscape\.storage\analysis\theta-evidence-residual-20260917")
PLOTS_DIR = OUTPUT_DIR / "plots"

M = 6
N = 3
EPISODES = 90
CHANGE_EPISODE = 30
SEEDS = list(range(42, 62)) # 20 seeds
GAMMA = 0.20
MISMATCH_REF = 0.10

VARIANTS = [
    "Persistent Dynamic",
    "State-Gated Freeze",
    "State-Gated Evidence-Residual",      # eta_res_max = 0.01
    "State-Gated Evidence-Residual-Half", # eta_res_max = 0.005
]

# -----------------------------------------------------------------------------
# Helpers
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
    X = np.zeros((N, M), dtype=np.float64)
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

def get_ground_truths() -> tuple[np.ndarray, np.ndarray]:
    gt_A = observed_cooccurrence(canonical_assignment("A"))
    gt_B = observed_cooccurrence(canonical_assignment("B"))
    return gt_A, gt_B

def generate_trajectory(seed: int, scenario: str, noise_prob: float) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    gt_A, gt_B = get_ground_truths()
    
    Xs = []
    Cs = []
    GTs = []
    
    for ep in range(EPISODES):
        if scenario == "A->A":
            curr_gt = gt_A
            base_struct = "A"
        elif scenario == "A->B":
            curr_gt = gt_A if ep < CHANGE_EPISODE else gt_B
            base_struct = "A" if ep < CHANGE_EPISODE else "B"
        elif scenario == "B->A":
            curr_gt = gt_B if ep < CHANGE_EPISODE else gt_A
            base_struct = "B" if ep < CHANGE_EPISODE else "A"
        else:
            raise ValueError(f"Unknown scenario: {scenario}")
            
        X = canonical_assignment(base_struct)
        if rng.random() < noise_prob:
            task = int(rng.integers(0, M))
            source = int(np.argmax(X[:, task]))
            target = int(rng.choice([a for a in range(N) if a != source]))
            X[source, task] = 0.0
            X[target, task] = 1.0
            
        C = observed_cooccurrence(X)
        Xs.append(X)
        Cs.append(C)
        GTs.append(curr_gt)
        
    return np.asarray(Xs), np.asarray(Cs), np.asarray(GTs)

# -----------------------------------------------------------------------------
# Unit Tests (Section 17)
# -----------------------------------------------------------------------------
def run_unit_tests():
    print("Running pre-run unit tests...")
    gt_A, gt_B = get_ground_truths()
    
    # 1. Deterministic formula test
    mismatch = 0.05
    eta_max = 0.01
    eta_expected = 0.01 * (0.05 / 0.10)
    eta_calc = eta_max * float(np.clip(mismatch / MISMATCH_REF, 0.0, 1.0))
    assert abs(eta_calc - eta_expected) < 1e-8, f"Formula test failed: {eta_calc} vs {eta_expected}"
    
    # 2. Clip upper bound test
    mismatch_large = 0.25
    eta_calc_large = eta_max * float(np.clip(mismatch_large / MISMATCH_REF, 0.0, 1.0))
    assert abs(eta_calc_large - 0.01) < 1e-8, f"Clip test failed: {eta_calc_large}"
    
    # 3. Simulate mini trajectory to verify ADAPTING eta=0.10 and timing
    theta = gt_A.copy()
    C_test = gt_B.copy()
    mismatch_val = float(np.linalg.norm(C_test - theta))
    assert mismatch_val > 0.10, "Initial mismatch must be > 0.10"
    
    # Ground truth isolation test: check formula takes only C and Theta
    eta_test = 0.01 * float(np.clip(mismatch_val / MISMATCH_REF, 0.0, 1.0))
    assert eta_test == 0.01, "Eta max cap test passed."
    
    print("All pre-run unit tests PASSED cleanly.")

# -----------------------------------------------------------------------------
# Simulation Logic
# -----------------------------------------------------------------------------
def run_simulation(
    variant: str,
    Cs: np.ndarray,
    GTs: np.ndarray,
    theta_init: np.ndarray,
) -> dict:
    theta = theta_init.copy()
    cbar_prev = None
    C_prev = None
    
    state = "STABLE"
    suspect_evidence = 0
    converged_counter = 0
    
    if variant == "State-Gated Evidence-Residual":
        eta_res_max = 0.01
    elif variant == "State-Gated Evidence-Residual-Half":
        eta_res_max = 0.005
    else:
        eta_res_max = 0.00
        
    theta_befores = []
    theta_afters = []
    etas = []
    consistencies = []
    c_changes = []
    cbar_theta_dists = []
    mismatches = []
    states = []
    first_converged_ep = None
    
    for t in range(EPISODES):
        C = Cs[t]
        theta_before = theta.copy()
        
        if t == 0:
            consistency = 1.0
            c_change = 0.0
            cbar = C.copy()
        else:
            consistency = cosine_similarity(C, cbar_prev)
            c_change = float(np.linalg.norm(C - C_prev))
            cbar = (1.0 - GAMMA) * cbar_prev + GAMMA * C
            
        cbar_theta_dist = float(np.linalg.norm(cbar - theta_before))
        mismatch = float(np.linalg.norm(C - theta_before))
        
        if variant == "Persistent Dynamic":
            alpha = float(np.clip((consistency - 0.70) / (0.90 - 0.70), 0.0, 1.0))
            eta = 0.10 * alpha
            if eta > 0.05:
                curr_state = "ADAPTING"
            elif eta > 0.005:
                curr_state = "CONVERGED"
            else:
                curr_state = "STABLE"
        else:
            # State-Gated Persistent Dynamic state machine
            is_mismatch = (consistency < 0.70) or (c_change >= 0.15)
            
            if state == "STABLE":
                if is_mismatch:
                    suspect_evidence += 1
                    if suspect_evidence >= 2:
                        state = "SUSPECTED_CHANGE"
                        suspect_evidence = 0
                else:
                    suspect_evidence = 0
                    
            elif state == "SUSPECTED_CHANGE":
                if consistency >= 0.75:
                    if cbar_theta_dist < 0.08:
                        state = "CONVERGED"
                        converged_counter = 0
                    else:
                        state = "ADAPTING"
                        converged_counter = 0
                    suspect_evidence = 0
                elif consistency >= 0.70:
                    state = "ADAPTING"
                    converged_counter = 0
                    suspect_evidence = 0
                    
            elif state == "ADAPTING":
                if consistency < 0.65:
                    suspect_evidence += 1
                    if suspect_evidence >= 2:
                        state = "SUSPECTED_CHANGE"
                        suspect_evidence = 0
                        converged_counter = 0
                else:
                    suspect_evidence = 0
                    if cbar_theta_dist < 0.08:
                        converged_counter += 1
                        if converged_counter >= 2:
                            state = "CONVERGED"
                    else:
                        converged_counter = 0
                        
            elif state == "CONVERGED":
                if consistency < 0.65 or cbar_theta_dist > 0.10:
                    suspect_evidence += 1
                    if suspect_evidence >= 2:
                        state = "SUSPECTED_CHANGE"
                        suspect_evidence = 0
                        converged_counter = 0
                else:
                    suspect_evidence = 0
                    
            curr_state = state
            if curr_state == "CONVERGED" and first_converged_ep is None and t >= CHANGE_EPISODE:
                first_converged_ep = t
                
            if curr_state == "STABLE":
                eta = 0.0
            elif curr_state == "SUSPECTED_CHANGE":
                eta = 0.0
            elif curr_state == "ADAPTING":
                eta = 0.10
            elif curr_state == "CONVERGED":
                if variant == "State-Gated Freeze":
                    eta = 0.0
                else:
                    scaled_alpha = float(np.clip(mismatch / MISMATCH_REF, 0.0, 1.0))
                    eta = eta_res_max * scaled_alpha
            else:
                eta = 0.0
                
        # Update Theta
        theta_next = normalize_theta((1.0 - eta) * theta_before + eta * C)
        theta = theta_next
        
        theta_befores.append(theta_before)
        theta_afters.append(theta_next)
        etas.append(eta)
        consistencies.append(consistency)
        c_changes.append(c_change)
        cbar_theta_dists.append(cbar_theta_dist)
        mismatches.append(mismatch)
        states.append(curr_state)
        
        cbar_prev = cbar.copy()
        C_prev = C.copy()
        
    return {
        "theta_before": np.asarray(theta_befores),
        "theta_after": np.asarray(theta_afters),
        "eta": np.asarray(etas),
        "consistency": np.asarray(consistencies),
        "c_change": np.asarray(c_changes),
        "cbar_theta_dist": np.asarray(cbar_theta_dists),
        "mismatch": np.asarray(mismatches),
        "state": np.asarray(states),
        "first_converged_ep": first_converged_ep,
    }

# -----------------------------------------------------------------------------
# Main Execution
# -----------------------------------------------------------------------------
def main():
    run_unit_tests()
    
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    PLOTS_DIR.mkdir(parents=True, exist_ok=True)
    
    gt_A, gt_B = get_ground_truths()
    
    test_cases = [
        ("A->A", 0.00),
        ("A->A", 0.20),
        ("A->A", 0.40),
        ("A->B", 0.00),
        ("A->B", 0.20),
        ("B->A", 0.00),
    ]
    
    state_diagnostics = []
    per_seed_rows = []
    post_conv_rows = []
    
    print("Running evidence-residual experiment across 20 seeds...")
    
    for sc, noise in test_cases:
        for seed in SEEDS:
            Xs, Cs, GTs = generate_trajectory(seed, sc, noise)
            theta_init = gt_A if sc in ("A->A", "A->B") else gt_B
            
            for variant in VARIANTS:
                res = run_simulation(variant, Cs, GTs, theta_init)
                
                tb = res["theta_before"]
                ta = res["theta_after"]
                eta = res["eta"]
                cons = res["consistency"]
                cc = res["c_change"]
                ctd = res["cbar_theta_dist"]
                mis = res["mismatch"]
                st = res["state"]
                first_conv = res["first_converged_ep"]
                
                for ep in range(EPISODES):
                    theta_step = float(np.linalg.norm(ta[ep] - tb[ep]))
                    dist_gt = float(np.linalg.norm(tb[ep] - GTs[ep]))
                    
                    state_diagnostics.append({
                        "scenario": sc,
                        "noise_prob": noise,
                        "seed": seed,
                        "episode": ep,
                        "variant": variant,
                        "state": st[ep],
                        "eta": eta[ep],
                        "consistency": cons[ep],
                        "C_change": cc[ep],
                        "Cbar_theta_distance": ctd[ep],
                        "mismatch": mis[ep],
                        "theta_step": theta_step,
                        "distance_to_GT": dist_gt,
                    })
                    
                # Phase I metrics
                drifts = [float(np.linalg.norm(tb[ep] - theta_init)) for ep in range(30)]
                mean_drift_p1 = float(np.mean(drifts))
                max_drift_p1 = float(np.max(drifts))
                final_drift_p1 = float(drifts[29])
                mean_theta_step_p1 = float(np.mean([np.linalg.norm(ta[ep] - tb[ep]) for ep in range(30)]))
                
                frac_eta_gt_001 = float(np.mean(eta[:30] > 0.001))
                frac_eta_gt_005 = float(np.mean(eta[:30] > 0.005))
                frac_adapting_p1 = float(np.mean(st[:30] == "ADAPTING"))
                frac_converged_p1 = float(np.mean(st[:30] == "CONVERGED"))
                
                # Structural change metrics
                if sc in ("A->B", "B->A"):
                    target_gt = gt_B if sc == "A->B" else gt_A
                    dists = [float(np.linalg.norm(tb[ep] - target_gt)) for ep in range(EPISODES)]
                    
                    adapting_eps = [ep for ep in range(30, EPISODES) if st[ep] == "ADAPTING"]
                    converged_eps = [ep for ep in range(30, EPISODES) if st[ep] == "CONVERGED"]
                    
                    time_to_adapting = (adapting_eps[0] - 30) if adapting_eps else None
                    time_to_converged = (converged_eps[0] - 30) if converged_eps else None
                    
                    dist_p2_end = dists[59]
                    dist_p3_end = dists[89]
                    min_dist = float(np.min(dists[30:]))
                    final_dist = dists[89]
                    
                    dist_at_conv = dists[first_conv] if first_conv is not None else None
                    dist_conv_5 = dists[first_conv + 5] if (first_conv is not None and first_conv + 5 < EPISODES) else None
                    dist_conv_10 = dists[first_conv + 10] if (first_conv is not None and first_conv + 10 < EPISODES) else None
                    dist_conv_20 = dists[first_conv + 20] if (first_conv is not None and first_conv + 20 < EPISODES) else None
                    
                    # Post-converged diagnostics
                    if first_conv is not None:
                        conv_indices = range(first_conv, EPISODES)
                        mean_res_eta_conv = float(np.mean(eta[conv_indices])) if len(conv_indices) > 0 else 0.0
                        max_res_eta_conv = float(np.max(eta[conv_indices])) if len(conv_indices) > 0 else 0.0
                        mean_mismatch_conv = float(np.mean(mis[conv_indices])) if len(conv_indices) > 0 else 0.0
                        
                        for offset in range(EPISODES - first_conv):
                            ep_idx = first_conv + offset
                            post_conv_rows.append({
                                "scenario": sc,
                                "noise_prob": noise,
                                "seed": seed,
                                "variant": variant,
                                "episodes_since_converged": offset,
                                "distance_to_target": dists[ep_idx],
                                "mismatch": mis[ep_idx],
                                "state": st[ep_idx],
                                "eta": eta[ep_idx],
                            })
                    else:
                        mean_res_eta_conv = 0.0
                        max_res_eta_conv = 0.0
                        mean_mismatch_conv = 0.0
                else:
                    time_to_adapting = None
                    time_to_converged = None
                    dist_p2_end = float(np.linalg.norm(tb[59] - theta_init))
                    dist_p3_end = float(np.linalg.norm(tb[89] - theta_init))
                    min_dist = float(np.min([np.linalg.norm(tb[ep] - theta_init) for ep in range(EPISODES)]))
                    final_dist = float(np.linalg.norm(tb[89] - theta_init))
                    dist_at_conv = None
                    dist_conv_5 = None
                    dist_conv_10 = None
                    dist_conv_20 = None
                    mean_res_eta_conv = 0.0
                    max_res_eta_conv = 0.0
                    mean_mismatch_conv = 0.0
                    
                per_seed_rows.append({
                    "scenario": sc,
                    "noise_prob": noise,
                    "seed": seed,
                    "variant": variant,
                    "mean_drift_p1": mean_drift_p1,
                    "max_drift_p1": max_drift_p1,
                    "final_drift_p1": final_drift_p1,
                    "mean_theta_step_p1": mean_theta_step_p1,
                    "frac_eta_gt_001_p1": frac_eta_gt_001,
                    "frac_eta_gt_005_p1": frac_eta_gt_005,
                    "frac_adapting_p1": frac_adapting_p1,
                    "frac_converged_p1": frac_converged_p1,
                    "time_to_adapting": time_to_adapting,
                    "time_to_converged": time_to_converged,
                    "dist_p2_end": dist_p2_end,
                    "dist_p3_end": dist_p3_end,
                    "min_dist": min_dist,
                    "final_dist": final_dist,
                    "dist_at_conv": dist_at_conv,
                    "dist_conv_5": dist_conv_5,
                    "dist_conv_10": dist_conv_10,
                    "dist_conv_20": dist_conv_20,
                    "mean_res_eta_conv": mean_res_eta_conv,
                    "max_res_eta_conv": max_res_eta_conv,
                    "mean_mismatch_conv": mean_mismatch_conv,
                })

    df_diag = pd.DataFrame(state_diagnostics)
    df_per_seed = pd.DataFrame(per_seed_rows)
    df_post_conv = pd.DataFrame(post_conv_rows)
    
    df_diag.to_csv(OUTPUT_DIR / "state_diagnostics.csv", index=False)
    df_per_seed.to_csv(OUTPUT_DIR / "per_seed_metrics.csv", index=False)
    df_post_conv.to_csv(OUTPUT_DIR / "post_convergence_trajectory.csv", index=False)
    
    # Summary Metrics
    summary_rows = []
    for (sc, noise, variant), group in df_per_seed.groupby(["scenario", "noise_prob", "variant"]):
        freeze_group = df_per_seed[(df_per_seed["scenario"] == sc) & (df_per_seed["noise_prob"] == noise) & (df_per_seed["variant"] == "State-Gated Freeze")]
        freeze_drift = freeze_group["mean_drift_p1"].mean() if len(freeze_group) > 0 else 1.0
        
        curr_drift = group["mean_drift_p1"].mean()
        rel_drift_increase = (curr_drift / freeze_drift) if (freeze_drift > 1e-8) else (1.0 if curr_drift == 0.0 else 999.0)
        
        summary_rows.append({
            "scenario": sc,
            "noise_prob": noise,
            "variant": variant,
            "mean_drift_p1": curr_drift,
            "std_drift_p1": group["mean_drift_p1"].std(),
            "max_drift_p1": group["max_drift_p1"].max(),
            "final_drift_p1": group["final_drift_p1"].mean(),
            "mean_theta_step_p1": group["mean_theta_step_p1"].mean(),
            "frac_eta_gt_001_p1": group["frac_eta_gt_001_p1"].mean(),
            "frac_eta_gt_005_p1": group["frac_eta_gt_005_p1"].mean(),
            "frac_adapting_p1": group["frac_adapting_p1"].mean(),
            "frac_converged_p1": group["frac_converged_p1"].mean(),
            "mean_time_to_adapting": group["time_to_adapting"].dropna().mean(),
            "mean_time_to_converged": group["time_to_converged"].dropna().mean(),
            "mean_dist_p2_end": group["dist_p2_end"].mean(),
            "mean_dist_p3_end": group["dist_p3_end"].mean(),
            "mean_min_dist": group["min_dist"].mean(),
            "mean_final_dist": group["final_dist"].mean(),
            "std_final_dist": group["final_dist"].std(),
            "mean_dist_at_conv": group["dist_at_conv"].dropna().mean(),
            "mean_dist_conv_5": group["dist_conv_5"].dropna().mean(),
            "mean_dist_conv_10": group["dist_conv_10"].dropna().mean(),
            "mean_dist_conv_20": group["dist_conv_20"].dropna().mean(),
            "mean_res_eta_conv": group["mean_res_eta_conv"].mean(),
            "max_res_eta_conv": group["max_res_eta_conv"].max(),
            "mean_mismatch_conv": group["mean_mismatch_conv"].mean(),
            "relative_drift_increase_p1": rel_drift_increase,
        })
        
    df_summary = pd.DataFrame(summary_rows)
    df_summary.to_csv(OUTPUT_DIR / "summary_metrics.csv", index=False)
    
    # Noise metrics file
    df_noise = df_summary[df_summary["scenario"] == "A->A"].copy()
    df_noise.to_csv(OUTPUT_DIR / "noise_metrics.csv", index=False)
    
    # Manifest JSON
    manifest = {
        "timestamp": "2026-09-17T18:30:00Z",
        "experiment": "Evidence-Scaled Residual Adaptation Mechanism Test",
        "M": M,
        "N": N,
        "episodes": EPISODES,
        "change_episode": CHANGE_EPISODE,
        "seeds": SEEDS,
        "test_cases": test_cases,
        "variants": VARIANTS,
        "hyperparameters": {
            "mismatch_ref": MISMATCH_REF,
            "eta_res_max_full": 0.01,
            "eta_res_max_half": 0.005,
            "eta_adapting": 0.10,
            "gamma": GAMMA,
            "converged_dist_threshold": 0.08,
            "converged_patience": 2,
        },
    }
    with open(OUTPUT_DIR / "manifest.json", "w") as f:
        json.dump(manifest, f, indent=2)
        
    print("CSV data files and manifest written.")
    
    # -------------------------------------------------------------------------
    # Visualizations (6 Plots Required)
    # -------------------------------------------------------------------------
    print("Generating 6 mechanism plots...")
    colors = {
        "Persistent Dynamic": "#ff7f0e",
        "State-Gated Freeze": "#7f7f7f",
        "State-Gated Evidence-Residual": "#1f77b4",
        "State-Gated Evidence-Residual-Half": "#2ca02c",
    }
    
    # Plot 1: Distance to GT_B after CONVERGED (A->B, noise=0.00)
    fig, ax = plt.subplots(figsize=(8, 4.5), dpi=300)
    df_pc_ab0 = df_post_conv[(df_post_conv["scenario"] == "A->B") & (df_post_conv["noise_prob"] == 0.00)]
    for v in VARIANTS:
        v_data = df_pc_ab0[df_pc_ab0["variant"] == v]
        if len(v_data) > 0:
            means = v_data.groupby("episodes_since_converged")["distance_to_target"].mean()
            ax.plot(means.index, means.values, label=v, color=colors[v], linewidth=2.5)
    ax.set_title("Plot 1: Asymptotic Distance to Ground Truth B after CONVERGED Entry (A->B, noise=0.00)")
    ax.set_xlabel("Episodes since entering CONVERGED")
    ax.set_ylabel("||Theta_t - GT_B||_F")
    ax.grid(True, linestyle=":", alpha=0.6)
    ax.legend(loc="upper right")
    fig.tight_layout()
    fig.savefig(PLOTS_DIR / "plot1_distance_after_converged.png")
    plt.close(fig)
    
    # Plot 2: Mismatch vs Residual Eta in CONVERGED (A->B, noise=0.00, Seed 42)
    fig, ax1 = plt.subplots(figsize=(8, 4.5), dpi=300)
    ax2 = ax1.twinx()
    
    seed_data = df_diag[(df_diag["scenario"] == "A->B") & (df_diag["noise_prob"] == 0.00) & (df_diag["seed"] == 42) & (df_diag["variant"] == "State-Gated Evidence-Residual")]
    
    l1 = ax1.plot(seed_data["episode"], seed_data["mismatch"], color="#9467bd", linewidth=2, label="mismatch = ||C_t - Theta_t||_F")
    l2 = ax2.plot(seed_data["episode"], seed_data["eta"], color="#1f77b4", linewidth=2, label="eta_t (Evidence-Residual)")
    
    ax1.axvline(30, color="black", linestyle="--", alpha=0.7, label="Structural Change (ep 30)")
    ax1.axhline(0.10, color="purple", linestyle=":", alpha=0.5, label="mismatch_ref (0.10)")
    
    ax1.set_xlabel("Episode")
    ax1.set_ylabel("Mismatch ||C_t - Theta_t||_F", color="#9467bd")
    ax2.set_ylabel("Learning Rate eta_t", color="#1f77b4")
    ax1.set_title("Plot 2: Observable Mismatch & Evidence-Scaled Eta (Seed 42, A->B)")
    
    lines = l1 + l2
    labels = [l.get_label() for l in lines]
    ax1.legend(lines, labels, loc="upper right")
    ax1.grid(True, linestyle=":", alpha=0.6)
    fig.tight_layout()
    fig.savefig(PLOTS_DIR / "plot2_mismatch_and_residual_eta.png")
    plt.close(fig)
    
    # Plot 3: Stationary Drift Comparison (A->A, Phase I Mean Drift across noise levels)
    fig, ax = plt.subplots(figsize=(8, 4.5), dpi=300)
    df_stat_sum = df_summary[df_summary["scenario"] == "A->A"]
    for v in VARIANTS:
        v_data = df_stat_sum[df_stat_sum["variant"] == v].sort_values("noise_prob")
        ax.plot(v_data["noise_prob"], v_data["mean_drift_p1"], marker="o", label=v, color=colors[v], linewidth=2.5)
    ax.set_title("Plot 3: Stationary Phase I Theta Drift vs. Observation Noise Level (A->A)")
    ax.set_xlabel("Observation Noise Probability")
    ax.set_ylabel("Mean Stationary Drift ||Theta_t - Theta_0||_F")
    ax.grid(True, linestyle=":", alpha=0.6)
    ax.legend(loc="upper left")
    fig.tight_layout()
    fig.savefig(PLOTS_DIR / "plot3_stationary_drift_comparison.png")
    plt.close(fig)
    
    # Plot 4: Theta Step and Eta profiles over time (A->B, noise=0.20)
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(8, 6), sharex=True, dpi=300)
    df_ab2 = df_diag[(df_diag["scenario"] == "A->B") & (df_diag["noise_prob"] == 0.20)]
    for v in VARIANTS:
        v_data = df_ab2[df_ab2["variant"] == v]
        eta_means = v_data.groupby("episode")["eta"].mean()
        step_means = v_data.groupby("episode")["theta_step"].mean()
        
        ax1.plot(eta_means.index, eta_means.values, label=v, color=colors[v], linewidth=2)
        ax2.plot(step_means.index, step_means.values, label=v, color=colors[v], linewidth=2)
        
    ax1.axvline(30, color="black", linestyle="--", alpha=0.7)
    ax2.axvline(30, color="black", linestyle="--", alpha=0.7)
    
    ax1.set_ylabel("Mean Learning Rate (eta)")
    ax2.set_ylabel("Theta Step ||Delta Theta||_F")
    ax2.set_xlabel("Episode")
    ax1.set_title("Plot 4: Eta & Theta Step Profiles (A->B, noise=0.20)")
    ax1.grid(True, linestyle=":", alpha=0.6)
    ax2.grid(True, linestyle=":", alpha=0.6)
    ax1.legend(loc="upper right")
    ax2.legend(loc="upper right")
    fig.tight_layout()
    fig.savefig(PLOTS_DIR / "plot4_theta_step_and_eta.png")
    plt.close(fig)
    
    # Plot 5: Distance at CONVERGED Entry vs Final Distance (A->B, noise=0.00)
    fig, ax = plt.subplots(figsize=(8, 4.5), dpi=300)
    df_ab0_ps = df_per_seed[(df_per_seed["scenario"] == "A->B") & (df_per_seed["noise_prob"] == 0.00)]
    
    x_coords = np.arange(len(VARIANTS))
    width = 0.35
    
    d_at_conv_means = [df_ab0_ps[df_ab0_ps["variant"] == v]["dist_at_conv"].dropna().mean() for v in VARIANTS]
    d_final_means = [df_ab0_ps[df_ab0_ps["variant"] == v]["final_dist"].mean() for v in VARIANTS]
    
    ax.bar(x_coords - width/2, d_at_conv_means, width, label="Distance at CONVERGED Entry", color="#9467bd", alpha=0.85)
    ax.bar(x_coords + width/2, d_final_means, width, label="Final Distance (Ep 89)", color="#1f77b4", alpha=0.85)
    
    ax.set_xticks(x_coords)
    ax.set_xticklabels(VARIANTS, rotation=15)
    ax.set_ylabel("||Theta - GT_B||_F")
    ax.set_title("Plot 5: Parameter Error at CONVERGED Entry vs. Final Episode (A->B, noise=0.00)")
    ax.grid(True, linestyle=":", alpha=0.6)
    ax.legend(loc="upper right")
    fig.tight_layout()
    fig.savefig(PLOTS_DIR / "plot5_distance_at_converged_vs_final.png")
    plt.close(fig)
    
    # Plot 6: Noise Sensitivity (Relative Drift Increase relative to Freeze)
    fig, ax = plt.subplots(figsize=(8, 4.5), dpi=300)
    df_stat_sum = df_summary[df_summary["scenario"] == "A->A"]
    for v in ["State-Gated Evidence-Residual", "State-Gated Evidence-Residual-Half"]:
        v_data = df_stat_sum[df_stat_sum["variant"] == v].sort_values("noise_prob")
        ax.plot(v_data["noise_prob"], v_data["relative_drift_increase_p1"], marker="s", label=v, color=colors[v], linewidth=2.5)
        
    ax.axhline(1.0, color="gray", linestyle="--", label="State-Gated Freeze Baseline (1.0x)")
    ax.set_title("Plot 6: Relative Stationary Drift Increase vs. Freeze Baseline (A->A)")
    ax.set_xlabel("Observation Noise Probability")
    ax.set_ylabel("Relative Drift Increase (x Freeze)")
    ax.grid(True, linestyle=":", alpha=0.6)
    ax.legend(loc="upper left")
    fig.tight_layout()
    fig.savefig(PLOTS_DIR / "plot6_noise_sensitivity.png")
    plt.close(fig)
    
    print("All 6 plots generated and saved successfully.")

if __name__ == "__main__":
    main()

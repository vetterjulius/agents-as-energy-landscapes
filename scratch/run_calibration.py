import sys
import os
import torch
import numpy as np
import pandas as pd

from controlled_benchmark.config import BenchmarkConfig
from controlled_benchmark.runner import ControlledBenchmarkRunner

def run_calibration():
    print("=== Starting 40-Run Calibration Test ===")
    cfg = BenchmarkConfig(
        seeds=[42, 43],
        scenarios=["Stationary", "Capability Drift", "Task Shift", "Dependency Change"],
        num_episodes=10,
        perturb_episode=5,
        output_dir="scratch/calibration_output",
        verbose_diagnostics=False,
    )
    
    runner = ControlledBenchmarkRunner(cfg)
    results = runner.run_benchmark()
    
    runs_df = results["runs_df"]
    episodes_df = results["episodes_df"]
    integrity = results["integrity"]
    
    print(f"Total runs executed: {len(runs_df)}")
    assert len(runs_df) == 40, f"Expected 40 runs, got {len(runs_df)}"
    
    print(f"All integrity checks passed: {integrity['all_checks_pass']}")
    assert integrity['all_checks_pass'] is True, f"Integrity failed: {integrity}"
    
    # Check conditions
    conditions = set(runs_df["condition_id"].unique())
    print(f"Conditions present in output: {conditions}")
    expected_conditions = {
        "conventional_greedy",
        "static_energy_greedy",
        "static_energy_sa",
        "adaptive_energy_greedy",
        "adaptive_energy_sa"
    }
    assert conditions == expected_conditions, f"Conditions mismatch: {conditions} vs {expected_conditions}"
    
    # Check for NaN / Inf in primary metrics
    for col in ["ppee_10"]:
        nan_count = runs_df[col].isna().sum()
        inf_count = np.isinf(runs_df[col]).sum()
        print(f"Metric {col} in runs_df: NaN={nan_count}, Inf={inf_count}")
        assert nan_count == 0, f"NaNs found in {col}"
        assert inf_count == 0, f"Infs found in {col}"

    for col in ["external_energy", "internal_energy"]:
        nan_count = episodes_df[col].isna().sum()
        inf_count = np.isinf(episodes_df[col]).sum()
        print(f"Metric {col} in episodes_df: NaN={nan_count}, Inf={inf_count}")
        assert nan_count == 0, f"NaNs found in {col}"
        assert inf_count == 0, f"Infs found in {col}"
        
    # Check B0 specific runs
    b0_runs = runs_df[runs_df["condition_id"] == "conventional_greedy"]
    print(f"B0 runs count: {len(b0_runs)}")
    assert len(b0_runs) == 8, f"Expected 8 B0 runs (4 scenarios x 2 seeds), got {len(b0_runs)}"
    
    # Second run for reproducibility test
    print("=== Re-running with identical seeds for Reproducibility Check ===")
    cfg_repro = BenchmarkConfig(
        seeds=[42, 43],
        scenarios=["Stationary", "Capability Drift", "Task Shift", "Dependency Change"],
        num_episodes=10,
        perturb_episode=5,
        output_dir="scratch/calibration_repro_output",
        verbose_diagnostics=False,
    )
    runner_repro = ControlledBenchmarkRunner(cfg_repro)
    results_repro = runner_repro.run_benchmark()
    runs_df_repro = results_repro["runs_df"]
    
    # Exclude runtime columns for exact deterministic reproducibility check
    non_time_cols = [c for c in runs_df.columns if "runtime" not in c]
    pd.testing.assert_frame_equal(runs_df[non_time_cols], runs_df_repro[non_time_cols])
    print("REPRODUCIBILITY CHECK: PASS (Identical outputs produced across duplicate runs)")
    print("=== CALIBRATION TEST SUCCESSFUL ===")

if __name__ == "__main__":
    run_calibration()

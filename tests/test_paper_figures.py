from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from scipy import stats

from paper_artifacts.analysis import make_paper_figures as figures


def valid_runs() -> pd.DataFrame:
    rows = []
    conditions = ["B0", "B1", "B2", "B3", "B4"]
    scenarios = ["Stationary", "Capability Drift", "Task Shift", "Dependency Change"]
    for scenario in scenarios:
        for condition in conditions:
            for seed in range(42, 62):
                rows.append({
                    "scenario_id": scenario, "condition": condition, "seed": seed,
                    "ppee_10": float((seed - 41) * (conditions.index(condition) + 1) / 1000),
                })
    return pd.DataFrame(rows)


def valid_episodes() -> pd.DataFrame:
    rows = []
    for scenario in ["Stationary", "Capability Drift", "Task Shift", "Dependency Change"]:
        for condition in ["B0", "B1", "B2", "B3", "B4"]:
            for seed in range(42, 62):
                for episode in range(50):
                    rows.append({
                        "scenario_id": scenario, "condition": condition, "seed": seed,
                        "episode": episode, "external_energy": 1.0 + episode / 100,
                        "reference_energy": 1.0, "reconfig_cost": 0.1,
                        "kappa_norm": np.nan if condition == "B0" else episode / 10,
                        "theta_norm": np.nan if condition == "B0" else 1.0,
                        "theta_diff_norm": np.nan if condition == "B0" else 0.5,
                    })
    return pd.DataFrame(rows)


def test_fresh_coverage_validation_accepts_minimal_data() -> None:
    figures.validate_runs(valid_runs())
    figures.validate_episodes(valid_episodes())


def test_missing_column_and_incomplete_coverage_are_rejected() -> None:
    runs = valid_runs().drop(columns=["ppee_10"])
    with pytest.raises(ValueError, match="missing required columns"):
        figures.validate_runs(runs)
    incomplete = valid_runs().iloc[:-1]
    with pytest.raises(ValueError, match="20 distinct seeds"):
        figures.validate_runs(incomplete)


def test_nonfinite_plotted_values_are_rejected() -> None:
    runs = valid_runs()
    runs.loc[0, "ppee_10"] = np.inf
    with pytest.raises(ValueError, match="non-finite"):
        figures.validate_runs(runs)


def test_manifest_contains_selected_run_and_all_figures(tmp_path: Path) -> None:
    import reproduce_controlled_benchmark as reproduction

    results = tmp_path / "results"
    artifacts = tmp_path / "artifacts"
    paper_figures = tmp_path / "paper_figures"
    results.mkdir()
    artifacts.mkdir()
    paper_figures.mkdir()
    (results / "config.json").write_text(json.dumps({"git_commit": "abc"}), encoding="utf-8")
    reproduction.write_provenance(artifacts, results, paper_figures)
    manifest = json.loads((artifacts / "figure_manifest.json").read_text(encoding="utf-8"))
    assert [entry["figure_id"] for entry in manifest["figures"]] == reproduction.FIGURE_NAMES
    assert manifest["run_identifier"] == str(results.resolve())


def test_numerical_generators_write_expected_outputs(tmp_path: Path) -> None:
    output = tmp_path / "figures"
    artifacts = tmp_path / "artifacts"
    runs = valid_runs()
    episodes = valid_episodes()
    figures.make_primary(runs, output, artifacts)
    figures.make_temporal(episodes, output, artifacts)
    figures.make_internal(episodes, output, artifacts)
    figures.make_ecdf(runs, output, artifacts)

    comparison_rows = []
    for name, a, b, family in figures.COMPARISONS:
        for scenario in figures.SCENARIOS:
            values = runs[runs.scenario_id == scenario].pivot(
                index="seed", columns="condition", values="ppee_10")
            diff = (values[a] - values[b]).to_numpy()
            half = stats.t.ppf(0.975, len(diff) - 1) * diff.std(ddof=1) / np.sqrt(len(diff))
            comparison_rows.append({
                "metric": "ppee_10", "comparison": name, "scenario": scenario,
                "n_pairs": len(diff), "mean_diff": diff.mean(),
                "median_diff": np.median(diff),
                "ci95_t_low": diff.mean() - half,
                "ci95_t_high": diff.mean() + half,
            })
    comparisons = pd.DataFrame(comparison_rows)
    figures.make_paired(runs, comparisons, output, artifacts)

    tables = artifacts / "tables"
    figdata = artifacts / "figure_data"
    tables.mkdir(parents=True)
    diag_rows = []
    for scenario in figures.SCENARIOS:
        for condition in ["B1", "B2", "B3", "B4"]:
            diag_rows.append({
                "scenario": scenario, "condition": condition,
                "kappa_change_post_mean": 0.1,
                "theta_change_post_mean": 0.2,
                "theta_diff_norm_final_mean": 0.5,
            })
    pd.DataFrame(diag_rows).to_csv(tables / "adaptation_diagnostics.csv", index=False)
    pd.DataFrame([
        {"scenario_id": scenario, "condition": condition,
         "window": "post-window(25-34)", "mean_reconfig": 0.1}
        for scenario in figures.SCENARIOS
        for condition in ["B1", "B2", "B3", "B4"]
    ]).to_csv(figdata / "assignment_churn_by_window.csv", index=False)
    pd.DataFrame([
        {"scenario": scenario, "condition": condition,
         "final_theta_diff_mean": 0.5}
        for scenario in figures.SCENARIOS
        for condition in ["B1", "B2", "B3", "B4"]
    ]).to_csv(figdata / "theta_accuracy_by_condition.csv", index=False)
    figures.make_diagnostics(episodes, artifacts, output)

    for name in [
        "main_ppee10", "paired_effects", "external_energy_trajectories",
        "internal_state_trajectories", "ppee_distribution", "adaptation_diagnostics",
    ]:
        assert (output / f"{name}.pdf").is_file()
        assert (output / f"{name}.png").is_file()
        assert (figdata / f"{name}.csv").is_file()

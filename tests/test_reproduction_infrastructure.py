from __future__ import annotations

import csv
from pathlib import Path

import pytest

import reproduce_controlled_benchmark as reproduction


def test_stage_order_and_shared_paths(tmp_path: Path) -> None:
    results = tmp_path / "results"
    artifacts = tmp_path / "artifacts"
    figures = tmp_path / "figures"

    stages = reproduction.stage_commands(results, artifacts, figures)
    names = [name for name, _ in stages]
    assert names == [
        "controlled benchmark",
        "integrity validation",
        "paper statistical analysis",
        "adaptation diagnostics",
        "setup tables",
        "all benchmark figures",
        "internal-state manuscript figure",
        "manuscript-facing figures",
        "primary table verification",
        "fresh-versus-frozen verification",
    ]
    for _, command in stages[1:]:
        assert str(results) in command
        assert str(artifacts) in command


def test_stage_uses_explicit_frozen_reference(tmp_path: Path) -> None:
    frozen = tmp_path / "frozen"
    stages = reproduction.stage_commands(
        tmp_path / "results", tmp_path / "artifacts", tmp_path / "figures", frozen)
    comparison = dict(stages)["fresh-versus-frozen verification"]
    assert "--reference-artifacts-dir" in comparison
    assert str(frozen) in comparison


def test_same_selected_and_frozen_artifacts_are_rejected(tmp_path: Path) -> None:
    artifacts = tmp_path / "artifacts"
    with pytest.raises(RuntimeError, match="separate"):
        reproduction.run_pipeline(
            tmp_path / "results", artifacts, tmp_path / "figures", artifacts)


def test_non_empty_results_directory_is_rejected(tmp_path: Path) -> None:
    path = tmp_path / "results"
    path.mkdir()
    (path / "old.csv").write_text("stale\n", encoding="utf-8")

    with pytest.raises(RuntimeError, match="non-empty reproduction directory"):
        reproduction.require_empty_or_absent(path)


def test_primary_statistics_validation_requires_24_pairs_of_20(tmp_path: Path) -> None:
    artifacts = tmp_path / "artifacts"
    table = artifacts / "tables" / "statistical_comparisons.csv"
    table.parent.mkdir(parents=True)
    fields = ["metric", "n_pairs", "permutation_p_exact",
              "holm_p_within_family", "ci95_t_low", "ci95_t_high"]
    rows = [
        {
            "metric": "ppee_10",
            "n_pairs": "20",
            "permutation_p_exact": "0.5",
            "holm_p_within_family": "1.0",
            "ci95_t_low": "0.0",
            "ci95_t_high": "1.0",
        }
        for _ in range(24)
    ]
    with table.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)

    reproduction.validate_primary_statistics(artifacts)

    rows[0]["n_pairs"] = "1"
    with table.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    with pytest.raises(RuntimeError, match="n_pairs=20"):
        reproduction.validate_primary_statistics(artifacts)


def test_missing_manuscript_artifact_is_rejected(tmp_path: Path) -> None:
    with pytest.raises(RuntimeError, match="Required fresh reproduction artifacts"):
        reproduction.validate_manuscript_artifacts(
            tmp_path / "results", tmp_path / "artifacts", tmp_path / "figures")

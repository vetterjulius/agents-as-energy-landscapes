"""Run the controlled-benchmark reproduction pipeline.

This command is intentionally separate from the LLM PoC. It refuses to reuse a
non-empty results directory, passes the same selected roots to every analysis
stage, and stops on the first failed required stage.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import subprocess
import sys
from pathlib import Path
from typing import Iterable

import pandas as pd


ROOT = Path(__file__).resolve().parent
FIGURE_NAMES = [
    "experimental_design_information_boundary",
    "main_ppee10",
    "paired_effects",
    "external_energy_trajectories",
    "internal_state_trajectories",
    "ppee_distribution",
    "adaptation_diagnostics",
]


def build_command(module: str, *arguments: str) -> list[str]:
    return [sys.executable, "-m", module, *arguments]


def current_commit() -> str:
    result = subprocess.run(
        ["git", "rev-parse", "--short", "HEAD"],
        cwd=ROOT,
        check=True,
        capture_output=True,
        text=True,
    )
    return result.stdout.strip()


def analysis_arguments(results_dir: Path, artifacts_dir: Path) -> list[str]:
    return ["--results-dir", str(results_dir), "--artifacts-dir", str(artifacts_dir)]


def stage_commands(
    results_dir: Path,
    artifacts_dir: Path,
    paper_figures_dir: Path,
    frozen_artifacts_dir: Path | None = None,
) -> list[tuple[str, list[str]]]:
    paths = analysis_arguments(results_dir, artifacts_dir)
    commit = current_commit()
    commands = [
        (
            "controlled benchmark",
            build_command(
                "run_controlled_benchmark",
                "--mode", "research",
                "--output-dir", str(results_dir),
            ),
        ),
        ("integrity validation", build_command(
            "analysis.audit_integrity", *paths,
            "--expected-commit", commit)),
        ("paper statistical analysis", build_command(
            "analysis.statistical_analysis", *paths)),
        ("adaptation diagnostics", build_command(
            "analysis.adaptation_diagnostics", *paths)),
        ("setup tables", build_command(
            "analysis.make_setup_tables", *paths)),
        ("experimental design figure", build_command(
            "analysis.make_experimental_design_figure", *paths,
            "--paper-figures-dir", str(paper_figures_dir))),
        ("numerical paper figures", build_command(
            "analysis.make_paper_figures", *paths,
            "--paper-figures-dir", str(paper_figures_dir)))
    ]
    if frozen_artifacts_dir is not None:
        commands.append(("optional historical comparison", build_command(
            "analysis.compare_fresh_vs_frozen", *paths,
            "--reference-artifacts-dir", str(frozen_artifacts_dir))))
    return commands


def require_empty_or_absent(path: Path) -> None:
    if path.exists() and any(path.iterdir()):
        raise RuntimeError(
            f"Refusing to reuse non-empty reproduction directory: {path}. "
            "Choose a new run-specific --results-dir."
        )


def require_clean_worktree() -> None:
    result = subprocess.run(
        ["git", "status", "--porcelain", "--untracked-files=normal"],
        cwd=ROOT,
        check=True,
        capture_output=True,
        text=True,
    )
    if result.stdout.strip():
        raise RuntimeError(
            "Refusing to start reproduction with a dirty Git worktree. "
            "Commit or stash source changes and remove unrelated untracked files first.\n"
            f"git status --porcelain:\n{result.stdout.rstrip()}"
        )


def require_files(paths: Iterable[Path]) -> None:
    missing = [str(path) for path in paths if not path.is_file()]
    if missing:
        raise RuntimeError("Required fresh reproduction artifacts are missing:\n" +
                           "\n".join(missing))


def validate_primary_statistics(artifacts_dir: Path) -> None:
    path = artifacts_dir / "tables" / "statistical_comparisons.csv"
    require_files([path])
    with path.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    primary = [row for row in rows if row.get("metric") == "ppee_10"]
    if len(primary) != 24:
        raise RuntimeError(f"Expected 24 primary comparisons, found {len(primary)}.")
    invalid = [row for row in primary if row.get("n_pairs") != "20"]
    if invalid:
        raise RuntimeError("Primary comparisons must all have n_pairs=20.")
    required_columns = {
        "permutation_p_exact", "holm_p_within_family",
        "ci95_t_low", "ci95_t_high",
    }
    missing = required_columns - set(rows[0]) if rows else required_columns
    if missing:
        raise RuntimeError(f"Primary statistics are missing columns: {sorted(missing)}")


def validate_manuscript_artifacts(results_dir: Path, artifacts_dir: Path,
                                  paper_figures_dir: Path) -> list[Path]:
    required = [
        results_dir / "runs.csv",
        results_dir / "episodes.csv",
        results_dir / "integrity.json",
        artifacts_dir / "tables" / "main_ppee_results.csv",
        artifacts_dir / "tables" / "statistical_comparisons.csv",
        artifacts_dir / "tables" / "adaptation_diagnostics.csv",
        artifacts_dir / "tables" / "experimental_setup.csv",
        artifacts_dir / "figure_manifest.json",
        artifacts_dir / "artifact_inventory.json",
    ]
    for name in FIGURE_NAMES:
        required.extend([
            paper_figures_dir / f"{name}.pdf",
            paper_figures_dir / f"{name}.png",
            artifacts_dir / "figure_data" / f"{name}.csv",
        ])
    require_files(required)
    validate_primary_statistics(artifacts_dir)
    validate_figure_data(artifacts_dir)
    return required


def validate_figure_data(artifacts_dir: Path) -> None:
    required_columns = {
        "main_ppee10": {"scenario", "condition", "seed", "ppee_10"},
        "paired_effects": {"comparison", "scenario", "seed", "diff",
                           "mean_diff", "median_diff", "ci95_t_low",
                           "ci95_t_high", "n_pairs"},
        "external_energy_trajectories": {"scenario", "condition", "episode",
                                          "mean", "ci95_t_low", "ci95_t_high"},
        "internal_state_trajectories": {"scenario", "condition", "episode",
                                         "theta_norm_mean", "kappa_norm_mean",
                                         "theta_diff_norm_mean"},
        "ppee_distribution": {"scenario", "condition", "seed_rank", "ppee_10", "ecdf"},
        "adaptation_diagnostics": {"panel", "scenario", "condition", "value"},
        "experimental_design_information_boundary": {"element", "value"},
    }
    for name, columns in required_columns.items():
        path = artifacts_dir / "figure_data" / f"{name}.csv"
        with path.open(newline="", encoding="utf-8") as handle:
            header = set(next(csv.reader(handle)))
        missing = columns - header
        if missing:
            raise RuntimeError(f"{path} is missing plotted-data columns: {sorted(missing)}")
    main = pd.read_csv(artifacts_dir / "figure_data" / "main_ppee10.csv")
    main_counts = main.groupby(["scenario", "condition"]).seed.nunique()
    if len(main_counts) != 20 or not (main_counts == 20).all():
        raise RuntimeError("Primary plotted data lacks 20 seeds per scenario/condition.")
    paired = pd.read_csv(artifacts_dir / "figure_data" / "paired_effects.csv")
    paired_counts = paired.groupby(["comparison", "scenario"]).seed.nunique()
    if len(paired_counts) != 24 or not (paired_counts == 20).all():
        raise RuntimeError("Paired plotted data lacks 20 paired seeds per contrast/scenario.")
    temporal = pd.read_csv(
        artifacts_dir / "figure_data" / "external_energy_trajectories.csv")
    if set(temporal.episode) != set(range(15, 35)):
        raise RuntimeError("Temporal plotted data must cover episodes 15..34.")
    internal = pd.read_csv(
        artifacts_dir / "figure_data" / "internal_state_trajectories.csv")
    internal_counts = internal.groupby(["scenario", "condition"]).episode.nunique()
    if len(internal_counts) != 8 or not (internal_counts == 50).all():
        raise RuntimeError("Internal-state plotted data lacks 50 episodes per cell.")


def write_provenance(artifacts_dir: Path, results_dir: Path,
                     paper_figures_dir: Path) -> None:
    config_path = results_dir / "config.json"
    config_bytes = config_path.read_bytes()
    fingerprint = hashlib.sha256(config_bytes).hexdigest()
    manifest = {
        "run_identifier": str(results_dir.resolve()),
        "results_dir": str(results_dir.resolve()),
        "artifacts_dir": str(artifacts_dir.resolve()),
        "paper_figures_dir": str(paper_figures_dir.resolve()),
        "benchmark_commit": json.loads(config_bytes).get("git_commit", "unknown"),
        "configuration_fingerprint_sha256": fingerprint,
        "figures": [],
    }
    roles = {
        "experimental_design_information_boundary": "main",
        "main_ppee10": "main",
        "paired_effects": "main",
        "external_energy_trajectories": "appendix",
        "internal_state_trajectories": "appendix",
        "ppee_distribution": "appendix",
        "adaptation_diagnostics": "appendix",
    }
    for name in FIGURE_NAMES:
        inputs = [str((results_dir / "runs.csv").resolve()),
                  str((results_dir / "episodes.csv").resolve())]
        if name in {"experimental_design_information_boundary"}:
            inputs = [str(config_path.resolve())]
        if name == "paired_effects":
            inputs.append(str((artifacts_dir / "tables" /
                               "statistical_comparisons.csv").resolve()))
        if name == "adaptation_diagnostics":
            inputs.extend([
                str((artifacts_dir / "tables" / "adaptation_diagnostics.csv").resolve()),
                str((artifacts_dir / "figure_data" /
                     "assignment_churn_by_window.csv").resolve()),
                str((artifacts_dir / "figure_data" /
                     "theta_accuracy_by_condition.csv").resolve()),
            ])
        manifest["figures"].append({
            "figure_id": name,
            "manuscript_role": roles[name],
            "generator": ("make_experimental_design_figure.py"
                          if name == "experimental_design_information_boundary"
                          else "make_paper_figures.py"),
            "selected_results_directory": str(results_dir.resolve()),
            "input_filenames": inputs,
            "output_pdf": str((paper_figures_dir / f"{name}.pdf").resolve()),
            "output_png": str((paper_figures_dir / f"{name}.png").resolve()),
            "plotted_data_csv": str((artifacts_dir / "figure_data" /
                                     f"{name}.csv").resolve()),
            "run_identifier": str(results_dir.resolve()),
            "benchmark_commit": manifest["benchmark_commit"],
            "configuration_fingerprint_sha256": fingerprint,
            "generation_status": "complete",
        })
    (artifacts_dir / "figure_manifest.json").write_text(
        json.dumps(manifest, indent=2), encoding="utf-8")
    inventory = []
    for root in (results_dir, artifacts_dir, paper_figures_dir):
        if root.is_dir():
            inventory.extend(str(path.resolve()) for path in root.rglob("*") if path.is_file())
    inventory_path = artifacts_dir / "artifact_inventory.json"
    inventory_path.write_text(
        json.dumps({"run_identifier": str(results_dir.resolve()),
                    "files": sorted(set(inventory + [str(inventory_path.resolve())]))},
                   indent=2), encoding="utf-8")


def run_pipeline(results_dir: Path, artifacts_dir: Path,
                 paper_figures_dir: Path,
                 frozen_artifacts_dir: Path | None = None) -> None:
    if frozen_artifacts_dir is not None and frozen_artifacts_dir.resolve() == artifacts_dir.resolve():
        raise RuntimeError(
            "Frozen reference artifacts must be separate from the selected output artifacts."
        )
    require_clean_worktree()
    require_empty_or_absent(results_dir)
    require_empty_or_absent(artifacts_dir)
    require_empty_or_absent(paper_figures_dir)
    results_dir.parent.mkdir(parents=True, exist_ok=True)
    artifacts_dir.mkdir(parents=True, exist_ok=True)
    paper_figures_dir.mkdir(parents=True, exist_ok=True)

    for name, command in stage_commands(
            results_dir, artifacts_dir, paper_figures_dir, frozen_artifacts_dir):
        print(f"[reproduce] {name}", flush=True)
        if name == "optional historical comparison":
            result = subprocess.run(command, cwd=ROOT)
            if result.returncode != 0:
                print("[reproduce] optional historical comparison failed; continuing")
            continue
        subprocess.run(command, cwd=ROOT, check=True)

    write_provenance(artifacts_dir, results_dir, paper_figures_dir)
    generated = validate_manuscript_artifacts(
        results_dir, artifacts_dir, paper_figures_dir)
    print("\nGenerated artifact inventory:")
    for path in generated:
        print(f"  {path}")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results-dir", type=Path, required=True)
    parser.add_argument("--artifacts-dir", type=Path, required=True)
    parser.add_argument("--paper-figures-dir", type=Path, required=True)
    parser.add_argument(
        "--frozen-artifacts-dir",
        type=Path,
        default=None,
        help="Existing frozen artifact directory used only for comparison.",
    )
    args = parser.parse_args(argv)
    run_pipeline(
        args.results_dir.resolve(),
        args.artifacts_dir.resolve(),
        args.paper_figures_dir.resolve(),
        (args.frozen_artifacts_dir.resolve()
         if args.frozen_artifacts_dir is not None else None),
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

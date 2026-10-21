"""Run the controlled-benchmark reproduction pipeline.

This command is intentionally separate from the LLM PoC. It refuses to reuse a
non-empty results directory, passes the same selected roots to every analysis
stage, and stops on the first failed required stage.
"""
from __future__ import annotations

import argparse
import csv
import subprocess
import sys
from pathlib import Path
from typing import Iterable


ROOT = Path(__file__).resolve().parent
ANALYSIS = ROOT / "paper_artifacts" / "analysis"
DEFAULT_RESULTS = ROOT / "results" / "controlled_landscape_solver"
DEFAULT_ARTIFACTS = ROOT / "paper_artifacts"
DEFAULT_PAPER_FIGURES = ROOT / "paper" / "figures"
DEFAULT_FROZEN_ARTIFACTS = ROOT / "paper_artifacts"


def build_command(script: Path, *arguments: str) -> list[str]:
    return [sys.executable, str(script), *arguments]


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
    frozen_artifacts_dir: Path = DEFAULT_FROZEN_ARTIFACTS,
) -> list[tuple[str, list[str]]]:
    paths = analysis_arguments(results_dir, artifacts_dir)
    commit = current_commit()
    return [
        (
            "controlled benchmark",
            build_command(
                ROOT / "controlled_landscape_solver_benchmark.py",
                "--mode", "research",
                "--output-dir", str(results_dir),
            ),
        ),
        ("integrity validation", build_command(
            ANALYSIS / "audit_integrity.py", *paths,
            "--expected-commit", commit)),
        ("paper statistical analysis", build_command(
            ANALYSIS / "statistical_analysis.py", *paths)),
        ("adaptation diagnostics", build_command(
            ANALYSIS / "adaptation_diagnostics.py", *paths)),
        ("setup tables", build_command(
            ANALYSIS / "make_setup_tables.py", *paths)),
        ("all benchmark figures", build_command(
            ANALYSIS / "make_figures.py", *paths)),
        ("internal-state manuscript figure", build_command(
            ANALYSIS / "make_internal_state_figure.py", *paths,
            "--output", str(paper_figures_dir / "internal_state_trajectories.pdf"))),
        ("manuscript-facing figures", build_command(
            ANALYSIS / "make_paper_figures.py", *paths,
            "--paper-figures-dir", str(paper_figures_dir))),
        ("primary table verification", build_command(
            ANALYSIS / "verify_results_table.py", *paths,
            "--manuscript", str(ROOT / "paper" / "results_section.tex"))),
        ("fresh-versus-frozen verification", build_command(
            ANALYSIS / "compare_fresh_vs_frozen.py", *paths,
            "--reference-artifacts-dir", str(frozen_artifacts_dir))),
    ]


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
        paper_figures_dir / "main_ppee10.pdf",
        paper_figures_dir / "ppee_distribution.pdf",
        paper_figures_dir / "paired_effects.pdf",
        paper_figures_dir / "internal_state_trajectories.pdf",
    ]
    require_files(required)
    validate_primary_statistics(artifacts_dir)
    return required


def run_pipeline(results_dir: Path, artifacts_dir: Path,
                 paper_figures_dir: Path,
                 frozen_artifacts_dir: Path = DEFAULT_FROZEN_ARTIFACTS) -> None:
    if frozen_artifacts_dir.resolve() == artifacts_dir.resolve():
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
        subprocess.run(command, cwd=ROOT, check=True)

    generated = validate_manuscript_artifacts(
        results_dir, artifacts_dir, paper_figures_dir)
    print("\nGenerated artifact inventory:")
    for path in generated:
        print(f"  {path}")
    print("\nManual checks remaining:")
    print("  - Compare all manuscript prose and LaTeX tables with the selected CSV outputs.")
    print("  - Compile paper/paper.tex and inspect all figure placement and captions.")
    print("  - The LLM PoC live-run table is not reproduced by this deterministic pipeline.")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results-dir", type=Path, default=DEFAULT_RESULTS)
    parser.add_argument("--artifacts-dir", type=Path, default=DEFAULT_ARTIFACTS)
    parser.add_argument("--paper-figures-dir", type=Path, default=DEFAULT_PAPER_FIGURES)
    parser.add_argument(
        "--frozen-artifacts-dir",
        type=Path,
        default=DEFAULT_FROZEN_ARTIFACTS,
        help="Existing frozen artifact directory used only for comparison.",
    )
    args = parser.parse_args(argv)
    run_pipeline(
        args.results_dir.resolve(),
        args.artifacts_dir.resolve(),
        args.paper_figures_dir.resolve(),
        args.frozen_artifacts_dir.resolve(),
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

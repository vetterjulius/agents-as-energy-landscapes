
from __future__ import annotations

import argparse
import csv
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch

import analysis.common as common


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    common.add_path_arguments(parser)
    parser.add_argument("--paper-figures-dir", type=Path, required=True)
    args = parser.parse_args(argv)
    common.configure_paths(args.results_dir, args.artifacts_dir)
    output_dir = args.paper_figures_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    config = common.load_config()
    if config["perturb_episode"] != common.PERTURB_EPISODE:
        raise ValueError("Selected benchmark configuration does not use episode 25")
    labels = [
        ("B0", "Conventional Greedy", "external baseline"),
        ("B1/B2", "Static landscape", "Greedy / SA"),
        ("B3/B4", "Adaptive landscape", "Greedy / SA; joint Theta and kappa"),
    ]
    fig, ax = plt.subplots(figsize=(12, 5.2))
    ax.axis("off")
    boxes = [
        (0.03, 0.55, 0.24, 0.25, "Benchmark conditions\nB0–B4"),
        (0.38, 0.55, 0.24, 0.25, "Decision at episode t\ncurrent (Theta_t, kappa_t)"),
        (0.73, 0.55, 0.24, 0.25, "Assignment X_t\nsolver output"),
        (0.38, 0.10, 0.24, 0.25, "Assignment-derived update\nstate for episode t+1"),
        (0.73, 0.10, 0.24, 0.25, "Post-hoc evaluation\nexternal G_t; not an input"),
    ]
    for x, y, w, h, text in boxes:
        ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.02",
                                    facecolor="#eaf2f8", edgecolor="#34495e"))
        ax.text(x + w / 2, y + h / 2, text, ha="center", va="center", fontsize=10)
    ax.annotate("", xy=(0.38, 0.675), xytext=(0.27, 0.675),
                arrowprops={"arrowstyle": "->", "lw": 1.4})
    ax.annotate("", xy=(0.73, 0.675), xytext=(0.62, 0.675),
                arrowprops={"arrowstyle": "->", "lw": 1.4})
    ax.annotate("", xy=(0.50, 0.35), xytext=(0.50, 0.55),
                arrowprops={"arrowstyle": "->", "lw": 1.4})
    ax.annotate("", xy=(0.73, 0.225), xytext=(0.62, 0.225),
                arrowprops={"arrowstyle": "->", "lw": 1.4})
    ax.text(0.50, 0.46, "X_t observed", ha="center", va="center", fontsize=9)
    ax.text(0.50, 0.04,
            "Episode 25 uses the pre-change state; the update from X_25 first affects episode 26",
            ha="center", fontsize=10, fontweight="bold")
    ax.text(0.03, 0.88, "Information boundary", fontsize=13, fontweight="bold")
    ax.text(0.03, 0.84,
            "Ground truth, reference optimum, and PPEE are unavailable during decision-making",
            fontsize=9)
    ax.text(0.03, 0.50, "\n".join(f"{a}: {b} ({c})" for a, b, c in labels),
            va="top", fontsize=8)
    fig.tight_layout()
    pdf = output_dir / "experimental_design_information_boundary.pdf"
    png = output_dir / "experimental_design_information_boundary.png"
    fig.savefig(pdf, bbox_inches="tight")
    fig.savefig(png, bbox_inches="tight")
    plt.close(fig)
    data_path = common.ARTIFACTS / "figure_data" / "experimental_design_information_boundary.csv"
    data_path.parent.mkdir(parents=True, exist_ok=True)
    with data_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=["element", "value"])
        writer.writeheader()
        for element, value in [
            ("perturbation_episode", config["perturb_episode"]),
            ("conditions", "; ".join(config["conditions"])),
            ("scenarios", "; ".join(config["scenarios"])),
            ("update_timing", "X_25 update first affects episode 26"),
            ("ground_truth_boundary", "post-hoc evaluation only"),
        ]:
            writer.writerow({"element": element, "value": value})
    print(f"generated conceptual figure in {output_dir}")


if __name__ == "__main__":
    main()

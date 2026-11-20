# Energy landscapes for multi-agent task allocation

This repository contains the implementation and analysis for a controlled
benchmark of static and assignment-adaptive energy-based task allocation.

## Prerequisites
Python should be installed on your computer.
**Create virtual environment and install Dependencies (for example with pip):**
```bash
# venv
python -m venv .venv
source .venv/bin/activate # .venv/Scripts/activate.ps1 on Windows

# dependencies
pip install -r requirements.txt
```

## Running the benchmark

Run this command from the repository root. The output directories must be new or empty:

```bash
python -m reproduce_controlled_benchmark \
  --results-dir <path> \
  --artifacts-dir <path> \
  --paper-figures-dir <path>
```

The pipeline runs the research benchmark and its integrity checks, statistical analysis, diagnostics, table generation, and figure generation. It writes raw results, analysis tables, figure data and figures to your specified paths.

## Statistical outputs

The benchmark's internal `results/statistics.csv` contains Monte Carlo
sign-flip and Wilcoxon results. This only for benchmark-internal logging. The paper uses the paired
sign-flip analysis in `artifacts/tables/statistical_comparisons.csv`, with all
`2^20` sign patterns enumerated and Holm correction applied within each
pre-specified test family. `results/statistics.csv` is not used for the
paper's inferential results.
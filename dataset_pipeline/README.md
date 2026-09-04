# Dataset Pipeline for Energy Landscape Multi-Agent Benchmark

This directory contains an automated pipeline to fetch real-world GitHub issues, convert them into vector embeddings & matrix topologies (without live LLM dependency), and evaluate the Energy-Based Orchestrator against state-of-the-art baselines.

---

## 🚀 Quickstart: Step-by-Step Guide

### Step 1: Fetch Raw GitHub Issues
### Datasets Included & Reproducibility

To ensure **100% full scientific reproducibility**, raw issue JSON files are generated with fixed limits and saved to disk.

1. **Quick Validation Dataset (3 Tasks, 4 Agents):**
   ```bash
   python dataset_pipeline/fetch_issues.py --repo pallets/flask --limit 15 --output dataset_pipeline/raw_issues.json
   python dataset_pipeline/process_issues.py --input dataset_pipeline/raw_issues.json --output dataset_pipeline/processed_gh_dataset.json --dim 8
   ```

2. **Publication Benchmark Dataset (50 Tasks, 10 Agents):**
   ```bash
   python dataset_pipeline/fetch_issues.py --repo pallets/flask --limit 50 --output dataset_pipeline/raw_issues_large.json
   python dataset_pipeline/process_issues.py --input dataset_pipeline/raw_issues_large.json --output dataset_pipeline/processed_gh_dataset_large.json --dim 8
   ```

---

### Step 3: Run Benchmark on Real Data
Runs all orchestrators (Energy, EBMAO, Market Auction SOTA, Greedy, Beam, Tabu) across all available real-world datasets:
```bash
python dataset_pipeline/run_real_benchmark.py
```

---

## 📊 Outputs
- **Markdown Report:** `results/real_github_benchmark_report.md`
- **CSV Data:** `results/real_github_benchmark_results.csv`
- **Plots:** Saved to `results/plots/`

import os
import pandas as pd
from benchmark.logging_config import get_logger
from benchmark.evaluation.report import generate_markdown_report, save_csv_results
from benchmark.evaluation.plots import plot_results

logger = get_logger("comprehensive_report")

def generate_comprehensive_report(all_results):
    logger.info("Generating comprehensive benchmark report with all experiments")
    
    generate_markdown_report(all_results)
    save_csv_results(all_results)
    plot_results(all_results)
    
    incorporate_scaling_results()
    
    incorporate_coupling_results()
    
    incorporate_dynamic_results()
    
    update_figure_catalog_complete()
    
    logger.info("Comprehensive report generation complete")


def incorporate_scaling_results():
    """Add scaling experiment results to the report if available"""
    tasks_csv = "results/scaling_tasks_results.csv"
    agents_csv = "results/scaling_agents_results.csv"
    
    if not (os.path.exists(tasks_csv) or os.path.exists(agents_csv)):
        logger.debug("No scaling results found, skipping")
        return
    
    logger.info("Incorporating scaling experiment results")
    
    report_path = "results/benchmark_report.md"
    with open(report_path, "r", encoding="utf-8") as f:
        content = f.read()
    
    # Check if scaling section already exists
    if "## Scalability Analysis" in content:
        logger.debug("Scalability section already exists")
        return
    
    # The section is already added by the updated generate_markdown_report
    logger.debug("Scalability section handled by main report generator")


def incorporate_coupling_results():
    """Add coupling sweep results to the report if available"""
    coupling_csv = "results/coupling_results.csv"
    
    if not os.path.exists(coupling_csv):
        logger.debug("No coupling sweep results found, skipping")
        return
    
    logger.info("Incorporating coupling sweep results")
    
    # The section is already added by the updated generate_markdown_report
    logger.debug("Coupling section handled by main report generator")


def incorporate_dynamic_results():
    """Add dynamic adaptation results to the report if available"""
    metrics_csv = "results/dynamic_adaptation_metrics.csv"
    
    if not os.path.exists(metrics_csv):
        logger.debug("No dynamic adaptation results found, skipping")
        return
    
    logger.info("Incorporating dynamic adaptation results")
    
    # The section is already added by the updated generate_markdown_report
    logger.debug("Dynamic adaptation section handled by main report generator")


def update_figure_catalog_complete():
    """Ensure the figure catalog includes all generated plots"""
    catalog_path = "results/figure_catalog.md"
    plots_dir = "results/plots"
    
    if not os.path.exists(catalog_path):
        logger.warning("Figure catalog not found")
        return
    
    if not os.path.exists(plots_dir):
        logger.warning("Plots directory not found")
        return
    
    logger.info("Verifying figure catalog completeness")
    
    # Get all PNG files in plots directory
    all_plots = [f for f in os.listdir(plots_dir) if f.endswith('.png')]
    
    with open(catalog_path, "r", encoding="utf-8") as f:
        catalog_content = f.read()
    
    # Check which plots are mentioned in catalog
    mentioned_plots = [plot for plot in all_plots if plot in catalog_content]
    missing_plots = [plot for plot in all_plots if plot not in catalog_content]
    
    if missing_plots:
        logger.info(f"Found {len(missing_plots)} plots not yet in catalog:")
        for plot in missing_plots:
            logger.debug(f"  - {plot}")
        
        # Add a miscellaneous section for any unaccounted plots
        with open(catalog_path, "a", encoding="utf-8") as f:
            f.write("\n\n## 9. Additional Visualizations\n\n")
            f.write("Additional plots generated during benchmark execution:\n\n")
            for plot in missing_plots:
                f.write(f"### {plot.replace('.png', '').replace('_', ' ').title()}\n")
                f.write(f"![{plot}](plots/{plot})\n\n")
    else:
        logger.info(f"All {len(all_plots)} plots are documented in catalog")
    
    logger.info(f"Figure catalog verified: {len(mentioned_plots)} plots documented")


def generate_experiment_summary():
    """Generate a summary of all completed experiments"""
    summary_path = "results/experiment_summary.md"
    
    experiments = []
    
    # Check which experiments have results
    if os.path.exists("results/benchmark_results.csv"):
        experiments.append("✓ Main Scenario & Solver Battle Benchmark (Independent, Interaction, Frustrated, Dynamic, Distribution Shift)")
    
    if os.path.exists("results/ablation_results.json"):
        experiments.append("✓ Ablation Studies (Representation & Solver Variants)")
    
    if os.path.exists("results/scaling_tasks_results.csv"):
        experiments.append("✓ Task Scalability Analysis")
    
    if os.path.exists("results/scaling_agents_results.csv"):
        experiments.append("✓ Agent Scalability Analysis")
    
    if os.path.exists("results/coupling_results.csv"):
        experiments.append("✓ Parameter Coupling Sweep")
    
    dynamic_files = [f for f in os.listdir("results") if f.startswith("dynamic_") and f.endswith(".csv")]
    if dynamic_files:
        experiments.append(f"✓ Dynamic Adaptation Experiments ({len(dynamic_files)} scenarios)")
    
    # Write summary
    with open(summary_path, "w", encoding="utf-8") as f:
        f.write("# Experiment Execution Summary\n\n")
        f.write(f"Generated: {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
        f.write("## Completed Experiments\n\n")
        for exp in experiments:
            f.write(f"{exp}\n\n")
        
        f.write("\n## Output Files\n\n")
        f.write("### Reports\n")
        f.write("- `benchmark_report.md` - Comprehensive benchmark report\n")
        f.write("- `figure_catalog.md` - Detailed figure documentation\n")
        f.write("- `experiment_summary.md` - This file\n\n")
        
        f.write("### Data Files\n")
        result_csvs = [f for f in os.listdir("results") if f.endswith(".csv")]
        for csv in sorted(result_csvs):
            f.write(f"- `{csv}`\n")
        
        f.write("\n### Visualizations\n")
        f.write("- `plots/` directory contains all generated figures\n")
        
        plots_dir = "results/plots"
        if os.path.exists(plots_dir):
            plot_count = len([f for f in os.listdir(plots_dir) if f.endswith('.png')])
            f.write(f"  - Total figures: {plot_count}\n")
    
    logger.info(f"Experiment summary written to {summary_path}")


if __name__ == "__main__":
    # Can be used standalone to regenerate reports from existing data
    logger.info("Regenerating comprehensive report from existing results")
    
    # Try to load main results if they exist
    if os.path.exists("results/benchmark_results.csv"):
        df = pd.read_csv("results/benchmark_results.csv")
        
        # Reconstruct all_results structure
        all_results = {}
        for scenario in df["Scenario"].unique():
            all_results[scenario] = {}
            scenario_df = df[df["Scenario"] == scenario]
            
            for orch in scenario_df["Orchestrator"].unique():
                orch_df = scenario_df[scenario_df["Orchestrator"] == orch]
                all_results[scenario][orch] = {
                    "energy": orch_df["Energy"].tolist(),
                    "load_balance": orch_df["LoadBalance"].tolist(),
                    "coordination": orch_df["Coordination"].tolist(),
                    "conflicts": orch_df["Conflicts"].tolist(),
                    "runtime": orch_df["Runtime"].tolist(),
                    "specialization": orch_df["Specialization"].tolist(),
                    "task_clustering": orch_df["TaskClustering"].tolist(),
                    "communication_cost": orch_df["CommunicationCost"].tolist(),
                    "conflict_rate": orch_df["ConflictRate"].tolist()
                }
        
        generate_comprehensive_report(all_results)
        generate_experiment_summary()
    else:
        logger.error("No benchmark results found. Run benchmark first.")

import json
import csv
import os

def generate_markdown_report(results, output_path="results/benchmark_report.md"):
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, "w") as f:
        f.write("# Energy-Based Orchestration Benchmark (EOB) Report\n\n")

        for scenario_name, scenario_results in results.items():
            f.write(f"## Scenario: {scenario_name}\n\n")
            f.write("| Orchestrator | Total Energy | Load Balance (std) | Coordination Score | Conflicts | Runtime (s) |\n")
            f.write("| :--- | :---: | :---: | :---: | :---: | :---: |\n")

            for orch_name, metrics in scenario_results.items():
                f.write(f"| {orch_name} | {metrics['energy']:.4f} | {metrics['load_balance']:.4f} | {metrics['coordination']:.2f} | {metrics['conflicts']:.2f} | {metrics['runtime']:.4f} |\n")
            f.write("\n")

def save_csv_results(results, output_path="results/benchmark_results.csv"):
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["Scenario", "Orchestrator", "Energy", "LoadBalance", "Coordination", "Conflicts", "Runtime"])
        for scenario_name, scenario_results in results.items():
            for orch_name, metrics in scenario_results.items():
                writer.writerow([
                    scenario_name, orch_name,
                    metrics['energy'], metrics['load_balance'],
                    metrics['coordination'], metrics['conflicts'],
                    metrics['runtime']
                ])

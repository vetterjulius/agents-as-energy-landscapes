import json
import csv
import os

def generate_markdown_report(results, output_path="results/benchmark_report.md"):
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, "w") as f:
        f.write("# Energy-Based Orchestration Benchmark (EOB) Report\n\n")

        for scenario_name, scenario_results in results.items():
            f.write(f"## Scenario: {scenario_name}\n\n")
            f.write("| Orchestrator | Total Energy | Load Balance (std) | Coordination Score | Conflicts | Runtime (s) | Energy Breakdown |\n")
            f.write("| :--- | :---: | :---: | :---: | :---: | :---: | :--- |\n")

            for orch_name, metrics in scenario_results.items():
                breakdown_str = ""
                if "energy_components" in metrics:
                    breakdown_str = ", ".join([f"{k}: {v:.4f}" for k, v in metrics["energy_components"].items()])

                f.write(f"| {orch_name} | {metrics['energy']:.4f} | {metrics['load_balance']:.4f} | {metrics['coordination']:.2f} | {metrics['conflicts']:.2f} | {metrics['runtime']:.4f} | {breakdown_str} |\n")
            f.write("\n")

def save_csv_results(results, output_path="results/benchmark_results.csv"):
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    # Collect all possible energy component keys
    component_keys = set()
    for scenario_results in results.values():
        for metrics in scenario_results.values():
            if "energy_components" in metrics:
                component_keys.update(metrics["energy_components"].keys())

    sorted_keys = sorted(list(component_keys))

    with open(output_path, "w", newline="") as f:
        writer = csv.writer(f)
        header = ["Scenario", "Orchestrator", "Energy", "LoadBalance", "Coordination", "Conflicts", "Runtime"] + sorted_keys
        writer.writerow(header)
        for scenario_name, scenario_results in results.items():
            for orch_name, metrics in scenario_results.items():
                row = [
                    scenario_name, orch_name,
                    metrics['energy'], metrics['load_balance'],
                    metrics['coordination'], metrics['conflicts'],
                    metrics['runtime']
                ]
                for k in sorted_keys:
                    row.append(metrics.get("energy_components", {}).get(k, 0.0))
                writer.writerow(row)

import matplotlib.pyplot as plt
import os
import pandas as pd

def plot_results(csv_path="results/benchmark_results.csv", output_dir="results/plots"):
    if not os.path.exists(csv_path):
        return

    os.makedirs(output_dir, exist_ok=True)
    df = pd.read_csv(csv_path)

    # 1. Energy Comparison per Scenario
    scenarios = df['Scenario'].unique()
    for scenario in scenarios:
        subset = df[df['Scenario'] == scenario]
        plt.figure(figsize=(10, 6))
        plt.bar(subset['Orchestrator'], subset['Energy'], color='skyblue')
        plt.title(f"Total Energy - {scenario}")
        plt.ylabel("Energy (Lower is better)")
        plt.xticks(rotation=45)
        plt.tight_layout()
        plt.savefig(f"{output_dir}/energy_{scenario}.png")
        plt.close()

    # 2. Constraint Violations
    plt.figure(figsize=(12, 7))
    for orch in df['Orchestrator'].unique():
        subset = df[df['Orchestrator'] == orch]
        plt.plot(subset['Scenario'], subset['Conflicts'], marker='o', label=orch)
    plt.title("Constraint Violations across Scenarios")
    plt.ylabel("Conflicts (Lower is better)")
    plt.legend()
    plt.tight_layout()
    plt.savefig(f"{output_dir}/conflicts_comparison.png")
    plt.close()

from . import (
    analyze_theta_persistence_mechanism,
    run_theta_change_gated_experiment,
    run_theta_evidence_residual_experiment,
    run_theta_residual_learning_experiment,
    run_theta_state_gated_experiment,
    theta_adaptation_ablation
)

EXPERIMENTS = [
    ("Theta persistence mechanism", analyze_theta_persistence_mechanism.main),
    ("Theta change gated", run_theta_change_gated_experiment.main),
    ("Theta evidence residual", run_theta_evidence_residual_experiment.main),
    ("Theta residual learning", run_theta_residual_learning_experiment.main),
    ("Theta state gated", run_theta_state_gated_experiment.main),
    ("Theta adaption ablation", theta_adaptation_ablation.main)
]

def main() -> int:
    print("Running all additional experiments with default configurations...")

    for e in EXPERIMENTS:
        print(f"\nStarting experiment {e[0]}.")
        e[1]()
        print(f"Finished experiment {e[0]}.")

    print("\nFinished all.")

    return 0

if __name__ == "__main__":
    raise SystemExit(main())
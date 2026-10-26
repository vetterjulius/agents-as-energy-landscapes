from . import run_theta_change_gated_experiment, run_theta_evidence_residual_experiment, run_theta_residual_learning_experiment, run_theta_state_gated_experiment, theta_adaptation_ablation

def main() -> int:
    print("Running all addtional experiments with default configurations...\n\n")

    run_theta_change_gated_experiment.main()
    run_theta_evidence_residual_experiment.main()
    run_theta_residual_learning_experiment.main()
    run_theta_state_gated_experiment.main()
    theta_adaptation_ablation.main()

    print("\nFinished.")

    return 0

if __name__ == "__main__":
    raise SystemExit(main())
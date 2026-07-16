# Configuration for the benchmark
config = {
    "dim": 8,
    "num_agents": 5,
    "num_tasks": 10,
    "seed": 42,

    # Core Evaluation Parameters
    "num_evaluation_seeds": 30,  # Run 30 seeds per scenario for statistical significance

    "solver": {
        "iterations": 10,  # Optimized iterations for fast and stable execution
        "temperature_init": 1.0,
        "min_temperature": 0.01,
        "max_temperature": 2.0,
        "target_accept_rate": 0.25
    },

    "model": {
        "lambda_align": 0.5,
        "interaction_weight": 1.0,
        "risk_weight": 1.0,
        "cost_weight": 1.0
    },

    # Robustness Switches and Noise Levels (Configurable toggles)
    "robustness": {
        "capability_noise": {
            "enabled": False,
            "level": 0.15  # Gaussian noise scale to inject into agent embeddings
        },
        "risk_weights_noise": {
            "enabled": False,
            "level": 0.15  # Noise level for risk weights perturbation
        },
        "agent_failure": {
            "enabled": False,
            "rate": 0.20   # Fraction of agents failing during solving
        },
        "comm_outages": {
            "enabled": False,
            "rate": 0.25   # Fraction of interaction graph channels to zero out
        }
    },

    # Scalability Sweep Parameters
    "scalability": {
        "agents_scaling": [5, 10, 20],  # Scale N
        "tasks_scaling": [20, 50, 100],  # Scale M up to 100 (for extremely fast sweeps)
        "sweep_iterations": 10,  # Fewer iterations for larger scalability runs to keep them fast
    },

    # Beam Search configuration
    "beam_search": {
        "beam_width": 3
    },

    # Tabu Search configuration
    "tabu_search": {
        "max_iterations": 20,
        "tabu_tenure": 4
    },

    "sweep": {
        "interaction_weights": [0.0, 0.5, 1.0, 2.0, 5.0],
        "risk_weights": [1.0],
        "cost_weights": [1.0],
        "scenario": "Interaction",
        "iterations": 10
    }
}

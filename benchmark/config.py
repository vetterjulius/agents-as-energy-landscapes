config = {
    "seed": 42,
    "num_evaluation_seeds": 30,

    "problem": {
        "dim": 8,
        "num_agents": 5,
        "num_tasks": 10,
    },

    "energy": {
        "lambda_align": 0.5,
        "lambda_memory": 0.5,
        "interaction_weight": 1.0,
        "risk_weight": 1.0,
        "risk_scale": 1.0,
        "cost_weight": 1.0,
    },

    # Main solver used by Experiment 1 for the proposed systems.
    "experiment_1": {
        "energy_solver": "hybrid",
        "iterations": 100,
    },

    # Solver comparison on the same energy landscape.
    "experiment_2": {
        "iterations": 100,

        "energy_greedy": {},
        "energy_sa": {
            "temperature_init": 4.0,
            "min_temperature": 1.0,
            "max_temperature": 6.0,
            "target_accept_rate": 0.3,
        },
        "energy_hybrid": {
            "temperature_init": 4.0,
            "min_temperature": 1.0,
            "max_temperature": 6.0,
            "target_accept_rate": 0.3,
        },

        "beam_search": {
            "beam_width": 3,
        },

        "tabu_search": {
            "max_iterations": 20,
            "tabu_tenure": 4,
        },
    },

    "ebmao": {
        "lambda_align": 0.5,
        "lambda_memory": 0.5,
        "eta_theta": 0.1,
        "eta_memory": 0.05,

        "temperature_init": 4.0,
        "min_temperature": 1.0,
        "max_temperature": 6.0,
        "target_accept_rate": 0.3,

        "proposal_candidates": 12,
        "proposal_task_sample": 8,
        "agent_sample_size": 6,
        "block_move_size": 4,

        "warm_start_steps": 6,
        "warm_start_type": "greedy",
        "hybrid_cleanup_prob": 0.25,
        "local_refine_steps": 2,
    },

    "robustness": {
        "capability_noise": {
            "enabled": False,
            "level": 0.15,
        },
        "risk_weights_noise": {
            "enabled": False,
            "level": 0.15,
        },
        "agent_failure": {
            "enabled": False,
            "rate": 0.20,
        },
        "comm_outages": {
            "enabled": False,
            "rate": 0.25,
        },
    },

    "scalability": {
        "agents_scaling": [5, 10, 20],
        "tasks_scaling": [20, 50, 100],
        "sweep_iterations": 10,
    },

    "sweep": {
        "interaction_weights": [0.0, 0.5, 1.0, 2.0, 5.0],
        "risk_weights": [1.0],
        "cost_weights": [1.0],
        "scenario": "Interaction",
        "iterations": 100,
    },
}
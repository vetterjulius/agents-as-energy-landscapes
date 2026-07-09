# Configuration for the benchmark
config = {
    "dim": 8,
    "num_agents": 5,
    "num_tasks": 10,
    "seed": 42,
    "solver": {
        "iterations": 25,
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
    "sweep": {
        "interaction_weights": [0.0, 0.5, 1.0, 2.0, 5.0],
        "risk_weights": [1.0],
        "cost_weights": [1.0],
        "scenario": "Interaction",
        "iterations": 100
    }
}

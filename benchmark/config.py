# Configuration for the benchmark
config = {
    "dim": 8,
    "num_agents": 5,
    "num_tasks": 10,
    "seed": 42,
    "solver": {
        "iterations": 100,
        "temperature_init": 1.0,
        "min_temperature": 0.01,
        "max_temperature": 2.0,
        "target_accept_rate": 0.25
    }
}

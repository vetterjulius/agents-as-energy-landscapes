import torch
import random
import yaml

from model.orchestrator import Orchestrator

def load_config(path="config.yaml"):
    with open(path, "r") as f:
        return yaml.safe_load(f)

if __name__ == "__main__":
    cfg = load_config()

    torch.manual_seed(cfg["training"]["seed"])
    random.seed(cfg["training"]["seed"])

    model = Orchestrator(cfg)

    for i in range(cfg["training"]["iterations"]):
        model.step()

        if i % cfg["training"]["log_every"] == 0:
            model.log(i)
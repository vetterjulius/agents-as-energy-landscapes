import torch

from adapters.single_agent import EnergySingleAgent


SQL_AGENTBENCH_AGENTS = [
    {"id": "planner", "role": "Planner", "capability_embedding": torch.randn(8), "memory_state": {}},
    {"id": "sql", "role": "SQL Expert", "capability_embedding": torch.randn(8), "memory_state": {}},
    {"id": "verifier", "role": "Verifier", "capability_embedding": torch.randn(8), "memory_state": {}}
]

agent = EnergySingleAgent(SQL_AGENTBENCH_AGENTS)

def infer(input):
    question = input[-1]["content"]
    return agent.invoke(question)

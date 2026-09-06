import torch
from benchmark.scenarios.base import ProblemInstance, Agent, Task
from benchmark.evaluation.metrics import estimate_llm_cost_and_tokens

def test_estimate_llm_cost_and_tokens():
    torch.manual_seed(42)
    agents = [Agent(id=0, role="worker", capability_embedding=torch.randn(4)) for _ in range(2)]
    tasks = [Task(id=i, embedding=torch.randn(4)) for i in range(4)]
    
    Theta = torch.tensor([[0.0, 1.0, 0.0, 0.0],
                          [1.0, 0.0, 1.0, 0.0],
                          [0.0, 1.0, 0.0, 1.0],
                          [0.0, 0.0, 1.0, 0.0]])
    C = torch.zeros(4, 4)
    W_risk = torch.randn(12, 1)
    
    problem = ProblemInstance(agents=agents, tasks=tasks, interaction_graph=Theta, co_assignment_costs=C, risk_weights=W_risk)
    
    # Assignment X: 2 agents, 4 tasks
    X = torch.zeros(2, 4)
    X[0, 0] = 1.0
    X[0, 1] = 1.0
    X[1, 2] = 1.0
    X[1, 3] = 1.0
    
    result = estimate_llm_cost_and_tokens(problem, X)
    
    assert "total_tokens" in result
    assert "estimated_usd" in result
    assert result["total_tokens"] > 0
    assert result["estimated_usd"] >= 0.0

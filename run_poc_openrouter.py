import os
import torch
from urllib import request
import json
from dotenv import load_dotenv

from benchmark.scenarios.base import ProblemInstance, Agent, Task
from orchestrator.ebmao_orchestrator import EBMAOOrchestrator

# Standard free model on OpenRouter: meta-llama/llama-3.2-1b-instruct:free or google/gemma-2-9b-it:free
DEFAULT_FREE_MODEL = "meta-llama/llama-3.2-1b-instruct:free"
load_dotenv()  # Load environment variables from .env file

def call_openrouter(prompt: str, model: str = DEFAULT_FREE_MODEL) -> str:
    api_key = os.environ.get("OPENROUTER_API_KEY", "")
    if not api_key:
        raise ValueError("OPENROUTER_API_KEY environment variable is missing.")

    url = "https://openrouter.ai/api/v1/chat/completions"
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json"
    }
    payload = {
        "model": model,
        "messages": [{"role": "user", "content": prompt}]
    }

    req = request.Request(url, data=json.dumps(payload).encode("utf-8"), headers=headers)
    with request.urlopen(req) as resp:
        res_data = json.loads(resp.read().decode("utf-8"))
        return res_data["choices"][0]["message"]["content"]


def run_poc():
    print("=" * 70)
    print("EBMAO Real-LLM Proof of Concept (OpenRouter)")
    print("=" * 70)

    # 1. Setup small problem: 2 Agents, 3 Tasks
    agents = [
        Agent(id=0, role="Frontend Specialist", capability_embedding=torch.tensor([0.9, 0.1, 0.1])),
        Agent(id=1, role="Backend & Database Engineer", capability_embedding=torch.tensor([0.1, 0.9, 0.8])),
    ]

    tasks = [
        Task(id=0, embedding=torch.tensor([0.85, 0.15, 0.10]), tags=["React UI Component"]),
        Task(id=1, embedding=torch.tensor([0.10, 0.85, 0.90]), tags=["PostgreSQL Query Optimization"]),
        Task(id=2, embedding=torch.tensor([0.15, 0.80, 0.75]), tags=["REST API Endpoint"]),
    ]

    Theta = torch.zeros(3, 3)
    Theta[1, 2] = 0.8  # Strong synergy link between DB query and REST API
    Theta[2, 1] = 0.8

    C = torch.zeros(3, 3)
    W_risk = torch.randn(9, 1)

    problem = ProblemInstance(agents=agents, tasks=tasks, interaction_graph=Theta, co_assignment_costs=C, risk_weights=W_risk)

    cfg = {
        "model": {
            "num_agents": 2,
            "num_tasks": 3,
            "dim": 3,
            "lambda_align": 0.5,
            "lambda_memory": 0.5,
            "eta_theta": 0.1,
            "eta_memory": 0.1,
            "risk_weight": 1.0,
            "interaction_weight": 1.0,
            "cost_weight": 1.0,
            "temperature_init": 1.0,
            "min_temperature": 0.1,
            "max_temperature": 2.0,
            "target_accept_rate": 0.3,
            "search_mode": "hybrid",
        },
        "solver": {"iterations": 20}
    }

    # 2. Run EBMAO Orchestrator to solve assignment
    print("Running EBMAO Energy Solver to compute optimal task allocation X*...")
    orchestrator = EBMAOOrchestrator(cfg)
    X_opt = orchestrator.solve(problem)
    
    print("\nOptimal Assignment Matrix X*:")
    print(X_opt)

    # 3. Execute LLM calls for each agent's assigned tasks
    print("\nExecuting real LLM calls for assigned tasks via OpenRouter...")
    for a_idx, agent in enumerate(agents):
        assigned_task_indices = (X_opt[a_idx] == 1.0).nonzero(as_tuple=True)[0]
        if len(assigned_task_indices) == 0:
            continue
        
        assigned_tasks = [tasks[t_idx] for t_idx in assigned_task_indices]
        task_descriptions = ", ".join([f"Task #{t.id} ({t.tags[0]})" for t in assigned_tasks])
        
        prompt = (
            f"You are an AI Agent with role: '{agent.role}'.\n"
            f"EBMAO has assigned you the following task(s): {task_descriptions}.\n"
            f"Briefly state in 2-3 sentences how you will execute your assigned task(s)."
        )

        print(f"\n--- Calling LLM for Agent #{agent.id} ({agent.role}) ---")
        try:
            response = call_openrouter(prompt)
            print(f"Response:\n{response}")
        except Exception as e:
            print(f"API Call failed (check OPENROUTER_API_KEY): {e}")

    print("\n" + "=" * 70)
    print("PoC Run Completed Successfully!")
    print("=" * 70)

if __name__ == "__main__":
    run_poc()

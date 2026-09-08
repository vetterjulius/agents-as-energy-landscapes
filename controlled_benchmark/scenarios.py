from __future__ import annotations

import random
from typing import List, Tuple
import torch

from benchmark.scenarios.base import Agent, ProblemInstance, Task
from landscape import LandscapeState, ProblemContext


def problem_instance_to_problem_context(
    problem: ProblemInstance,
    lambda_align: float = 0.5,
    lambda_memory: float = 0.5,
    interaction_weight: float = 1.0,
    cost_weight: float = 1.0,
    risk_weight: float = 1.0,
) -> ProblemContext:
    """Convert a benchmark ProblemInstance into an immutable ProblemContext."""
    s = torch.stack([a.capability_embedding for a in problem.agents])
    c = torch.stack([t.embedding for t in problem.tasks])
    N = len(problem.agents)
    M = len(problem.tasks)
    d = s.shape[1]

    return ProblemContext(
        s=s,
        c=c,
        C=problem.co_assignment_costs.clone(),
        W_risk=problem.risk_weights.clone(),
        N=N,
        M=M,
        d=d,
        lambda_align=lambda_align,
        lambda_memory=lambda_memory,
        interaction_weight=interaction_weight,
        cost_weight=cost_weight,
        risk_weight=risk_weight,
    )


def make_initial_landscape_state(problem: ProblemInstance) -> LandscapeState:
    """Construct a clean, unadapted initial LandscapeState."""
    N = len(problem.agents)
    d = problem.agents[0].capability_embedding.shape[0]
    return LandscapeState(
        kappa=torch.zeros(N, d),
        Theta=problem.interaction_graph.clone(),
    )


def generate_capability_drift_episode(
    episode: int,
    seed: int,
    perturb_episode: int,
    N: int,
    M: int,
    d: int,
) -> ProblemInstance:
    """
    Agent expertise changes abruptly at episode >= perturb_episode.
    Swaps Agent 0 and Agent 1 capability embeddings.
    Reuses existing dynamic benchmark semantics.
    """
    torch.manual_seed(seed + episode)
    random.seed(seed + episode)

    # Base embeddings generated from seed
    torch.manual_seed(seed)
    base_s = torch.randn(N, d)

    torch.manual_seed(seed + episode)
    s = base_s.clone()
    if episode >= perturb_episode and N >= 2:
        s[0], s[1] = base_s[1].clone(), base_s[0].clone()

    agents = [
        Agent(id=f"agent_{i}", role="drift_agent", capability_embedding=s[i])
        for i in range(N)
    ]
    tasks = [
        Task(
            id=f"task_{j}",
            embedding=torch.randn(d),
            estimated_cost=random.uniform(0.5, 1.5),
        )
        for j in range(M)
    ]

    interaction_graph = torch.zeros(M, M)
    for j in range(M):
        for k in range(j + 1, M):
            if random.random() < 0.3:
                val = random.uniform(0.1, 0.8)
                interaction_graph[j, k] = val
                interaction_graph[k, j] = val

    co_assignment_costs = torch.zeros(M, M)
    for j in range(M):
        for k in range(j + 1, M):
            if random.random() < 0.2:
                val = random.uniform(0.1, 0.5)
                co_assignment_costs[j, k] = val
                co_assignment_costs[k, j] = val

    risk_weights = torch.randn(3 * d, 1)

    return ProblemInstance(
        agents=agents,
        tasks=tasks,
        interaction_graph=interaction_graph,
        co_assignment_costs=co_assignment_costs,
        risk_weights=risk_weights,
    )


def generate_task_shift_episode(
    episode: int,
    seed: int,
    perturb_episode: int,
    N: int,
    M: int,
    d: int,
) -> ProblemInstance:
    """
    Task distribution shifts abruptly by +1.5 at episode >= perturb_episode.
    Reuses existing dynamic benchmark semantics.
    """
    torch.manual_seed(seed + episode)
    random.seed(seed + episode)

    torch.manual_seed(seed)
    s = torch.randn(N, d)

    torch.manual_seed(seed + episode)
    agents = [
        Agent(id=f"agent_{i}", role="shift_agent", capability_embedding=s[i])
        for i in range(N)
    ]

    shift = torch.zeros(d)
    if episode >= perturb_episode:
        shift = torch.ones(d) * 1.5

    tasks = [
        Task(
            id=f"task_{j}",
            embedding=torch.randn(d) + shift,
            estimated_cost=random.uniform(0.5, 1.5),
        )
        for j in range(M)
    ]

    interaction_graph = torch.zeros(M, M)
    for j in range(M):
        for k in range(j + 1, M):
            if random.random() < 0.3:
                val = random.uniform(0.1, 0.8)
                interaction_graph[j, k] = val
                interaction_graph[k, j] = val

    co_assignment_costs = torch.zeros(M, M)
    for j in range(M):
        for k in range(j + 1, M):
            if random.random() < 0.2:
                val = random.uniform(0.1, 0.5)
                co_assignment_costs[j, k] = val
                co_assignment_costs[k, j] = val

    risk_weights = torch.randn(3 * d, 1)

    return ProblemInstance(
        agents=agents,
        tasks=tasks,
        interaction_graph=interaction_graph,
        co_assignment_costs=co_assignment_costs,
        risk_weights=risk_weights,
    )


def generate_dependency_change_episode(
    episode: int,
    seed: int,
    perturb_episode: int,
    N: int,
    M: int,
    d: int,
) -> ProblemInstance:
    """
    Task dependency interaction graph Theta abruptly switches pattern at episode >= perturb_episode.
    Reuses existing dynamic benchmark semantics.
    """
    torch.manual_seed(seed + episode)
    random.seed(seed + episode)

    torch.manual_seed(seed)
    s = torch.randn(N, d)

    torch.manual_seed(seed + episode)
    agents = [
        Agent(id=f"agent_{i}", role="dep_agent", capability_embedding=s[i])
        for i in range(N)
    ]
    tasks = [
        Task(
            id=f"task_{j}",
            embedding=torch.randn(d),
            estimated_cost=random.uniform(0.5, 1.5),
        )
        for j in range(M)
    ]

    interaction_graph = torch.zeros(M, M)
    if episode < perturb_episode:
        # Pattern 1: Adjacent pairs have synergy
        for i in range(0, M, 2):
            if i + 1 < M:
                interaction_graph[i, i + 1] = 1.0
                interaction_graph[i + 1, i] = 1.0
    else:
        # Pattern 2: Shifted pairs have synergy
        for i in range(M):
            j = (i + 2) % M
            interaction_graph[i, j] = 1.0
            interaction_graph[j, i] = 1.0

    co_assignment_costs = torch.zeros(M, M)
    risk_weights = torch.randn(3 * d, 1)

    return ProblemInstance(
        agents=agents,
        tasks=tasks,
        interaction_graph=interaction_graph,
        co_assignment_costs=co_assignment_costs,
        risk_weights=risk_weights,
    )


def generate_stationary_episode(
    episode: int,
    seed: int,
    perturb_episode: int,
    N: int,
    M: int,
    d: int,
) -> ProblemInstance:
    """
    Control condition: environment distribution remains completely stationary across all episodes.
    Evaluates whether unnecessary adaptation introduces degradation.
    """
    torch.manual_seed(seed + episode)
    random.seed(seed + episode)

    torch.manual_seed(seed)
    s = torch.randn(N, d)

    torch.manual_seed(seed + episode)
    agents = [
        Agent(id=f"agent_{i}", role="stat_agent", capability_embedding=s[i])
        for i in range(N)
    ]
    tasks = [
        Task(
            id=f"task_{j}",
            embedding=torch.randn(d),
            estimated_cost=random.uniform(0.5, 1.5),
        )
        for j in range(M)
    ]

    interaction_graph = torch.zeros(M, M)
    for j in range(M):
        for k in range(j + 1, M):
            if random.random() < 0.3:
                val = random.uniform(0.1, 0.8)
                interaction_graph[j, k] = val
                interaction_graph[k, j] = val

    co_assignment_costs = torch.zeros(M, M)
    for j in range(M):
        for k in range(j + 1, M):
            if random.random() < 0.2:
                val = random.uniform(0.1, 0.5)
                co_assignment_costs[j, k] = val
                co_assignment_costs[k, j] = val

    risk_weights = torch.randn(3 * d, 1)

    return ProblemInstance(
        agents=agents,
        tasks=tasks,
        interaction_graph=interaction_graph,
        co_assignment_costs=co_assignment_costs,
        risk_weights=risk_weights,
    )


SCENARIO_GENERATORS = {
    "Stationary": generate_stationary_episode,
    "Capability Drift": generate_capability_drift_episode,
    "Task Shift": generate_task_shift_episode,
    "Dependency Change": generate_dependency_change_episode,
}


def generate_scenario_trajectory(
    scenario_id: str,
    seed: int,
    num_episodes: int,
    perturb_episode: int,
    N: int,
    M: int,
    d: int,
) -> List[ProblemInstance]:
    """
    Generate a full deterministic trajectory of ProblemInstance objects for a scenario.
    Crucial for fairness: the EXACT SAME trajectory is reused across all compared methods.
    """
    if scenario_id not in SCENARIO_GENERATORS:
        raise ValueError(
            f"Unknown scenario_id: {scenario_id}. Expected one of {list(SCENARIO_GENERATORS.keys())}"
        )

    generator_fn = SCENARIO_GENERATORS[scenario_id]
    trajectory = []
    for ep in range(num_episodes):
        inst = generator_fn(
            episode=ep,
            seed=seed,
            perturb_episode=perturb_episode,
            N=N,
            M=M,
            d=d,
        )
        trajectory.append(inst)

    return trajectory

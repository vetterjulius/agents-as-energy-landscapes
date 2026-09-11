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


def clone_problem_instance(base: ProblemInstance) -> ProblemInstance:
    """Perform an exact, isolated clone of a ProblemInstance."""
    agents = [
        Agent(
            id=a.id,
            role=a.role,
            capability_embedding=a.capability_embedding.clone(),
        )
        for a in base.agents
    ]
    tasks = [
        Task(
            id=t.id,
            embedding=t.embedding.clone(),
            estimated_cost=float(t.estimated_cost),
        )
        for t in base.tasks
    ]
    return ProblemInstance(
        agents=agents,
        tasks=tasks,
        interaction_graph=base.interaction_graph.clone(),
        co_assignment_costs=base.co_assignment_costs.clone(),
        risk_weights=base.risk_weights.clone(),
    )


def generate_base_problem(
    seed: int,
    N: int,
    M: int,
    d: int,
    scenario_id: str | None = None,
) -> ProblemInstance:
    """
    Generate exactly ONE deterministic base ProblemInstance per (seed, N, M, d).

    Strict fairness guarantee:
    - Base parameters (agents, tasks, costs, graph, risk) are generated once per seed.
    - Zero re-rolling of base parameters per episode.
    - Uses isolated torch.Generator and random.Random seeded by seed (no global side effects).
    """
    rng_torch = torch.Generator().manual_seed(seed)
    rng_py = random.Random(seed)

    base_s = torch.randn(N, d, generator=rng_torch)
    agents = [
        Agent(id=f"agent_{i}", role=f"agent_{i}", capability_embedding=base_s[i].clone())
        for i in range(N)
    ]

    tasks = [
        Task(
            id=f"task_{j}",
            embedding=torch.randn(d, generator=rng_torch),
            estimated_cost=rng_py.uniform(0.5, 1.5),
        )
        for j in range(M)
    ]

    interaction_graph = torch.zeros(M, M)
    if scenario_id == "Dependency Change":
        # Pattern 1: Adjacent pairs have synergy
        for i in range(0, M, 2):
            if i + 1 < M:
                interaction_graph[i, i + 1] = 1.0
                interaction_graph[i + 1, i] = 1.0
    else:
        for j in range(M):
            for k in range(j + 1, M):
                if rng_py.random() < 0.3:
                    val = rng_py.uniform(0.1, 0.8)
                    interaction_graph[j, k] = val
                    interaction_graph[k, j] = val

    co_assignment_costs = torch.zeros(M, M)
    for j in range(M):
        for k in range(j + 1, M):
            if rng_py.random() < 0.2:
                val = rng_py.uniform(0.1, 0.5)
                co_assignment_costs[j, k] = val
                co_assignment_costs[k, j] = val

    risk_weights = torch.randn(3 * d, 1, generator=rng_torch)

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
    base_problem: ProblemInstance | None = None,
) -> ProblemInstance:
    """
    Stationary Control Condition:
    All exogenous quantities remain 100% bitwise identical across all episodes.
    """
    if base_problem is None:
        base_problem = generate_base_problem(seed=seed, N=N, M=M, d=d, scenario_id="Stationary")
    return clone_problem_instance(base_problem)


def generate_capability_drift_episode(
    episode: int,
    seed: int,
    perturb_episode: int,
    N: int,
    M: int,
    d: int,
    base_problem: ProblemInstance | None = None,
) -> ProblemInstance:
    """
    Capability Drift:
    - Before T_perturb: identical to base problem.
    - At and after T_perturb: exclusively Agent 0 and Agent 1 swap capabilities.
    - Tasks, costs, risk weights, and interaction graph remain strictly unchanged.
    """
    if base_problem is None:
        base_problem = generate_base_problem(seed=seed, N=N, M=M, d=d, scenario_id="Capability Drift")

    inst = clone_problem_instance(base_problem)
    if episode >= perturb_episode and N >= 2:
        s0 = inst.agents[0].capability_embedding.clone()
        s1 = inst.agents[1].capability_embedding.clone()
        inst.agents[0].capability_embedding = s1
        inst.agents[1].capability_embedding = s0

    return inst


def generate_task_shift_episode(
    episode: int,
    seed: int,
    perturb_episode: int,
    N: int,
    M: int,
    d: int,
    base_problem: ProblemInstance | None = None,
) -> ProblemInstance:
    """
    Task Shift:
    - Before T_perturb: identical to base problem.
    - At and after T_perturb: exclusively task embeddings shift by +1.5.
    - Agents, costs, risk weights, and interaction graph remain strictly unchanged.
    """
    if base_problem is None:
        base_problem = generate_base_problem(seed=seed, N=N, M=M, d=d, scenario_id="Task Shift")

    inst = clone_problem_instance(base_problem)
    if episode >= perturb_episode:
        shift = torch.ones(d) * 1.5
        for task in inst.tasks:
            task.embedding = task.embedding + shift

    return inst


def generate_dependency_change_episode(
    episode: int,
    seed: int,
    perturb_episode: int,
    N: int,
    M: int,
    d: int,
    base_problem: ProblemInstance | None = None,
) -> ProblemInstance:
    """
    Dependency Change:
    - Before T_perturb: Pattern 1 (adjacent pair synergies).
    - At and after T_perturb: Pattern 2 (stride-2 shifted pair synergies).
    - Agents, tasks, costs, and risk weights remain strictly unchanged.
    """
    if base_problem is None:
        base_problem = generate_base_problem(seed=seed, N=N, M=M, d=d, scenario_id="Dependency Change")

    inst = clone_problem_instance(base_problem)
    if episode >= perturb_episode:
        new_graph = torch.zeros(M, M)
        for i in range(M):
            j = (i + 2) % M
            new_graph[i, j] = 1.0
            new_graph[j, i] = 1.0
        inst.interaction_graph = new_graph

    return inst


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
    base_problem = generate_base_problem(
        seed=seed,
        N=N,
        M=M,
        d=d,
        scenario_id=scenario_id,
    )

    trajectory = []
    for ep in range(num_episodes):
        inst = generator_fn(
            episode=ep,
            seed=seed,
            perturb_episode=perturb_episode,
            N=N,
            M=M,
            d=d,
            base_problem=base_problem,
        )
        trajectory.append(inst)

    return trajectory

import torch
from energy.registry import EnergyRegistry
from energy.assignment import AssignmentEnergy
from energy.interaction import InteractionEnergy
from energy.cost import CostEnergy
from energy.risk import RiskEnergy, RiskPredictor
from state.orchestration_state import OrchestrationState

def compute_energy(problem, X):
    N = len(problem.agents)
    M = len(problem.tasks)
    d = problem.agents[0].capability_embedding.shape[0]

    state = OrchestrationState(
        X=X,
        s=torch.stack([a.capability_embedding for a in problem.agents]),
        c=torch.stack([t.embedding for t in problem.tasks]),
        kappa=torch.zeros(N, d),
        Theta=problem.interaction_graph,
        C=problem.co_assignment_costs,
        N=N, M=M, d=d
    )

    risk_predictor = RiskPredictor(d, W_risk=problem.risk_weights)

    registry = EnergyRegistry()
    registry.add(AssignmentEnergy(lambda_align=0.5, weight=1.0))
    registry.add(InteractionEnergy(weight=1.0))
    registry.add(CostEnergy(weight=1.0))
    registry.add(RiskEnergy(risk_predictor, weight=1.0))

    total, components = registry.compute(state)
    return total.item(), components

def load_balance(X):
    workload = X.sum(dim=1)
    return torch.std(workload).item()

def coordination_score(problem, X):
    # Number of synergies exploited (Theta > 0 and tasks on same agent)
    co = X.T @ X
    synergies = (problem.interaction_graph > 0) * co
    return synergies.sum().item()

def constraint_violations(problem, X):
    # Number of conflicts (C > 0 and tasks on same agent)
    co = X.T @ X
    conflicts = (problem.co_assignment_costs > 0) * co
    return conflicts.sum().item()

def reconfiguration_cost(X_old, X_new):
    """Measures how many tasks were reassigned."""
    if X_old is None:
        return 0.0
    return (X_old - X_new).abs().sum().item() / 2.0

def brute_force_optimum(problem):
    """Finds global minimum energy by checking all assignments (only for very small M, N)."""
    import itertools
    N = len(problem.agents)
    M = len(problem.tasks)
    if N**M > 100000:
        return None # Too large

    best_E = float('inf')
    for assignment in itertools.product(range(N), repeat=M):
        X = torch.zeros(N, M)
        for t, a in enumerate(assignment):
            X[a, t] = 1.0
        E, _ = compute_energy(problem, X)
        if E < best_E:
            best_E = E
    return best_E

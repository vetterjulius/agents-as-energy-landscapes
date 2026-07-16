import torch
import numpy as np
import scipy.stats as stats
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


# --- New Emergent Behavior Metrics ---

def specialization_degree(problem, X):
    """
    Computes the specialization degree as the average cosine similarity
    between tasks and their assigned agents.
    """
    N = len(problem.agents)
    M = len(problem.tasks)
    if M == 0 or N == 0:
        return 0.0
    s = torch.stack([a.capability_embedding for a in problem.agents])
    c = torch.stack([t.embedding for t in problem.tasks])

    s_norm = s / (s.norm(dim=1, keepdim=True) + 1e-8)
    c_norm = c / (c.norm(dim=1, keepdim=True) + 1e-8)
    cos_sim = s_norm @ c_norm.T  # (N, M)

    assigned_sim = (X * cos_sim).sum().item()
    return assigned_sim / M

def task_clustering(problem, X):
    """
    Measures task clustering: the fraction of total positive synergy links
    that are successfully co-assigned to the same agent.
    """
    co = X.T @ X
    positive_theta = torch.clamp(problem.interaction_graph, min=0.0)
    total_synergy = positive_theta.sum().item()
    if total_synergy < 1e-6:
        return 0.0
    realized_synergy = (positive_theta * co).sum().item()
    return realized_synergy / total_synergy

def communication_cost(problem, X):
    """
    Measures communication overhead: synergy links that are NOT co-assigned
    to the same agent (requiring inter-agent coordination).
    """
    co = X.T @ X
    positive_theta = torch.clamp(problem.interaction_graph, min=0.0)
    comm_cost = (positive_theta * (1.0 - co)).sum().item()
    return comm_cost

def conflict_rate(problem, X):
    """
    Measures the conflict rate (violations) as the sum of co-assignment costs
    for conflicting tasks assigned to the same agent.
    """
    return constraint_violations(problem, X)


# --- Statistical Significance Helper ---

def compute_statistical_tests(ref_energies, baseline_energies):
    """
    Computes Welch's t-test and Mann-Whitney U test between a reference method
    (e.g., Energy Hybrid) and a baseline method across multiple runs/seeds.
    Also returns 95% Confidence Intervals.
    """
    ref = np.array(ref_energies)
    base = np.array(baseline_energies)

    # Welch's t-test (independent samples with unequal variances)
    if len(ref) > 1 and len(base) > 1 and np.var(ref) + np.var(base) > 1e-9:
        t_stat, p_val_t = stats.ttest_ind(ref, base, equal_var=False)
    else:
        t_stat, p_val_t = 0.0, 1.0

    # Mann-Whitney U test (non-parametric rank sum test)
    if len(ref) > 1 and len(base) > 1:
        try:
            u_stat, p_val_u = stats.mannwhitneyu(ref, base, alternative='two-sided')
        except Exception:
            u_stat, p_val_u = 0.0, 1.0
    else:
        u_stat, p_val_u = 0.0, 1.0

    # 95% Confidence Interval function
    def get_ci(data):
        if len(data) < 2:
            return (float(np.mean(data)), float(np.mean(data))) if len(data) == 1 else (0.0, 0.0)
        mean = np.mean(data)
        sem = stats.sem(data)
        ci_h = sem * stats.t.ppf((1 + 0.95) / 2., len(data) - 1)
        return float(mean - ci_h), float(mean + ci_h)

    ci_ref = get_ci(ref)
    ci_base = get_ci(base)

    return {
        "welch_t_stat": float(t_stat),
        "welch_p_val": float(p_val_t),
        "mann_whitney_u_stat": float(u_stat),
        "mann_whitney_p_val": float(p_val_u),
        "ci_ref": ci_ref,
        "ci_base": ci_base
    }

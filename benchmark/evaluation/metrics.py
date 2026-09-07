import torch
import numpy as np
import scipy.stats as stats
from energy.registry import EnergyRegistry
from energy.assignment import AssignmentEnergy
from energy.interaction import InteractionEnergy
from energy.cost import CostEnergy
from energy.risk import RiskEnergy, RiskPredictor
from state.orchestration_state import OrchestrationState

def compute_energy(
    problem,
    X,
    energy_cfg=None,
    kappa=None,
    theta=None,
    enabled_terms=None,
):
    energy_cfg = energy_cfg or {}

    active_terms = (
        {"assignment", "interaction", "cost", "risk"}
        if enabled_terms is None
        else set(enabled_terms)
    )

    interaction_weight = energy_cfg.get("interaction_weight", 1.0)
    lambda_align = energy_cfg.get("lambda_align", 0.5)
    lambda_memory = energy_cfg.get("lambda_memory", lambda_align)
    cost_weight = energy_cfg.get("cost_weight", 1.0)
    risk_weight = energy_cfg.get("risk_weight", 1.0)

    N = len(problem.agents)
    M = len(problem.tasks)
    d = problem.agents[0].capability_embedding.shape[0]

    state = OrchestrationState(
        X=X,
        s=torch.stack([a.capability_embedding for a in problem.agents]),
        c=torch.stack([t.embedding for t in problem.tasks]),
        kappa=torch.zeros(N, d) if kappa is None else kappa,
        Theta=problem.interaction_graph if theta is None else theta,
        C=problem.co_assignment_costs,
        N=N,
        M=M,
        d=d,
    )

    risk_predictor = RiskPredictor(
        d,
        W_risk=problem.risk_weights,
    )

    registry = EnergyRegistry()

    if "assignment" in active_terms:
        registry.add(
            AssignmentEnergy(
                lambda_align=lambda_align,
                lambda_memory=lambda_memory,
                weight=1.0,
            )
        )

    if "interaction" in active_terms:
        registry.add(
            InteractionEnergy(
                weight=interaction_weight,
            )
        )

    if "cost" in active_terms:
        registry.add(
            CostEnergy(
                weight=cost_weight,
            )
        )

    if "risk" in active_terms:
        registry.add(
            RiskEnergy(
                risk_predictor,
                weight=risk_weight,
            )
        )

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


# --- Real-World LLM Cost & Token Interpretation (GPT-4o Calibration) ---

# Pricing: GPT-4o ($2.50 / 1M Input Tokens, $10.00 / 1M Output Tokens)
# Assuming typical prompt/response ratio of 3:1 input:output -> Weighted avg: $4.375 / 1M tokens ($0.000004375 / token)
GPT4O_COST_PER_TOKEN = 0.000004375

def estimate_llm_cost_and_tokens(
    problem,
    X,
    avg_task_tokens: int = 3000,
    avg_msg_tokens: int = 800,
    retry_factor: float = 1.5
):
    """
    Translates an orchestration state X and problem instance into realistic estimated
    GPT-4o LLM token usage and USD costs.
    
    - Base execution: M tasks * avg_task_tokens
    - Inter-agent comms: communication_cost * avg_msg_tokens (for unaligned synergy links)
    - Risk retry overhead: risk_energy * retry_factor * avg_task_tokens
    """
    M = len(problem.tasks)
    
    # 1. Base prompt & context tokens for task execution
    base_tokens = M * avg_task_tokens
    
    # 2. Inter-agent communication tokens (synergy links assigned to different agents)
    comm_links = communication_cost(problem, X)
    comm_tokens = comm_links * avg_msg_tokens
    
    # 3. Risk-induced retry overhead tokens
    # Compute normalized risk energy for state X
    r_pred = RiskPredictor(problem.agents[0].capability_embedding.shape[0], W_risk=problem.risk_weights)
    s = torch.stack([a.capability_embedding for a in problem.agents])
    c = torch.stack([t.embedding for t in problem.tasks])
    k = torch.zeros_like(s)
    N = len(problem.agents)
    state_temp = OrchestrationState(X=X, s=s, c=c, kappa=k, Theta=problem.interaction_graph, C=problem.co_assignment_costs, N=N, M=M, d=s.shape[1])
    r_energy = RiskEnergy(r_pred).compute(state_temp).item()
    
    # Clamp risk energy to positive bounds for token estimation
    risk_tokens = max(0.0, r_energy) * M * avg_task_tokens * retry_factor
    
    total_tokens = base_tokens + comm_tokens + risk_tokens
    estimated_usd = total_tokens * GPT4O_COST_PER_TOKEN
    
    return {
        "total_tokens": int(total_tokens),
        "base_tokens": int(base_tokens),
        "comm_tokens": int(comm_tokens),
        "risk_tokens": int(risk_tokens),
        "estimated_usd": round(estimated_usd, 4)
    }



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

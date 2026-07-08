import torch
import random
import copy
from .baselines.energy_based import EnergyBasedOrchestrator
from energy.registry import EnergyRegistry
from energy.assignment import AssignmentEnergy
from energy.interaction import InteractionEnergy
from energy.cost import CostEnergy
from energy.risk import RiskEnergy, RiskPredictor
from state.orchestration_state import OrchestrationState
from model.orchestrator import Orchestrator as SystemOrchestrator

class AblationOrchestrator(EnergyBasedOrchestrator):
    def __init__(self, cfg, enabled_terms=None):
        super().__init__(cfg)
        self.enabled_terms = enabled_terms or ["assignment", "interaction", "cost", "risk"]

    def solve(self, problem) -> torch.Tensor:
        N = len(problem.agents)
        M = len(problem.tasks)
        d = problem.agents[0].capability_embedding.shape[0]

        s = torch.stack([a.capability_embedding for a in problem.agents])
        c = torch.stack([t.embedding for t in problem.tasks])
        kappa = torch.zeros(N, d)

        X_init = torch.zeros(N, M)
        for t in range(M):
            X_init[t % N, t] = 1.0

        state = OrchestrationState(
            X=X_init, s=s, c=c, kappa=kappa,
            Theta=problem.interaction_graph,
            C=problem.co_assignment_costs,
            N=N, M=M, d=d
        )

        model_cfg = {
            "model": {
                "num_agents": N, "num_tasks": M, "dim": d,
                "lambda_align": 0.5, "eta_theta": 0.1, "eta_memory": 0.05,
                "risk_weight": 1.0 if "risk" in self.enabled_terms else 0.0,
                "interaction_weight": 1.0 if "interaction" in self.enabled_terms else 0.0,
                "cost_weight": 1.0 if "cost" in self.enabled_terms else 0.0,
                "temperature_init": self.cfg["solver"]["temperature_init"],
                "min_temperature": self.cfg["solver"]["min_temperature"],
                "max_temperature": self.cfg["solver"]["max_temperature"],
                "target_accept_rate": self.cfg["solver"]["target_accept_rate"],
                "proposal_candidates": 12, "proposal_task_sample": 8, "agent_sample_size": 6, "block_move_size": 4,
                "warm_start_steps": 0, "warm_start_type": "greedy", "hybrid_cleanup_prob": 0.0, "local_refine_steps": 0
            }
        }

        orchestrator = SystemOrchestrator(model_cfg, initial_state=state, W_risk=problem.risk_weights)

        # Override registry to only include desired terms
        registry = EnergyRegistry()
        if "assignment" in self.enabled_terms:
            registry.add(AssignmentEnergy(lambda_align=0.5, weight=1.0))
        if "interaction" in self.enabled_terms:
            registry.add(InteractionEnergy(weight=model_cfg["model"]["interaction_weight"]))
        if "cost" in self.enabled_terms:
            registry.add(CostEnergy(weight=model_cfg["model"]["cost_weight"]))
        if "risk" in self.enabled_terms:
            registry.add(RiskEnergy(orchestrator.risk_predictor, weight=model_cfg["model"]["risk_weight"]))

        orchestrator.energy_registry = registry
        orchestrator.sampler.energy_registry = registry

        for _ in range(self.cfg["solver"]["iterations"]):
            orchestrator.step()

        return orchestrator.state.X

def run_representation_ablations(problem, cfg):
    experiments = {
        "Full Model": ["assignment", "interaction", "cost", "risk"],
        "Assignment Only": ["assignment"],
        "Assignment + Interaction": ["assignment", "interaction"],
        "Assignment + Cost": ["assignment", "cost"],
        "Assignment + Risk": ["assignment", "risk"]
    }

    results = {}
    for name, terms in experiments.items():
        orch = AblationOrchestrator(cfg, enabled_terms=terms)
        X = orch.solve(problem)
        from .evaluation.metrics import compute_energy
        energy, _ = compute_energy(problem, X)
        results[name] = energy
    return results

def run_solver_ablations(problem, cfg):
    """
    Step 6 - Solver Ablation
    Compares different optimization strategies on the same energy model.
    """
    N = len(problem.agents)
    M = len(problem.tasks)
    d = problem.agents[0].capability_embedding.shape[0]

    s = torch.stack([a.capability_embedding for a in problem.agents])
    c = torch.stack([t.embedding for t in problem.tasks])
    kappa = torch.zeros(N, d)

    X_init = torch.zeros(N, M)
    for t in range(M):
        X_init[random.randint(0, N-1), t] = 1.0

    state_template = OrchestrationState(
        X=X_init, s=s, c=c, kappa=kappa,
        Theta=problem.interaction_graph,
        C=problem.co_assignment_costs,
        N=N, M=M, d=d
    )

    # 1. Random Search
    best_X_random = X_init.clone()
    best_E_random = float('inf')
    for _ in range(cfg["solver"]["iterations"] * 10):
        X_prop = torch.zeros(N, M)
        for t in range(M):
            X_prop[random.randint(0, N-1), t] = 1.0
        temp_state = state_template.clone()
        temp_state.X = X_prop
        from .evaluation.metrics import compute_energy
        E, _ = compute_energy(problem, X_prop)
        if E < best_E_random:
            best_E_random = E
            best_X_random = X_prop

    # 2. Hill Climbing (Local Improvement)
    curr_X_hc = X_init.clone()
    from .baselines.greedy import GreedyOrchestrator
    # We can use a trick here: GreedyOrchestrator with local_improvement mode
    # But it needs to be initialized with our state.
    # Let's just implement a simple loop.
    best_E_hc, _ = compute_energy(problem, curr_X_hc)
    for _ in range(cfg["solver"]["iterations"]):
        improved = False
        for t in range(M):
            a_curr = torch.argmax(curr_X_hc[:, t]).item()
            for a in range(N):
                if a == a_curr: continue
                X_prop = curr_X_hc.clone()
                X_prop[a_curr, t] = 0.0
                X_prop[a, t] = 1.0
                E, _ = compute_energy(problem, X_prop)
                if E < best_E_hc - 1e-6:
                    best_E_hc = E
                    curr_X_hc = X_prop
                    improved = True
                    break
            if improved: break
        if not improved: break

    # 3. Simulated Annealing (Already our EnergyBasedOrchestrator)
    orch_sa = EnergyBasedOrchestrator(cfg)
    X_sa = orch_sa.solve(problem)
    E_sa, _ = compute_energy(problem, X_sa)

    return {
        "Random Search": best_E_random,
        "Hill Climbing": best_E_hc,
        "Simulated Annealing": E_sa
    }

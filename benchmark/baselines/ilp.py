import torch
import numpy as np
from scipy.optimize import milp, LinearConstraint, Bounds
from .base import Orchestrator
from ..scenarios.base import ProblemInstance
from ..evaluation.metrics import compute_energy
from energy.risk import RiskPredictor

class ILPOrchestrator(Orchestrator):
    """
    Exact Integer Linear Programming (ILP) / Mixed-Integer Linear Programming (MILP) Solver
    for EBMAO Task Allocation.
    
    Formulates the exact quadratic binary assignment problem as an MILP using standard
    Fortet/McCormick linearization:
      - Binary decision variables X[i, t] in {0, 1} for agent i, task t
      - Auxiliary binary variables Y[i, t, t'] in {0, 1} for co-assignment X[i, t] * X[i, t']
    
    Provides an exact MILP formulation when the solver reaches an optimal status;
    bounded runs may instead return a feasible incumbent or a solver limit result.
    """
    def __init__(self, energy_cfg=None, time_limit_sec=30.0):
        self.energy_cfg = energy_cfg or {}
        self.time_limit_sec = time_limit_sec

    def solve(self, problem: ProblemInstance) -> torch.Tensor:
        N = len(problem.agents)
        M = len(problem.tasks)
        d = problem.agents[0].capability_embedding.shape[0]

        # Weights
        lambda_align = self.energy_cfg.get("lambda_align", 0.5)
        lambda_memory = self.energy_cfg.get("lambda_memory", lambda_align)
        interaction_weight = self.energy_cfg.get("interaction_weight", 1.0)
        cost_weight = self.energy_cfg.get("cost_weight", 1.0)
        risk_weight = self.energy_cfg.get("risk_weight", 1.0)

        # 1. Compute Linear Cost Matrix L (N, M)
        s = torch.stack([a.capability_embedding for a in problem.agents])
        c = torch.stack([t.embedding for t in problem.tasks])
        kappa = torch.zeros(N, d)

        # Distance cost
        dist = torch.cdist(s, c) ** 2
        # Alignment score
        align_sc = s @ c.T
        align_mem = (s * kappa).sum(dim=1, keepdim=True)
        assign_score = dist - lambda_align * align_sc - lambda_memory * align_mem

        # Risk cost
        r_pred = RiskPredictor(d, W_risk=problem.risk_weights)
        s_exp = s.unsqueeze(1).expand(-1, M, -1)
        c_exp = c.unsqueeze(0).expand(N, -1, -1)
        k_exp = kappa.unsqueeze(1).expand(-1, M, -1)
        x_feat = torch.cat([s_exp, c_exp, k_exp], dim=-1)
        h = torch.matmul(x_feat, problem.risk_weights).squeeze(-1) / np.sqrt(max(d, 1e-8))
        risk_p = torch.sigmoid(h)
        risk_score = -risk_weight * torch.log(risk_p + 1e-8)

        L = (assign_score + risk_score).detach().cpu().numpy() / (N * M)

        # 2. Compute Quadratic Interaction Matrix Q (M, M)
        Theta = problem.interaction_graph.detach().cpu().numpy()
        C_cost = problem.co_assignment_costs.detach().cpu().numpy()

        # Upper triangular interaction matrix
        Theta_upper = np.triu(Theta, k=1)
        Q = (-interaction_weight * Theta_upper + cost_weight * C_cost) / (N * M)

        # 3. Formulate MILP Problem Variables
        # X: N * M variables (indexed as i * M + t)
        # Y: N * M * (M - 1) / 2 variables (indexed for pairs t < t')
        pair_indices = [(t1, t2) for t1 in range(M) for t2 in range(t1 + 1, M)]
        num_pairs = len(pair_indices)
        num_x = N * M
        num_y = N * num_pairs
        total_vars = num_x + num_y

        # Objective vector c_obj
        c_obj = np.zeros(total_vars)
        
        # Linear part
        for i in range(N):
            for t in range(M):
                c_obj[i * M + t] = L[i, t]

        # Quadratic part
        for i in range(N):
            for idx, (t1, t2) in enumerate(pair_indices):
                y_var_idx = num_x + i * num_pairs + idx
                c_obj[y_var_idx] = Q[t1, t2] + Q[t2, t1]

        # 4. Formulate Linear Constraints
        # Constraint set 1: Sum_i X[i, t] = 1 for all t (Assignment Constraint)
        # Constraint set 2: Linearization for Y[i, t1, t2] = X[i, t1] * X[i, t2]
        #   - Y <= X[i, t1]  => Y - X[i, t1] <= 0
        #   - Y <= X[i, t2]  => Y - X[i, t2] <= 0
        #   - Y >= X[i, t1] + X[i, t2] - 1 => X[i, t1] + X[i, t2] - Y <= 1

        row_list = []
        b_l_list = []
        b_u_list = []

        # Constraint set 1: Exactly 1 agent per task
        for t in range(M):
            row = np.zeros(total_vars)
            for i in range(N):
                row[i * M + t] = 1.0
            row_list.append(row)
            b_l_list.append(1.0)
            b_u_list.append(1.0)

        # Constraint set 2: Linearization constraints
        for i in range(N):
            for idx, (t1, t2) in enumerate(pair_indices):
                x1_idx = i * M + t1
                x2_idx = i * M + t2
                y_idx = num_x + i * num_pairs + idx

                # Y - X1 <= 0
                r1 = np.zeros(total_vars)
                r1[y_idx] = 1.0
                r1[x1_idx] = -1.0
                row_list.append(r1)
                b_l_list.append(-np.inf)
                b_u_list.append(0.0)

                # Y - X2 <= 0
                r2 = np.zeros(total_vars)
                r2[y_idx] = 1.0
                r2[x2_idx] = -1.0
                row_list.append(r2)
                b_l_list.append(-np.inf)
                b_u_list.append(0.0)

                # X1 + X2 - Y <= 1
                r3 = np.zeros(total_vars)
                r3[x1_idx] = 1.0
                r3[x2_idx] = 1.0
                r3[y_idx] = -1.0
                row_list.append(r3)
                b_l_list.append(-np.inf)
                b_u_list.append(1.0)

        A = np.vstack(row_list)
        constraints = LinearConstraint(A, b_l_list, b_u_list)
        integrality = np.ones(total_vars)  # All variables are binary integer (1)
        bounds = Bounds(0.0, 1.0)

        # Solve MILP with time limit
        options = {"time_limit": self.time_limit_sec}
        res = milp(c=c_obj, integrality=integrality, bounds=bounds, constraints=constraints, options=options)

        # Reconstruct X
        X_opt = torch.zeros(N, M)
        if res.success:
            x_sol = res.x[:num_x].reshape(N, M)
            for t in range(M):
                best_agent = np.argmax(x_sol[:, t])
                X_opt[best_agent, t] = 1.0
        else:
            # Fallback to greedy if solver fails
            for t in range(M):
                best_agent = np.argmin(L[:, t])
                X_opt[best_agent, t] = 1.0

        return X_opt

from __future__ import annotations

import math
import time
from dataclasses import dataclass
from typing import Any, List, Optional, Tuple
import numpy as np
import torch
from scipy.optimize import Bounds, LinearConstraint, milp

from ilp import enumerate_valid_assignments
from landscape import Landscape, LandscapeState, ProblemContext


class BudgetExceededError(RuntimeError):
    """Raised when an objective evaluation is attempted past max_energy_evaluations."""
    pass


class BudgetedLandscape:
    """
    Wraps a Landscape to strictly enforce and track the common evaluation budget.

    Fairness guarantees:
    - Every call to evaluate(X) counts as exactly 1 evaluation.
    - Raises BudgetExceededError if solver attempts to exceed the budget.
    - LandscapeState cannot be modified by the solver (returns isolated clones).
    """

    def __init__(self, landscape: Landscape, max_evaluations: int):
        self._landscape = landscape
        self.max_evaluations = max(1, int(max_evaluations))
        self.evaluations_used = 0

    @property
    def problem(self) -> ProblemContext:
        return self._landscape.problem

    @property
    def state(self) -> LandscapeState:
        # Isolated clone: solver can never mutate the actual landscape state
        return self._landscape.state.clone()

    def remaining_budget(self) -> int:
        return max(0, self.max_evaluations - self.evaluations_used)

    def evaluate(self, X: torch.Tensor) -> float:
        if self.evaluations_used >= self.max_evaluations:
            raise BudgetExceededError(
                f"Evaluation budget exceeded: {self.evaluations_used}/{self.max_evaluations} calls used."
            )
        self.evaluations_used += 1
        return self._landscape.evaluate(X)


@dataclass
class SolverResult:
    """Standardized result record returned by all solvers."""

    X: torch.Tensor
    energy: float
    energy_evaluations: int
    accepted_moves: int
    iterations: int
    runtime_sec: float
    termination_reason: str
    status: str
    is_optimal: Optional[bool] = None
    mip_gap: Optional[float] = None
    timeout: bool = False
    fallback_used: bool = False


class EnergyAwareSimulatedAnnealingSolver:
    """
    Fair Energy-aware Simulated Annealing solver operating under a strict evaluation budget.

    - Starts with evaluating initial assignment (1 eval).
    - Proposes single-task reassignment moves uniformly at random.
    - Evaluates each candidate move on BudgetedLandscape (1 eval).
    - Strictly stops when evaluation budget is exhausted.
    - Zero hidden warm starts, zero uncounted local refinements.
    """

    def __init__(
        self,
        temperature_init: float = 1.0,
        min_temperature: float = 0.01,
        cooling_rate: float = 0.98,
    ):
        self.temperature_init = temperature_init
        self.min_temperature = min_temperature
        self.cooling_rate = cooling_rate

    def solve(
        self,
        budgeted_landscape: BudgetedLandscape,
        initial_X: torch.Tensor,
        seed: int,
    ) -> SolverResult:
        start_time = time.perf_counter()
        rng = np.random.default_rng(seed)

        N = budgeted_landscape.problem.N
        M = budgeted_landscape.problem.M

        # Initial evaluation (counts as 1 evaluation against budget)
        X_curr = initial_X.clone().float()
        try:
            E_curr = budgeted_landscape.evaluate(X_curr)
        except BudgetExceededError:
            runtime = time.perf_counter() - start_time
            return SolverResult(
                X=X_curr,
                energy=float("inf"),
                energy_evaluations=budgeted_landscape.evaluations_used,
                accepted_moves=0,
                iterations=0,
                runtime_sec=runtime,
                termination_reason="budget_exhausted",
                status="budget_exhausted",
            )

        X_best = X_curr.clone()
        E_best = E_curr

        accepted_moves = 0
        iterations = 0
        T = float(self.temperature_init)
        termination_reason = "max_iterations"

        while budgeted_landscape.remaining_budget() > 0:
            iterations += 1

            # Propose moving one random task to a different agent
            task_idx = int(rng.integers(0, M))
            current_agent = int(torch.argmax(X_curr[:, task_idx]).item())
            candidates = [a for a in range(N) if a != current_agent]
            if not candidates:
                # N=1 case: no alternative agent exists
                termination_reason = "single_agent_no_moves"
                break

            new_agent = int(rng.choice(candidates))

            X_cand = X_curr.clone()
            X_cand[current_agent, task_idx] = 0.0
            X_cand[new_agent, task_idx] = 1.0

            try:
                E_cand = budgeted_landscape.evaluate(X_cand)
            except BudgetExceededError:
                termination_reason = "budget_exhausted"
                break

            dE = E_cand - E_curr
            if dE <= 0.0:
                accept = True
            else:
                prob = math.exp(-dE / max(T, 1e-8))
                accept = bool(rng.random() < prob)

            if accept:
                X_curr = X_cand
                E_curr = E_cand
                accepted_moves += 1

                if E_curr < E_best:
                    E_best = E_curr
                    X_best = X_curr.clone()

            # Cooling
            T = max(self.min_temperature, T * self.cooling_rate)

        if budgeted_landscape.remaining_budget() == 0 and termination_reason != "single_agent_no_moves":
            termination_reason = "budget_exhausted"

        runtime = time.perf_counter() - start_time
        return SolverResult(
            X=X_best,
            energy=E_best,
            energy_evaluations=budgeted_landscape.evaluations_used,
            accepted_moves=accepted_moves,
            iterations=iterations,
            runtime_sec=runtime,
            termination_reason=termination_reason,
            status="completed",
        )


class EnergyAwareGreedySolver:
    """
    Fair Energy-aware Greedy (steepest descent / best-improvement local search) solver.

    - Starts with evaluating initial assignment (1 eval).
    - Examines 1-move neighborhood (M * (N - 1) candidates).
    - Every candidate evaluation counts against the budget.
    - If no neighbor improves, terminates at local optimum.
    - If budget exhausts during neighborhood scan, terminates immediately.
    """

    def solve(
        self,
        budgeted_landscape: BudgetedLandscape,
        initial_X: torch.Tensor,
    ) -> SolverResult:
        start_time = time.perf_counter()

        N = budgeted_landscape.problem.N
        M = budgeted_landscape.problem.M

        # Initial evaluation (counts as 1 evaluation against budget)
        X_curr = initial_X.clone().float()
        try:
            E_curr = budgeted_landscape.evaluate(X_curr)
        except BudgetExceededError:
            runtime = time.perf_counter() - start_time
            return SolverResult(
                X=X_curr,
                energy=float("inf"),
                energy_evaluations=budgeted_landscape.evaluations_used,
                accepted_moves=0,
                iterations=0,
                runtime_sec=runtime,
                termination_reason="budget_exhausted",
                status="budget_exhausted",
            )

        X_best = X_curr.clone()
        E_best = E_curr

        accepted_moves = 0
        iterations = 0
        termination_reason = "converged_local_optimum"

        while budgeted_landscape.remaining_budget() > 0:
            iterations += 1
            improved = False
            best_neighbor_X = None
            best_neighbor_E = E_curr

            budget_exhausted_during_scan = False

            # Scan 1-move neighborhood
            for t in range(M):
                curr_agent = int(torch.argmax(X_curr[:, t]).item())
                for a in range(N):
                    if a == curr_agent:
                        continue

                    if budgeted_landscape.remaining_budget() <= 0:
                        budget_exhausted_during_scan = True
                        break

                    X_prop = X_curr.clone()
                    X_prop[curr_agent, t] = 0.0
                    X_prop[a, t] = 1.0

                    try:
                        E_prop = budgeted_landscape.evaluate(X_prop)
                    except BudgetExceededError:
                        budget_exhausted_during_scan = True
                        break

                    if E_prop < best_neighbor_E - 1e-6:
                        best_neighbor_E = E_prop
                        best_neighbor_X = X_prop
                        improved = True

                if budget_exhausted_during_scan:
                    break

            if improved and best_neighbor_X is not None:
                X_curr = best_neighbor_X
                E_curr = best_neighbor_E
                X_best = X_curr.clone()
                E_best = E_curr
                accepted_moves += 1
            else:
                if budget_exhausted_during_scan:
                    termination_reason = "budget_exhausted"
                else:
                    termination_reason = "converged_local_optimum"
                break

            if budgeted_landscape.remaining_budget() <= 0:
                termination_reason = "budget_exhausted"
                break

        runtime = time.perf_counter() - start_time
        return SolverResult(
            X=X_best,
            energy=E_best,
            energy_evaluations=budgeted_landscape.evaluations_used,
            accepted_moves=accepted_moves,
            iterations=iterations,
            runtime_sec=runtime,
            termination_reason=termination_reason,
            status="completed",
        )


class FixedLandscapeILPSolver:
    """
    Exact Reference Solver for a fixed Landscape.

    Formulates the exact quadratic binary assignment problem as an MILP via
    standard McCormick/Fortet linearization:
      min  sum_{i,t} L[i,t] X[i,t] + sum_i sum_{t1 < t2} (Q[t1,t2] + Q[t2,t1]) Y[i,t1,t2]
      s.t. sum_i X[i,t] = 1
           Y[i,t1,t2] <= X[i,t1]
           Y[i,t1,t2] <= X[i,t2]
           Y[i,t1,t2] >= X[i,t1] + X[i,t2] - 1
           X, Y binary.

    Operates under a wall-clock time limit.
    Never silently labels a fallback solution as ILP.
    """

    def __init__(self, time_limit_sec: float = 10.0):
        self.time_limit_sec = float(time_limit_sec)

    def solve(self, landscape: Landscape) -> SolverResult:
        start_time = time.perf_counter()
        problem = landscape.problem
        state = landscape.state

        N = problem.N
        M = problem.M
        d = problem.d

        # For very small search spaces (e.g. N^M <= 256), brute-force enumeration
        # is mathematically exact and instantaneous
        if N ** M <= 256:
            best_X = None
            best_E = float("inf")
            for X_cand in enumerate_valid_assignments(problem):
                E = landscape.evaluate(X_cand)
                if E < best_E:
                    best_E = E
                    best_X = X_cand.clone()

            runtime = time.perf_counter() - start_time
            return SolverResult(
                X=best_X,
                energy=best_E,
                energy_evaluations=N ** M,
                accepted_moves=0,
                iterations=N ** M,
                runtime_sec=runtime,
                termination_reason="exact_brute_force_optimum",
                status="optimal",
                is_optimal=True,
                mip_gap=0.0,
                timeout=False,
                fallback_used=False,
            )

        # 1. Compute Linear Cost Matrix L (N, M)
        dist = torch.cdist(problem.s, problem.c) ** 2
        align_sc = problem.s @ problem.c.T
        align_mem = (problem.s * state.kappa).sum(dim=1, keepdim=True)
        assign_score = dist - problem.lambda_align * align_sc - problem.lambda_memory * align_mem

        # Risk score
        s_exp = problem.s.unsqueeze(1).expand(-1, M, -1)
        c_exp = problem.c.unsqueeze(0).expand(N, -1, -1)
        k_exp = state.kappa.unsqueeze(1).expand(-1, M, -1)
        x_feat = torch.cat([s_exp, c_exp, k_exp], dim=-1)
        logits = torch.matmul(x_feat, problem.W_risk).squeeze(-1) / math.sqrt(max(d, 1))
        risk_p = torch.sigmoid(logits)
        risk_score = -problem.risk_weight * torch.log(risk_p + 1e-8)

        L = (assign_score + risk_score).detach().cpu().numpy() / (N * M)

        # 2. Quadratic Interaction Matrix Q (M, M)
        Theta = state.Theta.detach().cpu().numpy()
        C_cost = problem.C.detach().cpu().numpy()
        Theta_upper = np.triu(Theta, k=1)
        Q = (-problem.interaction_weight * Theta_upper + problem.cost_weight * C_cost) / (N * M)

        # 3. Formulate MILP variables
        pair_indices = [(t1, t2) for t1 in range(M) for t2 in range(t1 + 1, M)]
        num_pairs = len(pair_indices)
        num_x = N * M
        num_y = N * num_pairs
        total_vars = num_x + num_y

        c_obj = np.zeros(total_vars)
        for i in range(N):
            for t in range(M):
                c_obj[i * M + t] = L[i, t]

        for i in range(N):
            for idx, (t1, t2) in enumerate(pair_indices):
                y_var_idx = num_x + i * num_pairs + idx
                c_obj[y_var_idx] = Q[t1, t2] + Q[t2, t1]

        # 4. Constraints
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

        # Constraint set 2: McCormick linearization for Y[i, t1, t2] = X[i, t1] * X[i, t2]
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
        integrality = np.ones(total_vars)
        bounds = Bounds(0.0, 1.0)

        options = {"time_limit": self.time_limit_sec}
        res = milp(
            c=c_obj,
            integrality=integrality,
            bounds=bounds,
            constraints=constraints,
            options=options,
        )

        runtime = time.perf_counter() - start_time

        if res.success and res.x is not None:
            x_sol = res.x[:num_x].reshape(N, M)
            X_opt = torch.zeros(N, M)
            for t in range(M):
                best_agent = int(np.argmax(x_sol[:, t]))
                X_opt[best_agent, t] = 1.0

            energy_val = landscape.evaluate(X_opt)
            mip_gap = getattr(res, "mip_gap", 0.0)
            return SolverResult(
                X=X_opt,
                energy=energy_val,
                energy_evaluations=1,
                accepted_moves=0,
                iterations=getattr(res, "iteration", 0) or 1,
                runtime_sec=runtime,
                termination_reason="optimal",
                status="optimal",
                is_optimal=True,
                mip_gap=float(mip_gap) if mip_gap is not None else 0.0,
                timeout=False,
                fallback_used=False,
            )
        else:
            # Solver timed out or failed to find an optimal solution within limit.
            # Do NOT silently label a fallback as ILP!
            status_str = "timeout" if runtime >= self.time_limit_sec * 0.95 else "failed"
            # Return uniform round-robin fallback purely for data structure validity,
            # but EXPLICITLY flag fallback_used=True and is_optimal=False!
            X_fallback = torch.zeros(N, M)
            for t in range(M):
                X_fallback[t % N, t] = 1.0
            energy_val = landscape.evaluate(X_fallback)

            return SolverResult(
                X=X_fallback,
                energy=energy_val,
                energy_evaluations=1,
                accepted_moves=0,
                iterations=0,
                runtime_sec=runtime,
                termination_reason=status_str,
                status=status_str,
                is_optimal=False,
                mip_gap=None,
                timeout=(status_str == "timeout"),
                fallback_used=True,
            )

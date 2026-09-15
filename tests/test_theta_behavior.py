"""
test_theta_behavior.py
======================
Focused behavioral tests for the Theta (Θ) adaptation semantics.

These tests verify *behavior*, not just arithmetic correctness:
  - Θ converges to the observed co-assignment structure C under constant input.
  - Θ tracks a *changed* observed structure, not the old one.
  - Θ is blind to context changes that leave the observed co-assignment
    structure (C_t) unchanged.
  - Ground-truth matrices never enter the Θ update path.

No benchmark runs, no seed/scenario/results changes.
Existing tests are not modified.
"""
from __future__ import annotations

import torch
import pytest

from controlled_benchmark.adaptation import EpisodeAdaptationManager
from landscape import LandscapeState, ProblemContext


# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------

def _make_problem(N: int, M: int, d: int = 2) -> ProblemContext:
    """Minimal, neutral ProblemContext – all problem tensors are zeros."""
    return ProblemContext(
        s=torch.zeros(N, d),
        c=torch.zeros(M, d),
        C=torch.zeros(M, M),
        W_risk=torch.zeros(3 * d, 1),
        N=N,
        M=M,
        d=d,
        interaction_weight=1.0,
        cost_weight=1.0,
    )


def _make_state(M: int, N: int = 2, d: int = 2,
                theta_init: float = 0.0) -> LandscapeState:
    """LandscapeState with uniform Theta initialisation."""
    Theta = torch.full((M, M), theta_init)
    Theta.fill_diagonal_(0.0)
    return LandscapeState(
        kappa=torch.zeros(N, d),
        Theta=Theta,
    )


def _compute_C(X: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    """Reference implementation of C_t from X_t (mirrors adaptation.py exactly)."""
    co = X.T @ X
    co_sum = co.sum().item()
    if co_sum < eps:
        return torch.zeros_like(co)
    co_norm = co / (co_sum + eps)
    C = (co_norm + co_norm.T) / 2.0
    C.fill_diagonal_(0.0)
    return C


def _run_ema(
    manager: EpisodeAdaptationManager,
    state: LandscapeState,
    problem: ProblemContext,
    X: torch.Tensor,
    steps: int,
) -> LandscapeState:
    """Apply `steps` EMA updates using the same X each time."""
    for _ in range(steps):
        state = manager.step(state, X, problem)
    return state


# ---------------------------------------------------------------------------
# Test group 1 – Stationary: convergence toward C, no unbounded drift
# ---------------------------------------------------------------------------

class TestStationaryConvergence:
    """Θ converges to C under a constant assignment sequence."""

    # Fixed assignment: agent 0 holds tasks 0+1+2; agent 1 holds tasks 3+4.
    X_CONST = torch.tensor([
        [1., 1., 1., 0., 0.],
        [0., 0., 0., 1., 1.],
    ])
    M = 5
    N = 2

    def test_frobenius_distance_decreases_monotonically(self):
        """
        Semantic: each EMA step must strictly reduce ‖Θ − C‖_F when Θ ≠ C.
        Rationale: the EMA is a convex combination, so the distance to the
        fixed point cannot increase.
        """
        problem = _make_problem(self.N, self.M)
        state = _make_state(self.M, self.N, theta_init=0.5)  # starts away from C
        manager = EpisodeAdaptationManager("theta-only", eta_theta=0.2)
        C = _compute_C(self.X_CONST)

        prev_dist = torch.dist(state.Theta, C).item()
        for _ in range(30):
            state = manager.step(state, self.X_CONST, problem)
            dist = torch.dist(state.Theta, C).item()
            assert dist <= prev_dist + 1e-7, (
                f"Distance to C increased: {prev_dist:.6f} → {dist:.6f}"
            )
            prev_dist = dist

    def test_convergence_to_C_after_many_steps(self):
        """
        Semantic: after sufficiently many steps with constant X, Θ ≈ C.
        The EMA fixed-point under a constant target is exactly C.
        """
        problem = _make_problem(self.N, self.M)
        state = _make_state(self.M, self.N, theta_init=0.8)
        manager = EpisodeAdaptationManager("theta-only", eta_theta=0.3)
        C = _compute_C(self.X_CONST)

        state = _run_ema(manager, state, problem, self.X_CONST, steps=100)

        assert torch.allclose(state.Theta, C, atol=1e-3), (
            f"Θ did not converge to C after 100 steps.\n"
            f"Max deviation: {(state.Theta - C).abs().max().item():.6f}"
        )

    def test_no_unbounded_drift_under_constant_input(self):
        """
        Semantic: Θ must stay bounded under constant repeated input.
        Concretely, ‖Θ‖_F after many steps must be ≤ ‖C‖_F + tolerance.
        """
        problem = _make_problem(self.N, self.M)
        state = _make_state(self.M, self.N, theta_init=0.0)
        manager = EpisodeAdaptationManager("theta-only", eta_theta=0.1)
        C = _compute_C(self.X_CONST)
        C_norm = torch.norm(C, p="fro").item()

        state = _run_ema(manager, state, problem, self.X_CONST, steps=200)

        theta_norm = torch.norm(state.Theta, p="fro").item()
        assert theta_norm <= C_norm + 1e-4, (
            f"Θ norm ({theta_norm:.6f}) exceeded C norm ({C_norm:.6f}): drift detected."
        )

    def test_symmetry_and_zero_diagonal_preserved_throughout(self):
        """
        Semantic: Θ must remain symmetric with zero diagonal after every step.
        Verified at each step, not just the final one.
        """
        problem = _make_problem(self.N, self.M)
        state = _make_state(self.M, self.N, theta_init=0.3)
        manager = EpisodeAdaptationManager("theta-only", eta_theta=0.15)

        for step in range(50):
            state = manager.step(state, self.X_CONST, problem)
            assert torch.allclose(state.Theta, state.Theta.T, atol=1e-7), (
                f"Symmetry violated at step {step + 1}."
            )
            assert torch.allclose(
                state.Theta.diagonal(), torch.zeros(self.M), atol=1e-7
            ), f"Non-zero diagonal at step {step + 1}: {state.Theta.diagonal()}"


# ---------------------------------------------------------------------------
# Test group 2 – Dependency change: Θ tracks the new observed structure
# ---------------------------------------------------------------------------

class TestDependencyChange:
    """
    After observing a new co-assignment structure, Θ must move toward it
    and away from the old one – smoothly, not instantaneously.
    """
    M = 4
    N = 2

    # Pattern A: tasks {0,1} co-assigned; tasks {2,3} co-assigned.
    X_BEFORE = torch.tensor([
        [1., 1., 0., 0.],
        [0., 0., 1., 1.],
    ], dtype=torch.float32)

    # Pattern B: tasks {0,2} co-assigned; tasks {1,3} co-assigned.
    X_AFTER = torch.tensor([
        [1., 0., 1., 0.],
        [0., 1., 0., 1.],
    ], dtype=torch.float32)

    def _warm_up(self, manager, state, problem, steps=60):
        return _run_ema(manager, state, problem, self.X_BEFORE, steps)

    def test_theta_moves_toward_new_structure(self):
        """
        Semantic: after a dependency change, Θ's distance to C_after
        strictly decreases with each new observation.
        """
        problem = _make_problem(self.N, self.M)
        state = _make_state(self.M, self.N, theta_init=0.0)
        manager = EpisodeAdaptationManager("theta-only", eta_theta=0.2)

        # Warm-up: let Θ converge near C_before
        state = self._warm_up(manager, state, problem, steps=60)

        C_after = _compute_C(self.X_AFTER)
        dist_before_switch = torch.dist(state.Theta, C_after).item()

        # Switch to X_after: Θ must move closer each step
        prev_dist = dist_before_switch
        for step in range(20):
            state = manager.step(state, self.X_AFTER, problem)
            dist = torch.dist(state.Theta, C_after).item()
            assert dist <= prev_dist + 1e-7, (
                f"Step {step + 1}: distance to C_after increased "
                f"{prev_dist:.6f} → {dist:.6f}"
            )
            prev_dist = dist

    def test_theta_closer_to_C_after_than_before_switch(self):
        """
        Semantic: ‖Θ_after − C_after‖_F < ‖Θ_before_switch − C_after‖_F.
        After observing X_after repeatedly, Θ must have moved toward C_after.
        """
        problem = _make_problem(self.N, self.M)
        state = _make_state(self.M, self.N, theta_init=0.0)
        manager = EpisodeAdaptationManager("theta-only", eta_theta=0.2)

        state = self._warm_up(manager, state, problem, steps=60)
        C_after = _compute_C(self.X_AFTER)
        dist_at_switch = torch.dist(state.Theta, C_after).item()

        state = _run_ema(manager, state, problem, self.X_AFTER, steps=40)
        dist_after = torch.dist(state.Theta, C_after).item()

        assert dist_after < dist_at_switch, (
            f"Θ did not move toward C_after after 40 steps: "
            f"dist_at_switch={dist_at_switch:.6f}, dist_after={dist_after:.6f}"
        )

    def test_theta_does_not_jump_instantly_to_C_after(self):
        """
        Semantic: the EMA smooths the transition; Θ must NOT jump to C_after
        after a single observation.  The distance to C_after must still be
        substantial after just one step.

        With eta=0.2, one step yields Θ = 0.8*Θ_old + 0.2*C_after.
        So ‖Θ_new − C_after‖ = 0.8 * ‖Θ_old − C_after‖ > 0.
        We verify that at least 50 % of the distance remains.
        """
        problem = _make_problem(self.N, self.M)
        state = _make_state(self.M, self.N, theta_init=0.0)
        eta = 0.2
        manager = EpisodeAdaptationManager("theta-only", eta_theta=eta)

        state = self._warm_up(manager, state, problem, steps=60)
        C_after = _compute_C(self.X_AFTER)
        dist_before = torch.dist(state.Theta, C_after).item()

        state = manager.step(state, self.X_AFTER, problem)
        dist_after_one = torch.dist(state.Theta, C_after).item()

        # After exactly one EMA step: dist_after_one ≈ (1 - eta) * dist_before
        expected_fraction = (1.0 - eta)
        assert dist_after_one >= expected_fraction * dist_before - 1e-5, (
            f"Θ jumped too fast after one step: "
            f"dist_before={dist_before:.6f}, dist_after_one={dist_after_one:.6f}, "
            f"expected_min={(expected_fraction * dist_before):.6f}"
        )

    def test_theta_still_symmetric_and_zero_diagonal_after_switch(self):
        """
        Semantic: structural constraints must hold throughout a dependency change.
        """
        problem = _make_problem(self.N, self.M)
        state = _make_state(self.M, self.N, theta_init=0.0)
        manager = EpisodeAdaptationManager("theta-only", eta_theta=0.2)

        state = self._warm_up(manager, state, problem, steps=40)
        for _ in range(20):
            state = manager.step(state, self.X_AFTER, problem)

        assert torch.allclose(state.Theta, state.Theta.T, atol=1e-7), \
            "Θ not symmetric after dependency change."
        assert torch.allclose(
            state.Theta.diagonal(), torch.zeros(self.M), atol=1e-7
        ), f"Non-zero diagonal after dependency change: {state.Theta.diagonal()}"


# ---------------------------------------------------------------------------
# Test group 3 – Context-independence: Θ cares only about X, not context
# ---------------------------------------------------------------------------

class TestContextIndependence:
    """
    Changes in agent capabilities, task embeddings, risk weights, or any
    other ProblemContext fields must NOT affect the Θ update when X is
    identical.
    """
    M = 4
    N = 2
    D = 3

    X = torch.tensor([
        [1., 1., 0., 0.],
        [0., 0., 1., 1.],
    ], dtype=torch.float32)

    def _run_trajectory(self, problem: ProblemContext, steps: int = 30):
        state = _make_state(self.M, self.N, self.D, theta_init=0.1)
        manager = EpisodeAdaptationManager("theta-only", eta_theta=0.15)
        return _run_ema(manager, state, problem, self.X, steps)

    def test_different_agent_capabilities_same_X_same_theta(self):
        """
        Semantic: changing agent capability embeddings (s) does not affect
        the Θ update, because the Θ branch uses only X.T @ X.
        """
        problem_A = ProblemContext(
            s=torch.zeros(self.N, self.D),
            c=torch.zeros(self.M, self.D),
            C=torch.zeros(self.M, self.M),
            W_risk=torch.zeros(3 * self.D, 1),
            N=self.N, M=self.M, d=self.D,
        )
        problem_B = ProblemContext(
            s=torch.ones(self.N, self.D) * 99.0,   # very different capabilities
            c=torch.zeros(self.M, self.D),
            C=torch.zeros(self.M, self.M),
            W_risk=torch.zeros(3 * self.D, 1),
            N=self.N, M=self.M, d=self.D,
        )

        state_A = self._run_trajectory(problem_A)
        state_B = self._run_trajectory(problem_B)

        assert torch.allclose(state_A.Theta, state_B.Theta, atol=1e-7), (
            "Θ differs between contexts with identical X but different agent embeddings.\n"
            f"Max deviation: {(state_A.Theta - state_B.Theta).abs().max().item():.8f}"
        )

    def test_different_task_embeddings_same_X_same_theta(self):
        """
        Semantic: different task embeddings (c) with the same X produce
        identical Θ updates.
        """
        problem_A = ProblemContext(
            s=torch.zeros(self.N, self.D),
            c=torch.zeros(self.M, self.D),
            C=torch.zeros(self.M, self.M),
            W_risk=torch.zeros(3 * self.D, 1),
            N=self.N, M=self.M, d=self.D,
        )
        problem_B = ProblemContext(
            s=torch.zeros(self.N, self.D),
            c=torch.randn(self.M, self.D) * 5.0,   # very different task embeddings
            C=torch.zeros(self.M, self.M),
            W_risk=torch.zeros(3 * self.D, 1),
            N=self.N, M=self.M, d=self.D,
        )

        state_A = self._run_trajectory(problem_A)
        state_B = self._run_trajectory(problem_B)

        assert torch.allclose(state_A.Theta, state_B.Theta, atol=1e-7), (
            "Θ differs between contexts with identical X but different task embeddings.\n"
            f"Max deviation: {(state_A.Theta - state_B.Theta).abs().max().item():.8f}"
        )

    def test_different_risk_weights_same_X_same_theta(self):
        """
        Semantic: different risk weight matrices (W_risk) with the same X
        produce identical Θ updates.
        """
        problem_A = ProblemContext(
            s=torch.zeros(self.N, self.D),
            c=torch.zeros(self.M, self.D),
            C=torch.zeros(self.M, self.M),
            W_risk=torch.zeros(3 * self.D, 1),
            N=self.N, M=self.M, d=self.D,
        )
        problem_B = ProblemContext(
            s=torch.zeros(self.N, self.D),
            c=torch.zeros(self.M, self.D),
            C=torch.zeros(self.M, self.M),
            W_risk=torch.randn(3 * self.D, 1) * 10.0,   # very different risk weights
            N=self.N, M=self.M, d=self.D,
        )

        state_A = self._run_trajectory(problem_A)
        state_B = self._run_trajectory(problem_B)

        assert torch.allclose(state_A.Theta, state_B.Theta, atol=1e-7), (
            "Θ differs between contexts with identical X but different risk weights.\n"
            f"Max deviation: {(state_A.Theta - state_B.Theta).abs().max().item():.8f}"
        )

    def test_different_cost_weights_same_X_same_theta(self):
        """
        Semantic: interaction_weight and cost_weight in ProblemContext do not
        affect the Θ update, because C_t is derived purely from X.
        """
        problem_A = ProblemContext(
            s=torch.zeros(self.N, self.D),
            c=torch.zeros(self.M, self.D),
            C=torch.zeros(self.M, self.M),
            W_risk=torch.zeros(3 * self.D, 1),
            N=self.N, M=self.M, d=self.D,
            interaction_weight=1.0, cost_weight=1.0,
        )
        problem_B = ProblemContext(
            s=torch.zeros(self.N, self.D),
            c=torch.zeros(self.M, self.D),
            C=torch.zeros(self.M, self.M),
            W_risk=torch.zeros(3 * self.D, 1),
            N=self.N, M=self.M, d=self.D,
            interaction_weight=100.0, cost_weight=0.001,
        )

        state_A = self._run_trajectory(problem_A)
        state_B = self._run_trajectory(problem_B)

        assert torch.allclose(state_A.Theta, state_B.Theta, atol=1e-7), (
            "Θ differs between contexts with identical X but different energy weights.\n"
            f"Max deviation: {(state_A.Theta - state_B.Theta).abs().max().item():.8f}"
        )


# ---------------------------------------------------------------------------
# Test group 4 – Ground-truth isolation: GT never enters the Θ path
# ---------------------------------------------------------------------------

class TestGroundTruthIsolation:
    """
    Ground-truth interaction matrices must have zero influence on Θ.
    We simulate what happens inside the runner: the problem context carries
    the current problem's interaction graph, but Θ is updated from X only.
    The interaction_graph is only used to construct the ground-truth Landscape
    and to initialise the initial LandscapeState; it must not contaminate
    subsequent Θ updates.
    """
    M = 4
    N = 2
    D = 2

    X = torch.tensor([
        [1., 1., 0., 0.],
        [0., 0., 1., 1.],
    ], dtype=torch.float32)

    def _run_with_co_assignment_cost(
        self,
        C_matrix: torch.Tensor,
        steps: int = 40,
    ) -> LandscapeState:
        """
        Run the adaptation with a given co_assignment_costs (C) matrix,
        which is the closest analogue of a 'ground-truth' dependency matrix
        that lives inside ProblemContext.
        """
        problem = ProblemContext(
            s=torch.zeros(self.N, self.D),
            c=torch.zeros(self.M, self.D),
            C=C_matrix.clone(),
            W_risk=torch.zeros(3 * self.D, 1),
            N=self.N, M=self.M, d=self.D,
        )
        state = _make_state(self.M, self.N, self.D, theta_init=0.0)
        manager = EpisodeAdaptationManager("theta-only", eta_theta=0.2)
        return _run_ema(manager, state, problem, self.X, steps)

    def test_changing_co_assignment_cost_matrix_does_not_affect_theta(self):
        """
        Semantic: the ProblemContext.C field (co-assignment cost matrix —
        the nearest proxy for a 'ground-truth dependency' available inside
        the context) must not change the Θ update when X is identical.
        """
        C_zero = torch.zeros(self.M, self.M)

        # C_gt: a symmetric, offdiag 'ground-truth'-like matrix
        C_gt = torch.zeros(self.M, self.M)
        C_gt[0, 1] = C_gt[1, 0] = 0.9
        C_gt[2, 3] = C_gt[3, 2] = 0.9

        state_with_zero_gt = self._run_with_co_assignment_cost(C_zero)
        state_with_real_gt = self._run_with_co_assignment_cost(C_gt)

        assert torch.allclose(
            state_with_zero_gt.Theta,
            state_with_real_gt.Theta,
            atol=1e-7,
        ), (
            "Θ differed when only ProblemContext.C (co-assignment cost / GT proxy) "
            "was changed, with identical X. GT has leaked into the Θ update path.\n"
            f"Max deviation: "
            f"{(state_with_zero_gt.Theta - state_with_real_gt.Theta).abs().max().item():.8f}"
        )

    def test_extreme_gt_values_do_not_affect_theta(self):
        """
        Semantic: even with extreme GT values the Θ update is identical to
        the zero-GT case when X is held constant.
        """
        C_zero = torch.zeros(self.M, self.M)
        C_extreme = torch.ones(self.M, self.M) * 1e6
        C_extreme.fill_diagonal_(0.0)

        state_zero = self._run_with_co_assignment_cost(C_zero)
        state_extreme = self._run_with_co_assignment_cost(C_extreme)

        assert torch.allclose(state_zero.Theta, state_extreme.Theta, atol=1e-7), (
            "Extreme GT values in ProblemContext.C affected Θ with identical X.\n"
            f"Max deviation: "
            f"{(state_zero.Theta - state_extreme.Theta).abs().max().item():.8f}"
        )

    def test_theta_value_equals_pure_ema_of_C_t_regardless_of_gt(self):
        """
        Semantic: the final Θ after N steps must equal the pure EMA trajectory
        driven only by X — regardless of what GT/context values are present.
        This verifies the complete isolation numerically.
        """
        eta = 0.2
        steps = 25

        # Reference: compute expected Θ purely from X
        C_obs = _compute_C(self.X)
        Theta_ref = torch.zeros(self.M, self.M)
        for _ in range(steps):
            raw = (1.0 - eta) * Theta_ref + eta * C_obs
            sym = (raw + raw.T) / 2.0
            sym.fill_diagonal_(0.0)
            Theta_ref = sym

        # Run with large GT in context
        C_gt_large = torch.ones(self.M, self.M) * 500.0
        C_gt_large.fill_diagonal_(0.0)
        state_actual = self._run_with_co_assignment_cost(C_gt_large, steps=steps)

        assert torch.allclose(state_actual.Theta, Theta_ref, atol=1e-6), (
            "Θ deviated from pure EMA(X) trajectory even though only GT context changed.\n"
            f"Max deviation: {(state_actual.Theta - Theta_ref).abs().max().item():.8f}"
        )

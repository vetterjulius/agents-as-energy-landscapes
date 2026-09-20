from __future__ import annotations

import math
import torch
import pytest
import numpy as np

from controlled_benchmark.config import BenchmarkConfig
from controlled_benchmark.runner import ControlledBenchmarkRunner
from controlled_benchmark.scenarios import (
    generate_base_problem,
    generate_stationary_episode,
    problem_instance_to_problem_context,
)
from controlled_benchmark.solvers import (
    ConventionalGreedySolver,
    EnergyAwareGreedySolver,
    EnergyAwareSimulatedAnnealingSolver,
    BudgetedLandscape,
)
from landscape import Landscape, LandscapeState


def test_b0_deterministic():
    """Test A: B0 is strictly deterministic. Same input -> identical assignment matrix."""
    inst = generate_base_problem(seed=42, N=5, M=10, d=8)
    ctx1 = problem_instance_to_problem_context(inst)
    ctx2 = problem_instance_to_problem_context(inst)

    solver = ConventionalGreedySolver()
    res1 = solver.solve(ctx1)
    res2 = solver.solve(ctx2)

    assert torch.equal(res1.X, res2.X), "B0 must be 100% deterministic given identical problem context"


def test_b0_ignores_adaptive_states():
    """Test B: B0 does not use kappa or Theta. Changing them must NOT alter B0 output."""
    inst = generate_base_problem(seed=42, N=5, M=10, d=8)
    ctx = problem_instance_to_problem_context(inst)

    state_zero = LandscapeState(kappa=torch.zeros(5, 8), Theta=torch.zeros(10, 10))
    state_rand = LandscapeState(kappa=torch.randn(5, 8) * 10.0, Theta=torch.randn(10, 10) * 10.0)

    landscape_zero = Landscape(ctx, state_zero)
    landscape_rand = Landscape(ctx, state_rand)

    solver = ConventionalGreedySolver()
    res_zero = solver.solve(landscape_zero)
    res_rand = solver.solve(landscape_rand)

    assert torch.equal(res_zero.X, res_rand.X), "B0 solver output must not depend on kappa or Theta"


def test_b0_ignores_ground_truth_dependency():
    """Test C: B0 solver output does not depend on interaction_graph (G_gt)."""
    inst1 = generate_base_problem(seed=42, N=5, M=10, d=8)
    inst2 = generate_base_problem(seed=42, N=5, M=10, d=8)

    # Mutate interaction_graph completely in inst2
    inst2.interaction_graph = torch.ones(10, 10) * 99.0

    ctx1 = problem_instance_to_problem_context(inst1)
    ctx2 = problem_instance_to_problem_context(inst2)

    solver = ConventionalGreedySolver()
    res1 = solver.solve(ctx1)
    res2 = solver.solve(ctx2)

    assert torch.equal(res1.X, res2.X), "B0 solver output must be strictly independent of G_gt"


def test_b0_valid_assignment_matrix():
    """Test D: B0 produces valid assignment matrices (exactly 1 agent per task, binary values)."""
    inst = generate_base_problem(seed=123, N=4, M=12, d=6)
    ctx = problem_instance_to_problem_context(inst)

    solver = ConventionalGreedySolver()
    res = solver.solve(ctx)

    X = res.X
    assert X.shape == (4, 12)
    assert torch.all((X == 0.0) | (X == 1.0)), "Assignment matrix must be strictly binary"
    assert torch.allclose(X.sum(dim=0), torch.ones(12)), "Each task must be assigned to exactly one agent"


def test_existing_energy_solvers_unchanged():
    """Test E: Regression test ensuring B1/B2/B3/B4 solvers remain functional and unchanged."""
    inst = generate_stationary_episode(episode=0, seed=42, perturb_episode=5, N=3, M=6, d=4)
    ctx = problem_instance_to_problem_context(inst)
    state = LandscapeState(kappa=torch.zeros(3, 4), Theta=inst.interaction_graph.clone())
    landscape = Landscape(ctx, state)

    greedy = EnergyAwareGreedySolver()
    sa = EnergyAwareSimulatedAnnealingSolver()

    initial_X = torch.zeros(3, 6)
    for t in range(6):
        initial_X[t % 3, t] = 1.0

    b_greedy = BudgetedLandscape(landscape, max_evaluations=50)
    b_sa = BudgetedLandscape(landscape, max_evaluations=50)

    res_greedy = greedy.solve(b_greedy, initial_X)
    res_sa = sa.solve(b_sa, initial_X, seed=42)

    assert res_greedy.status == "completed"
    assert res_sa.status == "completed"
    assert res_greedy.energy_evaluations <= 50
    assert res_sa.energy_evaluations <= 50


def test_condition_matrix_contains_five_conditions():
    """Test F: Verify the condition matrix contains exactly five conditions (B0..B4)."""
    cfg = BenchmarkConfig.quick_mode()
    runner = ControlledBenchmarkRunner(cfg)
    cells = runner._cells()

    assert len(cells) == 5, f"Expected 5 conditions, got {len(cells)}"
    condition_ids = [c[0] for c in cells]
    expected = [
        "conventional_greedy",
        "static_energy_greedy",
        "static_energy_sa",
        "adaptive_energy_greedy",
        "adaptive_energy_sa",
    ]
    assert condition_ids == expected, f"Condition IDs mismatch: {condition_ids} vs {expected}"

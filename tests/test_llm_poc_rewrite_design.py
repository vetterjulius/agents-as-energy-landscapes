"""Scientific validity checks for the minimal rewrite experiment."""
from __future__ import annotations

import torch

from llm_poc_rewrite import config
from llm_poc_rewrite.orchestration import (
    _problem_context,
    make_initial_state,
    make_real_issue_initial_state,
    orchestrate,
    orchestrate_real_issue,
)
from llm_poc_rewrite.run import planned_real_issue_patch_calls, real_issue_episode_diagnostics
from llm_poc_rewrite.real_issue_design import check_transfer_design, current_design_report
from llm_poc_rewrite.scoring import deterministic_score, is_deterministic_task
from llm_poc_rewrite.tasks import build_agent_slots, build_episode, declared_dependency_pairs


def test_episode_answers_follow_actual_inputs():
    first = build_episode("stationary", 0)
    second = build_episode("stationary", 1)
    assert first.specs[0].reference_answer.endswith("largest increase 22 from week 3 to week 4")
    assert second.specs[0].reference_answer.endswith("largest increase 19 from week 5 to week 6")
    assert first.specs[2].reference_answer.endswith("tied")
    assert second.specs[2].reference_answer.endswith("Supplier B cheaper by 100")
    assert first.specs[3].reference_answer.startswith("-12.4%")
    assert second.specs[3].reference_answer.startswith("-14.3%")
    assert second.specs[8].reference_answer.startswith("BILLING +7pp")


def test_shift_occurs_early_and_dependency_task_prompts_match_graph():
    before = build_episode("shift_mid", 0)
    after = build_episode("shift_mid", 1)
    assert declared_dependency_pairs("shift_mid", 0) == [(1, 3), (6, 7), (8, 9)]
    assert declared_dependency_pairs("shift_mid", 1) == [(0, 3), (5, 7), (8, 9)]
    assert before.specs[3].depends_on == 1
    assert after.specs[3].depends_on == 0
    assert after.specs[7].depends_on == 5
    assert "maximum observed parcel count" in after.specs[3].prompt_template
    assert "inventory note" in after.specs[7].prompt_template


def test_deterministic_checks_cover_objective_tasks_and_accept_episode_answers():
    episode = build_episode("stationary", 1)
    samples = {
        0: "Up trend; maximum 190 in week 6; largest increase 19 from week 5 to week 6",
        1: "4270",
        2: "Supplier A total 4900; Supplier B total 4800; Supplier B cheaper by 100",
        3: "-14.3%",
        4: "LISBON, PORTO",
        5: "2026-10-01, 2026-11-15",
        6: "DESK=259,10\nCHAIR=95,108\nLAMP=49,88",
        8: "BILLING +7 percentage points",
    }
    for task_id, answer in samples.items():
        assert is_deterministic_task(task_id)
        assert deterministic_score(episode.specs[task_id], answer) == (2, None)
    assert not is_deterministic_task(7)
    assert not is_deterministic_task(9)


def test_shifted_task_3_has_a_deterministic_episode_specific_checker():
    episode = build_episode("shift_mid", 2)
    assert episode.specs[3].reference_answer == "YES; 1000 vs 250"
    assert deterministic_score(episode.specs[3], "YES; 1000 vs 250") == (2, None)
    assert deterministic_score(episode.specs[3], "YES; 990 vs 210") == (0, None)


def test_energy_conditions_use_same_solver_and_no_oracle_graph():
    slots = build_agent_slots()
    episode = build_episode("shift_mid", 0)
    state = make_initial_state(episode)
    context = _problem_context(slots, episode)
    assert torch.count_nonzero(state.Theta) == 0
    assert torch.count_nonzero(context.C) == 0

    greedy, _ = orchestrate("static_energy", slots, episode, state, rng_seed=77)
    annealed, _ = orchestrate("adaptive_energy", slots, episode, state, rng_seed=77)
    assert torch.equal(greedy.X, annealed.X)
    assert greedy.energy_evaluations == annealed.energy_evaluations


def test_real_issue_pilot_uses_fixed_g0_and_theta_only_boundary_update():
    tasks = build_episode("stationary", 0)
    # Reuse the two-task truncation only to exercise the pilot-specific state API.
    tasks.specs = tasks.specs[:2]
    state = make_real_issue_initial_state(tasks)
    expected_g0 = torch.tensor(config.REAL_ISSUE_G0)
    assert torch.equal(state.Theta, expected_g0)
    assert torch.count_nonzero(state.kappa) == 0

    pilot_tasks = type(tasks)(specs=tasks.specs, inputs=tasks.inputs[:2])
    before = state.clone()
    result, after = orchestrate_real_issue(
        "adaptive_energy", build_agent_slots(), pilot_tasks, state, rng_seed=7,
    )
    assert result.adaptation_mode == "theta-only"
    assert torch.equal(result.theta_before, expected_g0)
    assert torch.equal(result.theta_after, after.Theta)
    assert torch.equal(result.theta_before, before.Theta)
    assert torch.equal(after.kappa, before.kappa)
    assert not torch.equal(after.Theta, expected_g0)
    from llm_poc_rewrite.orchestration import orchestration_diagnostic
    controller_record = orchestration_diagnostic(result)
    assert "theta_distance_to_evaluator_after" not in controller_record
    evaluated = orchestration_diagnostic(
        result, evaluator_theta=torch.tensor(config.REAL_ISSUE_G1),
    )
    assert "theta_distance_to_evaluator_after" in evaluated
    assert evaluated["theta_distance_to_evaluator_before"] is not None
    assert evaluated["theta_distance_to_evaluator_after"] is not None
    assert torch.equal(result.theta_observation, result.theta_observation.T)
    assert torch.count_nonzero(torch.diagonal(result.theta_observation)) == 0

    static_result, static_after = orchestrate_real_issue(
        "static_energy", build_agent_slots(), pilot_tasks, before, rng_seed=7,
    )
    assert static_result.adaptation_mode == "static"
    assert torch.equal(static_after.Theta, expected_g0)


def test_real_issue_plan_is_fixed_and_offline_trace_marks_heuristic():
    assert planned_real_issue_patch_calls() == 36
    records = real_issue_episode_diagnostics("adaptive_energy", seed=9)
    assert len(records) == 2
    assert records[0]["graph_update_source"] == "assignment_history_only"
    assert records[0]["co_assignment_prior"] == "heuristic_not_empirically_validated"
    assert records[0]["positive_edge_means_shared_agent"] is True
    assert records[0]["assignment_synergy_validated"] is False
    assert records[1]["theta_before"] == records[0]["next_theta"]


def test_default_rewrite_initialization_and_adaptation_remain_unchanged():
    episode = build_episode("stationary", 0)
    state = make_initial_state(episode)
    assert torch.count_nonzero(state.Theta) == 0
    result, _ = orchestrate(
        "adaptive_energy", build_agent_slots(), episode, state, rng_seed=23,
    )
    assert result.adaptation_mode == "full"


def test_real_issue_pilot_fails_closed_on_unvalidated_synthetic_slots():
    report = current_design_report()
    assert report.ready is False
    assert report.task_count == len(config.REAL_ISSUE_TASK_IDS) == 2
    assert report.agent_count == config.N_AGENTS == 3
    assert any("not validated coding competencies" in issue for issue in report.issues)


def test_real_issue_design_gate_requires_shared_schema_and_rationale():
    issue_vectors = [[float(task == feature) for feature in range(8)] for task in range(2)]
    slot_vectors = [[float(slot == feature) for feature in range(8)] for slot in range(3)]
    names = [f"feature_{index}" for index in range(8)]
    rationale = "All eight dimensions encode the same preregistered routing attributes for issues and slots."

    valid = check_transfer_design(issue_vectors, names, slot_vectors, names, rationale)
    assert valid.ready
    assert valid.feature_dimensions == 8

    invalid = check_transfer_design(issue_vectors, names, slot_vectors, names[:-1], rationale)
    assert not invalid.ready
    assert any("feature names/order do not match" in issue for issue in invalid.issues)

    missing = check_transfer_design(None, None, slot_vectors, names, None)
    assert not missing.ready
    assert any("real-issue feature vectors" in issue for issue in missing.issues)
    assert any("feature-to-slot rationale" in issue for issue in missing.issues)

    wrong_task_count = check_transfer_design(
        [[0.0] * 8] * 3, names, slot_vectors, names, rationale,
        task_count=3,
    )
    assert not wrong_task_count.ready
    assert any("selected real-issue node set" in issue for issue in wrong_task_count.issues)


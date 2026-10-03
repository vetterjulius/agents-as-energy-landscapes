"""Smoke tests for the LLM PoC (offline; mock workers, no API access)."""
from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import pandas as pd
import pytest
import torch

ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture(scope="module")
def mock_run(tmp_path_factory):
    out = tmp_path_factory.mktemp("llm_poc_results")
    proc = subprocess.run(
        [sys.executable, "-m", "llm_poc.run", "--output-dir", str(out)],
        cwd=ROOT, capture_output=True, text=True, timeout=300,
    )
    assert proc.returncode == 0, proc.stderr
    df = pd.read_csv(out / "poc_episodes.csv")
    return df, out


def test_dry_run_scale_and_conditions(mock_run):
    df, _ = mock_run
    # 3 conditions x 2 blocks x 3 episodes x 3 reps x 10 tasks = 540 rows
    assert len(df) == 540
    assert set(df.condition) == {"baseline", "static_energy", "adaptive_energy"}
    assert set(df.block) == {"stationary", "shift_mid"}
    assert df.task_id.nunique() == 10


def test_mock_mode_is_offline(mock_run):
    df, _ = mock_run
    assert (df.scorer == "rubric_proxy").all()
    assert df.worker_ok.all()


def test_energy_conditions_co_locate_dependencies_baseline_does_not(mock_run):
    df, _ = mock_run
    eps = df.drop_duplicates(["condition", "block", "episode", "repetition"])
    rates = eps.groupby("condition").dependency_satisfaction_episode.mean()
    assert rates["static_energy"] > 0.8
    assert rates["adaptive_energy"] > 0.8
    assert rates["baseline"] < 0.2


def test_energy_assignment_differs_from_baseline(mock_run):
    df, _ = mock_run
    base = df[(df.condition == "baseline") & (df.block == "stationary") &
              (df.episode == 0) & (df.repetition == 0)].sort_values("task_id")
    en = df[(df.condition == "static_energy") & (df.block == "stationary") &
            (df.episode == 0) & (df.repetition == 0)].sort_values("task_id")
    # task 3 (cross-functional dependency task) is moved onto the extractor slot
    assert base[base.task_id == 3].assigned_agent.iloc[0] != \
        en[en.task_id == 3].assigned_agent.iloc[0]


def test_adaptive_theta_drifts_when_stationary(mock_run):
    df, _ = mock_run
    st = df[(df.condition == "adaptive_energy") & (df.block == "stationary")]
    by_ep = st.groupby("episode").theta_diff_from_gt.mean()
    assert by_ep[2] > by_ep[0]  # error grows without environmental change


def test_adaptive_theta_tracks_shift_better_than_static(mock_run):
    df, _ = mock_run
    sh = df[(df.block == "shift_mid") & (df.episode == 2)]
    adaptive = sh[sh.condition == "adaptive_energy"].theta_diff_from_gt.mean()
    static = sh[sh.condition == "static_energy"].theta_diff_from_gt.mean()
    assert adaptive < static  # one EMA step already moves Theta toward the new structure


def test_b0_uses_zero_energy_evaluations(mock_run):
    df, _ = mock_run
    assert (df[df.condition == "baseline"].solver_energy_evaluations == 0).all()
    assert (df[df.condition == "static_energy"].solver_energy_evaluations > 0).all()
    assert (df[df.condition == "adaptive_energy"].solver_energy_evaluations > 0).all()


def test_summary_and_report_written(mock_run):
    _, out = mock_run
    assert (out / "poc_summary.json").exists()
    assert (out / "poc_report.md").exists()
    import json
    summary = json.loads((out / "poc_summary.json").read_text(encoding="utf-8"))
    assert summary["total_worker_calls"] == 540
    assert summary["scorer"] == "rubric_proxy"


def test_task_bank_deterministic():
    from llm_poc.tasks import build_task_bank, build_episode, declared_dependency_pairs
    t1, t2 = build_task_bank(), build_task_bank()
    assert [x.embedding for x in t1] == [x.embedding for x in t2]
    assert declared_dependency_pairs("stationary", 0) == [(1, 3), (6, 7), (8, 9)]
    assert declared_dependency_pairs("shift_mid", 2) == [(0, 3), (5, 7), (8, 9)]
    ep = build_episode("shift_mid", 2)
    t3 = next(t for t in ep.specs if t.id == 3)
    assert t3.depends_on == 0


def test_orchestration_adaptation_chains():
    from llm_poc.orchestration import make_initial_state, orchestrate
    from llm_poc.tasks import build_agent_slots, build_episode, declared_dependency_pairs
    slots = build_agent_slots()
    et = build_episode("stationary", 0)
    deps = declared_dependency_pairs("stationary", 0)
    state = make_initial_state(et, deps)
    rng = 12345
    for _ in range(3):
        _, state = orchestrate("adaptive_energy", slots, et, deps, state, rng)
    assert float(state.Theta.norm()) > 0.0
    assert float(state.kappa.norm()) > 0.0

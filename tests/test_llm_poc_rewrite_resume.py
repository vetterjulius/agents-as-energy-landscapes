"""Offline tests for rewrite-run checkpoint and resume behavior."""
from __future__ import annotations

import csv
import json
from argparse import Namespace
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

from llm_poc_rewrite import config, run
from llm_poc_rewrite.api import Generation
from llm_poc_rewrite.tasks import EpisodeTasks, TaskSpec


def test_checkpoint_load_rejects_completed_or_changed_design(tmp_path):
    path = tmp_path / "checkpoint.json"
    checkpoint = run._new_checkpoint()
    run._write_checkpoint(path, checkpoint)
    assert run._load_checkpoint(path)["status"] == "running"
    assert not list(tmp_path.glob("*.tmp"))

    checkpoint["design"]["judge_model"] = "changed-model"
    run._write_checkpoint(path, checkpoint)
    with pytest.raises(SystemExit, match="does not match"):
        run._load_checkpoint(path)

    checkpoint = run._new_checkpoint()
    checkpoint["status"] = "complete"
    run._write_checkpoint(path, checkpoint)
    with pytest.raises(SystemExit, match="already complete"):
        run._load_checkpoint(path)


def test_resume_reuses_worker_result_when_judging_was_interrupted(tmp_path, monkeypatch):
    monkeypatch.setenv("GEMINI_API_KEY", "offline-test")
    monkeypatch.setattr(config, "CONDITIONS", ("baseline",))
    monkeypatch.setattr(config, "BLOCKS", ("stationary",))
    monkeypatch.setattr(config, "N_EPISODES", 1)
    monkeypatch.setattr(config, "N_REPETITIONS", 1)
    monkeypatch.setattr(config, "N_TASKS", 2)
    monkeypatch.setattr(config, "EXPECTED_WORKER_CALLS", 2)
    monkeypatch.setattr(config, "EXPECTED_JUDGE_CALLS", 2)
    monkeypatch.setattr(run.torch, "use_deterministic_algorithms", lambda enabled: None)

    specs = [
        TaskSpec(i, "analyze", (1.0,), 0.2, f"task {i}: {{input}}", "answer",
                 ("wrong", "right"))
        for i in range(2)
    ]
    episode = EpisodeTasks(specs=specs, inputs=["input-0", "input-1"])
    monkeypatch.setattr(run, "build_episode", lambda block, index: episode)
    monkeypatch.setattr(run, "declared_dependency_pairs", lambda block, index: [])
    monkeypatch.setattr(run, "make_initial_state", lambda tasks, dependencies: object())
    assignment = SimpleNamespace(X=torch.tensor([[1.0, 0.0], [0.0, 1.0], [0.0, 0.0]]))
    monkeypatch.setattr(run, "orchestrate", lambda *args: (assignment, object()))

    worker_calls = []

    def fake_generate(prompt, system_prompt, model, **kwargs):
        worker_calls.append(prompt)
        return Generation(f"worker-{len(worker_calls)}", model, 1, 1, 0.01)

    monkeypatch.setattr(run, "generate", fake_generate)
    judge_calls = []

    def interrupted_then_successful_score(task, candidate, *, task_prompt, seed=None):
        judge_calls.append(candidate)
        if len(judge_calls) == 2:
            raise RuntimeError("simulated interruption after worker response")
        return 1, Generation(
            "1", config.JUDGE_MODEL, 1, 1, 0.01, finish_reason="STOP",
        ), None

    monkeypatch.setattr(run, "score", interrupted_then_successful_score)
    monkeypatch.setattr(run.judge, "preflight", lambda **kwargs: Generation(
        "2", config.JUDGE_MODEL, 1, 1, 0.01, finish_reason="STOP",
    ))
    resume_flags = iter((False, True))
    monkeypatch.setattr(
        run, "parse_args",
        lambda: Namespace(output_dir=str(tmp_path), plan_only=False,
                          resume=next(resume_flags)),
    )

    with pytest.raises(RuntimeError, match="simulated interruption"):
        run.main()

    checkpoint_path = tmp_path / "llm_poc_rewrite_checkpoint.json"
    interrupted = json.loads(checkpoint_path.read_text(encoding="utf-8"))
    assert interrupted["status"] == "running"
    assert len(interrupted["rows"]) == 1
    assert interrupted["rows"][0]["worker_text"] == "worker-1"
    assert interrupted["pending"]["worker"]["text"] == "worker-2"
    assert len(worker_calls) == 2

    assert run.main() == 0
    completed = json.loads(checkpoint_path.read_text(encoding="utf-8"))
    assert completed["status"] == "complete"
    assert completed["pending"] is None
    assert len(completed["rows"]) == 2
    assert len(worker_calls) == 2, "resumed run must reuse completed and pending worker results"
    assert judge_calls == ["worker-1", "worker-2", "worker-2"]

    with (tmp_path / "llm_poc_rewrite.csv").open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    assert [row["worker_text"] for row in rows] == ["worker-1", "worker-2"]
    summary = json.loads((tmp_path / "llm_poc_rewrite_summary.json").read_text(encoding="utf-8"))
    assert summary["resumed"] is True

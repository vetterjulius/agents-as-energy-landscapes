"""Offline tests for the Gemma 4 judge response protocol and preflight."""
from __future__ import annotations

import json
from argparse import Namespace

import pytest
import torch

from llm_poc_rewrite import api, config, judge, run
from llm_poc_rewrite.api import Generation
from llm_poc_rewrite.tasks import TaskSpec


class FakeResponse:
    def __init__(self, body):
        self.body = body

    def __enter__(self):
        return self

    def __exit__(self, *args):
        return False

    def read(self):
        return json.dumps(self.body).encode("utf-8")


def _judge_body(text="2", finish_reason="STOP"):
    return {
        "modelVersion": config.JUDGE_MODEL,
        "candidates": [{
            "content": {"parts": [{"text": text}]},
            "finishReason": finish_reason,
        }],
        "usageMetadata": {
            "promptTokenCount": 31,
            "candidatesTokenCount": 1,
            "thoughtsTokenCount": 3,
            "totalTokenCount": 35,
        },
    }


def _task():
    return TaskSpec(
        id=0,
        kind="extract",
        embedding=(1.0,),
        difficulty=0.1,
        prompt_template="Extract a value: {input}",
        reference_answer="4270",
        rubric=("incorrect", "partially correct", "exactly 4270"),
    )


def test_generate_sets_gemma4_judge_thinking_and_records_response_metadata(monkeypatch):
    captured = {}

    def fake_urlopen(request, timeout):
        captured["payload"] = json.loads(request.data)
        return FakeResponse(_judge_body())

    monkeypatch.setenv("GEMINI_API_KEY", "offline-test")
    monkeypatch.setattr(api, "_last_request_at", 0.0)
    monkeypatch.setattr(api.urllib.request, "urlopen", fake_urlopen)

    result = api.generate("grade this", "system", config.JUDGE_MODEL, judge=True, seed=7)

    generation_config = captured["payload"]["generationConfig"]
    assert generation_config["maxOutputTokens"] == config.JUDGE_MAX_TOKENS == 32
    assert generation_config["thinkingConfig"] == {"thinkingLevel": "minimal"}
    assert result.text == "2"
    assert result.finish_reason == "STOP"
    assert result.prompt_tokens == 31
    assert result.completion_tokens == 1
    assert result.thoughts_tokens == 3
    assert result.total_tokens == 35


def test_worker_request_does_not_receive_judge_thinking_configuration(monkeypatch):
    captured = {}

    def fake_urlopen(request, timeout):
        captured["payload"] = json.loads(request.data)
        return FakeResponse({
            "modelVersion": config.AGENT_MODELS["analyst"],
            "candidates": [{"content": {"parts": [{"text": "ok"}]}, "finishReason": "STOP"}],
            "usageMetadata": {"promptTokenCount": 1, "candidatesTokenCount": 1},
        })

    monkeypatch.setenv("GEMINI_API_KEY", "offline-test")
    monkeypatch.setattr(api, "_last_request_at", 0.0)
    monkeypatch.setattr(api.urllib.request, "urlopen", fake_urlopen)

    api.generate("work", "system", config.AGENT_MODELS["analyst"])

    generation_config = captured["payload"]["generationConfig"]
    assert generation_config["maxOutputTokens"] == config.WORKER_MAX_TOKENS
    assert "thinkingConfig" not in generation_config


def test_score_rejects_non_stop_finish_reason_even_when_text_is_numeric(monkeypatch):
    monkeypatch.setattr(
        judge, "generate",
        lambda *args, **kwargs: Generation(
            "2", config.JUDGE_MODEL, 10, 1, 0.01, finish_reason="MAX_TOKENS",
        ),
    )

    level, generation, error = judge.score(_task(), "4270", task_prompt="prompt")

    assert level is None
    assert generation.finish_reason == "MAX_TOKENS"
    assert error and "did not finish normally" in error


def test_score_rejects_missing_finish_reason_and_out_of_rubric_score(monkeypatch):
    monkeypatch.setattr(
        judge, "generate",
        lambda *args, **kwargs: Generation("2", config.JUDGE_MODEL, 10, 1, 0.01),
    )
    _, _, missing_finish_error = judge.score(_task(), "4270", task_prompt="prompt")
    assert missing_finish_error and "finishReason=missing" in missing_finish_error

    monkeypatch.setattr(
        judge, "generate",
        lambda *args, **kwargs: Generation(
            "3", config.JUDGE_MODEL, 10, 1, 0.01, finish_reason="STOP",
        ),
    )
    level, _, range_error = judge.score(_task(), "4270", task_prompt="prompt")
    assert level is None
    assert range_error and "outside rubric range" in range_error


def test_preflight_requires_expected_score(monkeypatch):
    monkeypatch.setattr(
        judge, "score",
        lambda *args, **kwargs: (
            2, Generation("2", config.JUDGE_MODEL, 5, 1, 0.01, finish_reason="STOP"), None
        ),
    )
    assert judge.preflight(seed=1).finish_reason == "STOP"

    monkeypatch.setattr(
        judge, "score",
        lambda *args, **kwargs: (
            0, Generation("0", config.JUDGE_MODEL, 5, 1, 0.01, finish_reason="STOP"), None
        ),
    )
    with pytest.raises(RuntimeError, match="no worker calls were made"):
        judge.preflight(seed=1)


def test_preflight_failure_stops_before_any_worker_call(tmp_path, monkeypatch):
    monkeypatch.setenv("GEMINI_API_KEY", "offline-test")
    monkeypatch.setattr(config, "CONDITIONS", ("baseline",))
    monkeypatch.setattr(config, "BLOCKS", ("stationary",))
    monkeypatch.setattr(config, "N_EPISODES", 1)
    monkeypatch.setattr(config, "N_REPETITIONS", 1)
    monkeypatch.setattr(config, "N_TASKS", 1)
    monkeypatch.setattr(config, "EXPECTED_WORKER_CALLS", 1)
    monkeypatch.setattr(config, "EXPECTED_JUDGE_CALLS", 1)
    monkeypatch.setattr(run, "parse_args", lambda: Namespace(
        output_dir=str(tmp_path), plan_only=False, resume=False,
    ))
    worker_calls = []
    monkeypatch.setattr(run, "generate", lambda *args, **kwargs: worker_calls.append(args))
    monkeypatch.setattr(
        run.judge, "preflight",
        lambda **kwargs: (_ for _ in ()).throw(RuntimeError("simulated preflight failure")),
    )

    with pytest.raises(RuntimeError, match="simulated preflight failure"):
        run.main()

    assert worker_calls == []
    checkpoint = json.loads(
        (tmp_path / "llm_poc_rewrite_checkpoint.json").read_text(encoding="utf-8")
    )
    assert checkpoint["judge_preflight_calls"] == 1
    assert "simulated preflight failure" in checkpoint["judge_preflight_error"]
    assert checkpoint["rows"] == []
    assert checkpoint["pending"] is None


def test_invalid_task_judge_result_is_not_saved_as_a_zero_score(tmp_path, monkeypatch):
    monkeypatch.setenv("GEMINI_API_KEY", "offline-test")
    monkeypatch.setattr(config, "CONDITIONS", ("baseline",))
    monkeypatch.setattr(config, "BLOCKS", ("stationary",))
    monkeypatch.setattr(config, "N_EPISODES", 1)
    monkeypatch.setattr(config, "N_REPETITIONS", 1)
    monkeypatch.setattr(config, "N_TASKS", 1)
    monkeypatch.setattr(config, "EXPECTED_WORKER_CALLS", 1)
    monkeypatch.setattr(config, "EXPECTED_JUDGE_CALLS", 1)
    monkeypatch.setattr(run, "parse_args", lambda: Namespace(
        output_dir=str(tmp_path), plan_only=False, resume=False,
    ))
    monkeypatch.setattr(run.torch, "use_deterministic_algorithms", lambda enabled: None)
    task = _task()
    task = TaskSpec(
        id=0, kind=task.kind, embedding=task.embedding, difficulty=task.difficulty,
        prompt_template=task.prompt_template, reference_answer=task.reference_answer,
        rubric=task.rubric,
    )
    monkeypatch.setattr(run, "build_episode", lambda *args: type(
        "Episode", (), {"specs": [task], "inputs": ["4270"]}
    )())
    monkeypatch.setattr(run, "declared_dependency_pairs", lambda *args: [])
    monkeypatch.setattr(run, "make_initial_state", lambda *args: object())
    assignment = type("Assignment", (), {"X": torch.tensor([[1.0], [0.0], [0.0]])})()
    monkeypatch.setattr(run, "orchestrate", lambda *args: (assignment, object()))
    monkeypatch.setattr(run.judge, "preflight", lambda **kwargs: Generation(
        "2", config.JUDGE_MODEL, 1, 1, 0.01, finish_reason="STOP",
    ))
    monkeypatch.setattr(run, "generate", lambda *args, **kwargs: Generation(
        "4270", config.AGENT_MODELS["analyst"], 1, 1, 0.01,
    ))
    monkeypatch.setattr(run, "score", lambda *args, **kwargs: (
        None,
        Generation("* Data:", config.JUDGE_MODEL, 1, 0, 0.01, finish_reason="MAX_TOKENS"),
        "judge generation did not finish normally: finishReason=MAX_TOKENS",
    ))

    with pytest.raises(RuntimeError, match="Judge validation failed"):
        run.main()

    checkpoint = json.loads((tmp_path / "llm_poc_rewrite_checkpoint.json").read_text(encoding="utf-8"))
    assert checkpoint["rows"] == []
    assert checkpoint["pending"] is not None
    assert not (tmp_path / "llm_poc_rewrite.csv").exists()

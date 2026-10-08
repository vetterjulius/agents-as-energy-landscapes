"""Smoke tests for the LLM PoC (offline; mock workers, no API access)."""
from __future__ import annotations

import json
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
    assert base[base.task_id == 3].assigned_agent.iloc[0] != \
        en[en.task_id == 3].assigned_agent.iloc[0]


def test_adaptive_theta_drifts_when_stationary(mock_run):
    df, _ = mock_run
    st = df[(df.condition == "adaptive_energy") & (df.block == "stationary")]
    by_ep = st.groupby("episode").theta_diff_from_gt.mean()
    assert by_ep[2] > by_ep[0]


def test_adaptive_theta_tracks_shift_better_than_static(mock_run):
    df, _ = mock_run
    sh = df[(df.block == "shift_mid") & (df.episode == 2)]
    adaptive = sh[sh.condition == "adaptive_energy"].theta_diff_from_gt.mean()
    static = sh[sh.condition == "static_energy"].theta_diff_from_gt.mean()
    assert adaptive < static


def test_b0_uses_zero_energy_evaluations(mock_run):
    df, _ = mock_run
    assert (df[df.condition == "baseline"].solver_energy_evaluations == 0).all()
    assert (df[df.condition == "static_energy"].solver_energy_evaluations > 0).all()
    assert (df[df.condition == "adaptive_energy"].solver_energy_evaluations > 0).all()


def test_summary_report_and_provenance_written(mock_run):
    _, out = mock_run
    for name in ("poc_summary.json", "poc_report.md", "poc_judges.csv"):
        assert (out / name).exists()
    summary = json.loads((out / "poc_summary.json").read_text(encoding="utf-8"))
    assert summary["total_worker_calls"] == 540
    assert summary["scorer"] == "rubric_proxy"
    assert summary["total_judge_calls"] == 0
    assert summary["total_judge_tokens"] == 0
    assert summary["protocol"]["model_requested"] == "gemma-4-31b-it"
    assert summary["protocol"]["provider_required"] == "google-ai-studio"
    assert summary["protocol"]["planned_worker_calls"] == 540
    assert "source_sha256" in summary["environment"]
    judge_rows = pd.read_csv(out / "poc_judges.csv")
    assert len(judge_rows) == 540


def test_live_protocol_is_direct_gemini_pilot_without_judge():
    from llm_poc.config import DEFAULT_MODEL, MAX_LIVE_API_CALLS, PoCConfig

    assert DEFAULT_MODEL == "gemma-4-31b-it"
    assert MAX_LIVE_API_CALLS == 60
    from llm_poc import config
    assert config.MIN_REQUEST_INTERVAL_SEC >= 60 / 30
    assert config.MAX_RETRIES == 3
    assert config.API_TIMEOUT_SEC == 120.0
    assert config.MAX_RESUME_EXTRA_ATTEMPTS == 2
    assert config.MAX_LIVE_API_CALLS == 60
    assert config.RETRY_BASE_SEC == 10.0
    assert config.RETRY_MAX_SEC == 60.0
    cfg = PoCConfig(worker_mode="gemini")
    assert cfg.judge_enabled is False
    assert cfg.api_key_env == "GEMINI_API_KEY"
    with pytest.raises(ValueError, match="locked"):
        PoCConfig(worker_mode="gemini", model="gemma-4-26b-a4b-it")
    with pytest.raises(ValueError, match="disables LLM judging"):
        PoCConfig(worker_mode="gemini", judge_enabled=True)
    with pytest.raises(ValueError, match="at most 3 retries"):
        PoCConfig(worker_mode="gemini", max_retries=4)


def test_request_seed_is_stable_matched_and_role_specific():
    from llm_poc.workers import derive_request_seed

    a = derive_request_seed(123, "stationary", 1, 2, 3, "worker")
    assert a == derive_request_seed(123, "stationary", 1, 2, 3, "worker")
    assert a == derive_request_seed(123, "stationary", 1, 2, 3, "worker")
    assert a != derive_request_seed(123, "shift_mid", 1, 2, 3, "worker")
    assert a != derive_request_seed(123, "stationary", 1, 2, 3, "judge")
    assert 0 <= a < 2**31


def test_call_gemini_uses_direct_api_payload_and_verifies_model(monkeypatch):
    from llm_poc import config, workers

    captured = {}

    class FakeResponse:
        def __enter__(self):
            return self

        def __exit__(self, *args):
            return False

        def read(self):
            return json.dumps({
                "modelVersion": config.DEFAULT_MODEL,
                "candidates": [{"content": {"parts": [{"text": "ok"}]},
                                "finishReason": "STOP"}],
                "usageMetadata": {"promptTokenCount": 3, "candidatesTokenCount": 1,
                                  "totalTokenCount": 4},
            }).encode()

    def fake_urlopen(req, timeout):
        captured["url"] = req.full_url
        captured["payload"] = json.loads(req.data)
        captured["headers"] = dict(req.header_items())
        return FakeResponse()

    monkeypatch.setattr(workers.urllib.request, "urlopen", fake_urlopen)
    monkeypatch.setattr(workers.time, "sleep", lambda _: None)
    text, usage, _, _, _, error = workers.call_gemini(
        "prompt", "system", config.DEFAULT_MODEL, "test-key", seed=41,
        max_tokens=20, top_p=1.0)
    assert text == "ok" and error is None
    assert usage.provider == "google-ai-studio"
    assert usage.model == config.DEFAULT_MODEL
    assert usage.endpoint_model == config.DEFAULT_MODEL
    assert captured["url"] == config.GEMINI_API_URL.format(model=config.DEFAULT_MODEL)
    assert captured["payload"]["contents"][0]["parts"][0]["text"] == "prompt"
    assert captured["payload"]["systemInstruction"]["parts"][0]["text"] == "system"
    assert captured["payload"]["generationConfig"]["seed"] == 41
    assert captured["payload"]["generationConfig"]["maxOutputTokens"] == 20
    assert captured["headers"]["X-goog-api-key"] == "test-key"


def test_live_progress_journal_records_completion_without_api_key(tmp_path, monkeypatch):
    from llm_poc import config, workers

    class FakeResponse:
        def __enter__(self):
            return self

        def __exit__(self, *args):
            return False

        def read(self):
            return json.dumps({
            "modelVersion": config.DEFAULT_MODEL,
            "candidates": [{"content": {"parts": [{"text": "private output"}]},
                                "finishReason": "STOP"}],
                "usageMetadata": {"promptTokenCount": 4, "candidatesTokenCount": 2,
                                  "totalTokenCount": 6},
            }).encode()

    monkeypatch.setattr(workers.urllib.request, "urlopen", lambda req, timeout: FakeResponse())
    monkeypatch.setattr(workers.time, "sleep", lambda _: None)
    path = tmp_path / "progress.jsonl"
    workers.set_progress_log(path)
    workers.set_progress_context(condition="baseline", task_id=2)
    try:
        result = workers.call_gemini(
            "prompt", "system", config.DEFAULT_MODEL, "do-not-log-secret", max_retries=0)
    finally:
        workers.set_progress_log(None)
        workers.set_progress_context()

    assert result[0] == "private output"
    records = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]
    assert [record["event"] for record in records] == ["request_started", "request_completed"]
    assert records[-1]["prompt_tokens"] == 4
    assert records[-1]["context"] == {"condition": "baseline", "task_id": 2}
    assert "do-not-log-secret" not in path.read_text(encoding="utf-8")


def test_call_gemini_retries_transient_failure_and_logs_each_attempt(tmp_path, monkeypatch):
    from llm_poc import config, workers

    attempts = []
    class FakeResponse:
        def __enter__(self): return self
        def __exit__(self, *args): return False
        def read(self):
            return json.dumps({
                "modelVersion": config.DEFAULT_MODEL,
                "candidates": [{"content": {"parts": [{"text": "recovered"}]},
                                "finishReason": "STOP"}],
                "usageMetadata": {"promptTokenCount": 3, "candidatesTokenCount": 1,
                                  "totalTokenCount": 4},
            }).encode()

    def fake_urlopen(req, timeout):
        attempts.append(1)
        if len(attempts) == 1:
            raise workers.urllib.error.HTTPError(
                req.full_url, 500, "Internal Server Error", {}, None)
        return FakeResponse()

    sleeps = []
    monkeypatch.setattr(workers.urllib.request, "urlopen", fake_urlopen)
    monkeypatch.setattr(workers.time, "sleep", lambda delay: sleeps.append(delay))
    monkeypatch.setattr(workers.random, "uniform", lambda low, high: low)
    path = tmp_path / "retry.jsonl"
    workers.set_progress_log(path)
    try:
        text, usage, _, _, retries, error = workers.call_gemini(
            "prompt", "system", config.DEFAULT_MODEL, "secret", max_retries=1)
    finally:
        workers.set_progress_log(None)

    records = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]
    assert text == "recovered" and error is None and usage.model == config.DEFAULT_MODEL
    assert retries == 1 and len(attempts) == 2
    assert [r["event"] for r in records].count("request_failed") == 1
    assert [r["event"] for r in records].count("request_completed") == 1
    assert [r["event"] for r in records].count("retry_scheduled") == 1
    assert "secret" not in path.read_text(encoding="utf-8")
    assert any(delay == pytest.approx(8.0) for delay in sleeps)


def test_call_gemini_exponential_backoff_and_bounded_attempts(monkeypatch):
    from llm_poc import config, workers

    attempts = []
    sleeps = []

    class FakeResponse:
        def __enter__(self): return self
        def __exit__(self, *args): return False
        def read(self):
            return json.dumps({
                "modelVersion": config.DEFAULT_MODEL,
                "candidates": [{"content": {"parts": [{"text": "recovered"}]}}],
            }).encode()

    def fake_urlopen(req, timeout):
        attempts.append(1)
        if len(attempts) < 4:
            raise workers.urllib.error.HTTPError(
                req.full_url, 503, "Unavailable", {}, None)
        return FakeResponse()

    monkeypatch.setattr(workers.urllib.request, "urlopen", fake_urlopen)
    monkeypatch.setattr(workers.time, "sleep", lambda delay: sleeps.append(delay))
    monkeypatch.setattr(workers.time, "monotonic", lambda: 1_000.0)
    monkeypatch.setattr(workers.random, "uniform", lambda low, high: 1.0)
    workers._LAST_REQUEST_AT = 0.0
    text, _, _, _, retries, error = workers.call_gemini(
        "prompt", "system", config.DEFAULT_MODEL, "secret")

    assert text == "recovered" and error is None
    assert retries == 3 and len(attempts) == 4
    assert [delay for delay in sleeps if delay >= 10.0] == [10.0, 20.0, 40.0]


def test_retry_after_is_honored_and_bounded(monkeypatch):
    from email.message import Message
    from llm_poc import config, workers

    sleeps = []
    attempts = []

    def fail_once(req, timeout):
        attempts.append(1)
        if len(attempts) == 1:
            headers = Message()
            headers["Retry-After"] = "25"
            raise workers.urllib.error.HTTPError(
                req.full_url, 503, "Unavailable", headers, None)
        return type("Response", (), {
            "__enter__": lambda self: self,
            "__exit__": lambda self, *args: False,
            "read": lambda self: json.dumps({
                "modelVersion": config.DEFAULT_MODEL,
                "candidates": [{"content": {"parts": [{"text": "ok"}]}}],
            }).encode(),
        })()

    monkeypatch.setattr(workers.urllib.request, "urlopen", fail_once)
    monkeypatch.setattr(workers.time, "sleep", lambda delay: sleeps.append(delay))
    monkeypatch.setattr(workers.time, "monotonic", lambda: 1_000.0)
    monkeypatch.setattr(workers.random, "uniform", lambda low, high: 1.0)
    workers._LAST_REQUEST_AT = 0.0
    text, _, _, _, retries, error = workers.call_gemini(
        "retry-after", "system", config.DEFAULT_MODEL, "secret", max_retries=1)
    assert text == "ok" and retries == 1 and error is None
    assert 25.0 in sleeps

    headers = Message()
    headers["Retry-After"] = "900"
    attempts.clear()
    def fail_too_long(req, timeout):
        attempts.append(1)
        raise workers.urllib.error.HTTPError(
            req.full_url, 503, "Unavailable", headers, None)
    monkeypatch.setattr(workers.urllib.request, "urlopen", fail_too_long)
    text, _, _, _, retries, error = workers.call_gemini(
        "long-retry-after", "system", config.DEFAULT_MODEL, "secret", max_retries=1)
    assert text is None and retries == 0 and len(attempts) == 1
    assert "exceeds configured bound" in error


def test_resume_extends_only_exhausted_transient_task_by_two_attempts(tmp_path, monkeypatch):
    from llm_poc import config, workers

    attempts = []
    def fail_urlopen(req, timeout):
        attempts.append(timeout)
        raise workers.urllib.error.HTTPError(
            req.full_url, 500, "Internal Server Error", {}, None)

    monkeypatch.setattr(workers.urllib.request, "urlopen", fail_urlopen)
    monkeypatch.setattr(workers.time, "sleep", lambda delay: None)
    monkeypatch.setattr(workers.random, "uniform", lambda low, high: 1.0)
    path = tmp_path / "resume-attempts.jsonl"
    workers.set_progress_context(condition="baseline", block="stationary", episode=0,
                                repetition=0, task_id=6)
    workers.set_progress_log(path)
    original = workers.call_gemini(
        "resume prompt", "system", config.DEFAULT_MODEL, "secret", timeout=120)
    assert original[0] is None and len(attempts) == config.MAX_LIVE_REQUEST_ATTEMPTS
    workers.set_progress_log(path, resume=True)
    resumed = workers.call_gemini(
        "resume prompt", "system", config.DEFAULT_MODEL, "secret", timeout=120)
    assert resumed[0] is None
    assert "http_500" in resumed[-1]
    assert len(attempts) == config.MAX_LIVE_REQUEST_ATTEMPTS + config.MAX_RESUME_EXTRA_ATTEMPTS
    assert all(timeout == 120 for timeout in attempts)
    records = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]
    assert sum(record["event"] == "request_started" for record in records) == len(attempts)
    assert sum(record["event"] == "run_resumed" for record in records) == 1
    assert sum(record["event"] == "retry_scheduled" and record.get("resumed")
               for record in records) >= 1
    workers.set_progress_log(None)
    workers.set_progress_context()


def test_resume_does_not_extend_nonretryable_failures(tmp_path, monkeypatch):
    from llm_poc import config, workers

    attempts = []
    failure_code = {"value": 500}
    def fail_urlopen(req, timeout):
        attempts.append(failure_code["value"])
        raise workers.urllib.error.HTTPError(
            req.full_url, failure_code["value"], "failure", {}, None)

    monkeypatch.setattr(workers.urllib.request, "urlopen", fail_urlopen)
    monkeypatch.setattr(workers.time, "sleep", lambda delay: None)
    monkeypatch.setattr(workers.random, "uniform", lambda low, high: 1.0)
    path = tmp_path / "nonretry.jsonl"
    workers.set_progress_context(condition="baseline", block="stationary", episode=0,
                                repetition=0, task_id=1)
    workers.set_progress_log(path)
    exhausted = workers.call_gemini("p", "s", config.DEFAULT_MODEL, "secret")
    assert exhausted[0] is None and len(attempts) == config.MAX_LIVE_REQUEST_ATTEMPTS

    failure_code["value"] = 429
    workers.set_progress_log(path, resume=True)
    nonretryable = workers.call_gemini("p", "s", config.DEFAULT_MODEL, "secret")
    assert nonretryable[0] is None and "http_429" in nonretryable[-1]
    workers.set_progress_log(path, resume=True)
    stopped = workers.call_gemini("p", "s", config.DEFAULT_MODEL, "secret")
    assert stopped[0] is None and "non-retryable" in stopped[-1]
    assert attempts == [500] * config.MAX_LIVE_REQUEST_ATTEMPTS + [429]
    workers.set_progress_log(None)
    workers.set_progress_context()


def test_resume_attempt_cap_includes_historical_attempts(tmp_path, monkeypatch):
    from llm_poc import config, workers

    calls = []
    monkeypatch.setattr(workers.urllib.request, "urlopen",
                        lambda req, timeout: calls.append(1))
    monkeypatch.setattr(workers.time, "sleep", lambda _: None)
    path = tmp_path / "global-cap.jsonl"
    path.write_text("".join(json.dumps({"event": "request_started", "request_key": f"k{i}"}) + "\n"
                              for i in range(config.MAX_LIVE_API_CALLS)), encoding="utf-8")
    workers.set_progress_log(path, resume=True, max_attempts=config.MAX_LIVE_API_CALLS)
    workers.set_progress_context(condition="new", task_id=1)
    result = workers.call_gemini(
        "cap prompt", "system", config.DEFAULT_MODEL, "secret", max_retries=0)
    assert result[0] is None and "attempt cap reached" in result[-1]
    assert calls == []
    workers.set_progress_log(None)
    workers.set_progress_context()


def test_completed_live_request_is_reused_after_journal_reload(tmp_path, monkeypatch):
    from llm_poc import config, workers

    calls = []
    class FakeResponse:
        def __enter__(self): return self
        def __exit__(self, *args): return False
        def read(self):
            return json.dumps({
                "modelVersion": config.DEFAULT_MODEL,
                "candidates": [{"content": {"parts": [{"text": "cached"}]}}],
                "usageMetadata": {"totalTokenCount": 2},
            }).encode()

    monkeypatch.setattr(workers.urllib.request, "urlopen",
                        lambda req, timeout: (calls.append(1) or FakeResponse()))
    monkeypatch.setattr(workers.time, "sleep", lambda _: None)
    path = tmp_path / "resume.jsonl"
    workers.set_progress_log(path)
    workers.set_progress_context(condition="baseline", task_id=3)
    first = workers.call_gemini("same prompt", "system", config.DEFAULT_MODEL, "secret")
    workers.set_progress_log(path)
    workers.set_progress_context(condition="baseline", task_id=3)
    second = workers.call_gemini("same prompt", "system", config.DEFAULT_MODEL, "secret")
    assert first[0] == second[0] == "cached"
    assert len(calls) == 1
    assert [json.loads(line)["event"] for line in path.read_text().splitlines()][-1] == "request_reused"
    workers.set_progress_log(None)
    workers.set_progress_context()


def test_resume_rejects_changed_request_for_a_started_task(tmp_path, monkeypatch):
    from llm_poc import config, workers

    calls = []
    class FakeResponse:
        def __enter__(self): return self
        def __exit__(self, *args): return False
        def read(self):
            return json.dumps({
                "modelVersion": config.DEFAULT_MODEL,
                "candidates": [{"content": {"parts": [{"text": "ok"}]}}],
            }).encode()

    monkeypatch.setattr(workers.urllib.request, "urlopen",
                        lambda req, timeout: (calls.append(1) or FakeResponse()))
    monkeypatch.setattr(workers.time, "sleep", lambda _: None)
    path = tmp_path / "mismatch.jsonl"
    workers.set_progress_context(condition="baseline", block="stationary", episode=0,
                                repetition=0, task_id=2)
    workers.set_progress_log(path)
    workers.call_gemini("prompt-v1", "system", config.DEFAULT_MODEL,
                        "secret", max_retries=0)
    workers.set_progress_log(path, resume=True)
    workers.set_progress_context(condition="baseline", block="stationary", episode=0,
                                repetition=0, task_id=2)
    result = workers.call_gemini("prompt-v2", "system", config.DEFAULT_MODEL,
                                 "secret", max_retries=0)
    assert result[0] is None and "resume_request_mismatch" in result[-1]
    assert calls == [1]
    workers.set_progress_log(None)
    workers.set_progress_context()


def test_live_http_attempt_cap_stops_before_network(monkeypatch, tmp_path):
    from llm_poc import config, workers

    calls = []
    monkeypatch.setattr(workers.urllib.request, "urlopen",
                        lambda req, timeout: calls.append(1))
    monkeypatch.setattr(workers.time, "sleep", lambda _: None)
    path = tmp_path / "cap.jsonl"
    path.write_text("".join(json.dumps({"event": "request_started", "request_key": f"r{i}"}) + "\n"
                              for i in range(config.MAX_LIVE_API_CALLS)), encoding="utf-8")
    workers.set_progress_log(path)
    result = workers.call_gemini(
        "cap", "system", config.DEFAULT_MODEL, "secret", max_retries=0)
    assert result[0] is None and "attempt cap reached" in result[-1]
    assert calls == []
    workers.set_progress_log(None)


def test_live_run_manifest_allows_only_matching_resume(tmp_path):
    from llm_poc.run import prepare_live_run

    output_dir = tmp_path / "live"
    manifest = {"model": "pinned", "seed": 42, "attempt_cap": 60}
    prepare_live_run(output_dir, manifest)
    (output_dir / "poc_live_progress.jsonl").write_text("", encoding="utf-8")
    prepare_live_run(output_dir, manifest)
    prepare_live_run(output_dir, manifest)
    with pytest.raises(SystemExit, match="changed identity fields"):
        prepare_live_run(output_dir, {"model": "other", "seed": 42,
                                     "attempt_cap": 60}, resume=True)
    prior = prepare_live_run(output_dir, {"model": "pinned", "seed": 42,
                                         "attempt_cap": 60}, resume=True)
    assert prior["attempt_cap"] == 60
    with pytest.raises(SystemExit, match="cannot increase an existing run's cap"):
        prepare_live_run(output_dir, {"model": "pinned", "seed": 42,
                                     "attempt_cap": 61}, resume=True)

    blocked_dir = tmp_path / "not-resumable"
    blocked_dir.mkdir()
    (blocked_dir / "poc_episodes.csv").write_text("existing", encoding="utf-8")
    with pytest.raises(SystemExit, match="non-resumable artifacts"):
        prepare_live_run(blocked_dir, manifest)


def test_call_gemini_does_not_retry_quota_or_client_errors(monkeypatch):
    from llm_poc import config, workers

    calls = []
    def fail_urlopen(req, timeout):
        calls.append(1)
        raise workers.urllib.error.HTTPError(req.full_url, 429, "Too Many Requests", {}, None)

    monkeypatch.setattr(workers.urllib.request, "urlopen", fail_urlopen)
    monkeypatch.setattr(workers.time, "sleep", lambda _: None)
    text, _, _, _, retries, error = workers.call_gemini(
        "prompt", "system", config.DEFAULT_MODEL, "secret", max_retries=1)
    assert text is None and retries == 0 and "http_429" in error
    assert len(calls) == 1


def test_call_gemini_redacts_errors_and_fails_closed(monkeypatch):
    from llm_poc import config, workers

    def fail_urlopen(req, timeout):
        raise RuntimeError("reflected-secret")

    monkeypatch.setattr(workers.urllib.request, "urlopen", fail_urlopen)
    monkeypatch.setattr(workers.time, "sleep", lambda _: None)
    text, _, _, _, retries, error = workers.call_gemini(
        "prompt", "system", config.DEFAULT_MODEL, "reflected-secret")
    assert text is None and retries == 0
    assert "reflected-secret" not in error
    assert "[REDACTED]" in error


def test_live_worker_failure_stops_episode_after_first_failed_call(monkeypatch):
    from llm_poc import config, workers
    from llm_poc.tasks import build_agent_slots, build_episode

    attempts = []
    def fail_call(*args, **kwargs):
        attempts.append(1)
        return None, workers.Usage(), 0.0, 0.0, 0, "http_429: quota exceeded"

    monkeypatch.setattr(workers, "call_openrouter", fail_call)
    cfg = config.PoCConfig(worker_mode="gemini", judge_enabled=False)
    with pytest.raises(workers.LiveWorkerFailure, match="http_429"):
        workers.execute_episode(
            build_episode("stationary", 0), build_agent_slots(), torch.eye(10, 3).T,
            cfg, repetition=0, episode_seed=7)
    assert len(attempts) == 1


def test_outputs_and_scores_remain_aligned_after_dependency_ordering(monkeypatch):
    from llm_poc import config, workers
    from llm_poc.tasks import build_agent_slots, build_episode

    episode = build_episode("stationary", 0)
    assignments = torch.zeros(3, 10)
    for task_id in range(10):
        assignments[task_id % 3, task_id] = 1.0

    def fake_call(prompt, system_prompt, model, api_key, **kwargs):
        return f"answer-for-{prompt[:8]}", workers.Usage(), 0.0, 0.0, 0, None

    monkeypatch.setattr(workers, "call_openrouter", fake_call)
    cfg = config.PoCConfig(worker_mode="gemini", judge_enabled=False)
    execution = workers.execute_episode(
        episode, build_agent_slots(), assignments, cfg, repetition=0, episode_seed=7)

    assert [out.task_id for out in execution.outputs] == list(range(10))
    assert [score.task_id for score in execution.scores] == list(range(10))
    expected_agents = assignments.argmax(dim=0).tolist()
    assert [out.agent_id for out in execution.outputs] == expected_agents
    assert [score.agent_id for score in execution.scores] == expected_agents
    assert len({score.rubric_score for score in execution.scores}) == 1
    assert [out.usage.model for out in execution.outputs] == [""] * 10


def test_judge_usage_and_failures_are_counted(monkeypatch):
    from llm_poc import config, workers
    from llm_poc.tasks import build_agent_slots, build_episode

    episode = build_episode("stationary", 0)
    assignments = torch.zeros(3, 10)
    for task_id in range(10):
        assignments[task_id % 3, task_id] = 1.0

    def fake_call(prompt, system_prompt, model, api_key, **kwargs):
        return None, workers.Usage(prompt_tokens=7, completion_tokens=2), 0, 0, 0, "offline fake failure"

    monkeypatch.setattr(workers, "call_openrouter", fake_call)
    cfg = config.PoCConfig(worker_mode="mock", judge_enabled=True)
    execution = workers.execute_episode(
        episode, build_agent_slots(), assignments, cfg, repetition=0, episode_seed=7)
    assert execution.judge_calls == 10
    assert execution.judge_failures == 10
    assert execution.judge_usage.prompt_tokens == 70
    assert execution.judge_usage.completion_tokens == 20
    assert all(score.judge_error == "offline fake failure" for score in execution.scores)
    assert all(score.rubric_score == 0 for score in execution.scores)


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

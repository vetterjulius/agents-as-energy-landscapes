"""Validation and export tests for the repo-free text-only transfer pilot."""
from __future__ import annotations

import copy
import hashlib
import json
from pathlib import Path

import pytest

from llm_poc_rewrite import config, transfer_data as transfer

PACKAGE = Path(__file__).parents[1] / "llm_poc_rewrite" / "data" / "real_issue_transfer" / "v2"


def _read_tasks() -> list[dict]:
    return [json.loads(line) for line in (PACKAGE / "tasks.jsonl").read_text(encoding="utf-8").splitlines() if line]


def _write_package(tmp_path: Path, *, tasks: list[dict] | None = None, study: dict | None = None,
                   traces: list[dict] | None = None, rubrics: list[dict] | None = None) -> Path:
    root = tmp_path / "package"
    root.mkdir(parents=True, exist_ok=True)
    study_data = json.loads((PACKAGE / "study.json").read_text(encoding="utf-8")) if study is None else study
    task_data = _read_tasks() if tasks is None else tasks
    (root / "study.json").write_text(json.dumps(study_data), encoding="utf-8")
    (root / "tasks.jsonl").write_text("".join(json.dumps(row) + "\n" for row in task_data), encoding="utf-8")
    (root / "traces.jsonl").write_text("".join(json.dumps(row) + "\n" for row in (traces or [])), encoding="utf-8")
    (root / "rubrics.jsonl").write_text("".join(json.dumps(row) + "\n" for row in (rubrics or [])), encoding="utf-8")
    return root


def _make_trace(task: dict, *, repetition: int = 0, role: str = "analyst", model: str | None = None) -> dict:
    timestamp = "2026-10-17T12:00:00Z"
    prompt_hash = task["task_packet"]["prompt_sha256"]
    model = config.AGENT_MODELS[role] if model is None else model
    response = "Observed symptom summarized; diagnostic hypothesis clearly qualified; tests proposed."
    stage = {
        "role": role, "model_id": model, "model_version": f"{model}-2026-10",
        "system_prompt_sha256": hashlib.sha256(config.AGENT_SYSTEM_PROMPTS[role].encode("utf-8")).hexdigest(),
        "prompt_sha256": prompt_hash, "input_task_ids": [task["task_id"]],
        "output_text": response, "started_at_utc": timestamp, "ended_at_utc": timestamp,
        "latency_seconds": 0.5, "tokens_in": 200, "tokens_out": 40, "cost_usd": 0.001,
        "finish_reason": "STOP", "error_category": None,
    }
    return {
        "schema_version": "2.0", "trace_id": f"trace-{task['source']['issue_number']}-{role}-{repetition}",
        "task_id": task["task_id"], "condition": "independent_configured_agents", "repetition": repetition,
        "started_at_utc": timestamp, "ended_at_utc": timestamp, "stages": [stage],
        "final_response": response, "final_response_sha256": transfer._v2_hash(response),
        "api_requests": 1, "total_latency_seconds": 0.5, "total_tokens_in": 200,
        "total_tokens_out": 40, "total_cost_usd": 0.001, "termination_status": "completed",
    }


def _make_ready_package(tmp_path: Path) -> tuple[Path, list[dict], list[dict]]:
    tasks = _read_tasks()
    for task in tasks:
        task["eligibility"] = {"decision": "include_with_leakage_flag", "reviewers": ["reviewer-a", "reviewer-b"],
                               "reason": "Independently adjudicated as a text-only candidate; comments contain leakage."}
    study = json.loads((PACKAGE / "study.json").read_text(encoding="utf-8"))
    study["status"] = "ready"
    study["cohort"]["selected_tasks"] = len(tasks)
    study["agent_protocol"]["models"] = dict(config.AGENT_MODELS)
    traces = [
        _make_trace(task, role=role, repetition=rep)
        for task in tasks
        for role in study["agent_protocol"]["roles"]
        for rep in range(study["agent_protocol"]["repetitions"])
    ]
    root = _write_package(tmp_path, tasks=tasks, study=study, traces=traces)
    return root, tasks, traces


def test_checked_in_v2_package_is_valid_draft_but_not_ready():
    report = transfer.validate_v2_package(PACKAGE)
    assert report["valid"]
    assert not report["ready"]
    assert not report["complete"]
    assert report["task_records"] == 2
    assert report["eligible_tasks"] == 0
    assert report["response_count"] == 0
    assert report["expected_response_count"] == 0
    assert "study is still draft" in report["blockers"]
    assert any("eligibility pending" in blocker for blocker in report["blockers"])


def test_context_and_prompt_hashes_are_verified(tmp_path):
    tasks = _read_tasks()
    tasks[0]["task_packet"]["context_docs"][0]["content_text"] += " Tampered."
    with pytest.raises(transfer.PackageError, match="context document content hash mismatch"):
        transfer.validate_v2_package(_write_package(tmp_path, tasks=tasks))

    tasks = _read_tasks()
    tasks[0]["task_packet"]["issue_text"] += " Tampered."
    with pytest.raises(transfer.PackageError, match="prompt_sha256 does not match task packet"):
        transfer.validate_v2_package(_write_package(tmp_path, tasks=tasks))


def test_source_issue_url_cannot_disagree_with_identity(tmp_path):
    tasks = _read_tasks()
    tasks[0]["source"]["url"] = "https://github.com/pydantic/pydantic/issues/1"
    with pytest.raises(transfer.PackageError, match="source URL must match repository and issue identity"):
        transfer.validate_v2_package(_write_package(tmp_path, tasks=tasks))


def test_material_public_solution_or_causal_disclosure_requires_leakage_flag(tmp_path):
    tasks = _read_tasks()
    tasks[0]["eligibility"] = {"decision": "include", "reviewers": ["reviewer-a", "reviewer-b"], "reason": "Reviewed."}
    with pytest.raises(transfer.PackageError, match="disclosure requires leakage flag"):
        transfer.validate_v2_package(_write_package(tmp_path, tasks=tasks))


def test_protocol_requires_every_agent_and_exact_prompt_from_existing_config(tmp_path):
    report = transfer.validate_v2_package(PACKAGE)
    study = json.loads((PACKAGE / "study.json").read_text(encoding="utf-8"))
    protocol = study["agent_protocol"]
    assert protocol["roles"] == list(config.AGENT_MODELS)
    assert protocol["models"] == config.AGENT_MODELS
    assert protocol["system_prompt_sha256"] == {
        role: hashlib.sha256(prompt.encode("utf-8")).hexdigest()
        for role, prompt in config.AGENT_SYSTEM_PROMPTS.items()
    }
    assert protocol["temperature"] == config.WORKER_TEMPERATURE
    assert protocol["max_output_tokens_per_call"] == config.WORKER_MAX_TOKENS
    assert report["valid"]

    protocol["models"] = {"analyst": config.AGENT_MODELS["analyst"]}
    with pytest.raises(transfer.PackageError, match="must exactly match llm_poc_rewrite.config.AGENT_MODELS"):
        transfer.validate_v2_package(_write_package(tmp_path, study=study))

    protocol["models"] = dict(config.AGENT_MODELS)
    protocol["system_prompt_sha256"]["analyst"] = "0" * 64
    with pytest.raises(transfer.PackageError, match="system prompt hashes must match"):
        transfer.validate_v2_package(_write_package(tmp_path / "bad-prompt", study=study))


def test_intermediate_running_package_can_be_partial_but_complete_requires_full_design(tmp_path):
    tasks = _read_tasks()
    for task in tasks:
        task["eligibility"] = {"decision": "include_with_leakage_flag", "reviewers": ["reviewer-a", "reviewer-b"],
                               "reason": "Independent review completed; public hints are recorded."}
    study = json.loads((PACKAGE / "study.json").read_text(encoding="utf-8"))
    study["status"] = "running"
    study["cohort"]["selected_tasks"] = len(tasks)
    study["agent_protocol"]["models"] = dict(config.AGENT_MODELS)
    first_trace = _make_trace(tasks[0])
    partial = _write_package(tmp_path / "partial", tasks=tasks, study=study, traces=[first_trace])
    report = transfer.validate_v2_package(partial)
    assert report["valid"] and not report["complete"]
    assert report["trace_blockers"]
    assert any("running study is missing scheduled response" in blocker for blocker in report["blockers"])

    study["status"] = "complete"
    complete_candidate = _write_package(tmp_path / "incomplete", tasks=tasks, study=study, traces=[first_trace])
    report = transfer.validate_v2_package(complete_candidate)
    assert report["valid"] and not report["complete"]
    assert any("missing configured-agent" in blocker for blocker in report["blockers"])


def test_ready_requires_real_named_reviewers_and_pinned_model(tmp_path):
    tasks = _read_tasks()
    study = json.loads((PACKAGE / "study.json").read_text(encoding="utf-8"))
    study["status"] = "ready"
    study["cohort"]["selected_tasks"] = 2
    study["agent_protocol"]["models"] = dict(config.AGENT_MODELS)
    for task in tasks:
        task["eligibility"] = {"decision": "include_with_leakage_flag", "reviewers": ["pending-reviewer-a", "pending-reviewer-b"],
                               "reason": "Candidate reviewers are not actual adjudicators."}
    report = transfer.validate_v2_package(_write_package(tmp_path, tasks=tasks, study=study))
    assert not report["ready"]
    assert any("placeholder reviewer IDs" in blocker for blocker in report["blockers"])


def test_ready_export_contains_prompt_material_only_and_no_evaluator_metadata(tmp_path):
    root, tasks, _ = _make_ready_package(tmp_path)
    report = transfer.validate_v2_package(root)
    assert report["ready"]
    packets = transfer.export_v2_worker_packets(root)
    assert len(packets) == 2
    assert all(set(packet) == {"title", "issue_text", "acceptance_criteria", "context_docs", "transformation_note", "prompt_sha256"} for packet in packets)
    assert all("https://github.com/pydantic/pydantic/issues/" not in json.dumps(packet) for packet in packets)
    serialized = json.dumps(packets)
    for forbidden in ("disclosure", "eligibility", "reviewer_note", "linked_fix_ref", "issue_number", "trace_id"):
        assert forbidden not in serialized
    assert {task["task_id"] for task in tasks} == {task["task_id"] for task in _read_tasks()}


def test_configured_agent_traces_and_human_rubrics_complete(tmp_path):
    root, tasks, traces = _make_ready_package(tmp_path)
    trace = traces[0]
    rubric = {
        "schema_version": "2.0", "rating_id": "rating-1", "trace_id": trace["trace_id"],
        "rater_id": "human-rater-a", "blind_id": f"anon-{trace['trace_id']}",
        "scores": {dimension: 3 for dimension in transfer.PILOT_V2_RUBRIC_DIMENSIONS},
        "overall_note": "Useful and appropriately qualified.", "rated_at_utc": "2026-10-17T13:00:00Z",
    }
    rubrics = []
    for trace_index, response_trace in enumerate(traces, start=1):
        for rater_index, rater_id in enumerate(("human-rater-a", "human-rater-b"), start=1):
            rubrics.append({**rubric, "rating_id": f"rating-{trace_index}-{rater_index}",
                            "trace_id": response_trace["trace_id"], "rater_id": rater_id,
                            "blind_id": f"blind-{trace_index}"})
    study = json.loads((root / "study.json").read_text(encoding="utf-8"))
    study["status"] = "running"
    partially_scored = _write_package(tmp_path / "partially-scored", tasks=tasks, study=study,
                                     traces=traces, rubrics=rubrics[:2])
    report = transfer.validate_v2_package(partially_scored)
    assert report["valid"] and not report["complete"]
    assert report["unrated_trace_ids"]

    study["status"] = "complete"
    study["cohort"]["selected_tasks"] = len(tasks)
    study["agent_protocol"]["models"] = dict(config.AGENT_MODELS)
    root = _write_package(tmp_path / "copy", tasks=tasks, study=study, traces=traces, rubrics=rubrics)
    report = transfer.validate_v2_package(root)
    assert report["valid"] and report["ready"] and report["complete"]
    assert report["trace_records"] == 18
    assert report["rubric_records"] == 36


def test_trace_rejects_multiple_stages_and_bad_response_hash(tmp_path):
    root, tasks, traces = _make_ready_package(tmp_path)
    trace = copy.deepcopy(traces[0])
    trace["stages"].append(copy.deepcopy(trace["stages"][0]))
    with pytest.raises(transfer.PackageError, match="exactly one response record"):
        _write_package(tmp_path / "multiple", tasks=tasks, study=json.loads((root / "study.json").read_text()), traces=[trace])
        transfer.validate_v2_package(tmp_path / "multiple" / "package")

    bad = copy.deepcopy(traces[0])
    bad["final_response_sha256"] = "0" * 64
    with pytest.raises(transfer.PackageError, match="final response hash mismatch"):
        package = _write_package(tmp_path / "hash", tasks=tasks, study=json.loads((root / "study.json").read_text()), traces=[bad])
        transfer.validate_v2_package(package)


def test_trace_rejects_unpinned_model_and_mismatched_prompt(tmp_path):
    root, tasks, traces = _make_ready_package(tmp_path)
    study = json.loads((root / "study.json").read_text(encoding="utf-8"))

    wrong_model = copy.deepcopy(traces[0])
    wrong_model["stages"][0]["model_id"] = "other-model"
    with pytest.raises(transfer.PackageError, match="differs from config.AGENT_MODELS"):
        package = _write_package(tmp_path / "wrong-model", tasks=tasks, study=study, traces=[wrong_model])
        transfer.validate_v2_package(package)

    wrong_role_model = copy.deepcopy(traces[0])
    wrong_role_model["stages"][0]["role"] = "extractor"
    with pytest.raises(transfer.PackageError, match="differs from config.AGENT_MODELS"):
        package = _write_package(tmp_path / "wrong-role-model", tasks=tasks, study=study, traces=[wrong_role_model])
        transfer.validate_v2_package(package)

    wrong_system_prompt = copy.deepcopy(traces[0])
    wrong_system_prompt["stages"][0]["system_prompt_sha256"] = "0" * 64
    with pytest.raises(transfer.PackageError, match="system_prompt_sha256 differs from config.AGENT_SYSTEM_PROMPTS"):
        package = _write_package(tmp_path / "wrong-system-prompt", tasks=tasks, study=study, traces=[wrong_system_prompt])
        transfer.validate_v2_package(package)

    wrong_prompt = copy.deepcopy(traces[0])
    wrong_prompt["stages"][0]["prompt_sha256"] = "f" * 64
    with pytest.raises(transfer.PackageError, match="prompt hash differs from the frozen worker packet"):
        package = _write_package(tmp_path / "wrong-prompt", tasks=tasks, study=study, traces=[wrong_prompt])
        transfer.validate_v2_package(package)


def test_failed_trace_requires_error_category_and_cannot_be_rated(tmp_path):
    root, tasks, traces = _make_ready_package(tmp_path)
    study = json.loads((root / "study.json").read_text(encoding="utf-8"))
    failed = copy.deepcopy(traces[0])
    failed["termination_status"] = "failed"
    with pytest.raises(transfer.PackageError, match="incomplete trace must record the model request failure"):
        package = _write_package(tmp_path / "failed", tasks=tasks, study=study, traces=[failed])
        transfer.validate_v2_package(package)

    failed["stages"][0]["error_category"] = "api_error"
    failed["stages"][0]["output_text"] = ""
    failed["final_response"] = ""
    failed["final_response_sha256"] = transfer._v2_hash("")
    rating = {"schema_version": "2.0", "rating_id": "rating-failed", "trace_id": failed["trace_id"],
              "rater_id": "human-rater-a", "blind_id": "blind-failed",
              "scores": {dimension: 1 for dimension in transfer.PILOT_V2_RUBRIC_DIMENSIONS},
              "overall_note": "Should not score failed output.", "rated_at_utc": "2026-10-17T13:00:00Z"}
    with pytest.raises(transfer.PackageError, match="may rate completed responses only"):
        package = _write_package(tmp_path / "rated-failed", tasks=tasks, study=study, traces=[failed], rubrics=[rating])
        transfer.validate_v2_package(package)


def test_v1_schema_validation_and_default_path_are_preserved(tmp_path):
    import jsonschema

    root = Path(__file__).parents[1] / "llm_poc_rewrite"
    schema = json.loads((root / "schemas" / "transfer_study.schema.json").read_text(encoding="utf-8"))
    study = json.loads((root / "data" / "real_issue_transfer" / "v1" / "study.json").read_text(encoding="utf-8"))
    jsonschema.Draft202012Validator.check_schema(schema)
    assert list(jsonschema.Draft202012Validator(schema).iter_errors(study)) == []

    help_result = pytest.raises(SystemExit, transfer.main, ["--version", "1", "--help"])
    assert help_result.value.code == 0
    report = transfer.validate_package(root / "data" / "real_issue_transfer" / "v1")
    assert report["valid"] and not report["ready"]
    assert not report["complete"]
    assert report["missing_run_records"] == 0

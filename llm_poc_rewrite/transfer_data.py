"""Strict, dependency-free validation for the prospective real-issue study package."""
from __future__ import annotations

import argparse
from datetime import datetime
import hashlib
import json
import math
from pathlib import Path
import re
from typing import Any
from urllib.parse import urlparse

SCHEMA_VERSION = "1.0"
HEX_40 = re.compile(r"^[0-9a-f]{40}$")
HEX_64 = re.compile(r"^[0-9a-f]{64}$")
CONDITIONS = {"baseline", "static_energy", "adaptive_energy"}
RELATIONS = {
    "tracker_blocks", "verified_prerequisite", "shared_context",
    "related_only", "package_dependency",
}
SECRET_KEYS = {
    "apikey", "accesstoken", "bearertoken", "clientsecret", "password",
    "privatekey", "secret", "token",
}


class PackageError(ValueError):
    """Raised when a study record is malformed or unsafe to export."""


def _reject_constant(value: str) -> None:
    raise PackageError(f"non-standard JSON constant is not allowed: {value}")


def _unique_object(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise PackageError(f"duplicate JSON key: {key}")
        result[key] = value
    return result


def _load_json(path: Path) -> Any:
    try:
        if path.stat().st_size > 5_000_000:
            raise PackageError(f"{path.name} exceeds the 5 MB record limit")
        return json.loads(
            path.read_text(encoding="utf-8"),
            parse_constant=_reject_constant,
            object_pairs_hook=_unique_object,
        )
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise PackageError(f"cannot read {path}: {exc}") from exc


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise PackageError(message)


def _one_of(value: Any, choices: set[str]) -> bool:
    return isinstance(value, str) and value in choices


def _keys(record: dict[str, Any], required: set[str], allowed: set[str], label: str) -> None:
    missing = required - record.keys()
    unknown = record.keys() - allowed
    _require(not missing, f"{label}: missing fields {sorted(missing)}")
    _require(not unknown, f"{label}: unknown fields {sorted(unknown)}")


def _no_secrets(value: Any, path: str = "record") -> None:
    if isinstance(value, dict):
        for key, child in value.items():
            normalized_key = re.sub(r"[^a-z0-9]", "", key.lower())
            _require(normalized_key not in SECRET_KEYS, f"{path}.{key}: secret-like field is forbidden")
            _no_secrets(child, f"{path}.{key}")
    elif isinstance(value, list):
        for index, child in enumerate(value):
            _no_secrets(child, f"{path}[{index}]")


def _timestamp(value: Any, label: str) -> datetime:
    _require(isinstance(value, str), f"{label}: expected ISO 8601 string")
    try:
        parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError as exc:
        raise PackageError(f"{label}: invalid ISO 8601 timestamp") from exc
    _require(parsed.tzinfo is not None, f"{label}: timezone is required")
    _require(parsed.utcoffset() is not None and parsed.utcoffset().total_seconds() == 0,
             f"{label}: timestamps must be UTC")
    return parsed


def _sha(value: Any, pattern: re.Pattern[str], label: str) -> None:
    _require(isinstance(value, str) and pattern.fullmatch(value) is not None,
             f"{label}: invalid SHA-256/commit hash")


def _http_url(value: Any, label: str) -> None:
    parsed = urlparse(value if isinstance(value, str) else "")
    _require(parsed.scheme in {"http", "https"} and bool(parsed.netloc)
             and parsed.username is None and parsed.password is None,
             f"{label}: expected HTTP(S) URL without embedded credentials")


def _record_header(record: Any, label: str) -> dict[str, Any]:
    _require(isinstance(record, dict), f"{label}: each JSON value must be an object")
    _no_secrets(record, label)
    _require(record.get("schema_version") == SCHEMA_VERSION,
             f"{label}: schema_version must be {SCHEMA_VERSION}")
    return record


def _validate_study(study: Any) -> None:
    r = _record_header(study, "study.json")
    fields = {"schema_version", "study_id", "status", "created_at_utc", "aim",
              "cohort", "conditions", "evaluation", "claim_limits"}
    _keys(r, fields, fields, "study.json")
    _require(isinstance(r["study_id"], str) and r["study_id"].strip(), "study_id is required")
    _require(_one_of(r["status"], {"draft", "ready", "running", "complete"}), "invalid study status")
    _timestamp(r["created_at_utc"], "created_at_utc")
    _require(isinstance(r["aim"], str) and len(r["aim"].strip()) >= 20, "aim must be substantive")
    cohort = r["cohort"]
    _require(isinstance(cohort, dict), "cohort must be an object")
    _keys(cohort, {"target_min", "target_max", "repository", "intake_start", "intake_end"},
          {"target_min", "target_max", "repository", "intake_start", "intake_end"}, "cohort")
    _require(type(cohort["target_min"]) is int and 1 <= cohort["target_min"] <= 8,
             "cohort.target_min must be an integer from 1 to 8")
    _require(type(cohort["target_max"]) is int and cohort["target_min"] <= cohort["target_max"] <= 20,
             "cohort.target_max must be an integer between target_min and 20")
    for key in ("intake_start", "intake_end"):
        if cohort[key] is not None:
            _timestamp(cohort[key], f"cohort.{key}")
    _require(cohort["intake_start"] is None or cohort["intake_end"] is None
             or _timestamp(cohort["intake_start"], "cohort.intake_start") <=
                _timestamp(cohort["intake_end"], "cohort.intake_end"), "intake window is reversed")
    _require(cohort["repository"] is None or isinstance(cohort["repository"], str),
             "cohort.repository must be a string or null")
    _require(isinstance(r["conditions"], list) and len(r["conditions"]) > 0
             and all(_one_of(item, CONDITIONS) for item in r["conditions"]),
             "conditions must use supported names")
    _require(len(set(r["conditions"])) == len(r["conditions"]), "conditions must be unique")
    _require(set(r["conditions"]) == CONDITIONS,
             "conditions must include baseline, static_energy, and adaptive_energy")
    _require(isinstance(r["evaluation"], dict) and isinstance(r["claim_limits"], list),
             "evaluation must be an object and claim_limits a list")
    evaluation_fields = {"primary_contrast", "secondary_contrast", "repetitions",
                         "task_success_requires", "sandbox_required", "api_budget_usd",
                         "worker_skill_routing_claim"}
    _keys(r["evaluation"], evaluation_fields, evaluation_fields, "evaluation")
    _require(isinstance(r["evaluation"]["primary_contrast"], str)
             and r["evaluation"]["primary_contrast"].strip(), "evaluation.primary_contrast required")
    _require(isinstance(r["evaluation"]["secondary_contrast"], str)
             and r["evaluation"]["secondary_contrast"].strip(), "evaluation.secondary_contrast required")
    _require(isinstance(r["evaluation"]["task_success_requires"], list)
             and r["evaluation"]["task_success_requires"]
             and all(isinstance(item, str) and item.strip() for item in r["evaluation"]["task_success_requires"]),
             "evaluation.task_success_requires must contain non-empty strings")
    _require(type(r["evaluation"]["sandbox_required"]) is bool
             and r["evaluation"]["sandbox_required"], "evaluation.sandbox_required must be true")
    _require(type(r["evaluation"]["worker_skill_routing_claim"]) is bool,
             "evaluation.worker_skill_routing_claim must be boolean")
    _require(type(r["evaluation"].get("repetitions")) is int
             and r["evaluation"]["repetitions"] > 0, "evaluation.repetitions must be a positive integer")
    _require(type(r["evaluation"].get("api_budget_usd")) in {int, float, type(None)}
             and (r["evaluation"]["api_budget_usd"] is None
                  or (math.isfinite(r["evaluation"]["api_budget_usd"]) and r["evaluation"]["api_budget_usd"] >= 0)),
             "evaluation.api_budget_usd must be non-negative or null")
    _require(isinstance(r["claim_limits"], list)
             and all(isinstance(item, str) and item.strip() for item in r["claim_limits"]),
             "claim_limits must contain non-empty strings")


def _packet_sha256(packet: dict[str, Any]) -> str:
    frozen_fields = {key: packet[key] for key in ("title", "solver_visible_text", "acceptance_criteria")}
    canonical = json.dumps(frozen_fields, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


def _validate_issue(record: Any, index: int) -> None:
    label = f"issues.jsonl:{index}"
    r = _record_header(record, label)
    fields = {"schema_version", "task_id", "captured_at_utc", "source", "task_packet",
              "environment", "eligibility"}
    _keys(r, fields, fields, label)
    _require(isinstance(r["task_id"], str) and r["task_id"].strip(), f"{label}: task_id required")
    _timestamp(r["captured_at_utc"], f"{label}.captured_at_utc")
    source = r["source"]
    _require(isinstance(source, dict), f"{label}.source must be object")
    _keys(source, {"tracker", "repository", "issue_number", "issue_url", "title", "body_sha256"},
          {"tracker", "repository", "issue_number", "issue_url", "title", "body_sha256"}, f"{label}.source")
    _require(_one_of(source["tracker"], {"github", "gitlab", "bugzilla", "jira", "other"}),
             f"{label}.source.tracker invalid")
    _require(isinstance(source["repository"], str) and bool(source["repository"].strip()),
             f"{label}.source.repository required")
    _require(type(source["issue_number"]) is int and source["issue_number"] > 0,
             f"{label}.source.issue_number must be positive integer")
    _http_url(source["issue_url"], f"{label}.source.issue_url")
    _require(isinstance(source["title"], str) and source["title"].strip(), f"{label}.source.title required")
    _sha(source["body_sha256"], HEX_64, f"{label}.source.body_sha256")
    packet = r["task_packet"]
    _require(isinstance(packet, dict), f"{label}.task_packet must be object")
    _keys(packet, {"title", "solver_visible_text", "acceptance_criteria", "prompt_sha256"},
          {"title", "solver_visible_text", "acceptance_criteria", "prompt_sha256"}, f"{label}.task_packet")
    _require(isinstance(packet["title"], str) and packet["title"].strip(), f"{label}: task packet title required")
    _require(isinstance(packet["solver_visible_text"], str) and packet["solver_visible_text"].strip(),
             f"{label}: solver-visible task text required")
    _require(isinstance(packet["acceptance_criteria"], list) and packet["acceptance_criteria"]
             and all(isinstance(x, str) and x.strip() for x in packet["acceptance_criteria"]),
             f"{label}: non-empty acceptance_criteria required")
    _sha(packet["prompt_sha256"], HEX_64, f"{label}.task_packet.prompt_sha256")
    _require(packet["prompt_sha256"] == _packet_sha256(packet),
             f"{label}: prompt_sha256 does not match frozen worker-visible packet")
    env = r["environment"]
    _require(isinstance(env, dict), f"{label}.environment must be object")
    _keys(env, {"base_commit", "platform", "image_digest", "setup_ref", "test_commands"},
          {"base_commit", "platform", "image_digest", "setup_ref", "test_commands"}, f"{label}.environment")
    _sha(env["base_commit"], HEX_40, f"{label}.environment.base_commit")
    _require(isinstance(env["platform"], str) and env["platform"].strip(), f"{label}: platform required")
    _require(env["image_digest"] is None or (isinstance(env["image_digest"], str)
             and re.fullmatch(r"sha256:[0-9a-f]{64}", env["image_digest"]) is not None),
             f"{label}: image_digest must be immutable digest or null")
    _require(isinstance(env["setup_ref"], str) and env["setup_ref"].strip(), f"{label}: setup_ref required")
    _require(isinstance(env["test_commands"], list) and env["test_commands"]
             and all(isinstance(x, str) and x.strip() for x in env["test_commands"]),
             f"{label}: test_commands required")
    eligibility = r["eligibility"]
    _require(isinstance(eligibility, dict), f"{label}.eligibility must be object")
    _keys(eligibility, {"reviewers", "decision", "exclusion_reason"},
          {"reviewers", "decision", "exclusion_reason"}, f"{label}.eligibility")
    _require(isinstance(eligibility["reviewers"], list) and all(isinstance(x, str) for x in eligibility["reviewers"]),
             f"{label}: reviewers must be a list")
    _require(_one_of(eligibility["decision"], {"pending", "included", "excluded"}), f"{label}: invalid eligibility decision")
    _require(eligibility["exclusion_reason"] is None or isinstance(eligibility["exclusion_reason"], str),
             f"{label}: exclusion_reason must be string or null")


def _validate_evaluator(record: Any, index: int) -> None:
    label = f"evaluator.jsonl:{index}"
    r = _record_header(record, label)
    fields = {"schema_version", "task_id", "trusted_fix_ref", "regression_test_ref", "base_result", "fix_result"}
    _keys(r, fields, fields, label)
    _require(isinstance(r["task_id"], str) and r["task_id"].strip(), f"{label}.task_id required")
    for key in ("trusted_fix_ref", "regression_test_ref"):
        _require(r[key] is None or (isinstance(r[key], str) and r[key].strip()),
                 f"{label}.{key} must be a non-empty string or null")
    for key in ("base_result", "fix_result"):
        _require(_one_of(r[key], {"not_run", "fail", "pass", "timeout", "error"}), f"{label}.{key} invalid")
    if r["base_result"] == "fail" or r["fix_result"] == "pass":
        _require(bool(r["trusted_fix_ref"]) and bool(r["regression_test_ref"]),
                 f"{label}: base/fix evidence requires trusted fix and regression test references")


def _validate_relation(record: Any, index: int) -> None:
    label = f"relations.jsonl:{index}"
    r = _record_header(record, label)
    fields = {"schema_version", "relation_id", "source_task_id", "target_task_id", "relation_type",
              "tracker_relation", "observed_at_utc", "evidence_url", "rationale", "reviewers", "confidence"}
    _keys(r, fields, fields, label)
    for key in ("relation_id", "source_task_id", "target_task_id"):
        _require(isinstance(r[key], str) and r[key].strip(), f"{label}.{key} required")
    _require(r["source_task_id"] != r["target_task_id"], f"{label}: self-relations are invalid")
    _require(_one_of(r["relation_type"], RELATIONS), f"{label}: invalid relation_type")
    _require(isinstance(r["tracker_relation"], str), f"{label}: tracker_relation must preserve source relation")
    observed = r["observed_at_utc"]
    _require(isinstance(observed, dict), f"{label}.observed_at_utc must be object")
    _keys(observed, {"start", "end", "precision"}, {"start", "end", "precision"}, f"{label}.observed_at_utc")
    start = _timestamp(observed["start"], f"{label}.observed_at_utc.start")
    end = _timestamp(observed["end"], f"{label}.observed_at_utc.end")
    _require(start <= end, f"{label}: observation interval is reversed")
    _require(_one_of(observed["precision"], {"event", "snapshot", "interval"}), f"{label}: invalid time precision")
    _http_url(r["evidence_url"], f"{label}.evidence_url")
    _require(isinstance(r["rationale"], str) and r["rationale"].strip(), f"{label}: rationale required")
    _require(isinstance(r["reviewers"], list) and r["reviewers"], f"{label}: reviewer adjudication required")
    _require(_one_of(r["confidence"], {"high", "medium", "low"}), f"{label}: invalid confidence")


def _validate_run(record: Any, index: int) -> None:
    label = f"runs.jsonl:{index}"
    r = _record_header(record, label)
    fields = {"schema_version", "run_id", "task_id", "condition", "repetition", "model_id",
              "prompt_sha256", "seed", "started_at_utc", "ended_at_utc", "status",
              "request_attempts", "patch_sha256", "patch_application", "test_results",
              "tokens_in", "tokens_out", "cost_usd", "cost_basis", "runtime_seconds", "failure_category"}
    _keys(r, fields, fields, label)
    for key in ("run_id", "task_id", "model_id"):
        _require(isinstance(r[key], str) and r[key].strip(), f"{label}.{key} required")
    _require(_one_of(r["condition"], CONDITIONS), f"{label}: invalid condition")
    _require(type(r["repetition"]) is int and r["repetition"] >= 0, f"{label}: repetition must be non-negative integer")
    _require(type(r["seed"]) is int, f"{label}: seed must be integer")
    _sha(r["prompt_sha256"], HEX_64, f"{label}.prompt_sha256")
    start, end = _timestamp(r["started_at_utc"], f"{label}.started_at_utc"), _timestamp(r["ended_at_utc"], f"{label}.ended_at_utc")
    _require(start <= end, f"{label}: run end precedes start")
    _require(_one_of(r["status"], {"completed", "failed", "timeout", "cancelled"}), f"{label}: invalid status")
    _require(type(r["request_attempts"]) is int and r["request_attempts"] >= 0, f"{label}: invalid request_attempts")
    _require(r["patch_sha256"] is None or (isinstance(r["patch_sha256"], str) and HEX_64.fullmatch(r["patch_sha256"]) is not None),
             f"{label}: patch_sha256 must be SHA-256 or null")
    _require(_one_of(r["patch_application"], {"applied", "rejected", "not_applicable", "not_run"}),
             f"{label}: invalid patch_application")
    _require(isinstance(r["test_results"], list), f"{label}: test_results must be list")
    for test_index, result in enumerate(r["test_results"], start=1):
        test_label = f"{label}.test_results[{test_index}]"
        _require(isinstance(result, dict), f"{test_label} must be an object")
        test_fields = {"test_id", "command", "status", "exit_code", "duration_seconds", "log_sha256"}
        _keys(result, test_fields, test_fields, test_label)
        _require(isinstance(result["test_id"], str) and result["test_id"].strip(), f"{test_label}.test_id required")
        _require(isinstance(result["command"], str) and result["command"].strip(), f"{test_label}.command required")
        _require(_one_of(result["status"], {"pass", "fail", "timeout", "error", "not_run"}), f"{test_label}.status invalid")
        _require(result["exit_code"] is None or type(result["exit_code"]) is int, f"{test_label}.exit_code must be integer or null")
        duration = result["duration_seconds"]
        _require(duration is None or (type(duration) in {int, float} and math.isfinite(duration) and duration >= 0),
                 f"{test_label}.duration_seconds must be finite and non-negative or null")
        digest = result["log_sha256"]
        _require(digest is None or (isinstance(digest, str) and HEX_64.fullmatch(digest) is not None),
                 f"{test_label}.log_sha256 must be SHA-256 or null")
    for key in ("tokens_in", "tokens_out", "runtime_seconds"):
        number = r[key]
        _require(number is None or (type(number) in {int, float} and math.isfinite(number) and number >= 0),
                 f"{label}.{key} must be finite, non-negative number or null")
    _require(r["cost_usd"] is None or (type(r["cost_usd"]) in {int, float}
             and math.isfinite(r["cost_usd"]) and r["cost_usd"] >= 0), f"{label}: invalid cost_usd")
    _require(r["cost_basis"] is None or isinstance(r["cost_basis"], str), f"{label}: invalid cost_basis")
    _require(r["failure_category"] is None or isinstance(r["failure_category"], str), f"{label}: invalid failure_category")
    if r["status"] == "completed":
        _require(r["request_attempts"] > 0, f"{label}: completed run must record request attempts")


def _load_jsonl(path: Path, validator: Any) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    try:
        if path.stat().st_size > 50_000_000:
            raise PackageError(f"{path.name} exceeds the 50 MB file limit")
        with path.open(encoding="utf-8") as stream:
            for index, line in enumerate(stream, start=1):
                if not line.strip():
                    continue
                try:
                    record = json.loads(line, parse_constant=_reject_constant, object_pairs_hook=_unique_object)
                except (json.JSONDecodeError, PackageError) as exc:
                    raise PackageError(f"{path.name}:{index}: invalid JSON: {exc}") from exc
                validator(record, index)
                records.append(record)
    except (OSError, UnicodeError) as exc:
        raise PackageError(f"cannot read {path}: {exc}") from exc
    return records


def validate_package(directory: str | Path) -> dict[str, Any]:
    """Validate structure and report readiness separately from syntactic validity."""
    root = Path(directory)
    _require(root.is_dir(), f"study package directory does not exist: {root}")
    study = _load_json(root / "study.json")
    _validate_study(study)
    issues = _load_jsonl(root / "issues.jsonl", _validate_issue)
    evaluators = _load_jsonl(root / "evaluator.jsonl", _validate_evaluator)
    relations = _load_jsonl(root / "relations.jsonl", _validate_relation)
    runs = _load_jsonl(root / "runs.jsonl", _validate_run)
    task_ids = {item["task_id"] for item in issues}
    _require(len(task_ids) == len(issues), "issues.jsonl contains duplicate task_id values")
    evaluator_ids = {item["task_id"] for item in evaluators}
    _require(len(evaluator_ids) == len(evaluators), "evaluator.jsonl contains duplicate task_id values")
    _require(evaluator_ids <= task_ids, "evaluator.jsonl references unknown task_id")
    _require(len({item["relation_id"] for item in relations}) == len(relations),
             "relations.jsonl contains duplicate relation_id values")
    _require(len({item["run_id"] for item in runs}) == len(runs), "runs.jsonl contains duplicate run_id values")
    run_keys = [(item["task_id"], item["condition"], item["repetition"]) for item in runs]
    _require(len(set(run_keys)) == len(run_keys), "runs.jsonl contains duplicate task/condition/repetition records")
    _require(all(item["source_task_id"] in task_ids and item["target_task_id"] in task_ids for item in relations),
             "relations.jsonl references unknown task_id")
    _require(all(item["task_id"] in task_ids for item in runs), "runs.jsonl references unknown task_id")
    included = [item for item in issues if item["eligibility"]["decision"] == "included"]
    evaluator_by_task = {item["task_id"]: item for item in evaluators}
    failures = [item["task_id"] for item in included
                if item["task_id"] not in evaluator_by_task
                or evaluator_by_task[item["task_id"]]["base_result"] != "fail"
                or evaluator_by_task[item["task_id"]]["fix_result"] != "pass"]
    blockers = []
    if study["status"] == "draft":
        blockers.append("study is still draft")
    if study["status"] in {"running", "complete"} and len(included) < study["cohort"]["target_min"]:
        blockers.append("study status is running/complete without its minimum cohort")
    if len(included) < study["cohort"]["target_min"]:
        blockers.append(f"only {len(included)} included tasks; need {study['cohort']['target_min']}")
    if len(included) > study["cohort"]["target_max"]:
        blockers.append(f"{len(included)} included tasks exceed target_max {study['cohort']['target_max']}")
    if failures:
        blockers.append(f"base-fail/fix-pass gate missing for: {', '.join(failures)}")
    if study["cohort"]["repository"] is None or not study["cohort"]["repository"].strip():
        blockers.append("cohort repository has not been selected")
    if study["cohort"]["intake_start"] is None or study["cohort"]["intake_end"] is None:
        blockers.append("intake window has not been frozen")
    elif study["cohort"]["intake_start"] > study["cohort"]["intake_end"]:
        blockers.append("intake window is reversed")
    for item in issues:
        reviewers = item["eligibility"]["reviewers"]
        if len(reviewers) < 2 or len(set(reviewers)) < 2 or any(not reviewer.strip() for reviewer in reviewers):
            blockers.append(f"{item['task_id']} lacks two distinct, named eligibility reviewers")
        if item["eligibility"]["decision"] == "pending":
            blockers.append(f"{item['task_id']} eligibility is still pending")
        if item["eligibility"]["decision"] == "included" and item["eligibility"]["exclusion_reason"] is not None:
            blockers.append(f"{item['task_id']} is included but has an exclusion_reason")
        if item["eligibility"]["decision"] == "excluded" and not item["eligibility"]["exclusion_reason"]:
            blockers.append(f"{item['task_id']} is excluded without an exclusion_reason")
    if study["cohort"]["repository"]:
        outside_window = [item["task_id"] for item in issues
                          if not (_timestamp(study["cohort"]["intake_start"], "cohort.intake_start")
                                  <= _timestamp(item["captured_at_utc"], f"{item['task_id']}.captured_at_utc")
                                  <= _timestamp(study["cohort"]["intake_end"], "cohort.intake_end"))]
        if outside_window:
            blockers.append(f"issue captures fall outside the frozen intake window: {', '.join(outside_window)}")
        mismatched_repo = [item["task_id"] for item in issues
                           if item["source"]["repository"].casefold() != study["cohort"]["repository"].casefold()]
        if mismatched_repo:
            blockers.append(f"issues are outside selected repository: {', '.join(mismatched_repo)}")
    for item in included:
        if item["eligibility"]["decision"] != "included":
            blockers.append(f"{item['task_id']} is not eligible")
    if failures:
        missing_evaluator_ids = [task_id for task_id in failures if task_id not in evaluator_by_task]
        if missing_evaluator_ids:
            blockers.append(f"evaluator base/fix records missing for: {', '.join(missing_evaluator_ids)}")
        unverified_ids = [task_id for task_id in failures if task_id in evaluator_by_task]
        if unverified_ids:
            blockers.append(f"base-fail/fix-pass gate missing for: {', '.join(unverified_ids)}")
    incomplete_evaluator_refs = [item["task_id"] for item in included
                                 if item["task_id"] in evaluator_by_task
                                 and (not evaluator_by_task[item["task_id"]]["trusted_fix_ref"]
                                      or not evaluator_by_task[item["task_id"]]["regression_test_ref"])]
    if incomplete_evaluator_refs:
        blockers.append(f"evaluator references missing for: {', '.join(incomplete_evaluator_refs)}")
    run_keys = {(item["task_id"], item["condition"], item["repetition"]) for item in runs}
    expected_keys = {(item["task_id"], condition, repetition)
                     for item in included for condition in study["conditions"]
                     for repetition in range(study["evaluation"]["repetitions"])}
    missing_runs = expected_keys - run_keys
    complete = bool(expected_keys) and not missing_runs and not blockers
    return {"valid": True, "ready": not blockers, "complete": complete,
            "study_id": study["study_id"], "task_records": len(issues),
            "included_tasks": len(included), "evaluator_records": len(evaluators),
            "relation_records": len(relations), "run_records": len(runs),
            "missing_run_records": len(missing_runs), "blockers": blockers}


def export_worker_packets(directory: str | Path) -> list[dict[str, Any]]:
    """Return only frozen worker-visible task fields; never include evaluator records."""
    validate_package(directory)
    issues = _load_jsonl(Path(directory) / "issues.jsonl", _validate_issue)
    packets = []
    for issue in sorted(issues, key=lambda item: item["task_id"]):
        if issue["eligibility"]["decision"] != "included":
            continue
        packet = issue["task_packet"]
        packets.append({"task_id": issue["task_id"], "title": packet["title"],
                        "solver_visible_text": packet["solver_visible_text"],
                        "acceptance_criteria": packet["acceptance_criteria"],
                        "prompt_sha256": packet["prompt_sha256"]})
    return packets


def main(argv: list[str] | None = None) -> int:
    default_directory = Path(__file__).resolve().parent / "data" / "real_issue_transfer" / "v1"
    parser = argparse.ArgumentParser(description="Validate/export a real-issue transfer study package")
    parser.add_argument("directory", nargs="?", default=str(default_directory))
    parser.add_argument("--export-worker-packets", action="store_true",
                        help="write included, solver-visible packets as JSONL to stdout")
    args = parser.parse_args(argv)
    try:
        if args.export_worker_packets:
            report = validate_package(args.directory)
            if not report["ready"]:
                raise PackageError("worker packet export blocked: " + "; ".join(report["blockers"]))
            for packet in export_worker_packets(args.directory):
                print(json.dumps(packet, ensure_ascii=False, sort_keys=True))
            return 0
        report = validate_package(args.directory)
    except PackageError as exc:
        print(json.dumps({"valid": False, "ready": False, "error": str(exc)}, indent=2))
        return 1
    print(json.dumps(report, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

"""Validate legacy v1 patch packages and the active v2 text-transfer package."""
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

from . import config

SCHEMA_VERSION = "1.0"
PILOT_V2_VERSION = "2.0"
PILOT_V2_ROLES = tuple(config.AGENT_MODELS)
PILOT_V2_CONDITION = "independent_configured_agents"
MAX_V2_STUDY_OUTPUT_TOKENS = config.WORKER_MAX_TOKENS
MAX_V2_STUDY_TEMPERATURE = config.WORKER_TEMPERATURE
MAX_V1_TASKS = 8
TASK_PACKET_FIELDS = ("title", "issue_text", "acceptance_criteria", "context_docs", "transformation_note")
TASK_ID_PATTERN = re.compile(r"^[^#]+#[1-9][0-9]*$")
MAX_V2_CONTEXT_CHARS = 3000
MAX_V2_ISSUE_TEXT_CHARS = 12000
MAX_V2_RESPONSE_CHARS = 20000
PILOT_V2_RUBRIC_DIMENSIONS = {
    "issue_understanding", "evidence_grounding", "diagnostic_reasoning",
    "proposed_change_quality", "test_strategy", "uncertainty_and_scope",
}
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


def _v2_header(record: Any, label: str) -> dict[str, Any]:
    _require(isinstance(record, dict), f"{label}: each JSON value must be an object")
    _no_secrets(record, label)
    _require(record.get("schema_version") == PILOT_V2_VERSION,
             f"{label}: schema_version must be {PILOT_V2_VERSION}")
    return record


def _v2_hash(data: Any) -> str:
    encoded = json.dumps(data, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _prompt_hash(packet: dict[str, Any]) -> str:
    return _v2_hash({key: packet[key] for key in TASK_PACKET_FIELDS})


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
    _require(type(cohort["target_max"]) is int and cohort["target_min"] <= cohort["target_max"] <= MAX_V1_TASKS,
             f"cohort.target_max must be an integer between target_min and {MAX_V1_TASKS}")
    for key in ("intake_start", "intake_end"):
        if cohort[key] is not None:
            _timestamp(cohort[key], f"cohort.{key}")
    _require(cohort["intake_start"] is None or cohort["intake_end"] is None
             or _timestamp(cohort["intake_start"], "cohort.intake_start") <=
                _timestamp(cohort["intake_end"], "cohort.intake_end"), "intake window is reversed")
    _require(cohort["repository"] is None or isinstance(cohort["repository"], str),
             "cohort.repository must be a string or null")
    _require(isinstance(r["conditions"], list) and len(r["conditions"]) == len(CONDITIONS)
             and all(_one_of(item, CONDITIONS) for item in r["conditions"]),
             "conditions must contain exactly baseline, static_energy, and adaptive_energy")
    _require(len(set(r["conditions"])) == len(r["conditions"]), "conditions must be unique")
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
    _require(type(r["evaluation"]["worker_skill_routing_claim"]) is bool
             and r["evaluation"]["worker_skill_routing_claim"] is False,
             "evaluation.worker_skill_routing_claim must be false until validated capability evidence exists")
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
    _require(isinstance(eligibility["reviewers"], list)
             and len(set(eligibility["reviewers"])) == len(eligibility["reviewers"])
             and all(isinstance(x, str) and x.strip() for x in eligibility["reviewers"]),
             f"{label}: reviewers must be unique non-empty strings")
    if eligibility["decision"] == "pending":
        _require(not eligibility["reviewers"], f"{label}: pending eligibility must not invent reviewer identities")
    else:
        _require(len(eligibility["reviewers"]) >= 2,
                 f"{label}: two distinct reviewer IDs required for an eligibility decision")
    _require((eligibility["decision"] == "excluded") == bool(eligibility["exclusion_reason"]),
             f"{label}: exclusion_reason must be set exactly for excluded tasks")
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


def _validate_v2_study(study: Any) -> None:
    label = "v2 study.json"
    r = _v2_header(study, label)
    fields = {"schema_version", "study_id", "status", "created_at_utc", "research_question",
              "task_source_policy", "cohort", "agent_protocol", "evaluation", "claim_limits"}
    _keys(r, fields, fields, label)
    _require(_one_of(r["status"], {"draft", "ready", "running", "complete"}), f"{label}: invalid status")
    _timestamp(r["created_at_utc"], f"{label}.created_at_utc")
    policy = r["task_source_policy"]
    _require(isinstance(policy, dict), f"{label}.task_source_policy must be object")
    _keys(policy, {"source_snapshot", "include_comments", "include_repo_checkout", "include_web_browsing", "prior_public_disclosure_policy"},
          {"source_snapshot", "include_comments", "include_repo_checkout", "include_web_browsing", "prior_public_disclosure_policy"}, f"{label}.task_source_policy")
    _require(_one_of(policy["source_snapshot"], {"issue_report_only", "issue_and_comments"}), f"{label}: invalid source snapshot policy")
    _require(type(policy["include_comments"]) is bool and type(policy["include_repo_checkout"]) is bool
             and type(policy["include_web_browsing"]) is bool, f"{label}: source policy flags must be booleans")
    _require(policy["include_repo_checkout"] is False, f"{label}: repository checkout is out of scope")
    _require(policy["include_web_browsing"] is False,
             f"{label}: v2 workers receive only the frozen issue packet and attached context")
    _require(policy["source_snapshot"] == "issue_report_only" and policy["include_comments"] is False,
             f"{label}: solver input must be frozen issue report only; comments are evaluator-only")
    _require(isinstance(policy["prior_public_disclosure_policy"], str) and len(policy["prior_public_disclosure_policy"]) >= 20,
             f"{label}: public disclosure policy must be documented")
    cohort = r["cohort"]
    _require(isinstance(cohort, dict), f"{label}.cohort must be object")
    _keys(cohort, {"repository", "target_tasks", "selected_tasks", "captured_at_utc"},
          {"repository", "target_tasks", "selected_tasks", "captured_at_utc"}, f"{label}.cohort")
    _require(isinstance(cohort["repository"], str) and cohort["repository"].strip(), f"{label}: repository required")
    _require(type(cohort["target_tasks"]) is int and cohort["target_tasks"] > 0
             and type(cohort["selected_tasks"]) is int and 0 <= cohort["selected_tasks"] <= cohort["target_tasks"],
             f"{label}: invalid cohort size")
    _timestamp(cohort["captured_at_utc"], f"{label}.cohort.captured_at_utc")
    protocol = r["agent_protocol"]
    _require(isinstance(protocol, dict), f"{label}.agent_protocol must be object")
    protocol_fields = {"condition", "repetitions", "roles", "models", "system_prompt_sha256", "primary_contrast",
                       "max_output_tokens_per_call", "temperature", "max_api_requests", "tool_access",
                       "comparison_policy", "seed_policy"}
    _keys(protocol, protocol_fields, protocol_fields, f"{label}.agent_protocol")
    _require(protocol["condition"] == PILOT_V2_CONDITION and protocol["tool_access"] == "prompt_context_only",
             f"{label}: v2 runs the configured agents independently with prompt/context only")
    _require(protocol["primary_contrast"] == "descriptive_role_model_bundle_transfer_only",
             f"{label}: v2 cannot claim isolated model or allocation effects")
    _require(type(protocol["repetitions"]) is int and protocol["repetitions"] > 0,
             f"{label}: repetitions must be positive")
    _require(protocol["roles"] == list(PILOT_V2_ROLES),
             f"{label}: roles must exactly match config.AGENT_MODELS in configured order")
    _require(type(protocol.get("temperature")) in {int, float} and protocol["temperature"] == MAX_V2_STUDY_TEMPERATURE,
             f"{label}: temperature must match llm_poc_rewrite.config.WORKER_TEMPERATURE")
    _require(isinstance(protocol["models"], dict)
             and all(isinstance(k, str) and isinstance(v, str) and v.strip() for k, v in protocol["models"].items()),
             f"{label}: models must be a mapping of names to IDs")
    _require(protocol["models"] == config.AGENT_MODELS,
             f"{label}: models must exactly match llm_poc_rewrite.config.AGENT_MODELS")
    _require(protocol["system_prompt_sha256"] == {
        role: hashlib.sha256(config.AGENT_SYSTEM_PROMPTS[role].encode("utf-8")).hexdigest()
        for role in PILOT_V2_ROLES
    }, f"{label}: system prompt hashes must match llm_poc_rewrite.config.AGENT_SYSTEM_PROMPTS")
    _require(tuple(protocol["models"]) == PILOT_V2_ROLES,
             f"{label}: model IDs must include every configured agent in the declared order")
    _require(protocol["primary_contrast"] == "descriptive_role_model_bundle_transfer_only",
             f"{label}: v2 cannot claim isolated model or allocation effects")
    _require(isinstance(protocol["comparison_policy"], str) and len(protocol["comparison_policy"]) >= 20,
             f"{label}: comparison policy must be documented")
    _require(isinstance(protocol["seed_policy"], str) and len(protocol["seed_policy"].strip()) >= 20,
             f"{label}: deterministic seed policy must be documented")
    _require(type(protocol["max_output_tokens_per_call"]) is int
             and protocol["max_output_tokens_per_call"] == MAX_V2_STUDY_OUTPUT_TOKENS
             and type(protocol["max_api_requests"]) is int and protocol["max_api_requests"] > 0,
             f"{label}: output-token limit must match config.WORKER_MAX_TOKENS and request budget must be positive")
    _require(r["status"] != "ready" or bool(protocol["models"]),
             f"{label}: pin every configured agent model before marking the study ready")
    _require(protocol["max_api_requests"] == len(PILOT_V2_ROLES),
             f"{label}: per-task API budget must allow exactly one request per configured agent")
    if r["status"] in {"running", "complete"}:
        _require(type(r["agent_protocol"].get("temperature")) in {int, float},
                 f"{label}: temperature must be frozen before any model run")
        _require(r["agent_protocol"].get("seed_policy") is not None,
                 f"{label}: seed policy must be frozen before any model run")
    evaluation = r["evaluation"]
    _require(isinstance(evaluation, dict), f"{label}.evaluation must be object")
    evaluation_fields = {"method", "blinding", "raters_per_response", "dimensions", "no_patch_claim",
                         "rating_scale", "claims_not_evaluated"}
    _keys(evaluation, evaluation_fields, evaluation_fields, f"{label}.evaluation")
    _require(evaluation["method"] == "human_rubric" and evaluation["no_patch_claim"] is True,
             f"{label}: v2 evaluates text with human rubric only, not patches")
    _require(type(evaluation["raters_per_response"]) is int and evaluation["raters_per_response"] >= 1,
             f"{label}: raters_per_response must be positive")
    _require(isinstance(evaluation["dimensions"], list) and set(evaluation["dimensions"]) == PILOT_V2_RUBRIC_DIMENSIONS,
             f"{label}: rubric dimensions do not match the registered set")
    for key in ("blinding", "rating_scale"):
        _require(isinstance(evaluation[key], str) and len(evaluation[key]) >= 20,
                 f"{label}: {key} must be documented")
    _require(isinstance(evaluation["claims_not_evaluated"], list) and evaluation["claims_not_evaluated"],
             f"{label}: claims_not_evaluated must be non-empty")
    _require(isinstance(r["research_question"], str) and len(r["research_question"].strip()) >= 20,
             f"{label}: research_question must be substantive")
    _require(isinstance(r["claim_limits"], list) and all(isinstance(x, str) and x.strip() for x in r["claim_limits"]),
             f"{label}: claim_limits must contain non-empty strings")


def _validate_v2_task(record: Any, index: int) -> None:
    label = f"tasks.jsonl:{index}"
    r = _v2_header(record, label)
    fields = {"schema_version", "task_id", "source", "captured_at_utc", "task_packet", "context", "disclosure", "eligibility"}
    _keys(r, fields, fields, label)
    _require(isinstance(r["task_id"], str) and TASK_ID_PATTERN.fullmatch(r["task_id"]),
             f"{label}: task_id must be a canonical repository#number identifier")
    source = r["source"]
    _require(isinstance(source, dict), f"{label}.source must be object")
    source_fields = {"url", "repository", "issue_number", "issue_family", "title", "status_at_capture", "created_at_utc", "updated_at_utc"}
    _keys(source, source_fields, source_fields, f"{label}.source")
    _require(isinstance(source["repository"], str) and source["repository"].strip()
             and type(source["issue_number"]) is int and source["issue_number"] > 0,
             f"{label}: invalid source repository/issue identity")
    _http_url(source["url"], f"{label}.source.url")
    parsed_source_url = urlparse(source["url"])
    _require(parsed_source_url.scheme == "https" and parsed_source_url.hostname == "github.com",
             f"{label}: source URL must use the official GitHub HTTPS host")
    expected_issue_path = f"/{source['repository']}/issues/{source['issue_number']}"
    _require(parsed_source_url.path.rstrip("/") == expected_issue_path,
             f"{label}: source URL must match repository and issue identity")
    _require(isinstance(source["issue_family"], str) and source["issue_family"].strip(),
             f"{label}: invalid issue family")
    _require(isinstance(source["title"], str) and source["title"].strip(), f"{label}: source title required")
    _require(_one_of(source["status_at_capture"], {"open", "closed"}), f"{label}: invalid issue status")
    for key in ("created_at_utc", "updated_at_utc"):
        _timestamp(source[key], f"{label}.source.{key}")
    _timestamp(r["captured_at_utc"], f"{label}.captured_at_utc")
    packet = r["task_packet"]
    _require(isinstance(packet, dict), f"{label}.task_packet must be object")
    packet_fields = {*TASK_PACKET_FIELDS, "issue_url", "prompt_sha256"}
    _keys(packet, packet_fields, packet_fields, f"{label}.task_packet")
    for key in ("title", "issue_text"):
        _require(isinstance(packet[key], str) and packet[key].strip(), f"{label}.task_packet.{key} required")
    _require(isinstance(packet["issue_text"], str) and len(packet["issue_text"]) <= MAX_V2_ISSUE_TEXT_CHARS,
             f"{label}: issue_text exceeds the {MAX_V2_ISSUE_TEXT_CHARS}-character packet limit")
    _require(isinstance(packet["acceptance_criteria"], list) and packet["acceptance_criteria"]
             and all(isinstance(x, str) and x.strip() for x in packet["acceptance_criteria"]),
             f"{label}: acceptance criteria required")
    _require(isinstance(packet["issue_url"], str) and packet["issue_url"] == source["url"],
             f"{label}: worker packet must carry the source issue URL")
    _require(isinstance(packet["transformation_note"], str) and len(packet["transformation_note"].strip()) >= 20,
             f"{label}: task packet transformation must be documented")
    docs = packet["context_docs"]
    _require(isinstance(docs, list), f"{label}.task_packet.context_docs must be a list")
    for doc in docs:
        _require(isinstance(doc, dict), f"{label}: context doc must be an object")
        doc_fields = {"url", "purpose", "content_text", "content_sha256"}
        _keys(doc, doc_fields, doc_fields, f"{label}.context_doc")
        _http_url(doc["url"], f"{label}.context_doc.url")
        _require(isinstance(doc["purpose"], str) and doc["purpose"].strip(), f"{label}: context purpose required")
        _require(isinstance(doc["content_text"], str) and doc["content_text"].strip()
                 and len(doc["content_text"]) <= MAX_V2_CONTEXT_CHARS,
                 f"{label}: context text must be non-empty and no longer than {MAX_V2_CONTEXT_CHARS:,} characters")
        _sha(doc["content_sha256"], HEX_64, f"{label}.context_doc.content_sha256")
        _require(doc["content_sha256"] == hashlib.sha256(doc["content_text"].encode("utf-8")).hexdigest(),
                 f"{label}: context document content hash mismatch")
    _sha(packet["prompt_sha256"], HEX_64, f"{label}.task_packet.prompt_sha256")
    expected_hash = _prompt_hash(packet)
    _require(packet["prompt_sha256"] == expected_hash, f"{label}: prompt_sha256 does not match task packet")
    context = r["context"]
    _require(isinstance(context, dict), f"{label}.context must be object")
    _keys(context, {"python_version_reported", "package_versions_reported", "execution_environment_available", "repo_checkout_provided"},
          {"python_version_reported", "package_versions_reported", "execution_environment_available", "repo_checkout_provided"}, f"{label}.context")
    _require(context["execution_environment_available"] is False and context["repo_checkout_provided"] is False,
             f"{label}: this study has no code execution or repo checkout")
    _require(context["python_version_reported"] is None or isinstance(context["python_version_reported"], str),
             f"{label}: invalid reported Python version")
    _require(isinstance(context["package_versions_reported"], dict), f"{label}: reported package versions must be object")
    disclosure = r["disclosure"]
    _require(isinstance(disclosure, dict), f"{label}.disclosure must be object")
    disclosure_fields = {"fix_disclosed_in_issue", "workaround_disclosed_in_comments", "material_causal_hint_in_comments", "linked_fix_ref", "reviewer_note"}
    _keys(disclosure, disclosure_fields, disclosure_fields, f"{label}.disclosure")
    _require(type(disclosure["fix_disclosed_in_issue"]) is bool
             and type(disclosure["workaround_disclosed_in_comments"]) is bool
             and type(disclosure["material_causal_hint_in_comments"]) is bool,
             f"{label}: disclosure flags must be boolean")
    _require(disclosure["linked_fix_ref"] is None or isinstance(disclosure["linked_fix_ref"], str),
             f"{label}: linked_fix_ref must be string or null")
    _require(isinstance(disclosure["reviewer_note"], str) and disclosure["reviewer_note"].strip(),
             f"{label}: disclosure reviewer note required")
    eligibility = r["eligibility"]
    _require(isinstance(eligibility, dict), f"{label}.eligibility must be object")
    _keys(eligibility, {"decision", "reviewers", "reason"}, {"decision", "reviewers", "reason"}, f"{label}.eligibility")
    _require(_one_of(eligibility["decision"], {"pending", "include_with_leakage_flag", "include", "exclude"}),
             f"{label}: invalid eligibility decision")
    _require(isinstance(eligibility["reviewers"], list)
             and len(set(eligibility["reviewers"])) == len(eligibility["reviewers"])
             and all(isinstance(x, str) and x.strip() for x in eligibility["reviewers"]),
             f"{label}: reviewer IDs must be unique non-empty strings")
    if eligibility["decision"] == "pending":
        _require(not eligibility["reviewers"], f"{label}: pending eligibility must not invent reviewer identities")
    else:
        _require(len(eligibility["reviewers"]) >= 2, f"{label}: two distinct reviewer IDs required")
    _require(isinstance(eligibility["reason"], str) and eligibility["reason"].strip(),
             f"{label}: eligibility rationale required")


def _validate_v2_stage(record: Any, index: int, task_ids: set[str], expected_models: dict[str, str],
                       max_output_tokens: int) -> None:
    label = f"traces.jsonl.stage[{index}]"
    _require(isinstance(record, dict), f"{label}: stage must be an object")
    fields = {"role", "model_id", "model_version", "system_prompt_sha256", "prompt_sha256", "input_task_ids", "output_text",
              "started_at_utc", "ended_at_utc", "latency_seconds", "tokens_in", "tokens_out", "cost_usd", "finish_reason", "error_category"}
    _keys(record, fields, fields, label)
    _require(_one_of(record["role"], set(PILOT_V2_ROLES)), f"{label}: invalid configured-agent role")
    _require(isinstance(record["model_id"], str) and record["model_id"].strip(), f"{label}.model_id required")
    _require(record["model_id"] == expected_models[record["role"]],
             f"{label}: recorded model_id differs from config.AGENT_MODELS for role {record['role']}")
    _sha(record["system_prompt_sha256"], HEX_64, f"{label}.system_prompt_sha256")
    expected_system_prompt_hash = hashlib.sha256(config.AGENT_SYSTEM_PROMPTS[record["role"]].encode("utf-8")).hexdigest()
    _require(record["system_prompt_sha256"] == expected_system_prompt_hash,
             f"{label}: system_prompt_sha256 differs from config.AGENT_SYSTEM_PROMPTS for role {record['role']}")
    _sha(record["prompt_sha256"], HEX_64, f"{label}.prompt_sha256")
    _require(isinstance(record["input_task_ids"], list) and record["input_task_ids"]
             and set(record["input_task_ids"]) <= task_ids, f"{label}: unknown or missing input task IDs")
    _require(len(record["input_task_ids"]) == 1,
             f"{label}: one model response run must correspond to exactly one issue")
    _require(isinstance(record["output_text"], str) and len(record["output_text"]) <= MAX_V2_RESPONSE_CHARS,
             f"{label}.output_text must be a string of at most {MAX_V2_RESPONSE_CHARS:,} characters")
    start, end = _timestamp(record["started_at_utc"], f"{label}.started_at_utc"), _timestamp(record["ended_at_utc"], f"{label}.ended_at_utc")
    _require(start <= end, f"{label}: stage end precedes start")
    for key in ("latency_seconds", "cost_usd"):
        value = record[key]
        _require(value is None or (type(value) in {int, float} and math.isfinite(value) and value >= 0),
                 f"{label}.{key} must be finite, non-negative, or null")
    for key in ("tokens_in", "tokens_out"):
        value = record[key]
        _require(value is None or (type(value) is int and value >= 0), f"{label}.{key} must be non-negative integer or null")
    _require(record["tokens_out"] is None or record["tokens_out"] <= max_output_tokens,
             f"{label}.tokens_out exceeds the preregistered output token limit")
    for key in ("model_version", "finish_reason", "error_category"):
        _require(record[key] is None or isinstance(record[key], str), f"{label}.{key} must be string or null")
    if record["error_category"] is None:
        _require(bool(record["output_text"].strip()), f"{label}: successful model response must contain output text")
    else:
        _require(not record["output_text"], f"{label}: failed model request must not carry a success output")


def _validate_v2_trace(record: Any, index: int, task_ids: set[str], task_by_id: dict[str, dict[str, Any]],
                       max_api_requests: int, pinned_models: dict[str, str], max_output_tokens: int) -> None:
    label = f"traces.jsonl:{index}"
    r = _v2_header(record, label)
    fields = {"schema_version", "trace_id", "task_id", "condition", "repetition", "started_at_utc", "ended_at_utc", "stages", "final_response", "final_response_sha256", "api_requests", "total_latency_seconds", "total_tokens_in", "total_tokens_out", "total_cost_usd", "termination_status"}
    _keys(r, fields, fields, label)
    _require(r["task_id"] in task_ids, f"{label}: unknown task_id")
    _require(r["condition"] == PILOT_V2_CONDITION, f"{label}: unsupported condition")
    _require(type(r["repetition"]) is int and r["repetition"] >= 0, f"{label}: repetition must be non-negative")
    start, end = _timestamp(r["started_at_utc"], f"{label}.started_at_utc"), _timestamp(r["ended_at_utc"], f"{label}.ended_at_utc")
    _require(start <= end, f"{label}: trace end precedes start")
    _require(isinstance(r["stages"], list) and len(r["stages"]) == 1,
             f"{label}: each independent agent response must have exactly one response record")
    for stage_index, stage in enumerate(r["stages"], 1):
        _validate_v2_stage(stage, stage_index, task_ids, pinned_models, max_output_tokens)
    _require(r["stages"][0]["role"] in PILOT_V2_ROLES,
             f"{label}: trace must identify one configured agent role; agents run independently")
    _require(r["stages"][0]["input_task_ids"] == [r["task_id"]],
             f"{label}: agent response input task does not match trace task_id")
    _require(r["stages"][0]["prompt_sha256"] == task_by_id[r["task_id"]]["task_packet"]["prompt_sha256"],
             f"{label}: prompt hash differs from the frozen worker packet")
    _require(isinstance(r["final_response"], str) and len(r["final_response"]) <= MAX_V2_RESPONSE_CHARS,
             f"{label}: final_response must be a string of at most {MAX_V2_RESPONSE_CHARS:,} characters")
    _sha(r["final_response_sha256"], HEX_64, f"{label}.final_response_sha256")
    _require(r["final_response_sha256"] == _v2_hash(r["final_response"]), f"{label}: final response hash mismatch")
    _require(type(r["api_requests"]) is int and r["api_requests"] >= len(r["stages"]),
             f"{label}: api_requests must include at least one request per recorded stage")
    _require(r["api_requests"] <= max_api_requests,
             f"{label}: api_requests exceed the preregistered per-trace maximum")
    _require(r["api_requests"] >= 1,
             f"{label}: a recorded model response or failure must include at least one API request")
    _require(_one_of(r["termination_status"], {"completed", "failed", "timeout", "cancelled"}), f"{label}: invalid status")
    for key in ("total_latency_seconds", "total_cost_usd"):
        value = r[key]
        _require(value is None or (type(value) in {int, float} and math.isfinite(value) and value >= 0), f"{label}.{key} invalid")
    for key in ("total_tokens_in", "total_tokens_out"):
        value = r[key]
        _require(value is None or (type(value) is int and value >= 0), f"{label}.{key} invalid")
    _require(all(stage["error_category"] is None for stage in r["stages"])
             or r["termination_status"] != "completed",
             f"{label}: completed trace cannot contain failed stages")
    _require(r["termination_status"] == "completed" or r["stages"][0]["error_category"] is not None,
             f"{label}: incomplete trace must record the model request failure")
    if r["termination_status"] == "completed":
        response_stages = r["stages"]
        _require(bool(r["final_response"].strip()) and bool(response_stages[0]["output_text"].strip()),
                 f"{label}: completed trace must contain a final response")
        _require(r["final_response"] == response_stages[0]["output_text"],
                 f"{label}: final_response must match the model response record")


def _validate_v2_rubric(record: Any, index: int) -> None:
    label = f"rubrics.jsonl:{index}"
    r = _v2_header(record, label)
    fields = {"schema_version", "rating_id", "trace_id", "rater_id", "blind_id", "scores", "overall_note", "rated_at_utc"}
    _keys(r, fields, fields, label)
    for key in ("rating_id", "trace_id", "rater_id", "blind_id"):
        _require(isinstance(r[key], str) and r[key].strip(), f"{label}.{key} required")
    scores = r["scores"]
    _require(isinstance(scores, dict), f"{label}.scores must be object")
    _keys(scores, PILOT_V2_RUBRIC_DIMENSIONS, PILOT_V2_RUBRIC_DIMENSIONS, f"{label}.scores")
    _require(all(type(value) is int and 0 <= value <= 4 for value in scores.values()), f"{label}: scores must be integers from 0 to 4")
    _require(isinstance(r["overall_note"], str), f"{label}.overall_note must be string")
    _require(r["rater_id"].casefold() != r["blind_id"].casefold(),
             f"{label}: rater_id must be distinct from anonymized blind_id")
    _timestamp(r["rated_at_utc"], f"{label}.rated_at_utc")


def validate_v2_package(directory: str | Path) -> dict[str, Any]:
    root = Path(directory)
    _require(root.is_dir(), f"v2 package directory does not exist: {root}")
    study = _load_json(root / "study.json")
    _validate_v2_study(study)
    tasks = _load_jsonl(root / "tasks.jsonl", _validate_v2_task)
    task_ids = {task["task_id"] for task in tasks}
    task_by_id = {task["task_id"]: task for task in tasks}
    _require(len(task_ids) == len(tasks), "tasks.jsonl contains duplicate task_id values")
    _require(len({(task["source"]["repository"], task["source"]["issue_number"]) for task in tasks}) == len(tasks),
             "tasks.jsonl contains duplicate repository/issue identities")
    for task in tasks:
        expected_id = f"{task['source']['repository'].split('/')[-1]}#{task['source']['issue_number']}"
        _require(task["task_id"] == expected_id, f"{task['task_id']}: task_id does not match canonical repository#number ID")
        _require(task["source"]["repository"] == study["cohort"]["repository"],
                 f"{task['task_id']}: repository differs from study cohort")
        if task["eligibility"]["decision"] in {"include", "include_with_leakage_flag"}:
            disclosure = task["disclosure"]
            if disclosure["fix_disclosed_in_issue"] or disclosure["workaround_disclosed_in_comments"] or disclosure["material_causal_hint_in_comments"]:
                _require(task["eligibility"]["decision"] == "include_with_leakage_flag",
                         f"{task['task_id']}: public solution/workaround/cause disclosure requires leakage flag")
            _require(task["eligibility"]["decision"] != "include" or not task["disclosure"]["material_causal_hint_in_comments"],
                     f"{task['task_id']}: causal-hint leakage requires leakage-flagged eligibility")
        _require(task["source"]["issue_number"] > 0, f"{task['task_id']}: issue number must be positive")
    pinned_models = study["agent_protocol"]["models"]
    traces = _load_jsonl(root / "traces.jsonl", lambda record, index: _validate_v2_trace(
        record, index, task_ids, task_by_id, study["agent_protocol"]["max_api_requests"], pinned_models,
        study["agent_protocol"]["max_output_tokens_per_call"]
    ))
    rubrics = _load_jsonl(root / "rubrics.jsonl", _validate_v2_rubric)
    trace_ids = {trace["trace_id"] for trace in traces}
    _require(len(trace_ids) == len(traces), "traces.jsonl contains duplicate trace_id values")
    _require(all(rating["trace_id"] in trace_ids for rating in rubrics), "rubrics.jsonl references unknown trace_id")
    trace_by_id = {trace["trace_id"]: trace for trace in traces}
    _require(all(trace_by_id[rating["trace_id"]]["termination_status"] == "completed" for rating in rubrics),
             "rubrics.jsonl may rate completed responses only")
    _require(all(rating["rater_id"].casefold() != trace_by_id[rating["trace_id"]]["stages"][0]["role"].casefold()
                 for rating in rubrics),
             "a response's configured agent cannot rate its own output")
    _require(all(rating["blind_id"] not in {trace_by_id[rating["trace_id"]]["task_id"], rating["trace_id"]}
                 for rating in rubrics), "blind_id must not disclose task or trace identity")
    rating_keys = [(rating["trace_id"], rating["rater_id"]) for rating in rubrics]
    _require(len(set(rating_keys)) == len(rating_keys), "rubrics.jsonl contains duplicate trace/rater records")
    rating_ids = [rating["rating_id"] for rating in rubrics]
    _require(len(set(rating_ids)) == len(rating_ids), "rubrics.jsonl contains duplicate rating_id values")
    included = [task for task in tasks if task["eligibility"]["decision"] in {"include", "include_with_leakage_flag"}]
    study_blockers = []
    if study["status"] == "draft":
        study_blockers.append("study is still draft")
    if len(included) < study["cohort"]["target_tasks"]:
        study_blockers.append(f"only {len(included)} eligible tasks; need {study['cohort']['target_tasks']}")
    if len(included) > study["cohort"]["target_tasks"]:
        study_blockers.append(f"{len(included)} eligible tasks exceed the frozen target of {study['cohort']['target_tasks']}")
    family_counts: dict[str, int] = {}
    for task in included:
        family = task["source"]["issue_family"]
        family_counts[family] = family_counts.get(family, 0) + 1
    max_family_share = max(family_counts.values(), default=0) / max(1, len(included))
    if included and max_family_share > 0.5:
        study_blockers.append("eligible cohort is dominated by one related issue family")
    if study["cohort"]["selected_tasks"] != len(included):
        study_blockers.append("cohort.selected_tasks does not match eligible task records")
    if study["status"] in {"ready", "running", "complete"} and len(included) != study["cohort"]["target_tasks"]:
        study_blockers.append("study cannot be ready until target cohort size is reached")
    if study["status"] in {"ready", "running", "complete"} and family_counts and max_family_share > 0.5:
        study_blockers.append("related issue family concentration exceeds 50 percent")
    expected = {
        (task["task_id"], role, repetition)
        for task in included
        for role in study["agent_protocol"]["roles"]
        for repetition in range(study["agent_protocol"]["repetitions"])
    }
    observed = {
        (trace["task_id"], trace["stages"][0]["role"], trace["repetition"])
        for trace in traces
    }
    _require(len(observed) == len(traces), "traces.jsonl contains duplicate task/agent/repetition entries")
    trace_blockers = []
    if expected - observed:
        trace_blockers.append(f"{len(expected - observed)} expected configured-agent trace records missing")
    if observed - expected:
        trace_blockers.append(f"{len(observed - expected)} unexpected configured-agent trace records")
    if study["status"] == "running" and not traces:
        study_blockers.append("running study must have at least one response trace")
    if study["status"] == "running" and any(trace["termination_status"] != "completed" for trace in traces):
        study_blockers.append("running study cannot continue with failed traces; resolve failed calls first")
    if study["status"] == "complete" and not traces:
        study_blockers.append("complete study has no response traces")
    if study["status"] in {"running", "complete"} and expected - observed:
        study_blockers.append(f"{study['status']} study is missing scheduled response records")
    if study["status"] == "complete" and any(trace["termination_status"] != "completed" for trace in traces):
        study_blockers.append("complete study has incomplete or failed response traces")
    if study["status"] in {"ready", "running", "complete"} and study["agent_protocol"]["models"] != config.AGENT_MODELS:
        study_blockers.append("all configured agent model IDs must be frozen before the study is ready")
    if study["status"] in {"running", "complete"} and not study["agent_protocol"]["seed_policy"].strip():
        study_blockers.append("sampling settings must be frozen before model runs")
    if any(
        trace["stages"][0]["model_id"] != study["agent_protocol"]["models"].get(trace["stages"][0]["role"])
        for trace in traces
    ):
        study_blockers.append("trace model ID differs from the preregistered model for its configured agent role")
    pending = [task["task_id"] for task in tasks if task["eligibility"]["decision"] == "pending"]
    if study["status"] in {"ready", "running", "complete"} and pending:
        study_blockers.append("eligibility must be adjudicated before study is ready")
    for task in tasks:
        reviewers = task["eligibility"]["reviewers"]
        if task["eligibility"]["decision"] != "pending" and len(reviewers) < 2:
            study_blockers.append(f"{task['task_id']} needs a second independent eligibility reviewer")
        if any(reviewer.casefold().startswith(("pending", "placeholder", "todo")) for reviewer in reviewers):
            study_blockers.append(f"{task['task_id']} uses placeholder reviewer IDs")
        if task["eligibility"]["decision"] == "include_with_leakage_flag" and not task["disclosure"]["reviewer_note"]:
            study_blockers.append(f"{task['task_id']} has no leakage note")
    if pending:
        study_blockers.append(f"eligibility pending for: {', '.join(pending)}")
    if len(tasks) > study["cohort"]["target_tasks"]:
        study_blockers.append("task ledger exceeds the frozen cohort target; explicitly resolve excess records")
    for task in included:
        if task["eligibility"]["decision"] == "include_with_leakage_flag":
            if not (task["disclosure"]["fix_disclosed_in_issue"]
                    or task["disclosure"]["workaround_disclosed_in_comments"]
                    or task["disclosure"]["material_causal_hint_in_comments"]):
                study_blockers.append(f"{task['task_id']} is leakage-flagged without a recorded disclosure basis")
    if len(traces) > len(expected):
        study_blockers.append("study has more response records than the frozen design permits")
    for rating in rubrics:
        _require(rating["rater_id"].casefold() != rating["blind_id"].casefold(),
                 f"{rating['rating_id']}: rater_id must be distinct from anonymized blind_id")

    ratings_per_trace = {trace_id: {rating["rater_id"] for rating in rubrics if rating["trace_id"] == trace_id}
                         for trace_id in trace_ids}
    rating_blockers = [trace_id for trace_id, raters in ratings_per_trace.items()
                       if trace_by_id[trace_id]["termination_status"] == "completed"
                       and (len(raters) != study["evaluation"]["raters_per_response"]
                            if rubrics else study["status"] in {"running", "complete"})]
    if any(trace["termination_status"] != "completed" and ratings_per_trace[trace["trace_id"]]
           for trace in traces):
        study_blockers.append("failed/incomplete responses must not receive rubric scores")
    if study["status"] == "complete" and len(rubrics) != len(traces) * study["evaluation"]["raters_per_response"]:
        study_blockers.append("each completed response must have exactly the preregistered number of ratings")
    if study["status"] == "complete" and expected - observed:
        study_blockers.append("complete study is missing configured-agent response repetitions")
    if study["status"] == "complete" and (trace_blockers or rating_blockers):
        study_blockers.append("complete study has unexpected traces or incomplete rating coverage")
    ready = not study_blockers
    complete = (study["status"] == "complete" and ready and not trace_blockers
                and not rating_blockers and bool(traces))
    if study["status"] == "ready" and (traces or rubrics) and (trace_blockers or rating_blockers):
        study_blockers.append("ready study has missing or unexpected response records or incomplete ratings")
        ready = False
    if study["status"] == "complete" and not complete:
        study_blockers.append("study status is complete but traces or required ratings are missing")
        ready = False
        complete = False
    return {"valid": True, "ready": ready, "complete": complete,
            "study_id": study["study_id"], "task_records": len(tasks), "eligible_tasks": len(included),
            "response_count": len(traces), "expected_response_count": len(expected),
            "trace_records": len(traces), "rubric_records": len(rubrics), "trace_blockers": trace_blockers,
            "unrated_trace_ids": rating_blockers, "blockers": study_blockers}


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
        if item["eligibility"]["decision"] == "pending":
            blockers.append(f"{item['task_id']} eligibility is still pending")
        elif len(reviewers) < 2:
            blockers.append(f"{item['task_id']} lacks two distinct, named eligibility reviewers")
        if item["eligibility"]["decision"] == "included" and item["eligibility"]["exclusion_reason"] is not None:
            blockers.append(f"{item['task_id']} is included but has an exclusion_reason")
        if item["eligibility"]["decision"] == "excluded" and not item["eligibility"]["exclusion_reason"]:
            blockers.append(f"{item['task_id']} is excluded without an exclusion_reason")
    if study["cohort"]["repository"] and study["cohort"]["intake_start"] is not None and study["cohort"]["intake_end"] is not None:
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
    unexpected_runs = run_keys - expected_keys
    if unexpected_runs:
        blockers.append(f"runs.jsonl contains {len(unexpected_runs)} records outside the frozen study design")
    missing_runs = expected_keys - run_keys
    complete = (study["status"] == "complete" and bool(expected_keys)
                and not missing_runs and not blockers)
    if study["status"] == "complete" and missing_runs:
        blockers.append(f"complete study is missing {len(missing_runs)} expected run records")
        complete = False
    if study["status"] == "ready" and runs:
        blockers.append("study has run records but status is still ready")
    return {"valid": True, "ready": not blockers, "complete": complete,
            "study_id": study["study_id"], "task_records": len(issues),
            "included_tasks": len(included), "evaluator_records": len(evaluators),
            "relation_records": len(relations), "run_records": len(runs),
            "missing_run_records": len(missing_runs), "blockers": blockers}


def export_v2_worker_packets(directory: str | Path) -> list[dict[str, Any]]:
    report = validate_v2_package(directory)
    _require(report["ready"], "worker export is blocked until study and eligibility are ready")
    root = Path(directory)
    tasks = _load_jsonl(root / "tasks.jsonl", _validate_v2_task)
    return [
        {"title": task["task_packet"]["title"],
         "issue_text": task["task_packet"]["issue_text"],
         "acceptance_criteria": task["task_packet"]["acceptance_criteria"],
         "context_docs": [{key: doc[key] for key in ("url", "purpose", "content_text", "content_sha256")}
                          for doc in task["task_packet"]["context_docs"]],
         "transformation_note": task["task_packet"]["transformation_note"],
         "prompt_sha256": task["task_packet"]["prompt_sha256"]}
        for task in sorted(tasks, key=lambda item: item["task_id"])
        if task["eligibility"]["decision"] in {"include", "include_with_leakage_flag"}
    ]


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
    default_root = Path(__file__).resolve().parent / "data" / "real_issue_transfer"
    parser = argparse.ArgumentParser(description="Validate real-issue transfer study packages")
    parser.add_argument("directory", nargs="?", default=None)
    parser.add_argument("--version", choices=("1", "2"), default="2",
                        help="package schema version (default: text-only v2)")
    parser.add_argument("--export-worker-packets", action="store_true",
                        help="export worker-visible packets if the selected package is ready")
    args = parser.parse_args(argv)
    directory = args.directory or str(default_root / ("v2" if args.version == "2" else "v1"))
    try:
        if args.version == "2":
            if args.export_worker_packets:
                for packet in export_v2_worker_packets(directory):
                    print(json.dumps(packet, ensure_ascii=False, sort_keys=True))
                return 0
            report = validate_v2_package(directory)
        elif args.export_worker_packets:
            report = validate_package(directory)
            if not report["ready"]:
                raise PackageError("worker packet export blocked: " + "; ".join(report["blockers"]))
            for packet in export_worker_packets(directory):
                print(json.dumps(packet, ensure_ascii=False, sort_keys=True))
            return 0
        else:
            report = validate_package(directory)
    except PackageError as exc:
        print(json.dumps({"valid": False, "ready": False, "error": str(exc)}, indent=2))
        return 1
    print(json.dumps(report, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

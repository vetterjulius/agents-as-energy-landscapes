"""Evaluation module for minimal_llm_poc.

Supports:
1. Deterministic pattern/keyword/schema verification for unambiguous tasks.
2. Resilient LLM-as-judge evaluation using the disjoint judge model.
3. Episode-level scoring including co-assignment synergy bonuses.
"""
from __future__ import annotations

import json
import re
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

from . import api, config
from .scenario import (
    JUDGE_SYSTEM_PROMPT,
    TaskDef,
    get_interaction_graph,
    get_synergy_partners,
)


@dataclass
class TaskEvaluation:
    task_id: str
    slot_index: int
    assigned_agent: str
    base_score: float         # 0.0 or 1.0
    synergy_bonus: float      # 0.0 or 0.5
    total_score: float        # base_score + synergy_bonus
    eval_method: str          # "deterministic" or "llm_judge"
    details: str
    raw_output: str


def evaluate_deterministic(task: TaskDef, output: str) -> Tuple[float, str]:
    """Check required and forbidden patterns in the generated output."""
    output_lower = output.lower()

    missing: List[str] = []
    for pat in task.required_patterns:
        # Case-insensitive check
        if pat.lower() not in output_lower:
            missing.append(pat)

    forbidden_found: List[str] = []
    for pat in task.forbidden_patterns:
        if pat.lower() in output_lower:
            forbidden_found.append(pat)

    if missing:
        return 0.0, f"Missing required patterns: {missing[:3]}"
    if forbidden_found:
        return 0.0, f"Found forbidden patterns: {forbidden_found[:3]}"

    return 1.0, "All required patterns present and valid"


def evaluate_with_judge(
    task: TaskDef,
    output: str,
    *,
    mock: bool = False,
) -> Tuple[float, str]:
    """Evaluate non-deterministic tasks with the disjoint LLM judge."""
    if mock:
        return 1.0, "Mock judge pass"

    prompt = (
        f"Task Description:\n{task.prompt}\n\n"
        f"Agent Output to Evaluate:\n```\n{output}\n```\n\n"
        "Evaluate if the output correctly and completely satisfies the task requirements.\n"
        "Respond with a JSON object: {\"score\": 1, \"reason\": \"...\"} if passing, "
        "or {\"score\": 0, \"reason\": \"...\"} if failing."
    )

    generation = api.generate(
        prompt=prompt,
        system_prompt=JUDGE_SYSTEM_PROMPT,
        model=config.JUDGE_MODEL,
        judge=True,
        mock=mock,
    )

    raw = generation.text.strip()
    cleaned = raw
    if "```json" in cleaned:
        cleaned = cleaned.split("```json", 1)[1].split("```", 1)[0].strip()
    elif "```" in cleaned:
        cleaned = cleaned.split("```", 1)[1].split("```", 1)[0].strip()

    # Resilient JSON extraction
    try:
        data = json.loads(cleaned)
        score = float(data.get("score", 0))
        reason = str(data.get("reason", "Parsed JSON"))
        return (1.0 if score >= 0.5 else 0.0), reason
    except Exception:
        pass

    # Regex search for {"score": 1} or {"score": 0} anywhere in output
    match = re.search(r'["\']score["\']\s*:\s*([01])', raw)
    if match:
        score = float(match.group(1))
        return score, f"Regex extracted score {score} from: {raw[:60]}"

    # Fallback keyword match
    if '"score": 1' in raw or "'score': 1" in raw or 'score: 1' in raw:
        return 1.0, "Substring matched pass"

    return 0.0, f"Judge output unparseable: {raw[:100]}"


def evaluate_task(
    task: TaskDef,
    output: str,
    assigned_agent: str,
    *,
    mock: bool = False,
) -> TaskEvaluation:
    """Evaluate a single task trace using its specified mode."""
    if task.evaluation_mode == "deterministic":
        base_score, details = evaluate_deterministic(task, output)
        method = "deterministic"
    else:
        base_score, details = evaluate_with_judge(task, output, mock=mock)
        method = "llm_judge"

    return TaskEvaluation(
        task_id=task.id,
        slot_index=task.slot_index,
        assigned_agent=assigned_agent,
        base_score=base_score,
        synergy_bonus=0.0,  # Computed at episode level
        total_score=base_score,
        eval_method=method,
        details=details,
        raw_output=output,
    )


def compute_episode_evaluations(
    episode: int,
    evaluations: List[TaskEvaluation],
    assignment_map: Dict[int, str],  # slot_index -> agent_id
) -> List[TaskEvaluation]:
    """Compute co-assignment synergy bonuses across the episode's tasks.

    If task i and task j are synergistic in the active phase, both are assigned
    to the same agent, and both achieved base_score == 1.0, each gets +0.5 synergy.
    """
    updated: List[TaskEvaluation] = []
    eval_by_slot = {e.slot_index: e for e in evaluations}

    for ev in evaluations:
        slot = ev.slot_index
        partners = get_synergy_partners(episode, slot)
        agent = assignment_map.get(slot)

        has_synergy = False
        if ev.base_score >= 1.0 and agent is not None:
            for p_slot in partners:
                p_agent = assignment_map.get(p_slot)
                p_eval = eval_by_slot.get(p_slot)
                # Same agent and partner also solved correctly
                if p_agent == agent and p_eval is not None and p_eval.base_score >= 1.0:
                    has_synergy = True
                    break

        bonus = 0.5 if has_synergy else 0.0
        updated.append(
            TaskEvaluation(
                task_id=ev.task_id,
                slot_index=ev.slot_index,
                assigned_agent=ev.assigned_agent,
                base_score=ev.base_score,
                synergy_bonus=bonus,
                total_score=ev.base_score + bonus,
                eval_method=ev.eval_method,
                details=ev.details + (" [Synergy co-assignment bonus]" if bonus > 0 else ""),
                raw_output=ev.raw_output,
            )
        )

    return updated

"""Offline feature-schema gate for the currently selected real-issue pilot.

The pilot intentionally does not infer coding ability from the synthetic
analyst/extractor/synthesizer embeddings. This module checks that a future
issue/slot feature mapping has a matching named schema and an explicit rationale;
it does not choose or assign issues to slots.
"""
from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass
import json
from typing import Sequence

import numpy as np

from . import config
from .tasks import build_agent_slots


@dataclass(frozen=True)
class DesignGateResult:
    ready: bool
    issues: tuple[str, ...]
    task_count: int
    agent_count: int
    feature_dimensions: int | None


def check_transfer_design(
    issue_vectors: Sequence[Sequence[float]] | None,
    issue_feature_names: Sequence[str] | None,
    slot_vectors: Sequence[Sequence[float]] | None,
    slot_feature_names: Sequence[str] | None,
    semantic_rationale: str | None,
    *,
    task_count: int = config.REAL_ISSUE_TASK_COUNT,
    agent_count: int = config.N_AGENTS,
) -> DesignGateResult:
    """Check dimensions and declarations; passing is necessary, not sufficient."""
    issues: list[str] = []
    if task_count <= 0 or agent_count <= 0:
        issues.append("task and agent counts must be positive")
    if task_count != config.REAL_ISSUE_TASK_COUNT:
        issues.append("task count must match the selected real-issue node set")
    if issue_vectors is None:
        issues.append("real-issue feature vectors have not been supplied")
    if slot_vectors is None:
        issues.append("slot feature vectors have not been supplied")
    if issue_feature_names is None or slot_feature_names is None:
        issues.append("a shared, named feature schema has not been supplied")
    elif tuple(issue_feature_names) != tuple(slot_feature_names):
        issues.append("issue and slot feature names/order do not match")

    dimensions: int | None = None
    if issue_vectors is not None and slot_vectors is not None:
        try:
            issue_matrix = np.asarray(issue_vectors, dtype=float)
            slot_matrix = np.asarray(slot_vectors, dtype=float)
        except (TypeError, ValueError):
            issue_matrix = np.empty((0, 0))
            slot_matrix = np.empty((0, 0))
            issues.append("feature vectors must be rectangular numeric matrices")
        if issue_matrix.ndim != 2 or issue_matrix.shape[0] != task_count:
            issues.append(f"expected {task_count} issue vectors in a rectangular matrix")
        if slot_matrix.ndim != 2 or slot_matrix.shape[0] != agent_count:
            issues.append(f"expected {agent_count} slot vectors in a rectangular matrix")
        if issue_matrix.ndim == 2 and slot_matrix.ndim == 2:
            dimensions = int(issue_matrix.shape[1])
            if dimensions == 0 or slot_matrix.shape[1] != dimensions:
                issues.append("issue and slot vectors must have the same non-zero dimension")
            if not np.isfinite(issue_matrix).all() or not np.isfinite(slot_matrix).all():
                issues.append("feature vectors must contain only finite numbers")
            if issue_feature_names is not None and len(issue_feature_names) != dimensions:
                issues.append("issue feature schema length does not match vector dimension")
            if slot_feature_names is not None and len(slot_feature_names) != slot_matrix.shape[1]:
                issues.append("slot feature schema length does not match vector dimension")

    if semantic_rationale is None or len(semantic_rationale.strip()) < 40:
        issues.append("a substantive, preregistered feature-to-slot rationale is required")

    return DesignGateResult(
        ready=not issues,
        issues=tuple(issues),
        task_count=task_count,
        agent_count=agent_count,
        feature_dimensions=dimensions,
    )


def current_design_report() -> DesignGateResult:
    """Report why the current rewrite slots are not ready for issue routing."""
    slots = build_agent_slots()
    names = ", ".join(sorted(slot.name for slot in slots))
    result = check_transfer_design(
        issue_vectors=None,
        issue_feature_names=None,
        slot_vectors=[slot.capability for slot in slots],
        slot_feature_names=None,
        semantic_rationale=None,
    )
    findings = list(result.issues)
    findings.append(
        "current slot capabilities encode synthetic analyze/extract/synthesize roles "
        f"({names}), not validated coding competencies"
    )
    return DesignGateResult(
        ready=False,
        issues=tuple(findings),
        task_count=result.task_count,
        agent_count=result.agent_count,
        feature_dimensions=len(slots[0].capability),
    )


def main() -> int:
    parser = argparse.ArgumentParser(description="Offline real-issue feature-schema gate")
    parser.add_argument("--check", action="store_true", help="print current offline design-gate status")
    args = parser.parse_args()
    if not args.check:
        parser.error("use --check; this command never launches agents")
    print(json.dumps(asdict(current_design_report()), indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

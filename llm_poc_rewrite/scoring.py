"""Deterministic scoring for objective tasks in the rewrite benchmark."""
from __future__ import annotations

import re

from .tasks import TaskSpec

# Only short-form synthesis requires semantic grading; extraction and arithmetic
# are fully specified and can be checked reproducibly without a second LLM.
SEMANTIC_TASK_IDS = frozenset({7, 9})


def is_deterministic_task(task_id: int) -> bool:
    return task_id not in SEMANTIC_TASK_IDS


def deterministic_score(task: TaskSpec, candidate: str) -> tuple[int, str | None]:
    """Return exact-task rubric level and optional malformed-answer-key diagnostic."""
    text = candidate.strip()
    expected = task.reference_answer
    if task.id == 0:
        bits = re.search(
            r"maximum (\d+) in week (\d+); largest increase (\d+) from week (\d+) to week (\d+)",
            expected,
        )
        if not bits:
            return 0, "invalid task-0 answer key"
        maximum, max_week, increase, from_week, to_week = bits.groups()
        trend = expected.split(";", 1)[0]
        has_trend = trend in text.lower()
        has_maximum = bool(re.search(rf"\b{maximum}\b", text)) and bool(
            re.search(rf"week\s*{max_week}\b", text, re.I)
        )
        has_increase = bool(re.search(rf"\b{increase}\b", text)) and bool(
            re.search(rf"week\s*{from_week}\b.*week\s*{to_week}\b", text, re.I)
        )
        return (2 if has_trend and has_maximum and has_increase else
                1 if has_trend or has_maximum or has_increase else 0), None
    if task.id == 1:
        return (2 if text == expected else 0), None
    if task.id == 2:
        total_a = re.search(r"Supplier A total (\d+)", expected)
        total_b = re.search(r"Supplier B total (\d+)", expected)
        if not total_a or not total_b:
            return 0, "invalid task-2 answer key"
        comparison = expected.split("; ", 2)[-1]
        tied = "tied" in comparison.lower()
        correct_comparison = (
            (tied and "tie" in text.lower())
            or ("Supplier A cheaper" in comparison and bool(re.search(r"supplier\s*A", text, re.I)))
            or ("Supplier B cheaper" in comparison and bool(re.search(r"supplier\s*B", text, re.I)))
        )
        has_costs = total_a.group(1) in text and total_b.group(1) in text
        difference = abs(int(total_a.group(1)) - int(total_b.group(1)))
        exact_difference = tied or bool(re.search(rf"(?:by\s+|difference\s*(?:of|is)?\s*){difference}\b", text, re.I))
        return (2 if correct_comparison and has_costs and exact_difference else
                1 if correct_comparison else 0), None
    if task.id == 3 and task.depends_on == 0:
        match = re.fullmatch(r"\s*(YES|NO)[;,]?\s*(\d+)\s*(?:vs\.?|,|and)\s*(\d+)\s*", text, re.I)
        expected_match = re.fullmatch(r"(YES|NO); (\d+) vs (\d+)", expected, re.I)
        if not expected_match:
            return 0, "invalid shifted task-3 answer key"
        if not match:
            return 0, None
        return (2 if tuple(value.upper() for value in match.groups()) ==
                tuple(value.upper() for value in expected_match.groups()) else 0), None
    if task.id == 3:
        expected_pct = expected.split("%", 1)[0]
        percentages = re.findall(r"[-+]?\d+(?:\.\d+)?\s*%", text)
        if not percentages:
            return 0, None
        observed = float(percentages[0].replace("%", ""))
        return (2 if round(observed, 1) == round(float(expected_pct), 1) else 0), None
    if task.id == 4:
        expected_cities = [value.strip() for value in expected.split(",")]
        observed = [value.strip(" .,'\"*").upper() for value in text.split(",")]
        if observed == expected_cities:
            return 2, None
        if sorted(expected_cities) == sorted(observed):
            return 1, None
        return 0, None
    if task.id == 5:
        expected_dates = re.findall(r"\b\d{4}-\d{2}-\d{2}\b", expected)
        observed_dates = re.findall(r"\b\d{4}-\d{2}-\d{2}\b", text)
        if observed_dates == expected_dates and text == expected:
            return 2, None
        if observed_dates == expected_dates:
            return 1, None
        return 0, None
    if task.id == 6:
        expected_items = dict(re.findall(r"(DESK|CHAIR|LAMP)=(\d+,\d+)", expected))
        found = re.findall(r"(DESK|CHAIR|LAMP)\s*=\s*(\d+)\s*,\s*(\d+)", text)
        observed_items = {name: f"{price},{stock}" for name, price, stock in found}
        if observed_items == expected_items and len(found) == 3:
            return 2, None
        if len(observed_items) == 3 and all(name in expected_items for name in observed_items):
            return 1, None
        return 0, None
    if task.id == 8:
        match = re.search(r"([A-Z]+) ([+-]?\d+)pp", expected)
        if not match:
            return 0, "invalid task-8 answer key"
        category, amount = match.groups()
        if category in text.upper() and re.search(
            rf"\+?{amount}\s*(?:percentage points|pp)\b", text, re.I
        ):
            return 2, None
        if category in text.upper():
            return 1, None
        return 0, None
    return 0, f"task {task.id} has no deterministic scoring rule"

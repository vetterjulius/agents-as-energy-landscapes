"""LLM-as-judge scoring for every successful agent response."""
from __future__ import annotations

import re

from . import config
from .api import Generation, generate
from .tasks import TaskSpec

JUDGE_SYSTEM = (
    "You are a strict, consistent grader. Compare the candidate answer against the "
    "rubric levels and output ONLY a single integer: the index (0-based) of the "
    "highest rubric level the answer fully satisfies. Output nothing else."
)


def score(task: TaskSpec, candidate: str, *, task_prompt: str,
          seed: int | None = None) -> tuple[int, Generation, str | None]:
    """Return the rubric level, usage, and any output-validation error.

    Invalid judge text is retained for audit and assigned level zero so a single
    malformed response does not discard the remaining pilot results.
    """
    rubric = "\n".join(f"{index}: {anchor}" for index, anchor in enumerate(task.rubric))
    prompt = (
        f"Task presented to the agent:\n{task_prompt}\n\n"
        f"Reference-correct behaviour: {task.reference_answer}\n\n"
        f"Rubric levels:\n{rubric}\n\n"
        f"Candidate answer:\n{candidate[:1200]}\n\n"
        "Which rubric level (integer index) does the candidate fully satisfy?"
    )
    generation = generate(
        prompt, JUDGE_SYSTEM, config.JUDGE_MODEL, judge=True, seed=seed,
    )
    match = re.fullmatch(r"\s*(\d+)\s*", generation.text[:16])
    if not match:
        return 0, generation, "unparseable judge response; assigned conservative score 0"
    level = int(match.group(1))
    if not 0 <= level < len(task.rubric):
        return 0, generation, (
            f"judge score {level} outside rubric range 0..{len(task.rubric) - 1}; "
            "assigned conservative score 0"
        )
    return level, generation, None

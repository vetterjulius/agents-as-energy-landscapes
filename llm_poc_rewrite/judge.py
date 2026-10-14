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
          seed: int | None = None) -> tuple[int | None, Generation, str | None]:
    """Return the rubric level, usage, and any output-validation error.

    Invalid responses return no level; callers must fail closed and preserve
    the response for diagnosis rather than recording a score.
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
    if generation.finish_reason != "STOP":
        return None, generation, (
            f"judge generation did not finish normally: "
            f"finishReason={generation.finish_reason or 'missing'}"
        )
    match = re.fullmatch(r"\s*(\d+)\s*", generation.text)
    if not match:
        return None, generation, "unparseable judge response"
    level = int(match.group(1))
    if not 0 <= level < len(task.rubric):
        return None, generation, (
            f"judge score {level} outside rubric range 0..{len(task.rubric) - 1}"
        )
    return level, generation, None


def preflight(seed: int | None = None) -> Generation:
    """Verify the live judge can return a valid score before worker calls begin."""
    task = TaskSpec(
        id=-1,
        kind="extract",
        embedding=(1.0,),
        difficulty=0.0,
        prompt_template="Return the requested number.",
        reference_answer="4270",
        rubric=("incorrect number", "partially correct number", "exactly 4270"),
    )
    level, generation, error = score(
        task, "4270", task_prompt="What is 640 + 980 + 1020 + 890 + 740?", seed=seed,
    )
    if error is not None or level != 2:
        raise RuntimeError(
            "Gemini judge preflight failed; no worker calls were made. "
            f"Expected score 2, got {level}; finishReason="
            f"{generation.finish_reason or 'missing'}; output={generation.text!r}; "
            f"error={error or 'unexpected score'}"
        )
    return generation

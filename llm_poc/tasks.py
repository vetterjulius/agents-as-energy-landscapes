"""Task bank and agent definitions for the LLM PoC.

The task family is fixed at design time (deterministic): 10 tasks per episode,
3 specialization categories (analyze / extract / synthesize), each with
- a design-time 8-d feature embedding (the "capability" match target),
- a difficulty prior,
- a rubric with written anchors used for scoring,
and a set of *declared* dependency pairs (task B's prompt references task A's
output). Dependency pairs are the substrate the interaction term should exploit.

Embeddings are hand-designed, not random, so that category specialization is
interpretable: dims 0-2 = analysis, dims 3-5 = extraction, dims 6-7 = synthesis.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import torch

from . import config


@dataclass(frozen=True)
class AgentSlot:
    """One LLM agent slot (single fixed model, specialization via system prompt)."""
    id: int
    name: str
    specialization: str
    system_prompt: str
    capability: tuple[float, ...]


ANALYST_PROMPT = (
    "You are a precise analytical reasoning specialist. You break down problems, "
    "identify key quantities, compare alternatives, and state assumptions explicitly. "
    "Answer compactly and factually; never invent numbers."
)
EXTRACTOR_PROMPT = (
    "You are a meticulous data-extraction specialist. You follow extraction "
    "instructions to the letter, output exactly the requested fields, and copy "
    "values verbatim from the given text without paraphrasing numbers."
)
SYNTHESIST_PROMPT = (
    "You are a synthesis specialist. You combine provided bullet points into a "
    "coherent short paragraph that preserves every fact, resolves contradictions "
    "explicitly, and stays within the requested length."
)


def build_agent_slots() -> list[AgentSlot]:
    e = torch.eye(config.D)
    analyst = e[0] + e[1] + e[2]            # dims 0-2
    extractor = e[3] + e[4] + e[5]          # dims 3-5
    synthesist = e[6] + e[7]                # dims 6-7
    return [
        AgentSlot(0, "analyst", "analyze", ANALYST_PROMPT, tuple(analyst.tolist())),
        AgentSlot(1, "extractor", "extract", EXTRACTOR_PROMPT, tuple(extractor.tolist())),
        AgentSlot(2, "synthesist", "synthesize", SYNTHESIST_PROMPT, tuple(synthesist.tolist())),
    ]


@dataclass(frozen=True)
class TaskSpec:
    """A designed micro-task with embedding, difficulty and rubric."""
    id: int
    kind: str                    # "analyze" | "extract" | "synthesize"
    embedding: tuple[float, ...]
    difficulty: float            # 0..1 prior
    prompt_template: str         # {input} placeholder; dependency tasks also get {dep}
    reference_answer: str
    rubric: tuple[str, ...]      # written anchors; index == score level 0..len-1
    depends_on: int | None = None    # declared dependency: this task consumes that task's output
    unit_cost_hint: float = 1.0      # relative expected tokens


@dataclass
class EpisodeTasks:
    """The 10 tasks of one episode: fixed specs + per-episode input payloads."""
    specs: list[TaskSpec]
    inputs: list[str] = field(default_factory=list)
    episode: int = 0
    block: str = "stationary"


# ---------------------------------------------------------------------------
# Task bank: 10 designed tasks. ids 0..9. Declared dependencies:
#   task 3 consumes task 1's output (extract -> analyze)
#   task 7 consumes task 6's output (extract -> synthesize)
#   task 9 consumes task 8's output (analyze -> synthesize)
# These pairs are the "declared dependency graph" Theta_0 encodes in the
# energy conditions and that the adaptive arm must rediscover from co-assignment
# statistics.
# ---------------------------------------------------------------------------
def build_task_bank() -> list[TaskSpec]:
    e = torch.eye(config.D)
    def emb(kind: str, jitter: float = 0.0) -> tuple[float, ...]:
        base = {"analyze": e[0] + e[1] + e[2],
                "extract": e[3] + e[4] + e[5],
                "synthesize": e[6] + e[7]}[kind]
        v = base * (1.0 - jitter) + jitter / config.D
        return tuple((v / v.sum()).tolist())

    def emb_mixed(kind_a: str, kind_b: str) -> tuple[float, ...]:
        """Cross-functional task: equal affinity to two specialization slots.

        Dependency-downstream tasks are cross-functional by design so that
        co-locating a dependency pair on one slot is competitive with strict
        category specialization — the coordination trade-off the interaction
        term must resolve.
        """
        bases = {"analyze": e[0] + e[1] + e[2],
                 "extract": e[3] + e[4] + e[5],
                 "synthesize": e[6] + e[7]}
        v = (bases[kind_a] + bases[kind_b]) / 2.0
        return tuple((v / v.sum()).tolist())

    tasks = [
        TaskSpec(
            id=0, kind="analyze", embedding=emb("analyze"), difficulty=0.4,
            prompt_template=(
                "A delivery service recorded these weekly parcel counts: "
                "{input}. State the trend in one sentence and the week with the "
                "largest week-over-week increase."),
            reference_answer="INCREASE; largest increase WEEK4",
            rubric=(
                "no meaningful analysis",
                "identifies the trend OR the peak week, not both",
                "identifies both trend and peak week correctly, concise")),
        TaskSpec(
            id=1, kind="extract", embedding=emb("extract"), difficulty=0.3,
            prompt_template=(
                "From the following shipment log, extract the total number of "
                "units shipped in March as an integer, nothing else: {input}"),
            reference_answer="4270",
            rubric=(
                "no number or wrong number",
                "correct number but extra text",
                "exactly the integer 4270 and nothing else")),
        TaskSpec(
            id=2, kind="analyze", embedding=emb("analyze", 0.1), difficulty=0.6,
            prompt_template=(
                "Two suppliers quote unit prices: {input}. Which supplier is "
                "cheaper for 1000 units including the fixed fee, and by how much?"),
            reference_answer="SUPPLIER_B by 150",
            rubric=(
                "wrong supplier or wrong arithmetic",
                "correct supplier, arithmetic imprecise",
                "correct supplier and exact difference 150")),
        TaskSpec(
            id=3, kind="analyze", embedding=emb_mixed("analyze", "extract"), difficulty=0.5,
            depends_on=1,
            prompt_template=(
                "Using the March shipping total ({dep} units) and this April "
                "log: {input}, compute the April-over-March percentage change "
                "to one decimal."),
            reference_answer="-12.4%",
            rubric=(
                "no valid percentage",
                "percentage computed with wrong base or rounding",
                "correct percentage -12.4% with correct base")),
        TaskSpec(
            id=4, kind="extract", embedding=emb("extract", 0.1), difficulty=0.35,
            prompt_template=(
                "Extract the two cities with the highest sales from: {input}. "
                "Format: CITY1, CITY2"),
            reference_answer="LISBON, PORTO",
            rubric=(
                "wrong cities or wrong format",
                "correct cities, wrong format",
                "exactly 'LISBON, PORTO'")),
        TaskSpec(
            id=5, kind="extract", embedding=emb("extract", 0.2), difficulty=0.5,
            prompt_template=(
                "From the meeting minutes, list every deadline date verbatim, "
                "comma-separated: {input}"),
            reference_answer="2026-10-01, 2026-11-15",
            rubric=(
                "misses dates or invents dates",
                "all dates present but paraphrased/extra text",
                "exactly the two dates, verbatim, comma-separated")),
        TaskSpec(
            id=6, kind="extract", embedding=emb("extract", 0.05), difficulty=0.45,
            prompt_template=(
                "Extract the three product names with their unit prices from: "
                "{input}. Format: NAME=PRICE per line."),
            reference_answer="DESK=249\nCHAIR=89\nLAMP=45",
            rubric=(
                "missing or wrong items",
                "all items, format deviations",
                "exactly three NAME=PRICE lines, correct values")),
        TaskSpec(
            id=7, kind="synthesize", embedding=emb_mixed("synthesize", "extract"), difficulty=0.55,
            depends_on=6,
            prompt_template=(
                "Using the inventory list ({dep}) and this note: {input}, write "
                "2-3 sentences summarizing which items need restocking and why."),
            reference_answer="CHAIR and LAMP below threshold; DESK fine",
            rubric=(
                "summary contradicts the data",
                "partially correct: one item misclassified or vague",
                "correctly identifies CHAIR and LAMP as below threshold, DESK adequate")),
        TaskSpec(
            id=8, kind="analyze", embedding=emb("analyze", 0.15), difficulty=0.65,
            prompt_template=(
                "Support tickets by category this month: {input}. Identify the "
                "single category whose share grew the most versus last month "
                "and quantify the growth in percentage points."),
            reference_answer="BILLING +6pp (18% to 24%)",
            rubric=(
                "wrong category or no quantification",
                "correct category, growth imprecise",
                "BILLING with +6 percentage points correctly computed")),
        TaskSpec(
            id=9, kind="synthesize", embedding=emb_mixed("synthesize", "analyze"), difficulty=0.7,
            depends_on=8,
            prompt_template=(
                "Given the ticket analysis ({dep}) and this customer quote: "
                "{input}, write a 2-sentence customer-communication plan."),
            reference_answer="prioritize billing fixes; acknowledge quote",
            rubric=(
                "plan ignores the dominant category or the quote",
                "addresses category OR quote, not both",
                "addresses the billing growth and acknowledges the quote in 2 sentences")),
    ]
    return tasks


# Per-episode input payloads (fixed, deterministic strings).
EPISODE_INPUTS: list[list[str]] = [
    # episode 0
    [
        "[182, 195, 188, 210, 205, 221]",
        "2026-03-01: 640 units; 2026-03-08: 980 units; 2026-03-15: 1020 units; "
        "2026-03-22: 890 units; 2026-03-29: 740 units",
        "Supplier A: 4.20/unit + 900 fixed fee. Supplier B: 4.75/unit + 350 fixed fee.",
        "March total: 4270 units. April week 1: 990 units, week 2: 930 units, "
        "week 3: 910 units, week 4: 910 units",
        "SALES: Porto 12040; Lisbon 15600; Madrid 11200; Faro 6300",
        "Deadline for budget submission is 2026-10-01. The follow-up review is set for "
        "2026-11-15. AOE drafting begins immediately.",
        "DESK: 249 EUR (14 in stock). CHAIR: 89 EUR (112 in stock). LAMP: 45 EUR (96 in stock).",
        "note: chairs are below the 120-unit restocking threshold; lamps below 100; "
        "desks are fine at current demand",
        "billing 18%->24%, shipping 30%->27%, login 22%->21%, features 30%->28% "
        "(last month -> this month)",
        "quote: 'Your billing page has failed twice for me this month. I am losing patience.'",
    ],
    # episode 1 (same structure, different numbers)
    [
        "[140, 152, 149, 166, 171, 190]",
        "2026-03-01: 700 units; 2026-03-08: 720 units; 2026-03-15: 1010 units; "
        "2026-03-22: 940 units; 2026-03-29: 900 units",
        "Supplier A: 3.90/unit + 1000 fixed fee. Supplier B: 4.40/unit + 400 fixed fee.",
        "March total: 4270 units. April week 1: 950 units, week 2: 920 units, "
        "week 3: 900 units, week 4: 890 units",
        "SALES: Porto 13100; Lisbon 14900; Madrid 11800; Faro 7100",
        "Budget submission deadline: 2026-10-01. Design review: 2026-11-15.",
        "DESK: 259 EUR (10 in stock). CHAIR: 95 EUR (108 in stock). LAMP: 49 EUR (88 in stock).",
        "note: restock threshold is 120 for chairs and 100 for lamps; desks are adequate",
        "billing 15%->22%, shipping 32%->29%, login 23%->21%, features 30%->28%",
        "quote: 'The billing flow keeps erroring on my card. Please fix it or I cancel.'",
    ],
    # episode 2
    [
        "[201, 214, 209, 233, 228, 250]",
        "2026-03-01: 610 units; 2026-03-08: 990 units; 2026-03-15: 1050 units; "
        "2026-03-22: 920 units; 2026-03-29: 700 units",
        "Supplier A: 4.00/unit + 950 fixed fee. Supplier B: 4.55/unit + 380 fixed fee.",
        "March total: 4270 units. April week 1: 1000 units, week 2: 940 units, "
        "week 3: 920 units, week 4: 920 units",
        "SALES: Porto 12500; Lisbon 16100; Madrid 10900; Faro 6800",
        "Key dates: budget 2026-10-01, review 2026-11-15.",
        "DESK: 239 EUR (16 in stock). CHAIR: 85 EUR (101 in stock). LAMP: 42 EUR (99 in stock).",
        "note: chairs below 120 need restock; lamps below 100 need restock; desks fine",
        "billing 20%->27%, shipping 29%->26%, login 21%->20%, features 30%->27%",
        "quote: 'Billing failed again. Third time. I need this resolved today.'",
    ],
]


def build_episode(block: str, episode: int) -> EpisodeTasks:
    """Assemble the fixed task bank with the episode's input payloads.

    For the shift_mid block, episodes >= 2 swap the dependency structure:
    task 3 consumes task 0's output instead of task 1's, and task 7 consumes
    task 5's output instead of task 6's (mirrors the benchmark's Dependency
    Change perturbation).
    """
    specs = build_task_bank()
    inputs = EPISODE_INPUTS[min(episode, len(EPISODE_INPUTS) - 1)]
    if block == "shift_mid" and episode >= 2:
        rebuilt = []
        for t in specs:
            if t.id == 3:
                rebuilt.append(TaskSpec(
                    id=t.id, kind=t.kind, embedding=t.embedding, difficulty=t.difficulty,
                    prompt_template=(
                        "Using the week-over-week peak analysis ({dep}) and this April "
                        "log: {input}, state whether April's first week exceeds the "
                        "pre-March peak, yes or no, with the two numbers."),
                    reference_answer="YES; 990 vs 210",
                    rubric=(
                        "no valid comparison",
                        "correct direction, numbers imprecise",
                        "YES with 990 and 210 both cited"),
                    depends_on=0, unit_cost_hint=t.unit_cost_hint))
            elif t.id == 7:
                rebuilt.append(TaskSpec(
                    id=t.id, kind=t.kind, embedding=t.embedding, difficulty=t.difficulty,
                    prompt_template=(
                        "Using the extracted deadlines ({dep}) and this note: {input}, "
                        "write 2-3 sentences on which deadlines drive the schedule and "
                        "what should be restocked before them."),
                    reference_answer="budget date drives; CHAIR+LAMP restock",
                    rubric=(
                        "ignores deadlines or inventory",
                        "partial: one aspect missing",
                        "ties 2026-10-01 budget deadline to CHAIR+LAMP restocking"),
                    depends_on=5, unit_cost_hint=t.unit_cost_hint))
            else:
                rebuilt.append(t)
        specs = rebuilt
    return EpisodeTasks(specs=specs, inputs=inputs, episode=episode, block=block)


def declared_dependency_pairs(block: str, episode: int) -> list[tuple[int, int]]:
    """(upstream, downstream) declared dependency pairs for an episode."""
    if block == "shift_mid" and episode >= 2:
        return [(0, 3), (5, 7), (8, 9)]
    return [(1, 3), (6, 7), (8, 9)]

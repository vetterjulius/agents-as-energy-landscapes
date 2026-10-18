"""Task bank and deterministic episode data for the isolated LLM PoC rewrite."""
from __future__ import annotations

from dataclasses import dataclass, field, replace
import re

import torch

from . import config


@dataclass(frozen=True)
class AgentSlot:
    id: int
    name: str
    specialization: str
    system_prompt: str
    capability: tuple[float, ...]
    model: str | None = None


ANALYST_PROMPT = config.AGENT_SYSTEM_PROMPTS["analyst"]
EXTRACTOR_PROMPT = config.AGENT_SYSTEM_PROMPTS["extractor"]
SYNTHESIST_PROMPT = config.AGENT_SYSTEM_PROMPTS["synthesist"]


def build_agent_slots() -> list[AgentSlot]:
    """Build the three pinned worker-agent slots."""
    e = torch.eye(config.D)
    return [
        AgentSlot(0, "analyst", "analyze", ANALYST_PROMPT,
                  tuple((e[0] + e[1] + e[2]).tolist()), config.AGENT_MODELS["analyst"]),
        AgentSlot(1, "extractor", "extract", EXTRACTOR_PROMPT,
                  tuple((e[3] + e[4] + e[5]).tolist()), config.AGENT_MODELS["extractor"]),
        AgentSlot(2, "synthesist", "synthesize", SYNTHESIST_PROMPT,
                  tuple((e[6] + e[7]).tolist()), config.AGENT_MODELS["synthesist"]),
    ]


@dataclass(frozen=True)
class TaskSpec:
    id: int
    kind: str
    embedding: tuple[float, ...]
    difficulty: float
    prompt_template: str
    reference_answer: str
    rubric: tuple[str, ...]
    depends_on: int | None = None


@dataclass
class EpisodeTasks:
    specs: list[TaskSpec]
    inputs: list[str] = field(default_factory=list)
    episode: int = 0
    block: str = "stationary"


# Each row is an independently repeated episode; within a block, the shift changes
# the dependency graph after episode 0, leaving episode 2 to measure post-shift use.
EPISODE_INPUTS: list[list[str]] = [
    [
        "[182, 195, 188, 210, 205, 221]",
        "2026-03-01: 640 units; 2026-03-08: 980 units; 2026-03-15: 1020 units; 2026-03-22: 890 units; 2026-03-29: 740 units",
        "Supplier A: 4.20/unit + 900 fixed fee. Supplier B: 4.75/unit + 350 fixed fee.",
        "March total: 4270 units. April week 1: 990 units, week 2: 930 units, week 3: 910 units, week 4: 910 units",
        "SALES: Porto 12040; Lisbon 15600; Madrid 11200; Faro 6300",
        "Deadline for budget submission is 2026-10-01. The follow-up review is set for 2026-11-15. AOE drafting begins immediately.",
        "DESK: 249 EUR (14 in stock). CHAIR: 89 EUR (112 in stock). LAMP: 45 EUR (96 in stock).",
        "Restock thresholds: CHAIR < 120 units; LAMP < 100 units; DESK is adequate regardless of stock.",
        "billing 18%->24%, shipping 30%->27%, login 22%->21%, features 30%->28% (last month -> this month)",
        "quote: 'Your billing page has failed twice for me this month. I am losing patience.'",
    ],
    [
        "[140, 152, 149, 166, 171, 190]",
        "2026-03-01: 700 units; 2026-03-08: 720 units; 2026-03-15: 1010 units; 2026-03-22: 940 units; 2026-03-29: 900 units",
        "Supplier A: 3.90/unit + 1000 fixed fee. Supplier B: 4.40/unit + 400 fixed fee.",
        "March total: 4270 units. April week 1: 950 units, week 2: 920 units, week 3: 900 units, week 4: 890 units",
        "SALES: Porto 13100; Lisbon 14900; Madrid 11800; Faro 7100",
        "Budget submission deadline: 2026-10-01. Design review: 2026-11-15.",
        "DESK: 259 EUR (10 in stock). CHAIR: 95 EUR (108 in stock). LAMP: 49 EUR (88 in stock).",
        "Restock thresholds: CHAIR < 120 units; LAMP < 100 units; DESK is adequate regardless of stock.",
        "billing 15%->22%, shipping 32%->29%, login 23%->21%, features 30%->28%",
        "quote: 'The billing flow keeps erroring on my card. Please fix it or I cancel.'",
    ],
    [
        "[201, 214, 209, 233, 228, 250]",
        "2026-03-01: 610 units; 2026-03-08: 990 units; 2026-03-15: 1050 units; 2026-03-22: 920 units; 2026-03-29: 700 units",
        "Supplier A: 4.00/unit + 950 fixed fee. Supplier B: 4.55/unit + 380 fixed fee.",
        "March total: 4270 units. April week 1: 1000 units, week 2: 940 units, week 3: 920 units, week 4: 920 units",
        "SALES: Porto 12500; Lisbon 16100; Madrid 10900; Faro 6800",
        "Key dates: budget 2026-10-01, review 2026-11-15.",
        "DESK: 239 EUR (16 in stock). CHAIR: 85 EUR (101 in stock). LAMP: 42 EUR (99 in stock).",
        "Restock thresholds: CHAIR < 120 units; LAMP < 100 units; DESK is adequate regardless of stock.",
        "billing 20%->27%, shipping 29%->26%, login 21%->20%, features 30%->27%",
        "quote: 'Billing failed again. Third time. I need this resolved today.'",
    ],
]


def _embeddings() -> tuple[tuple[float, ...], ...]:
    e = torch.eye(config.D)
    categories = {
        "analyze": e[0] + e[1] + e[2],
        "extract": e[3] + e[4] + e[5],
        "synthesize": e[6] + e[7],
    }

    def pure(kind: str, jitter: float = 0.0) -> tuple[float, ...]:
        vec = categories[kind] * (1 - jitter) + jitter / config.D
        return tuple((vec / vec.sum()).tolist())

    def mixed(a: str, b: str) -> tuple[float, ...]:
        vec = (categories[a] + categories[b]) / 2
        return tuple((vec / vec.sum()).tolist())

    return (
        pure("analyze"), pure("extract"), pure("analyze", .1),
        mixed("analyze", "extract"), pure("extract", .1), pure("extract", .2),
        pure("extract", .05), mixed("synthesize", "extract"),
        pure("analyze", .15), mixed("synthesize", "analyze"),
    )


def build_task_bank() -> list[TaskSpec]:
    """Return the ten task templates; episode-specific references are filled later."""
    embeddings = _embeddings()
    definitions = [
        ("analyze", .4, "A delivery service recorded these weekly parcel counts: {input}. State the first-to-last trend, the week with the highest count, and the largest week-over-week increase with its adjacent weeks.", "", ("no meaningful analysis", "some requested facts correct", "trend, maximum and largest increase all correct"), None),
        ("extract", .3, "From this shipment log, extract the total units shipped in March as an integer, nothing else: {input}", "", ("wrong number", "correct number with extra text", "exact integer only"), None),
        ("analyze", .6, "Two suppliers quote unit prices: {input}. Which is cheaper for 1000 units including fixed fees, or are they tied? State both totals and the exact difference.", "", ("wrong comparison or arithmetic", "correct comparison but incomplete costs", "correct comparison and both exact totals"), None),
        ("analyze", .5, "Using the March shipping total ({dep} units) and this April log: {input}, compute the April-over-March percentage change to one decimal.", "", ("no valid percentage", "percentage with wrong base or rounding", "correct percentage to one decimal"), 1),
        ("extract", .35, "Extract the two cities with highest sales from: {input}. Format: CITY1, CITY2", "", ("wrong cities", "correct cities, format deviations", "exactly the two cities in order"), None),
        ("extract", .5, "List every ISO date mentioned in these minutes verbatim, comma-separated: {input}", "", ("misses or invents dates", "all dates with extra text", "exact dates, comma-separated"), None),
        ("extract", .45, "Extract the product names, unit prices and stock counts from: {input}. Format: NAME=PRICE,STOCK per line.", "", ("missing or wrong items", "all items with format deviations", "all exact product, price and stock values"), None),
        ("synthesize", .55, "Using the inventory ({dep}) and note: {input}, state which products need restocking and why in 2-3 sentences.", "", ("contradicts the data", "one item misclassified or vague", "correctly identifies all products below threshold"), 6),
        ("analyze", .65, "Support-ticket shares: {input}. Identify the category with the greatest share growth and quantify it in percentage points.", "", ("wrong category or no amount", "correct category, imprecise amount", "correct category and exact percentage-point growth"), None),
        ("synthesize", .7, "Given the ticket analysis ({dep}) and customer quote: {input}, write a 2-sentence customer-communication plan.", "prioritize the leading issue and acknowledge the quote", ("ignores issue or quote", "addresses only one", "addresses the leading issue and acknowledges the quote in 2 sentences"), 8),
    ]
    return [
        TaskSpec(i, kind, embeddings[i], difficulty, prompt, reference, rubric, depends_on)
        for i, (kind, difficulty, prompt, reference, rubric, depends_on) in enumerate(definitions)
    ]


def _episode_reference_answers(specs: list[TaskSpec], inputs: list[str]) -> list[TaskSpec]:
    """Derive all numeric/extraction gold answers from the input shown to the worker."""
    counts = [int(value) for value in re.findall(r"\d+", inputs[0])]
    deltas = [b - a for a, b in zip(counts, counts[1:])]
    march = [int(x) for x in re.findall(r"(\d+) units", inputs[1])]
    march_total = sum(march)
    supplier_quotes = re.findall(r"Supplier [AB]: ([\d.]+)/unit \+ (\d+) fixed fee", inputs[2])
    total_a = 1000 * float(supplier_quotes[0][0]) + int(supplier_quotes[0][1])
    total_b = 1000 * float(supplier_quotes[1][0]) + int(supplier_quotes[1][1])
    april = [int(x) for x in re.findall(r"week \d+: (\d+) units", inputs[3])]
    percent_change = round((sum(april) - march_total) / march_total * 100, 1)
    cities = re.findall(r"([A-Za-z]+) (\d+)", inputs[4])
    top_cities = [name.upper() for name, _ in sorted(cities, key=lambda pair: int(pair[1]), reverse=True)[:2]]
    dates = re.findall(r"\b\d{4}-\d{2}-\d{2}\b", inputs[5])
    products = re.findall(r"(DESK|CHAIR|LAMP): (\d+) EUR \((\d+) in stock\)", inputs[6])
    ticket_changes = re.findall(r"([a-z]+) (\d+)%->(\d+)%", inputs[8])
    leading = max(ticket_changes, key=lambda row: int(row[2]) - int(row[1]))
    restock = [name for name, _, stock in products if
               (name == "CHAIR" and int(stock) < 120) or (name == "LAMP" and int(stock) < 100)]
    ref = {
        0: (f"{'up' if counts[-1] > counts[0] else 'down' if counts[-1] < counts[0] else 'unchanged'}; "
            f"maximum {max(counts)} in week {counts.index(max(counts)) + 1}; "
            f"largest increase {max(deltas)} from week {deltas.index(max(deltas)) + 1} "
            f"to week {deltas.index(max(deltas)) + 2}"),
        1: str(march_total),
        2: f"Supplier A total {total_a:.0f}; Supplier B total {total_b:.0f}; " +
           ("tied" if total_a == total_b else f"Supplier {'A' if total_a < total_b else 'B'} cheaper by {abs(total_a-total_b):.0f}"),
        3: f"{percent_change:+.1f}% ({sum(april)} vs {march_total})",
        4: ", ".join(top_cities),
        5: ", ".join(dates),
        6: "\n".join(f"{name}={price},{stock}" for name, price, stock in products),
        7: ", ".join(restock) + " need restocking; DESK is adequate",
        8: f"{leading[0].upper()} +{int(leading[2])-int(leading[1])}pp ({leading[1]}% to {leading[2]}%)",
        9: f"prioritize {leading[0]} and acknowledge the customer's billing quote",
    }
    results = []
    for task in specs:
        answer = ref[task.id]
        if task.id == 7 and task.depends_on == 5:
            answer = f"{', '.join(dates)}; {', '.join(restock)} need restocking"
        if task.id == 3 and task.depends_on == 0:
            april_first = april[0]
            parcel_peak = max(counts)
            answer = f"{'YES' if april_first > parcel_peak else 'NO'}; {april_first} vs {parcel_peak}"
        results.append(replace(task, reference_answer=answer))
    return results


def build_episode(block: str, episode: int) -> EpisodeTasks:
    """Build an episode with input-derived answers and the mid-block graph shift."""
    if block not in ("stationary", "shift_mid"):
        raise ValueError(f"unknown task block {block!r}")
    if episode < 0:
        raise ValueError("episode must be non-negative")
    inputs = list(EPISODE_INPUTS[min(episode, len(EPISODE_INPUTS) - 1)])
    specs = build_task_bank()
    if block == "shift_mid" and episode >= 1:
        specs[3] = replace(
            specs[3], depends_on=0,
            prompt_template=(
                "Using the weekly parcel counts in {dep} and this April log: {input}, "
                "state whether April week 1 exceeds the maximum observed parcel count, "
                "yes or no, and give both exact counts."
            ),
            rubric=("no valid comparison", "correct direction, numbers imprecise",
                    "correct comparison and both exact counts"),
        )
        specs[7] = replace(
            specs[7], depends_on=5,
            prompt_template=(
                "Using the extracted deadline dates ({dep}) and this inventory note: "
                "{input}, name both dates and identify which products need restocking "
                "before the first deadline."
            ),
            rubric=("misses dates and restocking decision", "one aspect partly correct",
                    "both dates and all products below threshold are correct"),
        )
        stock = {name: count for name, _, count in re.findall(
            r"(DESK|CHAIR|LAMP): (\d+) EUR \((\d+) in stock\)", inputs[6]
        )}
        inputs[7] = (
            f"Inventory stock: DESK {stock['DESK']}; CHAIR {stock['CHAIR']}; LAMP {stock['LAMP']}. "
            "Restock thresholds: CHAIR < 120; LAMP < 100; DESK is adequate regardless of stock."
        )
    specs = _episode_reference_answers(specs, inputs)
    return EpisodeTasks(specs=specs, inputs=inputs, episode=episode, block=block)


def declared_dependency_pairs(block: str, episode: int) -> list[tuple[int, int]]:
    """Return upstream/downstream pairs; the shift starts after episode 0."""
    if block not in ("stationary", "shift_mid"):
        raise ValueError(f"unknown task block {block!r}")
    if episode < 0:
        raise ValueError("episode must be non-negative")
    if block == "shift_mid" and episode >= 1:
        return [(0, 3), (5, 7), (8, 9)]
    return [(1, 3), (6, 7), (8, 9)]

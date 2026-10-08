import pytest
from llm_poc.tasks import build_agent_slots

def test_per_slot_model_overrides():
    slots = build_agent_slots({
        "analyst": "gemini-3.5-flash-lite",
        "extractor": "gemma-4-26b-a4b-it",
    })
    assert slots[0].model == "gemini-3.5-flash-lite"
    assert slots[1].model == "gemma-4-26b-a4b-it"
    assert slots[2].model == "gemini-2.5-flash"


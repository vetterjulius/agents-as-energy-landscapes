"""LLM PoC package: external-validation experiment for orchestration transfer.

Layer 1 (energy formulation) and Layer 2 (solver) reuse the frozen benchmark
components unchanged; Layer 3 executes tasks with LLM workers (OpenRouter live
or deterministic mock). See paper_artifacts/llm_poc_design.md.
"""

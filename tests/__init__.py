"""
Author: Nigel J. Navarro
Test Suite

A set of automated tests for the ai-framework-sh framework. Run with pytest.

Contents:
    test_registry.py - ToolRegisry logic: register/get/schemas/execute, and its guardrails.
    test_safety.py   - Governance: Tests the built-in toolset against the model's ability to bypass guardrails. (ex. "ignore your instructions" prompt injection)
    test_memory.py   - MemoryStore: add/recall/delete round-trip, using a stubbed ModelClient

These are hermetic: they use a fake ModelClient and temporary directories where needed, so they run fast and deterministically without a live model 
or pre-built knowledge base.

Run: .venv/bin/pytest -v
"""
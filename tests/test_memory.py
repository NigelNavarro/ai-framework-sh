"""
Author: Nigel J. Navarro
Tests: MemoryStore (add / recall / list / delete round-trip)

Hermetic: swaps a stubbed ModelClient in place of the real one, and writes to a temp data_dir.
Tests the store's logic (dual write, recall ranking, delete-from-both, the client guard) all without
a live model.

Run: .venv/bin/pytest tests/test_memory.py -v
"""
from __future__ import annotations

import hashlib
import pytest

from framework.config import load_config
from framework.models.base import ModelClient
from framework.memory.store import MemoryStore

class StubModelClient(ModelClient):
    """
    A stub ModelClient that returns a deterministic embedding for testing.
    The same text always embeds to teh same vector (identical text -> distance 0).
    """
    def chat(self, messages):
        return "stub"           # Required by the ModelClient interface, but not used in these tests.

    def embed(self, texts):
        return [self._vec(text) for text in texts]  # self._vec(text) returns a deterministic vector for the given text.

    def _vec(self, text: str) -> list[float]:
        digest = hashlib.sha256(text.encode("utf-8")).digest()  # Hash the text to get a deterministic byte sequence.
        return [byte / 255.0 for byte in digest[:16]]           # Use the first 16 bytes of the hash to create a vector of floats in [0, 1].

@pytest.fixture
def store(tmp_path):
    """
    A fresh MemoryStore in a temporary directory with a stubbed ModelClient for each test.
    """
    config = load_config()
    config.data_dir = tmp_path
    return MemoryStore(config, StubModelClient())               # Use the stubbed ModelClient for deterministic embeddings.

def test_add_then_list(store):
    """
    Add a saved fact which should appear in the list() of memories with its text intact
    """
    store.add("This project is randomly named Bluefin Tuna.")
    facts = store.list()
    assert len(facts) == 1                                                      # Check that there is exactly one fact in the list after adding one.
    assert facts[0]["text"] == "This project is randomly named Bluefin Tuna."   # Check that the text of the fact matches what was added.

def test_recall_ranks_matching_fact_first(store):
    """
    The recall() Returns the nearest fact first. The exact match text has a distance of zero. 
    """
    store.add("This project is randomly named Bluefin Tuna.")
    store.add("An apple a day keeps the doctor away.")
    results = store.recall("This project is randomly named Bluefin Tuna.")
    assert results[0] == "This project is randomly named Bluefin Tuna."         # Check that the first result from recall is the exact match we added.

def test_delete_removes_from_both_stores(store):
    """
    Using delete() should clear the fact from the JSON list() and the Chroma collection (recall)
    """
    fact_id = store.add("An ephemeral fact.")
    assert store.delete(fact_id) is True                                        # Check that delete returns True, indicating successful deletion.
    assert store.list() == []                                                   # Check that the JSON list of facts is now empty after deletion.
    assert store.recall("An ephemeral fact.") == []                             # Check that recall returns an empty list, indicating the fact is no longer retrievable.

def test_add_without_client_raises(tmp_path):
    """
    If the MemoryStore is constructed without a ModelClient, add() should raise an error.
    """
    config = load_config()
    config.data_dir = tmp_path
    clientless = MemoryStore(config) # Construct a MemoryStore without a ModelClient, which should not allow adding facts.
    with pytest.raises(ValueError):
        clientless.add("Cannot embed this:")
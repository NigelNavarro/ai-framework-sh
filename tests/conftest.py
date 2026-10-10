"""
Author: Nigel J. Navarro
Shared test fixtures and stubs.
Hermetic with temp. directories only.

Fixtures defined here are auto-discovered by every test file in the folder with no import.
This allows tets like test_retrieval.py and test_memory.py to just name stub_client and tmp_config
instead of having to define them each time in their tests respectively.
"""
from __future__ import annotations
import hashlib
import pytest

from framework.config import load_config
from framework.models.base import ModelClient

class StubModelClient(ModelClient):
    """
    Deterministic stand-in for a real model. Hashes text to a vector, 
    so the same text always embeds to the same vector (distance 0). It otherwise makes it so that
    if you query with the exact text of a seeded chunk, that chunk is guaranteed to be the 
    nearest, allowing you to confirm which chunk comes back with no randomness applied.

    Same StubModelClient as seen in test_memory.py
    """
    def chat(self, messages):
        return "stub"           # Required by the ModelClient interface, but not used in these tests.

    def embed(self, texts):
        return [self._vec(text) for text in texts]  # self._vec(text) returns a deterministic vector for the given text.

    def _vec(self, text: str) -> list[float]:
        digest = hashlib.sha256(text.encode("utf-8")).digest()  # Hash the text to get a deterministic byte sequence.
        return [byte / 255.0 for byte in digest[:16]]           # Use the first 16 bytes of the hash to create a vector of floats in [0, 1].

@pytest.fixture
def stub_client():
    """
    A fresh deterministic model stub.
    When a test asks for "stub_client", hand it a fresh StubModelClient instance.
    Ensures tests do not each have to write StubModelClient() themselves.
    """
    return StubModelClient()

@pytest.fixture
def tmp_config(tmp_path):
    """
    A fresh temporary directory that the test can use so that test writes store nowhere real.
    """
    config = load_config()
    config.data_dir = tmp_path
    return config
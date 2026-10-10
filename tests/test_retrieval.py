"""
Author: Nigel J. Navarro
Tests: Document Retrieval Pipeline (Retriever + ChromaVectorStore)

A Hermetic Regression Test that features a stub ModelClient with deterministic vectors 
and a real Chroma store pointing a temporary directory.
The test checks that the retrieval returns the right document chunk for a query, and that
each chunk keeps it source and position as it makes the round trip through the vector store.

Run: .venv/bin/pytest tests/test_retrieval.py -v
"""
from __future__ import annotations
import pytest

from framework.knowledge.ingest import Chunk
from framework.knowledge.retrieve import Retriever
from framework.knowledge.vectorstore.chroma import ChromaVectorStore

@pytest.fixture
def seeded_retriever(stub_client, tmp_config):
    """
    A Retriever over a Chroma store seeded with three distinct (and fake) chunks.
    """
    store = ChromaVectorStore(tmp_config) # Creates a real vector store in the temp folder.
    chunks = [
        Chunk(text="Apples and Oranges are fruit.", source="fruit.txt", chunk_idx=0),
        Chunk(text="Ahab hunts the white whale.", source="whale.txt", chunk_idx=0),
        Chunk(text="Elizabeth marries Mr. Darcy.", source="pride.txt", chunk_idx=3),
    ]
    embeddings = stub_client.embed([chunk.text for chunk in chunks]) # Turns each chunk text into a vector using the stub, though in this test, each text always gets the same vector.
    store.add(chunks, embeddings) # Saves the three chunks and their vectors into the store.
    return Retriever(stub_client, store, top_k=3) # Hands back a ready-to-use Retriever class function, set to return 3 results

def test_retrieve_returns_matching_chunk_first(seeded_retriever):
    """
    Querying a chunk's exact text instead of a question. Returns that chunk first (distance 0)
    """
    results = seeded_retriever.retrieve("Ahab hunts the white whale.")
    assert results[0].source == "whale.txt" # Tests what is true. In this case, the closest chunk received must be the one from whale.txt.
                                            # Fails if changes break retrieval (wrong ranking, lost metadata, mangled query, etc.)

def test_retrieve_preserves_metadata(seeded_retriever):
    """
    Source and chunk_idx survive the roundm trip through ChromaDB.
    """
    # top = seeded_retriever.retrieve("Elizabeth marries Mr. Darcy.")[0]
    results = seeded_retriever.retrieve("Elizabeth marries Mr. Darcy.") # If at any point this doesn't match the chunks above (extra spaces, wrong title, etc.), assert will fire with an error.
    top = results[0]
    assert top.source == "pride.txt"    # Name of book is arbritary. What is more important is the consistency between fixture and assertion
    assert top.chunk_idx == 3           # Check to see if the chunk's position number survived the round trip through the DB, just like 'source' did.

def test_retrieve_respects_top_k(stub_client, tmp_config):
    """
    top_k caps how many chunks come back.
    """
    store = ChromaVectorStore(tmp_config)
    
    # Build Chunk objects like query() in the ChromaVectorStore
    # Create an object (in this case, just a filler document) for each pass, total of 5
    # We are checking top_k, the cap on how many results come back, so we're testing fort more items than the cap allows.
    chunks = [Chunk(text=f"doc number {i}", source=f"{i}.txt", chunk_idx=0)for i in range(5)]
    store.add(chunks, stub_client.embed([chunk.text for chunk in chunks]))
    retriever = Retriever(stub_client, store, top_k=2)
    assert len(retriever.retrieve("doc number 0")) == 2
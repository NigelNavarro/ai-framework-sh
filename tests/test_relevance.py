"""
Author: Nigel J. Navarro
Tests: Relevance Scores & the Knowledge Boundary (ChromaVectorStore + Retriever)

Methodology
-----------

These tests use chosen 2-dimensional vectors (arrows on a flat XY plane) so that the correct cosine score is known
exactly in advance.

    Stored vector vs query [1,0]                                    angle   expected score
    [10, 0] confirms vector is in the same direction,           0 degrees,      1.0
    [0,  1] confirms vector is perpendicular to the meaning,    90 degrees,     0.0
    [-1, 0] confirms vector is the opposite of the meaning,     180 degrees,    -1.0

The test hands these vectors to a real Chroma store kept in a temporary folder. It checks that the store gives back
these exact same scores. Doing so proves out the following:
  1. The collection is in fact using the cosine metric,
  2. query_with_scores flips distance to similarity correctly, and
  3. the vector length doesn't affect the overall score.

Run: .venv/bin/pytest tests/test_relevance.py -v
"""
from __future__ import annotations
import pytest

from framework.knowledge.ingest import Chunk
from framework.knowledge.retrieve import Retriever
from framework.knowledge.vectorstore.chroma import ChromaVectorStore
from framework.models.base import ModelClient

class FixedVectorClient(ModelClient):
    """
    A stub whose embed() returns the same chosen vector for every text. Each query's direction is known in advance.
    The stub lets each test decide which direction the question points so that the scores against the stored vectors
    are known before the test runs.
    """
    def __init__(self, vector: list[float]):
        self.vector = vector                    # The direction every question will point (ex. [1.0,0.0])

    def chat(self, messages):
        return "stub"                           # Satisfies the model interface. It is unused in these tests.

    def embed(self, texts):
        # embed() returns one vector per input text, in order.
        # We ignore what the text says and return our fixed vector once for each.
        # Therefore, a question is always embedded as self.vector.
        return [self.vector for _text in texts]


@pytest.fixture
def scored_store(tmp_config):
    """
    Uses a real Chroma store (in a temporary folder) seeded with three distinct (fake) chunks.

    Their vectors point in known directions. Every test that asks for 'scored_store' gets a fresh copy.
    Ensures tests can't affect each other. 
    """
    store = ChromaVectorStore(tmp_config)       # Creates a real vector store in the temp folder.

    # Three chunks. Their text doesn't matter here. Only their vectors below decide the scores.
    chunks = [
        Chunk(text="Same direction, ten times longer", source="same.txt", chunk_idx=0),
        Chunk(text="Perpendicular.", source="perpendicular.txt", chunk_idx=0),
        Chunk(text="Opposite direction.", source="opposite.txt", chunk_idx=0),
    ]

    # We skip over the model and hand-pick each chunk's vector.
    # embeddings[i] belongs to chunks[i].
    #   same.txt          -> [10, 0]   points right, 10x longer than the query will be
    #   perpendicular.txt -> [0, 1]    points straight up (90° from "right")
    #   opposite.txt      -> [-1, 0]   points left (180° from "right")
    store.add(chunks, [[10.0,0.0],[0.0,1.0],[-1.0,0.0]])
    return store

def test_scores_are_cosine_similarity(scored_store):
    # Search with a query pointing right ([1, 0]) and ask for all 3 chunks back, each paired with its score.
    results = scored_store.query_with_scores([1.0, 0.0], top_k=3)
    
    # result is a list of (chunk, score) pairs. 
    # Split it into two parallel lists so that we can check each separately.
    sources = [chunk.source for chunk, _score in results]       # Just the file names, in the order returned
    scores = [score for _chunk, score in results]               # Just the scores, in the same order

    # Order check: nearest-first means same direction, then perpendicular, then opposite.
    assert sources == ["same.txt","perpendicular.txt","opposite.txt"]

    # Value check: the exact cosine similarities from the table in the module docstring.
    # 'pytest.approx' allows a tiny tolerance (abs=1e-6) since floating-point math can be slightly off
    # ex. -- 0.9999999
    assert scores == pytest.approx([1.0, 0.0, -1.0], abs=1e-6)

def test_scores_ignore_vector_length(scored_store):
    """
    In this test, same.txt is stored as [10,0] but queried with [1, 0].
    The idea: Same direction, very different length.

    L2 (straight-line distance) - The distance between two vectors is 9.
    Cosine (angle only)         - Both vectors point straight along the x-axis, so the angle is 0 degrees 
                                with a similarity of 1.0.

    Ask for just the single best match (k=1), then take one pair from the list with [0] and unpack it
    """
    top_chunk, top_score = scored_store.query_with_scores([1.0, 0.0], top_k=1)[0]
    
    assert top_chunk.source == "same.txt"                # The longer vector still wins
    assert top_score == pytest.approx(1.0, abs=1e-6)    # with a perfect score since length is ignored.

def test_scores_stay_in_range(scored_store):
    """
    Query an in-between direction ([0.6, 0.8]). This is about 53 degrees up from the right.
    This ensures the scores are not neat round numbers for this test.
    Whatever the scores are, the cosine similarity must not leave the range -1.0 to 1.0.
    A score outside that range indicates a bug (ex. L2 distances being read like cosine)
    """
    for _chunk, score in scored_store.query_with_scores([0.6, 0.8], top_k=3):
        assert -1.0 - 1e-6 <= score <= 1.0 + 1e-6

def test_retrieve_drops_chunks_below_threshold(scored_store):
    """
    A Retriever whose questions always point to the right ([1, 0]) with a relevance threshold of 0.5.
    Expected scores:
        - same.txt = 1.0 (relevance is kept)
        - perpendicular.txt = 0.0 (relevance should be dropped)
        - opposite.txt = -1.0 (relevance should be dropped)
    """
    retriever = Retriever(FixedVectorClient([1.0, 0.0]), scored_store, top_k=3, min_score=0.5)
    
    # The question text is irrelevant. The stub turns any question into [1, 0].
    # retrieve() returns only the chunks at or above 0.5. We expect only same.txt should survive.
    assert [chunk.source for chunk in retriever.retrieve("anything")] == ["same.txt"]

def test_retrieve_returns_nothing_when_nothing_is_relevant(scored_store):
    """
    Questions this time point straight down ([0, -1]), away from every stored chunk.
    Expected scores:
        - same.txt = 0.0
        - perpendicular.txt = -1.0
        - opposite.txt = 0.0
    In this case, the best possible score is 0.0 which is below the 0.5 threshold.
    This tests the knowledge boundary.
    """
    retriever = Retriever(FixedVectorClient([0.0, -1.0]), scored_store, top_k=3, min_score=0.5)

    # An empty list is the boundary signal. The `ask` and `search_knowledge` tools 
    # report "nothing relevant" instead of passing on weak chunks.
    assert retriever.retrieve("something the corpus does not cover") == []

def test_retrieve_with_scores_stays_unfiltered(scored_store):
    """
    Questions this time point straight down ([0, -1]), away from every stored chunk.
    Expected scores:
        - same.txt = 0.0
        - perpendicular.txt = -1.0
        - opposite.txt = 0.0
    In this case, the best possible score is 0.0 which is below the 0.5 threshold.
    In this test: 
        - retrieve() returns nothing.
        - retrieve_with_scores() skips the threshold and returns all 3 scored chunks, so the raw 
        evidence stays available.
    """
    retriever = Retriever(FixedVectorClient([0.0, -1.0]), scored_store, top_k=3, min_score=0.5)

    # retrieve() returns nothing here. 
    # retrieve_with_scores() deliberately skips the threshold and returns all 3 scored chunks
    # with raw scores.
    # This keeps the boundary decision visible, as manual evaluations with a real model and known
    # questions require every score (incl. the low ones) to see where relevant and irrelevant
    # qeustions separate, and to choose the threshold from there.
    # A trace can also record the scores to explain why a question was judged out of bounds. 
    assert len(retriever.retrieve_with_scores("anything")) == 3



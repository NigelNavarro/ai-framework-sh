"""
Author Nigel J. Navarro
Grounding judge tests (hermetic)

The judge only reads text and records, so these tests build its inputs directly: an answer string, a Provenance, and a citation map ({citation number: filename}). No model, no loop, no database.

Each test sets up one situation and checks the graded fields of the GroundingReport it produces.
"""
from __future__ import annotations

import pytest

from framework.agent.grounding import judge_grounding, citation_map_from_chunks, citation_map_from_run
from framework.agent.loop import AgentRun, ToolCall
from framework.agent.provenance import Provenance, DOCUMENTS, MEMORY
from framework.knowledge.ingest import Chunk
from framework.tools.builtin import NO_KNOWLEDGE_RESULT

# This is what a boundary run looks like: the knowledge base was searched and nothing was relevant.
NOTHING = Provenance(boundary_hit=True)

# This is what a typical knowledge-route run looks like: two chunks retrieved from Moby Dick.
RETRIEVED = Provenance(sources=[DOCUMENTS], documents=["mobydick.txt"])
MAP = {1: "mobydick.txt", 2: "mobydick.txt"}

def test_knowledge_answer_citing_the_right_book_passes():
    """
    The answer cites [1], which exists and points to the expected file, so every check passes.
    """
    report = judge_grounding("Ahab captains the Pequod [1].", RETRIEVED, MAP, "knowledge", "mobydick.txt")

    assert report.checks == {"route": True, "citations_exist": True, "cites_expected": True}
    assert report.passed is True
    assert report.score == 1.0

def test_citation_to_a_source_that_was_never_retrieved_fails():
    """
    Only [1] and [2] were retrieved, so a [7] is invented: citations_exist fails and [7] is reported.
    """
    report = judge_grounding("Ahab [1] hunts the whale [7].", RETRIEVED, MAP, "knowledge", "mobydick.txt")

    assert report.citations_exist is False
    assert report.unknown_citations == [7]
    assert report.passed is False
    assert report.score == pytest.approx(2 / 3)        # Graded: 2 of 3 checks still passed.

def test_grouped_citations_are_read_as_separate_numbers():
    """
    Models sometimes group citations as "[1, 2]". Both numbers must be read, not skipped.
    """
    report = judge_grounding("Ahab captains the Pequod [1, 2].", RETRIEVED, MAP, "knowledge", "mobydick.txt")

    assert report.citations == [1, 2]
    assert report.passed is True

def test_citing_the_wrong_book_fails_cites_expected():
    """
    The retrieved chunks came from Frankenstein, but the question expects Moby Dick.
    """
    provenance = Provenance(sources=[DOCUMENTS], documents=["frankenstein.txt"])
    report = judge_grounding("Victor [1].", provenance, {1: "frankenstein.txt"}, "knowledge", "mobydick.txt")

    assert report.cites_expected is False
    assert report.cited_sources == ["frankenstein.txt"]

def test_knowledge_question_with_nothing_retrieved_fails_route():
    """
    A knowledge question should retrieve documents. Answering from general knowledge is the wrong route.
    """
    report = judge_grounding("Ahab is a captain.", NOTHING, {}, "knowledge", "mobydick.txt")

    assert report.route is False
    assert report.cites_expected is False

def test_boundary_answer_from_general_knowledge_passes():
    """
    Out-of-corpus question, nothing retrieved, answered without citations: allowed, because it is labeled.
    """
    report = judge_grounding("Leonardo da Vinci painted it.", NOTHING, {}, "boundary")

    assert report.provenance_label == "general knowledge"
    assert report.passed is True
    assert report.cites_expected is None                # Not applicable to a boundary question.
    assert report.declined is False

def test_boundary_answer_that_declines_passes_and_is_flagged():
    """
    Declining is also correct at the boundary. The declined flag records it without changing the score.
    """
    report = judge_grounding("The sources don't contain that information.", NOTHING, {}, "boundary")

    assert report.passed is True
    assert report.declined is True

def test_boundary_question_that_retrieved_documents_fails_route():
    """
    If documents came back for an out-of-corpus question, the threshold let something through.
    """
    report = judge_grounding("Something [1].", RETRIEVED, MAP, "boundary")

    assert report.route is False

def test_citation_with_nothing_retrieved_is_invented():
    """
    Nothing was retrieved, so there is nothing to cite. Any [n] in the answer was made up.
    """
    report = judge_grounding("Leonardo [1].", NOTHING, {}, "boundary")

    assert report.citations_exist is False
    assert report.unknown_citations == [1]

def test_citations_are_not_checkable_when_memory_was_also_recalled():
    """
    'search_knowledge' and 'recall' both number from '[1]', so [1] is ambiguous. The checks report None instead of a guess.
    """
    provenance = Provenance(sources=[DOCUMENTS, MEMORY], documents=["mobydick.txt"])
    report = judge_grounding("Ahab [1], and you like whales [1].", provenance, MAP, "knowledge", "mobydick.txt")

    assert report.citations_exist is None
    assert report.cites_expected is None
    assert report.score == 1.0                          # Only the route check applied, and it passed.

def test_capped_run_fails_route():
    """
    The round cap was hit, so there is no real answer to grade.
    """
    report = judge_grounding("I couldn't complete the request.", Provenance(capped=True), {}, "knowledge", "mobydick.txt")

    assert report.provenance_label == "no answer"
    assert report.route is False

def test_unknown_expected_route_raises():
    """
    A typo in the golden set should fail loudly, not produce a misleading grade.
    """
    with pytest.raises(ValueError):
        judge_grounding("Answer.", RETRIEVED, MAP, "knowlege", "mobydick.txt")

def test_citation_map_from_chunks_matches_the_ask_numbering():
    """
    'ask' numbers its chunks from [1] in retrieval order, so the map must do the same.
    """
    chunks = [Chunk(text="a", source="tests/books/mobydick.txt", chunk_idx=3),
              Chunk(text="b", source="tests/books/frankenstein.txt", chunk_idx=9)]

    assert citation_map_from_chunks(chunks) == {1: "mobydick.txt", 2: "frankenstein.txt"}

def test_citation_map_from_run_uses_the_last_successful_search():
    """
    Each search restarts at [1], so the answer's citations refer to the most recent search that found chunks.
    """
    first = ToolCall("search_knowledge", {"query": "a"}, "[1] frankenstein.txt (chunk 1):\ntext")
    second = ToolCall("search_knowledge", {"query": "b"}, "[1] mobydick.txt (chunk 2):\ntext")
    empty = ToolCall("search_knowledge", {"query": "c"}, NO_KNOWLEDGE_RESULT)
    run = AgentRun(answer="x", run_id="test", tool_calls=[first, second, empty])

    assert citation_map_from_run(run) == {1: "mobydick.txt"}
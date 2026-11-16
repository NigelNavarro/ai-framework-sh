"""
Author: Nigel J. Navarro
Provenance labeling tests (hermetic)

Each test performs the following:
  1. In advance, the test writes out the replies a fake model will give. For example, the first reply
     asks to call 'search_knowledge', and the second reply is the final answer.
  2. The real agent loop runs using those replies. When a reply asks for a tool, the loop runs 
     the real 'search_knowledge' or 'recall' tool and records what it returned. 
  3. The test checks the provenance label that the framework assigns from that record.

These tests do not touch the model or the database directly. 
The tests run the real loop and real tools, but we use the following to swap out the outside pieces:
the model (ScriptedClient) and the data sources (FakeRetriever and FakeMemory). These fakes feed in
chosen replies and data. This is so every test sets up an exact situation wihtout needing the real
model and database.
  - 'ScriptedClient' replays a fixed list of replies instead of calling a model. 
  - 'FakeRetriever' and 'FakeMemory' stand in for the real Retriever and MemoryStore.

Visual Diagram:

Real (The actual code)                    Fake (The test stand-ins)
──────────────────────                     ─────────────────────
agent loop (run_agent_with_record)    ←──  ScriptedClient    (replaces gemma4)
      │
      ▼
search_knowledge tool ─── calls ─────────► FakeRetriever     (replaces Retriever + Chroma)
recall tool ───────────── calls ─────────► FakeMemory        (replaces MemoryStore)
      ▲
built by the real factories:
  - make_search_knowledge_tool() 
  - make_recall_tool()

"""
from __future__ import annotations
import json

from framework.agent.loop import run_agent, run_agent_with_record
from framework.agent.provenance import label_provenance, DOCUMENTS, MEMORY, GENERAL, NO_ANSWER
from framework.knowledge.ingest import Chunk
from framework.models.base import ModelClient
from framework.tools.builtin import make_search_knowledge_tool, make_recall_tool, parse_knowledge_sources
from framework.tools.registry import ToolRegistry

class ScriptedClient(ModelClient):
    """
    Returns the given replies in order one per chat() call. A tool call is just a JSON reply.
    """
    model = "scripted"

    def __init__(self, replies: list[str]):
        self.replies = list(replies)

    def chat(self, messages):
        return self.replies.pop(0)          # Take the first remaining reply and cross it off. 
        # We go with this instead of self.replies[0] since the latter would read the first item without removing it,
        # meaning it would return the same tool call every round. The model would never reach its answer, and the loop
        # would always hit the cap.
        # pop() with no argument would just take from the very end, and the replies would come out backwards, answer first.
        # Therefore, the loop would finish in round one without ever calling the tool. 

    def embed(self, texts):
        return [[0.0] for _text in texts]   # Required by the interface, but otherwise not used here.

class FakeRetriever:
    """
    Stands in for the real Retriever.
    Returns the same chunks for any query. An empty list means a boundary miss. 
    """
    def __init__(self, chunks: list[Chunk]):
        self.chunks = chunks

    def retrieve(self, query):
        return self.chunks

class FakeMemory:
    """
    Stands in for the MemoryStore.
    Returns the same facts for any query. 
    """
    def __init__(self, facts: list[str]):
        self.facts = facts

    def recall(self, query):
        return self.facts

def tool_call(name: str, **arguments) -> str:
    """
    The JSON reply a model sends to request a tool.
    """
    return json.dumps({"tool": name, "arguments": arguments})

def make_registry(chunks=(), facts=()) -> ToolRegistry:
    """
    Build a registry with the real 'search_knowledge' and 'recall' tools, wired to fakes that return given data. 
    """
    registry = ToolRegistry()
    registry.register(make_search_knowledge_tool(FakeRetriever(list(chunks))))  # empty = boundary miss
    registry.register(make_recall_tool(FakeMemory(list(facts))))                # empty = nothing remembered
    return registry

AHAB = Chunk(text="Ahab is the captain of the Pequod.", source="tests/books/mobydick.txt", chunk_idx=12)
VICTOR = Chunk(text="Victor studied at Ingolstadt.", source="tests/books/frankenstein.txt", chunk_idx=40)

def test_no_tool_called_is_general_knowledge():
    """
    The model answers straight away, so nothing was retrieved.
    """
    run = run_agent_with_record(ScriptedClient(["Paris."]), make_registry(), "Capital of France?")
    provenance = label_provenance(run)

    assert provenance.label == GENERAL
    assert provenance.boundary_hit is False

def test_search_with_results_is_documents():
    """
    The model searches the knowledge base, which returns two chunks, then answers.
    Search returned chunks, so the label is "documents", with both files listed in first-seen order.
    """
    replies = [tool_call("search_knowledge", query="Ahab"), "Ahab captains the Pequod [1]."]
    run = run_agent_with_record(ScriptedClient(replies), make_registry(chunks=[AHAB, VICTOR]), "Who is Ahab?")
    provenance = label_provenance(run)

    assert provenance.label == DOCUMENTS
    assert provenance.documents == ["mobydick.txt", "frankenstein.txt"]

def test_search_with_no_results_is_general_knowledge_with_boundary_hit():
    """
    The knowledge base was searched, but nothing met the threshold, so any answer comes from the model directly.
    """
    replies = [tool_call("search_knowledge", query="Mona Lisa"), "Leonardo da Vinci."]
    run = run_agent_with_record(ScriptedClient(replies), make_registry(chunks=[]), "Who painted the Mona Lisa?")
    provenance = label_provenance(run)

    assert provenance.label == GENERAL
    assert provenance.boundary_hit is True
    assert "nothing relevant was found" in provenance.summary()

def test_recall_with_facts_is_memory():
    """
    The model calls recall, which finds a saved fact, then answers.
    Recall returned something and nothing else was retrieved, so the label is "memory".
    """
    replies = [tool_call("recall", query="editor"), "You prefer VS Code [1]."]
    run = run_agent_with_record(ScriptedClient(replies), make_registry(facts=["Prefers VS Code"]), "My editor?")

    assert label_provenance(run).label == MEMORY

def test_documents_and_memory_together():
    """
    The model searches the knowledge base (finds a chunk), then calls recall (finds a fact), then answers.
    Both sources returned something, so the label names both: "documents + memory".
    """
    replies = [tool_call("search_knowledge", query="Ahab"), tool_call("recall", query="book"), "Answer."]
    registry = make_registry(chunks=[AHAB], facts=["Is reading Moby Dick"])
    run = run_agent_with_record(ScriptedClient(replies), registry, "Who is Ahab, and what am I reading?")

    assert label_provenance(run).label == f"{DOCUMENTS} + {MEMORY}"

def test_failed_tool_call_does_not_count_as_a_source():
    """
    An unknown tool raises inside the loop and is recorded as an error, so it returned no information.
    """
    replies = [tool_call("no_such_tool", query="x"), "Answer."]
    run = run_agent_with_record(ScriptedClient(replies), make_registry(chunks=[AHAB]), "Q?")

    assert run.tool_calls[0].error is not None
    assert label_provenance(run).label == GENERAL

def test_round_cap_is_no_answer():
    """
    The model keeps asking for tools and never answers, so the loop stops at max_rounds.
    """
    replies = [tool_call("search_knowledge", query="Ahab")] * 2
    run = run_agent_with_record(ScriptedClient(replies), make_registry(chunks=[AHAB]), "Q?", max_rounds=2)
    provenance = label_provenance(run)

    assert run.capped is True
    assert provenance.label == NO_ANSWER

def test_run_agent_still_returns_plain_text():
    """
    Existing callers of run_agent() keep getting a str, not an AgentRun.
    """
    answer = run_agent(ScriptedClient(["Paris."]), make_registry(), "Capital of France?")
    assert answer == "Paris."

def test_parse_knowledge_sources_reads_the_formatter_back():
    """
    Whatever 'search_knowledge' writes, 'parse_knowledge_sources' must read back: number -> filename.
    """
    observation = make_search_knowledge_tool(FakeRetriever([AHAB, VICTOR])).handler(query="anything")
    assert parse_knowledge_sources(observation) == {1: "mobydick.txt", 2: "frankenstein.txt"}
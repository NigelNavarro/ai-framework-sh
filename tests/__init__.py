"""
Author: Nigel J. Navarro
Test Suite

A set of automated tests for the ai-framework-sh framework. Run with pytest.

Contents:
    test_grounding.py   - Grounding judge logic: route / citations / expected source, graded not pass-fail.
                        - The judge reads an answer, its provenance, and a citation map, so tests build those directly (no model, no loop).
                        - A correct citation to the expected book passes every check (score 1.0).
                        - An invented citation (e.g. [7] when only [1] and [2] were retrieved) fails citations_exist.
                        - Grouped citations like [1, 2] are read as separate numbers.
                        - Citing the wrong book fails cites_expected; a knowledge question with nothing retrieved fails route.
                        - At the boundary, answering from general knowledge or declining both pass, and declining is flagged.
                        - When memory was also recalled, citations are ambiguous, so those checks report None instead of guessing.
                        - The citation maps number chunks the same way ask does, and the agent's map uses its last successful search.

    test_memory.py      - MemoryStore: add / recall / delete round-trip, & "no client" guard.
                        - Facts added with add() show up in list() with its text intact.
                        - Then recall() ranks the exact match fact ahead of an unrelated one.
                        - After, delete() removes the fact from both places it lives (memory.json and Chroma memory collection).
                        - Lastly, this test confirms a MemoryStore built wthout a ModelClient refuses to add facts, since it can't embed them.

    test_provenance.py  - Provenance logic: the label the framework derives from what the agent loop observed.
                        - Each test scripts a fake model's replies and runs the real loop with the real tools (fake retriever and memory, so no Ollama or Chroma).
                        - Covers documents, memory, both together, general knowledge (no search, or a boundary miss), failed tool calls, and the round cap ("no answer").
                        - run_agent() still returns plain text, and parse_knowledge_sources() reads back exactly what search_knowledge writes.
                        - An empty model reply is retried instead of being returned as the answer, and only-empty replies stop at the round cap.

    test_registry.py    - ToolRegisry logic: register / get / schemas / execute, and its guardrails (duplicates, unknown names, handler never advertised)
                        - A registered tool can be fetched back by its name.
                        - It rejects registering another tool under the same name of one that already exists.
                        - An error is rased for cases where a name is fetched but was never registered.
                        - schemas() advertises each tool's name, description, parameters, but never its handler.
                        - execute() calls the right handler with the given arguments.

    test_relevance.py   - Relevance logic: cosine scores / threshold / knowledge boundary.
                        - Hand-picked 2D vectors have cosine scores known in advance, and the store returns exactly those scores, nearest-first.
                        - Scores ignore vector length (same direction, different length = 1.0).
                        - Scores always stay within -1.0 to 1.0.
                        - The Retriever's threshold keeps only chunks at or above it.
                        - When no chunk meets the threshold, retrieve() returns nothing (the knowledge boundary).
                        - retrieve_with_scores() skips the threshold, so the raw scores stay available.

    test_retrieval.py   - Retrieval logic: match ranking / metadata round-trip / top_k limit.
                        - A query brings back its matching chunk first.
                        - Each chunk keeps its source file and chunk index after the round trip through the vector store.
                        - The Retriever never returns more than top_k results.

    test_safety.py      - Governance logic: safe toolset / drafts sandbox.
                        - The built-in toolset is exactly the known safe set, so adding a new tool forces a deliberate update to this test.
                        - draft_document, the only tool that writes files, cannot be tricked into writing outside data/drafts/ with a 
                        path-traversal filename (e.g. "../../secrets.txt").    

Support files:
    conftest.py         - Shared fixtures: stub_client / tmp_config, loaded automatically by pytest.
                        - stub_client is a fake ModelClient that turns text into a repeatable vector by hashing it, so identical text always gets an identical vector.
                        - tmp_config is a Config whose data_dir points to a temporary folder, so tests never touch real data.

    books/              - Fixture texts: Frankenstein / Moby Dick / Pride and Prejudice.
                        - The source texts the golden Q&A set (eval/golden_qa.json) was written from.           

Run: .venv/bin/pytest -v
"""
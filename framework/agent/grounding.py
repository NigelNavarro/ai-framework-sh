"""
Author: Nigel J. Navarro
Grounding Judge: Grade an answer against the source it should have come from.

Provenance (provenance.py) records what was retrieved for an answer. This module checks what the answer did with the answer, and returns a graded report (see: GroundingReport below) instead of a single pass or fail.

The checks are deterministic (no model call), so they give us the same result every time:
  - route:              Did the answer take the route the golden set expects?
                        "knowledge" expects documents to have been retrieved. 
                        "boundary" expects no documents (Declining or answering from general knowledge are both acceptable as long as the provenance label says so)
  - citations_exist:    Does every [n] in the answer, point to a source that was actually retrieved?
  - cites_expected:      For a knowledge question, does the answer cite the expected file? 

A check that doesn't apply is reported as None and is left out of the score. 

Limitation: 
  - Citation numbers are only unambiguous when they come from one numbered list.
  - 'search_knowledge' and 'recall' both number their results from [1], so when memory is also recalled, a [1] in the answer could mean either source.
  - In this case, the citation checks are reported as None (not checkable) rather than guessed.
  - When the agent searches more than once, the citations are mapped against the last successful search. 

Futurestate:
  - Explore "guessing smarter". The complexity here may not be worth it if in the end it is still a guess, but worth investigating nevertheless.
  - Avoid the problem at the source: make 'search_knowledge' keep numbering across calls ([6], [7], ... on the second search).
"""
from __future__ import annotations

import re

from dataclasses import dataclass
from pathlib import Path

from framework.agent.loop import AgentRun
from framework.agent.provenance import Provenance, DOCUMENTS, MEMORY
from framework.knowledge.ingest import Chunk
from framework.tools.builtin import NO_KNOWLEDGE_RESULT, parse_knowledge_sources

# An inline citation in an answer: a single "[3]" or a grouped "[1, 2]". Captures the number(s) inside.
_CITATION = re.compile(r"\[(\d+(?:\s*,\s*\d+)*)\]")

# Phrases that suggest the answer declined because the sources didn't cover the question.
# This is a heuristic: it is reported for information only and never affects the score.
_DECLINE_PHRASES = (
    "don't contain", "do not contain", "doesn't contain", "does not contain",
    "not in the sources", "not in the provided", "no information", "not mentioned",
    "cannot find", "can't find", "couldn't find", "could not find",
    "unable to answer", "i don't know", "i do not know",
)

@dataclass
class GroundingReport:
    """
    The graded result of judging one answer. Each check is True, False, or None (not applicable).
    """
    expected_route: str                 # From the golden set: "knowledge" or "boundary".
    provenance_label: str               # What the framework observed, e.g. "documents" or "general knowledge".
    route: bool                         # Did the answer take the expected route?
    citations_exist: bool | None        # Does every [n] point to a retrieved source? None if not checkable.
    cites_expected: bool | None         # Knowledge only: is the expected file among the cited ones?
    citations: list[int]                # The [n] numbers found in the answer, sorted.
    unknown_citations: list[int]        # Cited numbers that match no retrieved source.
    cited_sources: list[str]            # Filenames the valid citations point to.
    declined: bool                      # Heuristic, informational only: did the answer say it couldn't answer?

    @property
    def checks(self) -> dict[str, bool | None]:
        """
        Every check by name, so callers can print or count them the same way.
        """
        return {"route": self.route, "citations_exist": self.citations_exist, "cites_expected": self.cites_expected}

    @property
    def score(self) -> float:
        """
        The fraction of applicable checks that passed, from 0.0 to 1.0. Checks that are None don't count.
        """
        applicable = [result for result in self.checks.values() if result is not None]
        return sum(applicable) / len(applicable)

    @property
    def passed(self) -> bool:
        """
        True only when every applicable check passed.
        """
        return all(result for result in self.checks.values() if result is not None)

def citation_map_from_chunks(chunks: list[Chunk]) -> dict[int, str]:
    """
    Number chunks the same way the 'ask' prompt does (context.format_sources, starting at [1]),
    giving {citation number: filename}. Example: [chunk from mobydick.txt] becomes {1: "mobydick.txt"}.
    """
    return {number: Path(chunk.source).name for number, chunk in enumerate(chunks, start=1)}

def citation_map_from_run(run: AgentRun) -> dict[int, str]:
    """
    Rebuild {citation number: filename} from an agent run's LAST successful search_knowledge call,
    since each search restarts its numbering at [1]. An empty dict means no search returned chunks.
    """
    for call in reversed(run.tool_calls):                       # Walk backwards to find the most recent search.
        if call.name == "search_knowledge" and call.error is None and call.observation != NO_KNOWLEDGE_RESULT:
            return parse_knowledge_sources(call.observation)
    return {}

def judge_grounding(answer: str, provenance: Provenance, citation_map: dict[int, str], expected_route: str, expected_source: str | None = None) -> GroundingReport:
    """
    Grade one answer.

    Args:
        answer:             The final answer text.
        provenance:         What was retrieved for it (from label_provenance, or built directly for `ask`).
        citation_map:       {citation number: filename} for the sources the answer could cite.
        expected_route:     "knowledge" or "boundary", from the golden set.
        expected_source:    For a knowledge question, the filename that should be cited.
    """
    if expected_route not in ("knowledge", "boundary"):
        raise ValueError(f"Unknown expected_route {expected_route!r}")

    # Route: knowledge needs retrieved documents; boundary needs none. A capped run gave no answer at all.
    retrieved_documents = DOCUMENTS in provenance.sources
    if provenance.capped:
        route = False
    elif expected_route == "knowledge":
        route = retrieved_documents
    else:
        route = not retrieved_documents

    # Each match is the text inside one pair of brackets, e.g. "1, 2". Split it on commas to get every number.
    citations = sorted({int(number) for group in _CITATION.findall(answer) for number in group.split(",")})
    unknown = [number for number in citations if number not in citation_map]
    cited_sources = list(dict.fromkeys(citation_map[n] for n in citations if n in citation_map))

    # Citation checks only make sense when [n] can mean just one thing (see the module docstring).
    checkable = MEMORY not in provenance.sources
    citations_exist = (not unknown) if checkable else None
    
    if expected_route == "knowledge" and checkable:
        cites_expected = expected_source in cited_sources
    else:
        cites_expected = None

    return GroundingReport(
        expected_route=expected_route,
        provenance_label=provenance.label,
        route=route,
        citations_exist=citations_exist,
        cites_expected=cites_expected,
        citations=citations,
        unknown_citations=unknown,
        cited_sources=cited_sources,
        declined=any(phrase in answer.lower() for phrase in _DECLINE_PHRASES),
    )
"""
Author: Nigel J. Navarro
Provenance: Label every answer with where its information came from.

The label is decided by the framework from what the agent loop actually observed (the AgentRun record), and never by asking the model. 

Labels:
   - "documents":           'search_knowledge' returned chunks from 'knowledge_base'
   - "memory":              'recall' returned saved facts
   - "general knowledge":   No source returned anything (no tool was called or KB was searched but nothing met the relevance threshold.)
   - "no answer":           The round cap was hit, so the reply is the fixed fallback message.

The label tells us what was retrieved, not what the answer used. 
Retrieved chunks can be offered to the model, but the model can still choose to ignore them.
Checking whether the answer actually stayed grounded in those chunks is checked separately by the grounding judge (see framework/agent/grounding.py)
"""
from __future__ import annotations
from dataclasses import dataclass, field

from framework.agent.loop import AgentRun
from framework.tools.builtin import NO_KNOWLEDGE_RESULT, NO_MEMORY_RESULT, parse_knowledge_sources

DOCUMENTS = "documents"
MEMORY = "memory"
GENERAL = "general knowledge"
NO_ANSWER = "no answer"

@dataclass
class Provenance:
    """
    Where one answer's information came from as observed by the framework. 
    """
    sources: list[str] = field(default_factory=list)    # The routes that returned something: DOCUMENTS and/or MEMORY.
    documents: list[str] = field(default_factory=list)  # Filenames of retrieved chunks, (ex. ["mobydick.txt"]).
    boundary_hit: bool = False                          # TRUE if the knowledge base was searched and nothing was relevant.
    capped: bool = False                                # TRUE if the run hit 'max_rounds' without an answer.

    @property
    def label(self) -> str:
        """
        The short label: "documents", "memory", "documents + memory", "general knowledge", or "no answer".
        """
        if self.capped:
            return NO_ANSWER
        return " + ".join(self.sources) if self.sources else GENERAL

    def summary(self) -> str:
        """
        A one line human-readable explanation of the label.
        For printing under an answer.
        """
        if self.capped:
            return f"{NO_ANSWER} (the round limit was reached before the agent answered.)"
        if self.sources:
            parts = []
            if DOCUMENTS in self.sources:
                parts.append(f"documents retrieved ({', '.join(self.documents)})")
            if MEMORY in self.sources:
                parts.append("memories recalled")
            return " + ".join(parts)
        if self.boundary_hit:
            return f"{GENERAL} (the knowledge base was searched, but nothing relevant was found.)"
        return f"{GENERAL} (no documents or memories were retrieved.)"

def label_provenance(run: AgentRun) -> Provenance:
    """
    Read an AgentRun's tool calls and decide where the answer's information came from.

    A tool call only counts as a source when it succeeded and returned something:
        - 'search_knowledge' counts when its result held chunks (anything other than NO_KNOWLEDGE_RESULT).
        - 'recall' counts when its result held facts (anything other than NO_MEMORY_RESULT).
    
    Errors, empty results, and Write-only tools (remember, draft_document) never count.
    """
    provenance = Provenance(capped=run.capped)

    for call in run.tool_calls:
        if call.error is not None:
            continue                                    # A failed call returned no information.

        if call.name == "search_knowledge":
            if call.observation == NO_KNOWLEDGE_RESULT:
                provenance.boundary_hit = True          # Was searched, but nothing met the threshold
                continue
            if DOCUMENTS not in provenance.sources:     # Stops document from being added to the list twice
                provenance.sources.append(DOCUMENTS)
            for name in parse_knowledge_sources(call.observation).values():
                if name not in provenance.documents:        # Keep each filename once, first-seen ordering
                    provenance.documents.append(name)

        elif call.name == "recall":
            if call.observation != NO_MEMORY_RESULT and MEMORY not in provenance.sources:
                provenance.sources.append(MEMORY)

    return provenance
# Design Overview

`ai-framework-sh` is a self-hosted, single-user AI assistant built as a learning project. It is a **modular monolith**: one local application, internally decoupled into independent pillars behind clean interfaces, so components (the model, the vector store) can be swapped without touching the rest.

This document is a map. Each pillar has its own README with the detailed design notes, and the governance decisions live in [DECISIONS.md](DECISIONS.md).

## The app/framework boundary

The application depends on the framework, never the reverse. The CLI is simply the first _consumer_; a web or voice front-end could sit on the same framework unchanged.

- **Application:** [`apps/cli`](../apps/cli/README.md), the reference consumer (an `argparse` CLI).
- **Framework:** everything under `framework/`, app-agnostic and reusable.

## Architecture at a glance

Each labeled group is a pillar, and the indented names are the `.py` files inside it. Dependencies flow downward (upper layers use lower ones), and the application only ever talks to the framework, never the reverse.

```text
APPLICATION
  apps/cli/main.py   ·   chat   ingest   ask   agent   memory   trace
        │
        │  the app depends on the framework (never the reverse)
        ▼
FRAMEWORK
  │
  ├─ agent/                    orchestration
  │    context.py   grounded, cited prompt assembly          [verb: ask]
  │    loop.py      run_agent: think -> act -> observe       [verb: agent]
  │                   │  advertises tool schemas, dispatches tool calls
  │                   ▼
  ├─ tools/                    model-callable capabilities
  │    registry.py  Tool registry: name -> schema + handler
  │    builtin.py   search_knowledge, remember, recall, draft_document
  │                   │        │              │
  │           wraps ──┘        │              └──► writes data/drafts/
  │                            ▼
  ├─ knowledge/                RAG (documents)
  │    ingest.py    documents -> Chunks
  │    retrieve.py  query -> nearest chunks
  │    vectorstore/
  │      base.py    VectorStore interface
  │      chroma.py  Chroma implementation   (data/chroma/)
  │
  ├─ memory/                   durable facts
  │    store.py     dual store: memory.json + a Chroma collection
  │
  ├─ models/                   the LLM runtime
  │    base.py      ModelClient interface   (chat, embed)
  │    ollama.py    Ollama implementation
  │
  └─ cross-cutting (used by every pillar)
       config.py               typed settings (model, paths, chunk size, top_k)
       observability/trace.py  append-only JSONL audit of every action

Shared dependencies (arrows mean "uses"):
  retrieve.py      ─► vectorstore + models.embed()
  memory/store.py  ─► its own Chroma collection + models.embed()
  builtin tools    ─► retrieve.py (search), store.py (remember/recall), drafts
  loop.py          ─► registry.execute()  and  models.chat()
```

## The pillars

| Pillar        | Responsibility                                                 | README                                                                          |
| ------------- | -------------------------------------------------------------- | ------------------------------------------------------------------------------- |
| Models        | The LLM runtime, behind a `ModelClient` interface (pluggable). | [framework/models](../framework/models/README.md)                               |
| Knowledge     | Ingest and retrieve documents (RAG).                           | [framework/knowledge](../framework/knowledge/README.md)                         |
| Vector Store  | Semantic storage behind a `VectorStore` interface (pluggable). | [framework/knowledge/vectorstore](../framework/knowledge/vectorstore/README.md) |
| Agent         | Prompt assembly and the tool-using loop.                       | [framework/agent](../framework/agent/README.md)                                 |
| Tools         | The registry and model-callable capabilities.                  | [framework/tools](../framework/tools/README.md)                                 |
| Memory        | Durable, inspectable facts (dual JSON and vector store).       | [framework/memory](../framework/memory/README.md)                               |
| Observability | Append-only JSONL tracing of every action.                     | [framework/observability](../framework/observability/README.md)                 |

## How a request flows

- **`ask` (fixed RAG, Level 1):** take the question, embed it, retrieve the top-k chunks, assemble a grounded and cited prompt, then answer.
- **`agent` (agentic, Level 1-2):** the model decides each round whether to answer or call a tool (`search_knowledge`, `remember`, `recall`, `draft_document`); the loop runs the tool, feeds the result back, and repeats up to a hard cap.

Every model call, retrieval, and command is recorded to the trace, so any answer can be reconstructed after the fact.

## Cross-cutting principles

- **Dependency inversion.** The framework depends on interfaces (`ModelClient`, `VectorStore`), not implementations, which is what makes components swappable and testable (the test suite substitutes a stub `ModelClient` for the real one).
- **Safety by constraint.** Every capability sits at Autonomy Levels 0 to 2 (read-only or local drafting); nothing acts on the outside world. See [DECISIONS.md](DECISIONS.md) for the full governance model.

## See also

- [DECISIONS.md](DECISIONS.md): the ten governance questions answered for this project.

# ai-framework-sh

A self-hosted AI assistant **framework**, a reusable Python substrate for building local, private AI assistants on top of [Ollama](https://ollama.com).

This is a personal project: a learning-driven, from-scratch build of the core pillars an AI assistant needs: a model runtime, retrieval-augmented generation (RAG) over your own
documents, and an agent loop with tools, durable memory, and observability. A command-line app is its first consumer.

> **Status:** ✅ **Working MVP (`v0.2`).** All core pillars (model runtime, RAG, agent loop, tools, memory, and observability) are built, tested, and verified
> end-to-end. It runs entirely locally and stays within a read-only + drafting safety envelope (no real-world actions). Built as a learning project; capable and
> usable, but not yet hardened for production or multi-user use.

## What it can do

- **Cited Q&A over your documents:** ingest a folder of notes, ask questions, and get answers grounded strictly in those sources with `[n]` citations (and a plain "not in the sources" when the answer isn't there, instead of a hallucination).
- **An agent that decides:** in `agent` mode the model chooses when to search your documents, save or recall a memory, or draft a document, through a bounded tool-calling loop.
- **Durable, inspectable memory:** tell it a preference or fact once and it persists across sessions, stored in a human-readable file you can list, review, and delete.
- **Safe drafting:** it writes drafts (emails, notes) to a local folder for you to review. It never sends or shares anything.
- **Observability by default:** every model call is appended to a per-day JSONL trace, so any answer can be reconstructed after the fact.

## This is a framework, not an application

When I set out to do this project, I wanted to learn how to leverage the power of AI beyond that of a simple chat interface to a specific subscription-service LLM. From the beginning, I was most interested in creating a harness that can leverage the power of multiple LLMs and achieve creating a system that can handle multiple tasks at once.

The build in this current state is just the first step in that learning (though even this took a little while longer to put together than I initially expected it would take). Its current core functions are:

- The **framework** (`framework/`) is app-agnostic, with reusable machinery: model abstraction, RAG, agent loop, tools, memory, observability.
- An **application** (`apps/`) is one _consumer_ of the framework, a CLI today, but perhaps a web or voice front-end later.

The dependency only points one way: _the app depends on the framework, never the reverse._ That's what makes the same core reusable across different interfaces. For more information, see [`docs/DESIGN.md`](docs/DESIGN.md) to see the full architecture.

## Requirements

- **macOS / Linux** (developed on Apple Silicon)
- **Python 3.12** (pinned, as newer versions lag on ML library support)
- **[Ollama](https://ollama.com)** installed and running, with two models pulled:

```bash
ollama pull gemma4            # reasoning / chat
ollama pull nomic-embed-text  # embeddings for RAG and memory
```

Two models by design: one runtime (Ollama) serves both reasoning and embeddings.

## Setup

```bash
# 1. Clone, then enter the project
git clone https://github.com/NigelNavarro/ai-framework-sh.git
cd ai-framework-sh

# 2. Create an isolated virtual environment with Python 3.12
python3.12 -m venv .venv
source .venv/bin/activate         # prompt shows (.venv) when active

# 3. Install the framework (plus test tooling)
pip install -e ".[dev]"
```

## Configuration

Settings live in `framework/config.py` as typed defaults, the framework runs with **zero configurations** out of the box (model names, paths, chunk size, `top_k`, the agent's `max_tool_rounds`). To override anything, copy the example and edit it:

```bash
cp config.example.toml config.toml
```

`config.toml` is gitignored; `config.example.toml` is the committed template showing every available option. Because the model name lives in config, swapping models is a config edit, not a code change.

## Usage

All commands are subcommands of the CLI. Make sure that Ollama is running first.

```bash
# Ingest a file or a folder of .md/.txt documents into the knowledge base
.venv/bin/python apps/cli/main.py ingest path/to/notes/

# Ask a grounded, cited question (always retrieves from the knowledge base)
.venv/bin/python apps/cli/main.py ask "What does the onboarding doc say about VPN setup?"

# Agent mode: the model decides which tools to use (search / remember / recall / draft)
.venv/bin/python apps/cli/main.py agent "Remember that I prefer concise answers."
.venv/bin/python apps/cli/main.py agent "Draft a short thank-you email to my team and save it as thanks.md"

# Plain conversational REPL (no tools, no retrieval)
.venv/bin/python apps/cli/main.py chat
```

Memory and trace commands are **human controls** and do not need Ollama running:

```bash
# Inspect and forget stored memories
.venv/bin/python apps/cli/main.py memory list
.venv/bin/python apps/cli/main.py memory delete <id>

# See what the assistant recently did (the observability trail)
.venv/bin/python apps/cli/main.py trace tail -n 10
```

### `ask` vs. `agent`

Both answer questions, but differently:

- **`ask`** _always_ retrieves and prints a `Sources:` list. You get deterministic, guaranteed citations. Good when you want a grounded answer every time.
- **`agent`** lets the model _decide_ whether to search, remember, recall, or draft. A general question is answered directly; a document question triggers a search. This is the agentic path.

## Where your data lives

Everything stays on your machine, under `data/` (gitignored):

| Path                  | What                                           |
| --------------------- | ---------------------------------------------- |
| `data/chroma/`        | The embedded vector store (documents + memory) |
| `data/memory.json`    | Durable memories, human-readable               |
| `data/drafts/`        | Drafts written by `draft_document`             |
| `data/traces/*.jsonl` | The append-only audit log, one file per day    |

There is no cloud service, external API, or network egress in normal operation in this version. Inference runs locally through Ollama.

## Testing

The suite is hermetic (a stub `ModelClient` stands in for Ollama, temp dirs for disk), so it runs fast and offline:

```bash
.venv/bin/pytest -v
```

It covers the tool registry, the memory store round-trip, and a safety check that the built-in toolset stays read-only / write-only-inside-`data/` (including that `draft_document` cannot escape its folder via a path-traversal filename).

## Architecture

A **modular monolith**: one local app, internally decoupled into independent pillars behind clean interfaces.

```text
apps/cli/main.py     chat · ingest · ask · agent · memory · trace   (the app)
        │  depends on ↓ (never the reverse)
framework/
  agent/        context.py (grounded prompts) · loop.py (the tool-using loop)
  tools/        registry.py · builtin.py (search / remember / recall / draft)
  knowledge/    ingest.py · retrieve.py · vectorstore/ (Chroma behind an interface)
  memory/       store.py (dual: memory.json + a Chroma collection)
  models/       base.py (ModelClient interface) · ollama.py (implementation)
  config.py     typed settings   ·   observability/trace.py  (JSONL audit)
```

Each pillar has its own README with detailed design notes. Full map: [`docs/DESIGN.md`](docs/DESIGN.md).

## Safety & autonomy

Every capability sits at **Autonomy Levels 0 to 2** (chat, read-only, local drafting), and nothing acts on the outside world. The one writing tool (`draft_document`) is confined to `data/drafts/`. Higher autonomy (sending, scheduling, external access) is deliberately out of scope for this MVP, and would sit behind approval gates and an audit log. The full governance model (the ten questions of what to log, what to never access, what would need approval) is in [`docs/DECISIONS.md`](docs/DECISIONS.md).

## Beyond the MVP

Deferred by design, in rough priority order:

1. **Web / voice front-ends:** a second app on the same framework, unchanged.
2. **Supervised actions (Level 3):** a write-capable tool behind an approval gate and
   audit log.
3. **Richer, per-tool-call tracing:** replay the agent loop step by step.
4. **RAG quality:** reranking, PDF ingestion, a golden-Q&A regression set.
5. **Memory:** in-place `memory edit`, categories, optional expiry.

## Documentation

- [`docs/DESIGN.md`](docs/DESIGN.md): architecture and the app/framework boundary
- [`docs/DECISIONS.md`](docs/DECISIONS.md): the ten governance questions, answered
- Per-pillar READMEs under `framework/*/`: detailed design notes for each module

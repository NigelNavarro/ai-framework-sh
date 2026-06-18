# ai-framework-sh

A self-hosted AI assistant **framework** — a reusable Python substrate for building
local, private AI assistants on top of [Ollama](https://ollama.com).

This is a learning-driven, from-scratch build of the core pillars an AI assistant
needs: a model runtime, retrieval-augmented generation (RAG) over your own
documents, an agent loop with tools, memory, and observability. A command-line app
is its first consumer.

> **Status:** 🚧 Early development. Day 1 complete — project scaffolding, config
> system, and a verified connection to a local model. See the roadmap below.

## Framework vs. application

This repo is a **framework**, not a finished app. The distinction matters:

- The **framework** (`framework/`) is app-agnostic reusable machinery: model
  abstraction, RAG, agent loop, memory.
- An **application** (`apps/`, planned) is one _consumer_ of the framework — a CLI
  to start, perhaps a web UI later.

The dependency only points one way: **the app depends on the framework, never the
reverse.** That's what makes the same core reusable across different interfaces.

## Requirements

- **macOS / Linux** (developed on Apple Silicon)
- **Python 3.12** (pinned — newer versions lag on ML library support)
- **[Ollama](https://ollama.com)** installed and running, with a model pulled:
  ```bash
  ollama pull gemma4
  ```

## Setup

```bash
# 1. Clone, then enter the project
cd ai-framework-sh

# 2. Create an isolated virtual environment with Python 3.12
python3.12 -m venv .venv
source .venv/bin/activate        # prompt shows (.venv) when active

# 3. (Later) install dependencies as they're added
#    pip install -e ".[dev]"
```

## Configuration

Settings live in `framework/config.py` as typed defaults — the framework runs with
**zero config** out of the box. To override anything, copy the example and edit it:

```bash
cp config.example.toml config.toml
```

`config.toml` is gitignored (your personal settings); `config.example.toml` is the
committed template showing every available option.

## Verify your setup

A small smoke test confirms your config can reach Ollama and get a reply:

```bash
.venv/bin/python smoke_test.py
```

Expected: the model introduces itself. If so, your environment is working.

## Roadmap

| Pillar          | What it does                                          | Status     |
| --------------- | ----------------------------------------------------- | ---------- |
| Config          | Typed settings, one source of truth                   | ✅ Done    |
| Model runtime   | Ollama adapter (`chat`, `embed`), streaming           | ⬜ Planned |
| Knowledge / RAG | Ingest docs → chunk → embed → retrieve with citations | ⬜ Planned |
| Agent loop      | Bounded tool-calling loop                             | ⬜ Planned |
| Tools           | Registry + read-only built-in tools                   | ⬜ Planned |
| Memory          | Inspectable, editable semantic memory                 | ⬜ Planned |
| Observability   | Append-only trace of every model/tool step            | ⬜ Planned |

**Autonomy target (MVP):** read-only Q&A + drafting. No real-world actions.

## License

_TBD._

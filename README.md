# ai-framework-sh

A self-hosted AI assistant **framework**, a reusable Python substrate for building local, private AI assistants on top of [Ollama](https://ollama.com).

This is a personal project: a learning-driven, from-scratch build of the core pillars an AI assistant needs: a model runtime, retrieval-augmented generation (RAG) over your own documents, and an agent loop with tools, durable memory, and observability. A command-line app is its first consumer.

> **Status:** ✅ **`v0.3`: Harden the Core.** All core pillars (model runtime, RAG, agent loop, tools, memory, and
> observability) are built, tested, and verified end-to-end, and v0.3 makes them measurable: relevance scores with a
> knowledge boundary, provenance on every answer, and a golden-set eval with a grounding judge. It runs entirely
> locally and stays within a read-only + drafting safety envelope (no real-world actions). Built as a learning
> project; capable and usable, but not yet hardened for production or multi-user use.

## What it can do

- **Knows what it doesn't know:** retrieval scores every chunk, and when nothing in your documents is relevant enough, it says so instead of stretching unrelated text into an answer.
- **Labels where every answer came from:** each answer ends with a provenance line (documents, memory, or general knowledge), decided by the framework from what actually happened, not by the model's own claim.
- **Cited Q&A over your documents:** ingest a folder of notes, ask questions, and get answers grounded strictly in those sources with `[n]` citations (and a plain "not in the sources" when the answer isn't there, instead of a hallucination).
- **An agent that decides:** in `agent` mode the model chooses when to search your documents, save or recall a memory, or draft a document, through a bounded tool-calling loop.
- **Durable, inspectable memory:** tell it a preference or fact once and it persists across sessions, stored in a human-readable file you can list, review, and delete.
- **Safe drafting:** it writes drafts (emails, notes) to a local folder for you to review. It never sends or shares anything.
- **Observability by default:** every model call and every agent step (each tool call, its result, and any error) is appended to a per-day JSONL trace under a shared run id, so any answer can be reconstructed and replayed after the fact.

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

Settings live in `framework/config.py` as typed defaults, the framework runs with **zero configurations** out of the box (model names, paths, chunk size, `top_k`, `relevance_threshold`, the agent's `max_tool_rounds`). To override anything, copy the example and edit it:

```bash
cp config.example.toml config.toml
```

`config.toml` is gitignored; `config.example.toml` is the committed template showing every available option. The model name lives in the config, making swapping models therefore a configuration change as opposed to a code change.

`relevance_threshold` (default `0.60`) is the cosine similarity a chunk must reach to count as relevant. It is specific to the embedding model: if you change `embed_model`, re-ingest your documents and re-calibrate it with `eval/run_eval.py --threshold`.

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

# Replay one agent run, step by step (the run id is shown in trace tail)
.venv/bin/python apps/cli/main.py trace show <run-id>
```

### `ask` vs. `agent`

Both answer questions, but differently:

- **`ask`** _always_ retrieves and prints a `Sources:` list. You get deterministic, guaranteed citations. Good when you want a grounded answer every time. If nothing in the knowledge base is relevant, it says so without calling the model.
- **`agent`** lets the model _decide_ whether to search, remember, recall, or draft. It is told to search your documents first, and falls back to general knowledge only when the search finds nothing relevant. This is the agentic path.

Both end with a `Provenance:` line saying where the answer's information came from.

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

It covers the tool registry, the memory store round-trip, retrieval ranking, cosine relevance scores and the threshold, provenance labeling, the grounding judge, and a safety check that the built-in toolset stays read-only / write-only-inside-`data/` (including that `draft_document` cannot escape its folder via a path-traversal filename).

## Evaluation

`tests/` checks that the code works; `eval/` measures how well the live system performs, against a golden set of questions with known answers (`eval/golden_qa.json`). Both scripts need Ollama running and an ingested corpus, and only read:

```bash
.venv/bin/python eval/run_eval.py        # retrieval: right book found, out-of-corpus questions refused
.venv/bin/python eval/run_grounding.py   # full agent answers, graded by the grounding judge
```

See [`eval/README.md`](eval/README.md) for how to read the output and calibrate the threshold.

## Architecture

A **modular monolith**: one local app, internally decoupled into independent pillars behind clean interfaces.

```text
apps/cli/main.py      the app: chat · ingest · ask · agent · memory · trace
eval/                 live measurement: golden_qa.json · run_eval.py · run_grounding.py

framework/            the reusable core (both apps/ and eval/ import from it; it imports from neither)
  agent/              context.py (grounded prompts) · loop.py (the tool-using loop)
                      provenance.py (where an answer came from) · grounding.py (the grounding judge)
  tools/              registry.py · builtin.py (search / remember / recall / draft)
  knowledge/          ingest.py · retrieve.py (scores + threshold) · vectorstore/ (Chroma behind an interface)
  memory/             store.py (dual: memory.json + a Chroma collection)
  models/             base.py (ModelClient interface) · ollama.py (implementation)
  observability/      trace.py (JSONL audit)
  config.py           typed settings
```

Each pillar has its own README with detailed design notes. Full map: [`docs/DESIGN.md`](docs/DESIGN.md).

## Safety & autonomy

Every capability sits at **Autonomy Levels 0 to 2** (chat, read-only, local drafting), and nothing acts on the outside world. The one writing tool (`draft_document`) is confined to `data/drafts/`. Higher autonomy (sending, scheduling, external access) is deliberately out of scope for this version, and would sit behind approval gates and an audit log. The full governance model (the ten questions of what to log, what to never access, what would need approval) is in [`docs/DECISIONS.md`](docs/DECISIONS.md).

## What's next

Deferred by design, in rough priority order (see [`ROADMAP.md`](ROADMAP.md)):

1. **Supervised actions (Level 3):** a write-capable tool behind an approval gate and audit log.
2. **External knowledge sources (governed):** consult something like web search when a question falls outside your documents, with source labeling and injection safeguards.
3. **Web / voice front-ends:** a second app on the same framework, unchanged.
4. **RAG quality:** reranking, PDF ingestion, and a fresh golden set.
5. **Memory:** in-place `memory edit`, categories, optional expiry.

## Documentation

- [`docs/DESIGN.md`](docs/DESIGN.md): architecture and the app/framework boundary
- [`docs/DECISIONS.md`](docs/DECISIONS.md): the ten governance questions, answered
- Per-pillar READMEs under `framework/*/`: detailed design notes for each module
- [`ROADMAP.md`](ROADMAP.md) and [`CHANGELOG.md`](CHANGELOG.md): where it's headed, and what shipped when
- [`eval/README.md`](eval/README.md): the golden set and the two eval scripts

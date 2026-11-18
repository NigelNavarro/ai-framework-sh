# Changelog

All notable changes to `ai-framework-sh` are recorded here. The format follows
[Keep a Changelog](https://keepachangelog.com/), and this project aims to follow
[Semantic Versioning](https://semver.org/) (pre-1.0, so minor versions may include
larger changes). For where the project is headed, see [ROADMAP.md](ROADMAP.md).

## [Unreleased]

_Work toward v0.4 (Supervised Actions) will be recorded here as it lands._

## [0.3.0] - 2026-11-16

Tag: `v0.3-harden-core`. Milestone 3: Harden the Core. Makes the existing pillars measurably good and visible before any new power is added. Stays within Autonomy Levels 0 to 2.

### Added

- **Per-run tracing** (`framework/agent/loop.py`): every agent run carries a short `run_id`, and the
  loop records `agent_start`, `tool_call` (with latency and an `error` field), `agent_end`, and
  `cap_reached` events, so a run can be filtered out of the trace and replayed (#1, #2, #3).
- **Trace replay** (`trace show <run-id>`): prints one run's events in order (#4).
- **Golden Q&A set** (`eval/golden_qa.json`): 9 in-corpus questions with an expected source book and
  5 out-of-corpus questions, each with an `expected_route` of `knowledge` or `boundary` (#5).
- **Hermetic retrieval regression test** (`tests/test_retrieval.py`) with a shared stub client in
  `tests/conftest.py`, and the book fixtures in `tests/books/` (#6).
- **Relevance scores and the knowledge boundary** (#9): `VectorStore.query_with_scores()` and
  `Retriever.retrieve_with_scores()` return cosine similarity scores; `Retriever.retrieve()` keeps
  only chunks at or above `relevance_threshold`, and an empty result means the question is outside
  what the knowledge base covers. `ask` and `search_knowledge` say so plainly.
- **Retrieval eval** (`eval/run_eval.py`): scores the golden set by source hit rate and boundary
  accuracy, with `--threshold` to compare values (#8).
- **Provenance labels** (`framework/agent/provenance.py`): every `ask` and `agent` answer is labeled
  with where its information came from (documents, memory, general knowledge, or no answer),
  decided by the framework from what the loop observed, never by the model (#7).
- **Grounding judge** (`framework/agent/grounding.py`): grades an answer with separate checks
  (route, citations exist, expected source cited) and a score, instead of a single pass or fail;
  `eval/run_grounding.py` runs it over the golden set (#7).
- **Tests:** `test_relevance.py`, `test_provenance.py`, and `test_grounding.py` (45 tests in total).

### Changed

- **Cosine distance** for the `knowledge_base` collection (was L2), with a guardrail that refuses a
  collection built on another metric. Requires a one-time re-index.
- **`relevance_threshold`** defaults to **0.60**, calibrated for `nomic-embed-text` against the golden
  set. Model specific: re-calibrate when `embed_model` changes.
- **`run_agent()`** is now a thin wrapper over `run_agent_with_record()`, which returns an `AgentRun`
  record (the answer plus every tool call). `run_agent()` still returns plain text.
- **Agent system prompt**: tells the model to search the user's documents before answering from
  general knowledge (golden-set route accuracy went from 12/14 to 14/14).
- The test corpus now lives in the repo at `tests/books/` instead of a separate project.
- Bumped version to 0.3.0.

### Fixed

- **Empty model replies**: a thinking model (gemma4 in testing) occasionally returns no visible text;
  the loop now retries with a nudge instead of returning an empty answer.

### Known Limitations

- The golden set has been used to tune both the threshold and the prompt, so fresh questions are
  needed before its scores count as independent.
- One golden question ("Where does Victor Frankenstein go to university?") fails because retrieval
  does not surface the passage with the answer; the model declines rather than inventing one.
- Citations are mapped against the agent's last successful search, and are not checkable when
  memory was also recalled (both number their results from [1]).

## [0.2.0] - 2026-08-18

Tag: `v0.2-mvp`. Milestone 2: read-only + drafting MVP, tested and documented.

### Added

- **Agent loop** (`framework/agent/loop.py`): a bounded think -> act -> observe loop with
  prompt-based tool calling (model-agnostic; no native tool API), hard-capped by
  `config.max_tool_rounds`.
- **Tool registry** (`framework/tools/registry.py`): a `Tool` dataclass and `ToolRegistry`
  that advertises tool schemas to the model and dispatches calls to handlers.
- **Built-in tools** (`framework/tools/builtin.py`): `search_knowledge`, `remember`,
  `recall`, and `draft_document`.
- **Durable memory** (`framework/memory/store.py`): a dual store (human-readable
  `memory.json` plus a Chroma collection) with add/recall/list/delete, keyed by a shared id.
- **Safe drafting**: `draft_document` writes to `data/drafts/` and never sends; filenames
  are sanitized to block path traversal.
- **CLI commands**: `agent`, `memory list|delete`, and `trace tail`.
- **Test suite** (`tests/`): hermetic tests for the registry, a safety allowlist plus a
  path-traversal check, and a memory round-trip using a stubbed `ModelClient` (no Ollama).
- **Documentation**: `docs/DESIGN.md`, `docs/DECISIONS.md`, and a README per pillar.

### Changed

- Rewrote `README.md` to reflect the finished MVP (capabilities, setup, usage, safety).
- Relocated inline teaching notes from source files into per-module READMEs.
- Scoped pytest to `tests/` so bare `pytest` no longer collects Ollama-dependent scripts.
- Bumped version to 0.2.0.

## [0.1.0] - 2026-07-10

Tag: `v0.1-readonly`. Milestone 1: read-only cited Q&A over local documents.

### Added

- **Config** (`framework/config.py`): typed settings with sensible defaults, optionally
  overridden by `config.toml`.
- **Model runtime** (`framework/models/`): a `ModelClient` interface and an Ollama adapter
  with chat, streaming, and embeddings.
- **Knowledge / RAG** (`framework/knowledge/`): document ingestion with overlapping chunks,
  a `VectorStore` interface, an embedded Chroma implementation, and a retriever.
- **Cited Q&A**: the `ask` command retrieves relevant chunks, assembles a grounded prompt,
  and answers strictly from those sources with `[n]` citations (declining when the answer
  is not present).
- **Observability** (`framework/observability/trace.py`): an append-only, per-day JSONL
  trace of model calls.
- **CLI**: `chat`, `ingest`, and `ask` subcommands.

[Unreleased]: https://github.com/NigelNavarro/ai-framework-sh/compare/v0.3-harden-core...HEAD
[0.3.0]: https://github.com/NigelNavarro/ai-framework-sh/compare/v0.2-mvp...v0.3-harden-core
[0.2.0]: https://github.com/NigelNavarro/ai-framework-sh/compare/v0.1-readonly...v0.2-mvp
[0.1.0]: https://github.com/NigelNavarro/ai-framework-sh/releases/tag/v0.1-readonly

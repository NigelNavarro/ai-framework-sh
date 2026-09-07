# Changelog

All notable changes to `ai-framework-sh` are recorded here. The format follows
[Keep a Changelog](https://keepachangelog.com/), and this project aims to follow
[Semantic Versioning](https://semver.org/) (pre-1.0, so minor versions may include
larger changes). For where the project is headed, see [ROADMAP.md](ROADMAP.md).

## [Unreleased]

_Work toward v0.3 (Harden the Core) will be recorded here as it lands._

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

[Unreleased]: https://github.com/NigelNavarro/ai-framework-sh/compare/v0.2-mvp...HEAD
[0.2.0]: https://github.com/NigelNavarro/ai-framework-sh/compare/v0.1-readonly...v0.2-mvp
[0.1.0]: https://github.com/NigelNavarro/ai-framework-sh/releases/tag/v0.1-readonly

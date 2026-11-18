# Roadmap

A directional view of where `ai-framework-sh` is headed, by version. This is a living
document: themes and ordering may shift as the project learns. For what has already
shipped, see [CHANGELOG.md](CHANGELOG.md); for architecture and governance, see
[docs/DESIGN.md](docs/DESIGN.md) and [docs/DECISIONS.md](docs/DECISIONS.md).

**Guiding principle:** never expand power faster than observability and governance can
keep up. Each release should either prove the framework, deepen trust in what it already
does, or carefully expand autonomy, in that order of priority.

## Shipped

- **v0.1: Read-only cited Q&A.** RAG over local documents: ingest, retrieve, and answer
  grounded strictly in sources with `[n]` citations. (Autonomy Level 1.)
- **v0.2: MVP.** Agent loop with prompt-based tool calling, durable memory
  (remember/recall + human controls), safe local drafting, observability tracing, a
  hermetic test suite, and full documentation. (Autonomy Levels 0 to 2.)
- **v0.3: Harden the Core.** Per-run tracing with replay (`trace show`), cosine relevance scores with
  a calibrated knowledge boundary, provenance labels on every answer, a golden Q&A set, a retrieval
  eval, and a route-aware grounding judge. (Autonomy Levels 0 to 2.)

## Planned

- **v0.4: Supervised Actions (Level 3).** The first action that reaches beyond
  `data/drafts/`, behind an explicit approval gate and audit log: the agent proposes, the
  system surfaces the exact action and arguments, it executes only on approval, and both
  the request and the decision are logged. Enabled by the v0.3 audit trail.

- **v0.4 or later: External knowledge sources (governed).** When a question falls outside
  the local corpus and memory, let the agent consult an external source such as web search,
  and cite it. This is read-only but still a governed capability: the query leaves the
  machine and untrusted text enters the model's context, so it ships with an explicit
  egress decision, source labeling, and injection safeguards. Enabled by the v0.3 boundary
  detection and provenance work. Deliberately outside v0.3.

## Under consideration (v0.5+)

- **A second interface (Web / voice front-end)** on the same framework, unchanged, to
  prove the app/framework boundary with a real second consumer.
- **Multi-model routing:** pick a model per task (a fast model to decide, a larger one to
  answer). Echoes the project's founding "leverage multiple LLMs" motivation.
- **Memory maturation:** in-place `memory edit`, categories/tags, auto-injection of
  relevant memories into context, dedup, optional expiry.
- **Pluggable vector-store backend** (for example a server-based or SQL/distributed store),
  enabled by the existing `VectorStore` interface.
- **RAG quality:** reranking, PDF ingestion, and a fresh golden set (the current one has been used
  for tuning).

_Dates are intentionally omitted; this is a learning project and milestones ship when they
are ready and verified._

# Memory: durable, inspectable facts

Durable memory: facts and preferences that survive across sessions (as opposed to chat history, which dies with the session). It is a dual store by design, so a fact can be reached two ways: as human-readable truth (saved data), and by meaning (references).

## Files

- `store.py`, which provides `MemoryStore` (`add`, `recall`, `list`, `delete`).

## Design notes

- **A dual store, kept in lockstep.** Every fact lives in two places, linked by a shared id:
  1. `data/memory.json`, the human-readable source of truth (open it, read it, edit or delete facts by hand).
  2. a Chroma `"memory"` collection, the semantic recall index (find facts by meaning, not keywords).

  Every write hits both, and `delete` removes from both, so a "forgotten" fact can never resurface from a leftover vector.

- **Reuses the framework's own pieces.** `MemoryStore` takes a `ModelClient` (the interface, not Ollama directly) to embed, and uses Chroma for vectors: the same read path as document RAG, pointed at a personal collection instead of the document one. It uses a _separate_ collection from `knowledge_base`, so personal facts never mix with document chunks.
- **The client is optional, and failures are honest.** `add` and `recall` embed, so they need a `ModelClient`; `list` and `delete` do not. The constructor accepts `ModelClient | None`, and `add`/`recall` fail fast with a clear error if it is missing. This makes the type signature tell the truth (memory can exist without a model; only two methods need one) and lets the CLI manage memory with no model running. (See DECISIONS.md §7.)
- **A unique id with no coordination.** `uuid.uuid4().hex` mints a distinct id per fact without tracking a counter, and that id is the glue linking the JSON record to its vector.

## Related

- **Exposed as tools:** `remember` / `recall` in [`tools`](../tools/README.md), plus the `memory` CLI command in [`apps/cli`](../../apps/cli/README.md).
- **Storage:** uses Chroma directly (see [`knowledge/vectorstore`](../knowledge/vectorstore/README.md)).
- **Governance:** [DECISIONS.md §7: How will stored memories be reviewed and deleted?](../../docs/DECISIONS.md#7-how-will-stored-memories-be-reviewed-and-deleted)

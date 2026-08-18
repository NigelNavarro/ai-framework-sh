# Vector Store: semantic storage (pluggable backend)

Stores chunk embeddings and retrieves them by similarity. Like the model runtime, the storage backend sits behind an interface (`VectorStore`), so it is a replaceable component: the RAG pipeline depends on the contract, and a different database can be swapped in without touching retrieval or ingestion.

This directory is intentionally self-contained, so that the current Chroma implementation (and any future backends) each get their own documented explanation here.

## Files

- `base.py`, which defines `VectorStore`, the abstract interface (`add`, `query`).
- `chroma.py`, which provides `ChromaVectorStore`, an implementation backed by an embedded, on-disk ChromaDB.

## The interface (`base.py`)

- **A contract, enforced.** `class VectorStore(ABC)` plus `@abstractmethod` defines an interface any concrete store must follow. `VectorStore` cannot be instantiated directly; only a subclass that implements both `add()` and `query()` can be.
- **`Chunk` in, `Chunk` out.** `add(chunks, embeddings)` stores document pieces alongside their vectors (`embeddings[i]` is the vector for `chunks[i]`). `query(query_embedding, top_k)` returns the `top_k` nearest chunks, nearest-first. The symmetry matters: a `Chunk` goes in via `add`, and a `Chunk` comes back out via `query`. The `Chunk` dataclass is the common currency flowing through the whole pipeline (chunked, stored, then retrieved), carrying its source metadata the entire way, which is exactly what makes citations possible at the end.

## Implementations

### ChromaVectorStore (`chroma.py`)

An embedded, on-disk Chroma database (a `PersistentClient`, no server required).

**Setup and guardrail**

- `chromadb.PersistentClient(path=...)` is the embedded client. It writes to `data/chroma` and survives restarts, with no server. (Everything under `data/` is gitignored, so vectors are never committed.)
- `get_or_create_collection(name, metadata={...})`: a collection is Chroma's equivalent of a table.
- **Embed-model guardrail.** The collection is stamped with the embedding model that built it. On startup the store reads that stamp back and compares it to the current config; if they differ it raises a clear "re-index required" error. This catches a model switch loudly instead of silently returning garbage from vectors that live in an incompatible coordinate space. This is also what would make a future model-switching feature safe.

**Writing (`add`)**

- Chroma's `add()` takes four parallel lists of equal length and order: `ids`, `embeddings`, `documents`, `metadatas`. Index `i` across all four describes one chunk.
- `ids` must be unique strings, so we build `"{source}:{chunk_idx}"` (for example, `frankenstein.txt:10`), unique per chunk.
- We pass our **own** vectors (`embeddings=embeddings`, produced by `nomic-embed-text`) rather than letting Chroma embed. This keeps control of which model encodes meaning, preserving the two-model design.
- `metadatas` carries the `source` and `chunk_idx` for each chunk (used for citations later).

**Reading (`query`)**

- `query_embeddings=[query_embedding]`: Chroma supports batch queries, but we send one vector, so we wrap it in a list (a batch of one).
- `results["documents"][0]`: because results are nested one level per query, and we sent a single query, we take the first (and only) element.
- **Rebuilding chunks.** We reconstruct `Chunk` objects from the returned documents and metadatas, so `query` returns the same type (`Chunk`) that `add` took in. This keeps the interface consistent and hides Chroma's internal shape from the caller.

## Adding a new backend

To add a store (for example, a server-based or SQL-backed database), write a new class in this directory that subclasses `VectorStore` and implements `add` and `query` against the same contract. Nothing in ingestion, retrieval, or the agent needs to change. Document the new store under its own subsection.

## Related

- **Used by:** [`knowledge/retrieve.py`](../README.md), and `memory/store.py` (which uses Chroma directly for its own memory collection).
- **Governance:** [DECISIONS.md §10: Which component should be replaceable first?](../../../docs/DECISIONS.md#10-which-component-should-be-replaceable-first-the-model-the-ui-the-vector-store-the-tool-layer-or-the-orchestration)

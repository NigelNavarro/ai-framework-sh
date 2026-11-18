# Knowledge: retrieval-augmented generation (RAG)

The knowledge pillar turns documents into searchable context. It reads and chunks source files (`ingest`), and turns a question into the most relevant stored chunks (`retrieve`). The vector storage itself lives one level down in [`vectorstore/`](vectorstore/README.md), behind an interface, so the storage backend is independently replaceable.

## Files

- `ingest.py`, which reads documents and splits them into overlapping `Chunk`s (`ingest_file`, `ingest_folder`, `chunk_text`, and the `Chunk` dataclass).
- `retrieve.py`, which provides `Retriever`, composing a model and a vector store to turn a question into the nearest stored chunks, scored and filtered by a relevance threshold.
- [`vectorstore/`](vectorstore/README.md), the storage layer (the `VectorStore` interface and its implementations).

## Design notes

### Ingestion (`ingest.py`)

- **Split into two stages.** Ingestion is split into `ingest_file` (reads a single file into chunks) and `ingest_folder` (walks a folder and returns all chunks from every supported file). Separating them allows flexible usage, such as testing with a single file or processing an entire dataset.
- **Keyword arguments name the destination.** In `Chunk(text=piece, source=str(path), chunk_idx=index)`, `text=piece` sets the `Chunk`'s `text` attribute (the destination, fixed by the `@dataclass`) to the current `piece` of text (the source, returned by `chunk_text`). Naming the field explicitly clarifies which attribute is being set, especially when the source and destination variable names differ.
- **`extend` flattens.** `chunks.extend(ingest_file(...))` adds every chunk from a file into one flat list (unlike `append`, which would nest a list). This lets every chunk be handled uniformly, regardless of which file it came from.

### Retrieval (`retrieve.py`)

- **Composed from interfaces, backend-agnostic.** `Retriever` is handed a `ModelClient` and a `VectorStore` when created, and it depends on those _interfaces_, not concrete classes. Any model and any store implementing the contracts can be composed in (for example, `Retriever(OllamaModelClient(cfg), ChromaVectorStore(cfg), cfg.top_k, cfg.relevance_threshold)`), and the retriever neither knows nor cares which backend it received. It just calls `embed()` and `query_with_scores()`.
- **`top_k: int | None = None`.** A `Union` type hint meaning "either an integer (how many chunks to return) or `None` (use the default)."
- **The read path in one line.** `self.model.embed([question])[0]` wraps the question in a list and takes the single vector back, then hands that vector to `store.query_with_scores`. `Retriever` hides the embed-then-query detail from the caller.
- **Two ways to retrieve (added in v0.3).** `retrieve_with_scores()` returns every top-k chunk with its cosine similarity score, unfiltered, so the raw evidence is always available (eval/run_eval.py uses it to report each question's top score). `retrieve()` is built on it and keeps only chunks scoring at or above `min_score` (from `config.relevance_threshold`). An empty list means nothing in the knowledge base is relevant: the knowledge boundary. Without this cutoff, retrieval would always return its top-k nearest chunks even for a question that the documents never mention, and the model would be handed unrelated text to stretch into an answer.
- **The threshold is model specific.** Cosine similarity runs from -1.0 to 1.0, but where "relevant" starts depends on the embedding model. With `nomic-embed-text`, unrelated text still scores about 0.4 to 0.57, and the calibrated threshold is 0.60. Changing `embed_model` means re-indexing and re-calibrating.

## Related

- **Storage layer:** [`vectorstore/`](vectorstore/README.md)
- **Uses:** [`models`](../models/README.md) (to embed text).
- **Feeds:** `agent/context.py` (grounded prompts) and the `search_knowledge` tool in [`tools`](../tools/README.md).

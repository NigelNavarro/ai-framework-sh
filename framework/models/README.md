# Models: the LLM runtime (pluggable)

The model runtime, kept behind an interface so it is a replaceable component. The rest of the framework talks to the `ModelClient` contract, never to a specific backend, so swapping models (or model providers) requires no changes elsewhere.

## Files

- `base.py`, which defines `ModelClient`, the abstract interface every model backend implements (`chat`, `embed`). This is the contract, and it has no behavior of its own.
- `ollama.py`, which provides `OllamaModelClient`, the concrete implementation that talks to a local Ollama server over HTTP (`chat`, `chat_stream`, `embed`).

## Design notes

- **Contract vs. implementation.** `base.py` is a pure interface (an `ABC` with `@abstractmethod`), so Python refuses to instantiate any backend that does not implement every method. `ollama.py` is one implementation. A different provider would be a new class written against the same contract, and nothing else in the framework would change. (See DECISIONS.md §10.)
- **What an embedding is.** `embed` turns text into a `list[float]`, a vector (a point in high-dimensional space that represents the text's meaning). For example, "the cat sat" might become `[0.0123, -0.4567, 0.8901, ...]` (often 768 or 1024 numbers long). Texts with similar meaning have vectors that sit close together, which is what makes semantic search (RAG) possible.
- **Streaming vs. non-streaming.** `chat` waits for the full reply and returns a string. `chat_stream` yields the reply piece by piece as Ollama produces it, so the CLI can print tokens in real time.
- **Defensive response parsing.** When reading streamed chunks, we use `.get(key, default)` rather than `chunk["message"]["content"]`. Square-bracket access raises `KeyError` and crashes if a key is missing, whereas `chunk.get("message", {}).get("content", "")` returns an empty string when the response shape is unexpected (something we can safely ignore). This keeps a single malformed chunk from bringing down the whole stream.

## Related

- **Consumed by:** `knowledge/retrieve.py` (embeds queries), `memory/store.py` (embeds facts), and `agent/loop.py` / `agent/context.py` (chat).
- **Governance:** [DECISIONS.md §10: Which component should be replaceable first?](../../docs/DECISIONS.md#10-which-component-should-be-replaceable-first-the-model-the-ui-the-vector-store-the-tool-layer-or-the-orchestration)

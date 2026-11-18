# Tools: the registry and the model-callable capabilities

Tools are what let the _model_ reach the framework's capabilities. Each tool pairs a schema (what the model reads to decide whether and how to call it) with a handler (the Python function that runs). The registry stores tools, advertises their schemas to the model, and dispatches a call by name to its handler. The builders in `builtin.py` turn existing framework capabilities (retrieval, memory, drafting) into tools.

## Files

- `registry.py`, which defines the `Tool` dataclass and `ToolRegistry` (`register`, `get`, `schemas`, `execute`).
- `builtin.py`, which builds the concrete tools: `search_knowledge`, `remember`, `recall`, and `draft_document`.

## Design notes

### The registry (`registry.py`)

- **Functions as values (`Callable[..., str]`).** A `Tool` stores its `handler` as a function value in a dataclass field. `Callable[..., str]` reads as "some function (any args) that returns a string." This is where the framework first treats a function as data, which is what makes a registry possible.
- **JSON Schema describes the inputs.** The `parameters` field is a dict describing a tool's inputs in JSON Schema, so the model knows what arguments to provide. For example, `search_knowledge` is described as:

```json
{"type": "object",
  "properties": {"query": {"type": "string", "description": "What to search for"}},
  "required": ["query"]}
```

- **`schemas()` omits the handler on purpose.** The model never sees or runs our Python; it only needs the name, description, and parameters to decide when and how to call a tool. So `schemas()` advertises those three and withholds the handler.
- **`execute(...)` with `handler(**arguments)`.** `**` is the unpacking operator: it expands the model's argument dict into keyword arguments, so `{"query": "..."}` becomes `handler(query="...")`. The schema's parameter names must therefore match the handler's parameter names.
- **`_tools` (the leading underscore).** A convention marking the attribute as internal to `ToolRegistry` (use the methods, do not reach in from outside).

### The tool builders (`builtin.py`)

- **A builder file, not a runtime.** Each `make_*_tool(...)` function returns a `Tool`; it does not run the model or the underlying capability. The dependency (a `Retriever`, a `MemoryStore`) is captured in a closure so the inner handler can use it, while the handler's own signature stays limited to the model-facing arguments.
- **Why the closure.** The handler needs, say, a `Retriever`, but the model only supplies a `query`. Passing the retriever into the outer builder lets the inner handler use it without it being a model-facing argument.
- **Schema name must match handler parameter.** The model sends arguments keyed by the schema's names (for example, `{"query": "Frankenstein"}`), and the registry runs `handler(**arguments)`, equivalent to `search_knowledge(query="Frankenstein")`. So the two must agree.
- **Handlers return a confirmation, not silence.** A tool's return value goes back into the model's context as an observation. `remember` returns `"Saved to memory: ..."` (rather than the raw id from `store.add`) so the model can tell the user it acknowledged and will remember.
- **`draft_document`, the first two-argument tool.** It takes `filename` and `content`. Like every tool it bundles a name, a description (why and when to use it), a parameter schema (what the model reads to know what to send), and a handler (the function that runs). It also sanitizes the filename so it cannot escape `data/drafts/` (verified by the safety tests).
- **"Nothing found" is a named constant.** `search_knowledge` and `recall` return `NO_KNOWLEDGE_RESULT` and `NO_MEMORY_RESULT` when they find nothing, and `parse_knowledge_sources()` reads a search result back into {citation number: filename}. Provenance and the grounding judge depend on both, so they live next to the code that writes them.

## Evolution

- **2026-08-15:** added the memory tools (`remember`, `recall`) and the first writing, two-argument tool (`draft_document`), extending the toolset from retrieval-only to memory and drafting.

## Related

- **Driven by:** [`agent/loop.py`](../agent/README.md), which advertises these schemas to the model and dispatches its tool calls.
- **Wraps:** [`knowledge`](../knowledge/README.md) (search), [`memory`](../memory/README.md) (remember/recall), and local drafting.
- **Governance:** [DECISIONS.md §4](../../docs/DECISIONS.md#4-which-systems-should-the-assistant-never-access) and [§5](../../docs/DECISIONS.md#5-what-tools-need-approval-before-execution).

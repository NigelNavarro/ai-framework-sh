# Agent: context assembly and the tool-using loop

The agent pillar decides how the model is prompted and, in the agentic path, lets the model drive. `context.py` assembles grounded, cite-instructed prompts for the fixed RAG path (`ask`). `loop.py` runs the bounded think/act/observe loop for the agentic path (`agent`), where the model chooses which tools to call.

## Files

- `context.py`, which builds the grounded prompt (`SYSTEM_PROMPT`, `format_sources`, `build_messages`).
- `loop.py`, which runs the agent loop (`build_system_prompt`, `parse_tool_call`, `run_agent`).

## Design notes

### Context assembly (`context.py`)

- **The system prompt sets standing rules.** `SYSTEM_PROMPT` is applied before the model sees the question. Its four rules keep answers grounded and cited:
  1. _Use only the given sources_ (grounding, which stops the model answering from training memory and avoids hallucination).
  2. _If the sources don't contain the answer, say so_ (an escape hatch; a model with no way to say "I don't know" will invent an answer to please the user).
  3. _Cite the sources you used by [number]_ (provides trust, so answers can be traced back to their source).
  4. _Be concise_ (tight, to-the-point responses).
- **`format_sources` numbers the chunks.** It turns retrieved chunks into a numbered block the model can cite (`[1]`, `[2]`, ...). The number is a short, clean label to cite instead of a long filename, and because each block records the chunk index, a citation can be mapped back to the originating document.
- **`build_messages` returns the shape `chat` expects.** It assembles a two-message list (the system rules, then a user message with the numbered Sources, the Question, and a reminder to cite), so the caller can just do `client.chat(build_messages(q, chunks))`. The shape is `list[Message]`, where `Message = dict[str, str]` (from `models/base.py`), so the return is a `list[dict[str, str]]`: an ordered sequence of messages (the outer list, where order matters) in which each message is a named record (the inner dict).

### The agent loop (`loop.py`)

- **Prompt-based tool calling, model-agnostic.** Rather than relying on a model's native tool API, we describe the tools in the system prompt and ask the model to reply with a JSON object when it wants a tool. `parse_tool_call` detects that JSON (returning `None` for a plain-text answer, which is the loop's exit signal). This works with any instruction-following model and requires no change to the model adapter. (See DECISIONS.md §2.)
- **Think, act, observe, repeat.** `run_agent` asks the model (think), runs any requested tool via the registry (act), feeds the result back as an observation (observe), and repeats. It is hard-capped at `config.max_tool_rounds` so a confused model cannot loop forever (see DECISIONS.md §8).
- **Schema key must match what the loop reads.** The prompt tells the model to put arguments under a specific key, and `run_agent` reads that same key. The two must agree, or the tool receives no arguments.

## Related

- **Uses:** [`tools`](../tools/README.md) (the registry it drives), [`knowledge`](../knowledge/README.md) (chunks feed `context.py`), and [`models`](../models/README.md) (chat).
- **Governance:** [DECISIONS.md §2](../../docs/DECISIONS.md#2-which-use-cases-are-read-only-drafting-only-supervised-action-or-autonomous) and [§9](../../docs/DECISIONS.md#9-how-will-model-outputs-be-evaluated-for-correctness-and-safety).

## Observability (introduced in v0.3)

`run_agent` takes an optional `Tracer`. When one is provided, the loop records structured events which are each stamped with a shared `run_id` so that an entire run can be filtered out of the trace log and replayed in order.

- **`agent_start`** — Once per run (the question, the model, and the tools available).
- **`tool_call`** — Runs once per tool the model calls (tool name, arguments, an observation preview, the tool's latency, and an `error` field that is `null` on success or has an error message when the tool is raised).
- **`agent_end`** — Runs once when the model answers (rounds used and total latency).
- **`cap_reached`** - Runs once when a run exhausts `max_rounds` without answering (rounds used and total latency).

Tracing is _optional_: with no tracer, the loop behaves identically and records nothing.

### Verification

The first live check confirmed the wiring end to end even though the run aborted early because the local model server was offline. The `agent_start` event was written to the trace before the run stopped:

```shell
    type: agent_start
    run_id: 8fe1c54a
    model: gemma4:latest
    question: Draft a haiku about whales and save it as whales.md
    tools_available: ['search_knowledge', 'remember', 'recall', 'draft_document']
```

From this event trace we can confirm the following:

1. The tracer is wired through `run_agent` and the loop is emitting events.
2. Each event carries a shared `run_id`, the thread that ties a run's events together.
3. The event carries a `command` or `type` label, which `trace tail` uses to head each entry.

The `tool_call` and `agent_end` events aren't reached not because of the tracing, but because of the original issue regarding the local model being offline.
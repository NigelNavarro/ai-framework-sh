# Agent: context, the tool-using loop, provenance, and grounding

The agent pillar decides how the model is prompted and, in the agentic path, lets the model drive. `context.py` assembles grounded, cite-instructed prompts for the fixed RAG path (`ask`). `loop.py` runs the bounded think/act/observe loop for the agentic path (`agent`), where the model chooses which tools to call. After an agent run, `provenance.py` labels where the answer's information came from, and `grounding.py` can grade the answer against the source it should have come from.

## Files

- `context.py`, which builds the grounded prompt (`SYSTEM_PROMPT`, `format_sources`, `build_messages`).
- `loop.py`, which runs the agent loop (`build_system_prompt`, `parse_tool_call`, `run_agent_with_record`, and the thin `run_agent` wrapper), and defines the `ToolCall` and `AgentRun` records.
- `provenance.py`, which labels where an answer's information came from (`label_provenance`, `Provenance`).
- `grounding.py`, which grades an answer against the source it should have come from (`judge_grounding`, `GroundingReport`).

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
- **Think, act, observe, repeat.** `run_agent_with_record` asks the model (think), runs any requested tool via the registry (act), feeds the result back as an observation (observe), and repeats. It is hard-capped at `config.max_tool_rounds` so a confused model cannot loop forever (see DECISIONS.md §8). It returns an `AgentRun`: the answer plus a `ToolCall` record of every tool call and what it returned. `run_agent()` is a one-line wrapper that returns only the answer text, for callers that don't need the record.
- **Schema key must match what the loop reads.** The prompt tells the model to put arguments under a specific key, and the loop reads that same key. The two must agree, or the tool receives no arguments.
- **Search first.** The system prompt tells the model the user's questions are usually about their own documents, and to call `search_knowledge` before answering from general knowledge. Without it, tests with gemma4 answered well-known topics from memory without searching (route accuracy 12/14 on the golden set, 14/14 with it).
- **Empty replies are retried.** A thinking model can spend a turn on hidden reasoning and return no visible text. An empty reply is neither an answer nor a tool call, so the loop sends `EMPTY_REPLY_NUDGE` and tries again. The retry uses up a round, so it still stops at `max_rounds`.

### Provenance (`provenance.py`)

- **The framework labels the answer, not the model.** `label_provenance(run)` reads the `AgentRun` record: a successful `search_knowledge` with chunks means documents, a successful `recall` with facts means memory, nothing returned means general knowledge, and a capped run means no answer.
- **"Retrieved", not "used".** The label says what was offered to the model. Whether the answer actually stayed with it is the grounding judge's job.
- **Shared constants, not copied wording.** "Nothing found" is detected by comparing against `NO_KNOWLEDGE_RESULT` and `NO_MEMORY_RESULT` from `tools/builtin.py`, so rewording a message can't silently break the label.

### Grounding judge (`grounding.py`)

- **Graded, not pass or fail.** `judge_grounding()` returns a `GroundingReport` with separate checks: `route` (did the answer take the route the golden set expects), `citations_exist` (does every `[n]` point to a retrieved chunk), and `cites_expected` (is the expected book cited). A check that doesn't apply is `None` and left out of the score.
- **Deterministic.** No model call, so the same inputs always give the same grade, and the tests need no Ollama.
- **Known limitations.** Citations are mapped against the last successful search (each search numbers from [1]), and are reported as not checkable when memory was also recalled.

## Related

- **Uses:** [`tools`](../tools/README.md) (the registry it drives), [`knowledge`](../knowledge/README.md) (chunks feed `context.py`), and [`models`](../models/README.md) (chat).
- **Governance:** [DECISIONS.md §2](../../docs/DECISIONS.md#2-which-use-cases-are-read-only-drafting-only-supervised-action-or-autonomous) and [§9](../../docs/DECISIONS.md#9-how-will-model-outputs-be-evaluated-for-correctness-and-safety).

## Observability (introduced in v0.3)

`run_agent_with_record` takes an optional `Tracer`. When one is provided, the loop records structured events which are each stamped with a shared `run_id` so that an entire run can be filtered out of the trace log and replayed in order.

- **`agent_start`**: Once per run (the question, the model, and the tools available).
- **`tool_call`**: Runs once per tool the model calls (tool name, arguments, an observation preview, the tool's latency, and an `error` field that is `null` on success or has an error message when the tool fails).
- **`agent_end`**: Runs once when the model answers (rounds used and total latency).
- **`cap_reached`**: Runs once when a run exhausts `max_rounds` without answering (rounds used and total latency).
- **`empty_reply`**: Runs once per empty model reply, before the loop retries (round number).
- **`provenance`**: Recorded by the `agent` command after the run, under the same `run_id` (the label, the documents retrieved, and whether the knowledge base was searched without a match).

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

1. The tracer is wired through `run_agent_with_record` and the loop is emitting events.
2. Each event carries a shared `run_id`, the thread that ties a run's events together.
3. The event carries a `command` or `type` label, which `trace tail` uses to head each entry.

The `tool_call` and `agent_end` events weren't reached because the local model was offline. This was not due to a tracing issue.

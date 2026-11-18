# CLI: the reference application

The command-line app is the framework's first _consumer_. The app/framework boundary is proven because Main is dependent on the framework, not vice-versa. `main.py` wires the framework's pieces behind an `argparse` CLI with several subcommands.

## Files

- `main.py`, the entry point. Parses arguments and dispatches to one command handler per verb.

## Commands

| Command                      | What it does                                                     | Autonomy        |
| ---------------------------- | ---------------------------------------------------------------- | --------------- |
| `chat`                       | Plain streaming REPL (Read-Eval-Print Loop) with the model.      | Level 0         |
| `ingest <path>`              | Read, chunk, and embed a file or folder into the knowledge base. | (build step)    |
| `ask "<question>"`           | Always-retrieve, grounded, cited Q&A.                            | Level 1         |
| `agent "<question>"`         | Agentic loop; the model chooses which tools to call.             | Level 1-2       |
| `memory list \| delete <id>` | Human inspect/forget controls for stored memory.                 | (management)    |
| `trace tail [-n N]`          | Show the most recent trace events.                               | (observability) |
| `trace show <run-id>`        | Replay one agent run's events in order.                          | (observability) |

## Design notes

- **Everything lives in functions behind a name guard.** Each command is a `cmd_*` function, and `main()` runs only under `if __name__ == "__main__"`. So importing the module has no side effects (nothing parses args or starts the REPL), which keeps it testable and reusable. The first versions of REPL was top-level code. Once there were reusable functions and a dispatcher, the guard became necessary.
- **`argparse` subcommands.** `add_subparsers(dest="command", required=True)` declares that the CLI has verbs. The chosen verb is stored in `args.command`, and a missing verb prints helpful usage instead of doing something random. _Every subcommand must be registered before `parse_args()`_ (under this concept of "build the entire menu, then take the order").
- **`choices` and `nargs` for the `memory` command.** `choices=["list", "delete"]` lets argparse validate the action for us, and `nargs="?"` makes the id an optional positional (present for `delete`, omitted for `list`), with a manual check filling the gap argparse cannot express.
- **Trace files sort chronologically.** They are named by ISO date, so string-sorting them _is_ chronological sorting, which is why `trace tail` takes `[-1]` (the last) as the newest day's log.

## Evolution

This file grew the most over the project, from a single REPL into a six-verb CLI:

- **Start:** a plain chat REPL (top-level code).
- **Restructure:** moved the REPL into `cmd_chat` and introduced `argparse` subcommands behind a `__main__` guard.
- **2026-07-08:** `cmd_ingest` upgraded to read a single file _or_ a whole folder (chunk, embed, store), with graceful handling of unsupported types and empty folders.
- **2026-07-10:** added `ask`, which composes a `Retriever`, retrieves chunks, and answers with citations that stay in lockstep between the prompt and the printed Sources.
- **2026-07-27:** added `agent`, the agentic loop that wraps retrieval as a tool the model may call (or not, or repeatedly), alongside the original always-retrieve `ask`.
- **2026-08-15:** registered the `remember` / `recall` and `draft_document` tools on `agent`; added the `memory list | delete` controls and the `trace tail` viewer.

## v0.3 - Harden the Core

This update deepens what the CLI can see and replay.

- **The agent now traces itself.** `cmd_agent` passes its `Tracer` into `run_agent_with_record` so the loop records structured events (`agent_start`, one `tool_call` per tool, and a terminal `agent_end` or `cap_reached`), all sharing a `run_id`. The older post-hoc summary was removed in favor of this richer event capture option.
- **`trace show <run_id>` replays one run.** Alongside `trace tail`, this new action gathers every event carrying a given `run_id` across all trace files, sorts them by time, and prints them in order, reconstructing a single run step by step.
- **`print_event` is shared.** The per-event formatting was extracted from the tail loop into a `print_event(event)` helper, so `tail` and `show` display events identically instead of keeping two copies of the code.
- **Every answer is labeled.** `ask` and `agent` print a `Provenance:` line under the answer (documents, memory, general knowledge, or no answer), and `agent` records a `provenance` trace event under the run's `run_id`.


### Python concepts to highlight in this milestone

- **`event.get(key, default)` is a dict method.** For example, `event.get("type", "event")` looks up the key `"type"` and returns its value if it exists. If the value does not exist, then instead it returns the default `"event"` (or, with no default passed in, a missing key here will return `None`).
  - This differs from the square bracket lookup: `event["type"]`. If the key value doesn't exist when using `event["type"]`, Python raises a `KeyError` indicating that the key is missing.
  - Both `event["type"]` and `.get("type", "event")` are runtime dictionary lookups, but the `.get("type", "event")` has a fallback default `"event"` which allows the code to keep going instead of raising a `KeyError`.
- **`trace_files = sorted(traces_dir.glob("*.jsonl"))`.** `.glob("*.jsonl")` finds every file ending in `.jsonl` and treats them as a generator of `Path`s in no guaranteed order. Then, `sorted(...)` consumes that generator and returns a new, ordered list. No sort key is needed because `Path`s are naturally orderable (chronologically since the files are also ISO-date named).
- **In `events.sort(key=lambda e: e.get("timestamp", ""))`,** `.sort()` reorders the list in place (it mutates `events` and returns `None`, unlike `sorted()`, which returns a new list). A `key` is required because dicts are not directly comparable, so the `lambda` maps each event to a scalar to sort by: its timestamp string (or `""` if missing, which keeps every key comparable). ISO timestamps sort chronologically as strings, so the run replays in the order it happened.

## Related

- **Consumes the framework:** [`models`](../../framework/models/README.md), [`knowledge`](../../framework/knowledge/README.md), [`agent`](../../framework/agent/README.md), [`tools`](../../framework/tools/README.md), [`memory`](../../framework/memory/README.md), [`observability`](../../framework/observability/README.md).
- **Design & governance:** [docs/DESIGN.md](../../docs/DESIGN.md), [docs/DECISIONS.md](../../docs/DECISIONS.md).

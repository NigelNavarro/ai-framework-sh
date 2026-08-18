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

## Related

- **Consumes the framework:** [`models`](../../framework/models/README.md), [`knowledge`](../../framework/knowledge/README.md), [`agent`](../../framework/agent/README.md), [`tools`](../../framework/tools/README.md), [`memory`](../../framework/memory/README.md), [`observability`](../../framework/observability/README.md).
- **Design & governance:** [docs/DESIGN.md](../../docs/DESIGN.md), [docs/DECISIONS.md](../../docs/DECISIONS.md).

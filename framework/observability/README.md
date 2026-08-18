# Observability: append-only tracing

Records every notable event (a model call, retrieval, or tool call) as one JSON line, so that nothing the framework does is invisible and any answer can be reconstructed and audited after the fact.

## Files

- `trace.py`, which provides `Tracer`, an append-only JSONL event recorder. One JSON object per line, one file per day under `data/traces/`.

## Design notes

- **Built once from config, then reused.** `Tracer` follows the same pattern as `OllamaModelClient` (constructed once from the `Config`, then reused). Its `__init__` resolves the traces folder (`data_dir/traces`) and creates it if needed (`mkdir(parents=True, exist_ok=True)`).
- **A deliberately generic `record`.** `record(event)` is the tracer's single method, and it takes _any_ dict (for example, `{"model": ..., "latency_s": ...}`), stamps it, and writes it. Because it does not hard-code a fixed set of fields, it can log model calls today and per-tool-call detail in the future **without changing the logger**. The only requirement is that the event can be serialized to JSON. The payoff is that extending observability means writing a different dict at the call site, not rewriting `Tracer`.
- **Timestamping via dict unpacking.** `{"timestamp": ..., **event}` "spills" all of the event's key/values into a new dict alongside the timestamp, a clean way to merge in a standard field without the caller having to remember it.
- **Unambiguous, standard format.** `datetime.now(timezone.utc).isoformat()` records the time in UTC as an ISO-8601 string, and `json.dumps(entry) + "\n"` turns the dict into a proper [JSON Lines](https://jsonlines.org/) entry appended to the day's file.

## Related

- **Viewer:** the `trace tail` CLI command (see [`apps/cli`](../../apps/cli/README.md)) reads the newest trace file and prints the most recent events.
- **Governance:** [DECISIONS.md §6: What should be logged for every action?](../../docs/DECISIONS.md#6-what-should-be-logged-for-every-action)

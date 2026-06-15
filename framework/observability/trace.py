"""
Author: Nigel J. Navarro

Observability: Model Call Tracing & History
Every notable event (a model call, retrieval, or toll call) is recorded as one JSON line in the <data_dir>/trace/<date>.jsonl file. This is a simple, append-only log of what happened, when, and with what details. Nothing that the framework does has to be invisible, and data can be used for debugging, monitoring, or analysis.
"""
from __future__ import annotations
import json
from datetime import datetime, timezone
from pathlib import Path

from framework.config import Config

class Tracer:
    """
    An append-only recorder for events.
    One JSON line per event.
    """

    def __init__(self, config:Config): # Initialize the tracer with the Config from the framework. We need the data_dir to know where to write the trace files.
        self.traces_dir = config.data_dir / "traces" #Trace files live in a "trace subfolder of the configured data_dir."
        self.traces_dir.mkdir(parents=True, exist_ok=True) # Ensure the trace directory exists at startup.
    
    def record(self, event: dict) -> None:
        """
        Append one event (plus a UTC timestamp) as a single JSON line.
        """
        entry = {
            "timestamp": datetime.now(timezone.utc).isoformat(), # Record the current time in UTC as an ISO-formatted string.
            **event, # Merge the event dict into the entry, so all event details are included alongside the timestamp.
        }

        # One file per day (e.g. traces/2026-06-14.jsonl). This keeps files manageable and organized by date.
        date_stamp = datetime.now(timezone.utc).date().isoformat()
        trace_file = self.traces_dir / f"{date_stamp}.jsonl" # Path to the current day's trace file.

        with open(trace_file, "a", encoding="utf-8") as f:
            f.write(json.dumps(entry) + "\n") # Append the JSON line to the file

"""
Notes
class Tracer with __init__ follows the same pattern as our OllamaModelClient, built once from config, then reused. 
Its __init__ figures out the traces folder (data_dir/traces) and creates it if needed (mkdir(parents=True, exist_ok=True).
The record(self, event) is the single method in this Tracer. It handles whatever dict it wants logged (e.g. {"model": ..., "latency_s":...}), then record stamping and writing it. By keeping it generic today, it can log model calls today and retrievals/tool calls in the future without needing changes. The only requirement is that the event is a dict that can be serialized to JSON.
The dict unpacking is hanled by `{timestamp:..., **event}` which "spills" all of the event's key-values (KV) into the new dict alongside the timestamp. It is a clean way to merge in a standard field (the timestamp) without the caller having to remember it.
datetime.now(timezone.utc).isoformat() records the current time in UTC as an ISO-formatted string. Makes timestamps unabiguious by standardizing format.
json.dumps(entry) turns the dict into a JSON string, then adds the newline that makes it a proper JSON line.
"""
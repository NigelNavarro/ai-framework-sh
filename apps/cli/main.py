"""
Author: Nigel J. Navarro
CLI Entry Point for the ai-framework-sh assistant.

Subcommands:
    chat                    Start an interactive chat REPL (Read-Eval-Print Loop) with the model.
    ingest <path>           Ingest a file or folder of documents into the knowledge base (chunk, embed, store).
    ask "<question>"        Retrieve and answer with citations (always searches the knowledge base)
    agent "<question>"      The agentic loop: the model decides which tools to use (search_knowledge, remember, recall, draft_document)
    memory list|delete <id> Inspect or forget stored memories
    trace tail [-n N]       Show the most recent trace events (default 10)

Run: .venv/bin/python apps/cli/main.py <command> [args]
"""
import argparse             # For parsing command-line arguments
import json                 # For reading trace lines (JSON per line) back into dicts
import time                 # For measuring latency of model responses
from pathlib import Path    # For handling file system paths in a platform-independent way

from framework.config import load_config
from framework.models.ollama import OllamaModelClient
from framework.observability.trace import Tracer
from framework.knowledge.ingest import SUPPORTED_EXTENSIONS, ingest_file, ingest_folder
from framework.knowledge.vectorstore import ChromaVectorStore
from framework.knowledge.retrieve import Retriever
from framework.agent.context import build_messages
from framework.agent.loop import run_agent
from framework.tools.registry import ToolRegistry
from framework.tools.builtin import make_search_knowledge_tool, make_remember_tool, make_recall_tool, make_draft_document_tool
from framework.memory.store import MemoryStore

def cmd_chat(config):
    """
    Interactive chat REPL.
    Streaming replies, conversation memory, and tracing.
    """
    client = OllamaModelClient(config) # Build the model client once, from our configuration.
    tracer = Tracer(config)            # The Tracer wraps the model client to record all interactions for observability.

    print(f"You are now chatting with {client.model}. Type 'exit' or 'quit' to stop.\n")

    messages = []   # Conversation History: A list of messages that the model expects. Each message is a dict with "role" (e.g. "user", "assistant") and "content" (the text). 
                    # This is the chat's memory. We resend the whole thing every turn so that the model has context.

    while True:
        user_input = input(">you ")                         # Read. Wait for the user to type and press Enter.
        if user_input.strip().lower() in ("exit", "quit"):  # Exit Condition
            print("Goodbye!")
            break

        messages.append({"role": "user", "content": user_input})

        print(f"{client.model}> ", end="", flush=True)
        reply = ""                              # Accumulate the full reply in case we want to use it later.
        start = time.perf_counter()             # Start the Stopwatch
        for chunk in client.chat_stream(messages):
            print(chunk, end="", flush=True)
            reply += chunk
        latency_s = time.perf_counter() - start # Stop the Stopwatch; calculate latency in seconds.
        print("\n")

        tracer.record({
            "model": client.model,              # Record the model name
            "user_msg": user_input,             # Record the user input
            "reply_chars": len(reply),          # Record the length of the model's reply in characters.
            "latency_s": round(latency_s, 3),   # Record latency rounded to 3 decimal places for readability.
        })

        messages.append({"role": "assistant", "content": reply}) # Record the model's fully reply for the next turn's context.

def cmd_ingest(config, path):
    """
    Build the knowledge base from a file or a folder of documents.
    
    Read the given path, then chunk each document, embed each chunk, and then store the vectors in a vector DB for later retrieval.
    This is the first stage of the RAG pipeline.
    """
    # Converts the string path into a Path object, as `argparse` returns strings. 
    # This allows us to use Path methods and properties throughout the ingest logic without having to convert it repeatedly. 
    # It also makes it easier to handle relative vs absolute paths, and to join paths in a platform-independent way.
    path = Path(path) 
    if not path.exists():
        print(f"Error: path '{path}' not found.")
        return
    
    # Decision: Walk an entire folder, or read a single supported file
    if path.is_dir():
        chunks = ingest_folder(path, config.chunk_size, config.chunk_overlap)
    elif path.is_file() and path.suffix.lower() in SUPPORTED_EXTENSIONS:
        chunks = ingest_file(path, config.chunk_size, config.chunk_overlap)
    else:
        print(f"Error: path '{path}' is not a supported file or folder.\n Supported file types: {', '.join(sorted(SUPPORTED_EXTENSIONS))}")
        return
    
    if not chunks: # If no supported files were found, we print a message and return early.
        print(f"No supported files ({', '.join(sorted(SUPPORTED_EXTENSIONS))}) found in {path}.")
        return
    print(f"Read {len(chunks)} chunks from {path}. Embedding...")

    # Client & Store are built once, and after the file(s) obtained above are validated.
    client = OllamaModelClient(config)  # Build the model client once. Provides embed()
    store = ChromaVectorStore(config)   # Build the vector store client once. Provides add()

    # EMBED
    # We extract the text from each chunk and send it to the model's embed() function, which returns a list of embeddings (vectors) corresponding to each chunk.
    texts = [c.text for c in chunks]
    embeddings = client.embed(texts)

    # STORE
    store.add(chunks, embeddings)
    print(f"Done. Stored {len(chunks)} chunks in the knowledge base.")

def cmd_ask(config, question):
    """
    Answer a question from the knowledge base using Retrieval Augmented Generation (RAG) with citations (read-only cited Q&A).

    This is the bridge that ties the RAG pipeline together.
    Retrieve relevant chunks -> Assemble a grounded, cite-intructed prompt -> Ask the model.
    """
    client = OllamaModelClient(config)
    store = ChromaVectorStore(config)
    retriever = Retriever(client, store, config.top_k)  # Composes the model and vector store into a retriever that can find relevant chunks for a given question.
    tracer = Tracer(config)                             # The Tracer wraps the model client to record all interactions for observability.

    # RETRIEVE - Embed the question, then pull the nearest stored chunks
    chunks = retriever.retrieve(question)
    if not chunks:
        print("No relevant information found. Has anything been ingested yet? Try: ingest <path>")
        return
    
    # ASSEMBLE - Build the system & user messages that ground the model in these chunks and instruct it to cite them by number.
    messages = build_messages(question, chunks)

    # ASK - Send one non-streaing call to the model so taht we have the full answer before printing citations.
    start = time.perf_counter()             # Start the Stopwatch
    answer = client.chat(messages)          # The non-streaming call returns the full answer in one go, which allows us to print citations after the answer is complete.
    latency_s = time.perf_counter() - start # Stop the Stopwatch; calculate latency in seconds.

    # PRINT - Provide the answer, then the source list. 
    # The [n] here matches the [n] model. Both number the same chunks in the same order (as both enumerate starting from 1).
    print(f"\n{answer}\n")
    print("Sources: ")
    for number, chunk in enumerate(chunks, start=1):
        print(f"  [{number}]  {Path(chunk.source).name}  (chunk {chunk.chunk_idx})")
    
    # TRACE - Record what was retrieved and why for observability.
    tracer.record({
        "command": "ask",
        "model": client.model,
        "question": question,
        "retrieved": [f"{Path(c.source).name} (chunk {c.chunk_idx})" for c in chunks],
        "answer_chars": len(answer),
        "latency_s": round(latency_s, 3),
    })

def cmd_agent(config, question):
    """
    Answer a question using the agentic loop.
    The model decides whether to call `search_knowledge` (whereas cmd_ask always calls it). 
    The model can call the tool multiple times, or not at all, and can also answer without calling it.
    Implementation is similar to cmd_ask, but the retrieval is wrapped as a tool, and run_agent() steers the behavioral loop (think -> act -> observe -> repeat).
    """
    client = OllamaModelClient(config)
    store = ChromaVectorStore(config)
    retriever = Retriever(client, store, config.top_k)
    tracer = Tracer(config)

    # Build the tool registry. Retrieval stops being automatic, and becomes a tool the model can choose to call.
    # It sits alongside the memory Read/Write tools.
    memory = MemoryStore(config, client)                        # Reuse the same client on L155 above for embedding memory facts to keep the model consistent across both knowledge and memory.
    registry = ToolRegistry()
    registry.register(make_search_knowledge_tool(retriever))    # Register the retrieval tool so the model can call it to search the knowledge base.
    registry.register(make_remember_tool(memory))               # Register the remember tool so the model can call it to store facts in memory.
    registry.register(make_recall_tool(memory))                 # Register the recall tool so the model can call it to retrieve facts from memory.
    registry.register(make_draft_document_tool(config))         # Register the draft_document tool so the model can use it to create draft documents.

    # `run_agent` asks the model, runs any tool it requests, feeds results back, and returns the plain-text answer.
    start = time.perf_counter()
    answer = run_agent(client, registry, question, config.max_tool_rounds) 
    latency_s = time.perf_counter() - start

    print(f'\n{answer}\n')

    # Trace - Retrieval happens inside the agent loop, so at this level, we can only record which tools were available, not the ones the model actually called.
    tracer.record({
        "command": "agent",
        "model": client.model,
        "question": question,
        "tools_available": [schema['function']['name'] for schema in registry.schemas()],
        "answer_chars": len(answer),
        "latency_s": round(latency_s, 3),
    })

def cmd_memory(config, action, memory_id):
    """
    These are the human controls for stored memory.
    Allows the user to list all facts or delete one by ID.
    These are not agent tools. They are our direct inspect/forget controls.

    `list` reads memory.json, and `delete` removes by ID from JSON + Chroma.
    No embedding happens here, so no model call is needed. This is a direct read/write to the memory store.
    """
    memory = MemoryStore(config)    # List/Delete never embed, so no client is needed.

    if action == "list":
        facts = memory.list()
        if not facts:
            print("No memories stored yet.")
            return
        for fact in facts:
            print(f"{fact['id']}  ({fact['created_at']})")
            print(f"    {fact['text']}")
    elif action == "delete":
        if not memory_id:
            print("Error 'memory delete' needs an ID. Run `memory list` to see IDs.")
            return
        if memory.delete(memory_id):
            print(f"Deleted memory {memory_id}")
        else:
            print(f"No memory found with id {memory_id}. Run `memory list` to see IDs.")

def cmd_trace(config, count):
    """
    View the most recent trace events (the 'tail' of the log).

    Reads the newest data/traces/<date>.jsonl file and prints its last 'count' events.
    This is the observability payoff which allows us to see what every chat/ask/agent command actually performed.
    """
    traces_dir = config.data_dir / "traces"
    if not traces_dir.exists():
        print("No traces found yet. Run a command first (chat, ask, agent) to generate traces.")
        return

    # Filenames are ISO dates (e.g. `2026-08-15.jsonl`), which sort chronologically as strings.
    # The last one is the most recent day's log.
    trace_files = sorted(traces_dir.glob("*.jsonl"))        # trace_files is a sorted list of Path objects for each trace file in the traces directory.
    if not trace_files:
        print("No trace files found.")
        return
    latest = trace_files[-1]                                # Get the most recent trace file (the last one in the sorted list).

    lines = latest.read_text(encoding="utf-8").splitlines() # Read the file and split it inoto one event per line.
    recent = lines[-count:]                                 # Get the last 'count' lines (the most recent events).

    print(f"Showing last {len(recent)} of {len(lines)} events from {latest.name}:\n")
    for line in recent:
        event = json.loads(line)                            # event turns JSON text into a dict
        timestamp = event.get("timestamp", "?")
        label = event.get("command", event.get("model", "event"))
        print(f"{timestamp}  [{label}]")
        for key, value in event.items():
            if key in ("timestamp", "command"):
                continue
            print(f"    {key}: {value}")
        print()

def main():
    # Build the top-level parser  and a slot for the subcommands ("verbs")
    parser = argparse.ArgumentParser(
        prog="ai-framework-sh",                                         # The name of the program (used in help messages)
        description="A self-hosted AI assistant framework CLI."
    )
    subparsers = parser.add_subparsers(dest="command", required=True)   # Create a subparser for each subcommand (e.g. "chat", "ingest", "ask")

    # Register the "chat" subcommand (ingest/ask will get added here in the future)
    subparsers.add_parser("chat", help="Start an interactive chat REPL.")

    # Register the "ingest" subcommand
    ingest_parser = subparsers.add_parser("ingest", help="Ingest a file or folder of documents into the knowledge base.")
    ingest_parser.add_argument("path", help="Path to the file or folder of documents to ingest.")

    # Register the "ask" subcommand
    ask_parser = subparsers.add_parser("ask", help="Ask a question and get a cited answer from the knowledge base.")
    ask_parser.add_argument("question", help="The question to answer (wrap it in quotes)")

    # Register the "agent" subcommand
    agent_parser = subparsers.add_parser("agent", help="Ask a question and let the model decide to search the knowledge base or not.")
    agent_parser.add_argument("question", help="The question to ask the agent (wrap it in quotes).")

    # Register the "memory" subcommand (human inspect/forget controls)
    memory_parser = subparsers.add_parser("memory", help="List or delete stored memories.")
    memory_parser.add_argument("action", choices =["list", "delete"], help="List all memories, or delete one by ID.")
    memory_parser.add_argument("id", nargs="?", help="The memory ID to delete (required for 'delete')")

    # Register the 'trace' subcommand (observability)
    trace_parser = subparsers.add_parser("trace", help="View recent trace events.")
    trace_parser.add_argument("action", choices=["tail"], help="tail: show the most recent events")
    trace_parser.add_argument("-n", "--count", type=int, default=10, help="How many recent events to show (default 10)")

    args = parser.parse_args() # Parse the command-line arguments; everything above must be registered before this line
    config = load_config()     # Load the configuration (e.g. model settings, API keys, etc.)

    # Dispatch to the right function based on the chosen subcommand.
    if args.command == "chat":
        cmd_chat(config)
    elif args.command == "ingest":
        cmd_ingest(config, args.path)
    elif args.command == "ask":
        cmd_ask(config, args.question)
    elif args.command == "agent":
        cmd_agent(config, args.question)
    elif args.command == "memory":
        cmd_memory(config, args.action, args.id)
    elif args.command == "trace":
        cmd_trace(config, args.count)

if __name__ == "__main__":
    main() # Run the CLI entry point

"""
Author: Nigel J. Navarro
CLI Entry Point for the ai-framework-sh assistant.

Subcommands:
    chat        Start a REPL (Read-Eval-Print Loop) for chatting with the model.
    ingest      Build the knowledge base from a file or a folder of documents.
    ask         Answer a question from the knowledge base using RAG with citations (read-only cited Q&A).

Run: .venv/bin/python apps/cli/main.py <command> [args]
"""
import argparse # For parsing command-line arguments
import time     # For measuring latency of model responses
from pathlib import Path # For handling file system paths in a platform-independent way

from framework.config import load_config
from framework.models.ollama import OllamaModelClient
from framework.observability.trace import Tracer
from framework.knowledge.ingest import SUPPORTED_EXTENSIONS, ingest_file, ingest_folder
from framework.knowledge.vectorstore import ChromaVectorStore
from framework.knowledge.retrieve import Retriever
from framework.agent.context import build_messages
from framework.agent.loop import run_agent
from framework.tools.registry import ToolRegistry
from framework.tools.builtin import make_search_knowledge_tool

def cmd_chat(config):
    """
    Interactive chat REPL.
    Streaming replies, conversation memory, and tracing.
    """
    client = OllamaModelClient(config) # Build the model client once, from our configuration.
    tracer = Tracer(config)            # The Tracer wraps the model client to record all interactions for observability.

    print(f"You are now chatting with {client.model}. Type 'exit' or 'quit' to stop.\n")

    messages = [] # Conversation History: A list of messages that the model expects. Each message is a dict with "role" (e.g. "user", "assistant") and "content" (the text). This is the chat's memory. We resend the whole thing every turn so that the model has context.

    while True:
        user_input = input(">you ") # Read. Wait for the user to type and press Enter.
        if user_input.strip().lower() in ("exit", "quit"): # Exit Condition
            print("Goodbye!")
            break

        messages.append({"role": "user", "content": user_input})

        print(f"{client.model}> ", end="", flush=True)
        reply = "" # Accumulate the full reply in case we want to use it later.
        start = time.perf_counter() # Start the Stopwatch
        for chunk in client.chat_stream(messages):
            print(chunk, end="", flush=True)
            reply += chunk
        latency_s = time.perf_counter() - start # Stop the Stopwatch; calculate latency in seconds.
        print("\n")

        tracer.record({
            "model": client.model,            # Record the model name
            "user_msg": user_input,           # Record the user input
            "reply_chars": len(reply),        # Record the length of the model's reply in characters.
            "latency_s": round(latency_s, 3), # Record latency rounded to 3 decimal places for readability.
        })

        messages.append({"role": "assistant", "content": reply}) # Record the model's fully reply for the next turn's context.

def cmd_ingest(config, path):
    """
    Build the knowledge base from a file or a folder of documents.
    
    Read the given path, then chunk each document, embed each chunk, and then store the vectors in a vector DB for later retrieval.
    This is the first stage of the RAG pipeline.
    """
    path = Path(path) # Converts the string path into a Path object, as `argparse` returns strings. This allows us to use Path methods and properties throughout the ingest logic without having to convert it repeatedly. It also makes it easier to handle relative vs absolute paths, and to join paths in a platform-independent way.
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
    client = OllamaModelClient(config) # Build the model client once. Provides embed()
    store = ChromaVectorStore(config) # Build the vector store client once. Provides add()

    # EMBED
    # We extract the text from each chunk and send it to the model's embed() function, which returns a list of embeddings (vectors) corresponding to each chunk.
    texts = [c.text for c in chunks]
    embeddings = client.embed(texts)

    # STORE
    store.add(chunks, embeddings)
    print(f"Done. Stored {len(chunks)} chunks in the knowledge base.")

def cmd_ask(config, question):
    """
    Answer a question from the knowledge base using RAG with citations (read-only cited Q&A).

    This is the bridge that ties the RAG pipeline together.
    Retrieve relevant chunks -> Assemble a grounded, cite-intructed prompt -> Ask the model.
    """
    client = OllamaModelClient(config)
    store = ChromaVectorStore(config)
    retriever = Retriever(client, store, config.top_k) # Composes the model and vector store into a retriever that can find relevant chunks for a given question.
    tracer = Tracer(config) # The Tracer wraps the model client to record all interactions for observability.

    # RETRIEVE - Embed the question, then pull the nearest stored chunks
    chunks = retriever.retrieve(question)
    if not chunks:
        print("No relevant information found. Has anything been ingested yet? Try: ingest <path>")
        return
    
    # ASSEMBLE - Build the system & user messages that ground the model in these chunks and instruct it to cite them by number.
    messages = build_messages(question, chunks)

    # ASK - Send one non-streaing call to the model so taht we have the full answer before printing citations.
    start = time.perf_counter() # Start the Stopwatch
    answer = client.chat(messages) # The non-streaming call returns the full answer in one go, which allows us to print citations after the answer is complete.
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

    # Build the tool registry and register search_knowledge as a tool, binding it to the retriever.
    # Retrieval stops being automatic, and instead becomes a tool that the model can choose to call.
    registry = ToolRegistry()
    registry.register(make_search_knowledge_tool(retriever))

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

def main():
    # Build the top-level parser  and a slot for the subcommands ("verbs")
    parser = argparse.ArgumentParser(
        prog="ai-framework-sh", # The name of the program (used in help messages)
        description="A self-hosted AI assistant framework CLI."
    )
    subparsers = parser.add_subparsers(dest="command", required=True) # Create a subparser for each subcommand (e.g. "chat", "ingest", "ask")

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

if __name__ == "__main__":
    main() # Run the CLI entry point

"""
Notes
We haven't changed the REPL (Read-Eval-Print Loop) logic. Instead, it moved from the top-level code into cnd_chat(config). It otherwise uses the same streaming, memory, and tracing logic as before.
The use of `argparse` adds new concepts into the mix.
  - argparse.ArgumentParser(...) — creates the CLI parser. prog is your program's name, description shows up in --help.
  - add_subparsers(dest="command", required=True) — declares "this CLI has subcommands (verbs)." 
    Two key parts:
    - dest="command" - whichever verb the user picks gets stored in args.command. So if they type
      chat, then args.command == "chat".                                                                   
    - required=True - the user must provide a subcommand; running bare main.py with nothing shows a
      helpful error instead of doing something random.                                                   
  - subparsers.add_parser("chat", help=...) — registers the chat verb. Later, "ingest" and "ask" will each get their own add_parser(...)
      line right here (and ask/ingest will also declare their arguments, like the question text or folder path).
  - parser.parse_args() - reads what the user actually typed (from sys.argv), validates it, and returns an args object. 
      Type something invalid, and `argparse` auto-prints usage help.
  - The dispatch — if args.command == "chat": cmd_chat(config) — routes to the right function.  As we add verbs, this grows into a small if/elif (or we can get fancier later).

Also new: `if __name__ == "__main__"`
  In short, it means "only run main() when this file is executed directly, not when it is imported as a module."
  This is a common Python idiom for CLI entry points. It allows the file to be imported without side effects such as starting the REPL or parsing command-line arguments. This is useful for testing or reusing functions in other scripts.
In practice:
  - `Run python apps/cli/main.py chat` → Python sets `__name__` to "__main__". If the guard is true, then main() runs.
  - If some other file did `import apps.cli.main`, then `__name__` is "apps.cli.main", so the guard is false, and main() does not run. 

Wrapping everything in functions means nothing executes on import. The file only does something when you explicitly run it and give it a command. This makes it clean and predictable. We didn't need it before because the old REPL was top level code that ran on site as is. Now that we have reusable functions and a dispatcher, we have to implement the guard. 
NEW 07-08-2026
cmd_ingest(file, path) is heavily upgraded. It is a new entry point for the first stage of the RAG pipeline. Details:
  - It can now read a single file or an entire folder of documents, chunk them, embed them, and store them in a vector database. 
  - The function is designed to be flexible and robust, handling various edge cases like unsupported file types and empty folders gracefully. 
  - It also provides clear feedback to the user about what it is performing, including how many chunks were read and stored
This makes it easier to build a knowledge base from a variety of document sources.
`ingest_parser = subparsers.add_parser()` help now cites file or folder ingestion.
Dispatch also changed to path: cmd_ingest(config, args.path)
NEW 07-10-2026
Retriever(client, store, config.top_k) n - This is where we are composing the model and storing into the retriever.
retriever.retrieve(question) - Handles the question embeddding and internal vector search so that `cmd_ask` doesn't touch the vectors directly.
`format_sources` - Comes from context.py via our `build_messages` function import. It numbers the chunks with `enumerate(..., start=1)` so that the model and the printed sources match. The model is instructed to cite chunks by number, and we print the same numbers for clarity.
  - This is for our observability purposes. The safeguard here is simply: Feed the exact same chunks list, untouched, to both `build_messages` and the print loop.
`"retrieved": [f"{Path(c.source).name} (chunk {c.chunk_idx})" for c in chunks],` - This is a list comprehension that formats the retrieved chunks for tracing. It creates a list of strings, each representing a chunk's source file name and its index within that file. 
  - This provides clear traceability of what information was used to answer the question.
`ask_parser.add_argument("question", help="The questiuon to answer (wrap it in quotes)")` - This is the current rendition for now. `ask` takes one positional question, and the shell needs quotes since questions have spaces within them.
NEW 07-27-2026
There are now two paths for answering questions: `ask` and `agent`. 
  - `ask` is the original RAG pipeline. It always retrieves relevant chunks and instructs the model to cite them (Sources). The model cannot choose to ignore retrieval, and it cannot call retrieval multiple times.
  - `agent` is the new agentic loop. It wraps retrieval as a tool that the model can choose to call or not, and it can call it multiple times if it wants. This is more flexible and allows for more complex reasoning, but it also requires careful prompt design to ensure the model behaves as intended.
"""
"""
Author: Nigel J. Navarro
CLI Entry Point for the ai-framework-sh assistant.

Subcommands:
    chat        Start a REPL (Read-Eval-Print Loop) for chatting with the model.
    (ingest, ask -- not implemented yet)

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

    args = parser.parse_args()  # Parse the command-line arguments
    config = load_config()      # Load the configuration (e.g. model settings, API keys, etc.)

    # Dispatch to the right function based on the chosen subcommand. For now, we only have "chat".
    if args.command == "chat":
        cmd_chat(config)
    elif args.command == "ingest":
        cmd_ingest(config, args.path)

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
NEW 07--08-2026
cmd_ingest(file, path) is heavily upgraded. It is a new entry point for the first stage of the RAG pipeline. Details:
  - It can now read a single file or an entire folder of documents, chunk them, embed them, and store them in a vector database. 
  - The function is designed to be flexible and robust, handling various edge cases like unsupported file types and empty folders gracefully. 
  - It also provides clear feedback to the user about what it is performing, including how many chunks were read and stored
This makes it easier to build a knowledge base from a variety of document sources.
`ingest_parser = subparsers.add_parser()` help now cites file or folder ingestion.
Dispatch also changed to path: cmd_ingest(config, args.path)
"""
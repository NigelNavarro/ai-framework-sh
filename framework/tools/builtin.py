"""
Author: Nigel J. Navarro
Built-In Tools

Tool objects are built here, then registered once. They sit in the registry until the model chooses to call one.

The current version builds the following tools:
  - "search_knowledge": Wraps the existing Retriever as a Tool so the model can choose to search the knowledge base for relevant information to answer a question.
  - "remember": Bound to MemoryStore, this lets the model save a durable fact/preference mid-conversation.
  - "recall": Bound to MemoryStore, this lets the model look up previously saved facts by meaning (semantic recall).
  - "draft_document": Writes a draft file inside data/drafts/ and nothing else (it never sends anything).

It also defines the fixed "nothing found" replies (NO_KNOWLEDGE_RESULT, NO_MEMORY_RESULT) and parse_knowledge_sources(), which reads a search_knowledge result back into {citation number: filename} 
for provenance and the grounding judge.

This is the bridge between our framework's capabilities and the model's ability to call them. 
The model sees only the tool's schema (name, description, parameters) and calls it by name with arguments. The framework handles the rest. 
This does not run the model or the Retriever; it just builds the tools and registers them for later use.
"""
from __future__ import annotations # Store annotations as strings so modern hints (list[str], etc.) work without runtime cost.
from pathlib import Path
import re                           # Leveraging power of REGEX. Used to read search_knowledge's header lines back (see _KNOWLEDGE_HEADER).

from framework.knowledge.retrieve import Retriever
from framework.tools.registry import Tool
from framework.memory.store import MemoryStore
from framework.config import Config

# The exact text each tool returns when it finds nothing. To ensure provenance, we need to know if a tool found something. 
# We turn the no knowledge or no memory results into constants to help with the accuracy checks.
NO_KNOWLEDGE_RESULT = "Nothing in the knowledge base is relevant to this question (no chunk met the relevance threshold)."
NO_MEMORY_RESULT = "No relevant memories found."

# The header line 'search_knowledge' writes above each chunk, e.g. "[3] frankenstein.txt (chunk 204):".
# Captures the citation number and the filename. Must match the f-string in search_knowledge below.
_KNOWLEDGE_HEADER = re.compile(r"^\[(\d+)\] (.+) \(chunk \d+\):$", re.MULTILINE)

def make_search_knowledge_tool(retriever: Retriever) -> Tool:
    """
    Builds the `search_knowledge` tool, and binds it to a specific Retriever. Runs once at wiring time.

    The Retriever is composed of a ModelClient and a VectorStore that embeds a question and returns the nearest stored chunks. 
    Since the model can only pass a `query` string, we take in the Retriever here as an argument and define the handler inside this function where it can see it (otherwise known as a closure). 
    This is a common pattern in Python for creating closures that capture variables from their enclosing scope. The Retriever gets baked into the returned tool, and the model never has to know that it exists.
    """

    def search_knowledge(query: str) -> str:
        """
        The handler for the `search_knowledge` tool. It takes in the `query` string, uses the Retriever to find relevant chunks, and returns them as a single string. 
        This is because the tool result is text that the model can read. It cannot read a list of chunks.
        """
        chunks = retriever.retrieve(query) # Use the Retriever to get the most relevant chunks for the query.
        if not chunks:
            return NO_KNOWLEDGE_RESULT
        
        # Formats the return
        # Example:
        #   [3] frankenstein.txt (chunk 204):
        #   It was on a...
        return "\n\n".join(
            f"[{number}] {Path(chunk.source).name} (chunk {chunk.chunk_idx}):\n{chunk.text}"
            for number, chunk in enumerate(chunks, start=1)
        )
    
    # The schema for the `search_knowledge` tool. This is what the model sees and uses to decide when and how to call it. 
    # A Python dict/list/str maps almost 1:1 onto JSON's object/array/string. This is why we author it as a dict, and the framework will convert and send it to the model as JSON.
    parameters = {
        "type": "object",
        "properties": {
            "query": {
                "type": "string",
                "description": "The question or query to search for in the knowledge base."
            }
        },
        "required": ["query"]
    }
    
    # Bundles the handler and the schema into a Tool object and returns it. The model will see the schema and call the handler when it wants to use this tool.
    return Tool(
        name="search_knowledge",
        description=(
            "Search the user's ingested documents (the knowledge base) for relevant information to answer the query. Use this whenever answering needs information from those documents rather than general knowledge."
        ),
        parameters=parameters,
        handler=search_knowledge,
    )

def parse_knowledge_sources(observation: str) -> dict[int, str]:
    """
    Read a 'search_knowledge' result back into {citation number: filename}. Example:
    
    "[1] mobydick.txt (chunk 12):\n..." becomes {1: "mobydick.txt"}.

    This is the reverse of the f-string inside make_search_knowledge_tool(), which writes those header lines. The two must agree on the format.
    If that f-string changes, then '_KNOWLEDGE_HEADER' must also be updated to match (otherwise this definition returns nothing).
    
    An empty dict means that the result held no chunks (ex. NO_KNOWLEDGE_RESULT).
    """
    return {int(number): name for number, name in  _KNOWLEDGE_HEADER.findall(observation)}

def make_remember_tool(store: MemoryStore) -> Tool:
    """
    Builds the 'remember' tool, binding it to a MemoryStore.

    Let's the model save a durable fact or preference mid-conversation. Wraps store.add, which writes to both memory.json and the Chroma memory collection.
    This is our first tool that writes to the framework's state, but only inside data/ (no network or other writes elsewhere).
    """

    def remember(text: str) -> str:
        """
        The handler: save one fact to memory and confirm it back to the model.
        """
        store.add(text)
        return f"Saved to memory: {text}" # Return a confirmation string to the model that the fact was saved.

    parameters = {
        "type": "object",
        "properties": {
            "text": {
                "type": "string",
                "description": "A fact or preference about the user worth remembering across sessions."
            }
        },
        "required": ["text"]
    }

    return Tool(
        name="remember",
        description=(
            "Save a durable fact or preference about the user so it persists across "
            "sessions. Use when the user states a lasting preference, detail, or goal."
        ),
        parameters=parameters,
        handler=remember,
    )

def make_recall_tool(store: MemoryStore) -> Tool:
    """
    Build the recall tool, binding it to a MemoryStore.
    
    This let's the model look up preivously saved facts by meaning (sematic recall).
    Wraps store.recall, which embeds the query and returns the most relevant (nearest neighbor) saved facts.
    """

    def recall(query: str) -> str:
        """
        Handler: Fetch the relevant memories and return them as text for the model.
        """
        facts = store.recall(query)
        if not facts:
            return NO_MEMORY_RESULT
        # One fact per line, numbered for clarity. The model can read this and decide which to use.
        return "\n".join(f"[{number}] {fact}" for number, fact in enumerate(facts, start=1))

    parameters = {
        "type": "object",
        "properties": {
            "query": {
                "type": "string",
                "description": "What to look up in the user's saved memories."
            }
        },
        "required": ["query"],
    }

    return Tool(
        name="recall",
        description=(
            "Look up previously saved facts or preferences about the user by meaning (semantic recall). "
            "Use when the model needs to remember something the user has told it in a previous session."
        ),
        parameters=parameters,
        handler=recall
    )

def make_draft_document_tool(config : Config) -> Tool:
    """
    Builds the `draft_document` tool. Writes a draft to a data/drafts and nothing else.

    This is built with safety boundary L2. The model can create an artifact, but it cannot send it.
    Human intervention outside of this command will decide to review and/or manipulate the file and its contents separately.
    This is a purposeful feature, and the tool is deliberately confined to a single folder for safety.
    """

    def draft_document(filename: str, content: str) -> str:
        """
        Handler: Save content as a file in data/drafts/ 
        Return the path.
        """
        drafts_dir = config.data_dir / "drafts"
        drafts_dir.mkdir(parents=True, exist_ok=True)  # Creates data/ and data/drafts if they do not exist.

        # Security: Models will provide just the filename. We will not allow any path traversal or absolute paths.
        # Strip any directory parts so it can never escape. "../../x" -> "x", "/tmp/x" -> "x".
        safe_name = Path(filename).name             # Extract just the filename, removing any directory components
        if not safe_name:
            return "Error: could not create a draft. Filename was empty or invalid."

        path = drafts_dir / safe_name
        path.write_text(content, encoding="utf-8")  # Write the content to the file
        return f"Draft saved to: {path}"            # Return the path to the model

    parameters = {
        "type": "object",
        "properties": {
            "filename": {
                "type": "string",
                "description": "A short filename for the draft, e.g. 'email_to_team.md'.",
            },
            "content": {
                "type": "string",
                "description": "The full text of the draft to write.",
            },
        },
        "required": ["filename","content"]
    }

    return Tool(
        name="draft_document",
        description=(
            "Write a draft document (email, note, letter, etc.) to a local drafts folder "
            "for the user to review. Use when the user asks you to draft or write something. "
            "This only saves a file. It never sends or shares anything."
        ),
        parameters=parameters,
        handler=draft_document,
    )




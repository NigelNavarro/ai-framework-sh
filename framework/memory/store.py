"""
Author: Nigel J. Navarro
Memory Store

Persists facts and preferences across sessions, stored as a human-readable JSON file plus a Chroma vector index for semantic recall.
Similar in framing to ChromaVectorStore() in chroma.py.
This will be a dual store (because we need two different ways to reach a fact).
  1. memory.json - Human Readable source of truth. Open, edit, or delete facts by hand (memory must be inspectable and/or forgettable).
  2. A Chroma "memory" collection - Semantic recall index. Allows the model to find facts by meaning, not just by exact string match. This is the "brain" of the memory system.
Both hold the same facts, linked by a shared ID. Every Write hits both, and they stay in step.
A ModelClient does the embedding, and Chroma stores the vectors.
"""
from __future__ import annotations

import json
import uuid
from datetime import datetime, timezone
import chromadb

from framework.config import Config
from framework.models.base import ModelClient

# We have a separate collection of knowledge from the knowledge base, as we don't want personal facts to be mixed in with those.
COLLECTION_NAME="memory"

class MemoryStore:
    """
    Dual store for durable facts; memory.json and a Chroma collection for semantic recall. Takes in a ModelClient so it stays backend-agnostic, like Retriever().
    """

    def __init__(self, config: Config, client: ModelClient):
        self.client = client                                    # used for an interface, embed()
        self.embed_model = config.embed_model  
        self.json_path = config.data_dir / "memory.json"        # The human-readable JSON file for facts

        config.data_dir.mkdir(parents=True, exist_ok=True)      # Ensure the data directory exists before trying to Read/Write files in it.

        # Load existing facts from JSON if the file is there; otherwise start empty.
        # self.facts is the in-memory mirror of memory.json. (a list of {id, text, created_at} dicts)
        if self.json_path.exists():
            self.facts = json.loads(self.json_path.read_text(encoding="utf-8"))
        else:
            self.facts = []                                     

        # Same on-disk Chroma database as the documents, but a different collection. 
        # PersistentClient is what makes the data durable, while self.collection is just a handle.
        chroma_path = config.data_dir / "chroma"                        # The Chroma database path for semantic recall
        chroma_path.mkdir(parents=True, exist_ok=True)                  # Ensure the Chroma directory exists before trying to create a PersistentClient in it.
        self.chroma = chromadb.PersistentClient(path=str(chroma_path))  # PersistentClient stores data.

        # Stamp the collection with the embedding model (same guardrail used in ChromaVectorStore).
        # IF the embed model changes, old vectors live in an incompatible space.
        self.collection = self.chroma.get_or_create_collection(
            name=COLLECTION_NAME,
            metadata={"embed_model": config.embed_model},       # Store the embedding model in the collection metadata for later reference
        )
        # Safely get the embed_model from metadata, defaulting to None if metadata is missing
        built_with = (self.collection.metadata or {}).get("embed_model")
        if built_with != config.embed_model:
            raise ValueError(
                f"Memory store was built with embedding model '{built_with}', but "
                f"config uses '{config.embed_model}'. "
                f"Delete data/chroma and re-add memories."
            )

    def _save(self) -> None:
        """
        Write and save the in-memory facts to memory.json. 
        """
        self.json_path.write_text(
            json.dumps(self.facts, indent=2), # indent=2 makes it human-readable and editable.
            encoding="utf-8"
        )

    def add(self, text: str) -> str:
        """
        Save one fact to both stores and return its ID.

        JSON get the readable record; Chroma gets the embedding for the sematic recall.
        The shared ID is what lets list/delete keep the two in sync later.
        """
        fact_id = uuid.uuid4().hex # A unique ID in hex format. Links JSON and Chroma records together.
        fact = {
            "id": fact_id,
            "text": text,
            "created_at": datetime.now(timezone.utc).isoformat() # ISO format with UTC timezone. This is human-readable and sortable.
        }

        # Write 1: JSON (source of truth)
        self.facts.append(fact) # Add the new fact to the in-memory list
        self._save()            # Persist the in-memory list to memory.json

        # Write 2: Chroma (recall index)
        # Embed the text into one vector. Wrap in a list because embed() expects a list of texts, 
        # and take the first (and only) result (like Retriever).
        embedding = self.client.embed([text])[0]
        self.collection.add(
            ids=[fact_id],                                  # Unique string ID for the fact, same as in JSON
            embeddings=[embedding],                         # The vector for the fact, in the same order
            documents=[text],                               # The text of the fact, in the same order
            metadatas=[{"created_at": fact["created_at"]}]  # Metadata for the fact, in the same order
        )
        return fact_id # Return the unique ID of the newly added fact, so it can be referenced later (e.g., for deletion).

    def recall(self, query: str, k: int = 5) -> list[str]:
        """
        Return up to 'k' stored facts most relevant to the 'query' by meaning.

        Similar to Retriever.retrieve(), but for facts instead of documents. 
        The Chroma path embeds the query and asks for the nearest fact vectors.
        This is what feeds relevant memories into the model's context later.
        """
        if not self.facts:
            return [] # If there are no facts to recall, return an empty list. Skips embed + query if there is nothing to search.

        # Embed the query into one vector. Wrap in a list because embed() expects a list of texts, 
        # and take the first (and only) result. Query text -> One Vector (like Retriever)
        query_embedding = self.client.embed([query])[0]
        results = self.collection.query(
            query_embeddings=[query_embedding], # Since Chroma supports batch queries, we wrap our single query_embedding in a list to make it a batch of one.
            n_results=min(k, len(self.facts))   # This carries the same meaning as top_k: how many results to return, ordered nearest-first. We also ensure we don't ask for more than we have.
        )
        return results["documents"][0]          # Unwrap the single query's results -> list of fact texts.

    def list(self) -> list[dict]:
        """
        Human-readable path: return every stored fact, straight from the JSON mirror.
        No embedding or vector search is needed here; we just want the readable records.        
        """
        # Return a copy of the in-memory list of facts. Makes it so that callers cannot mutate our internal list.
        return list(self.facts)

    def delete(self, fact_id: str) -> bool:
        """
        Forgets one fact by ID, removing it from both stores. Returns True if it existed.

        Keeping the two in lockstep is the whole point of the shared ID.
        Delete the JSON record and its Chroma vector, so that a recall can never surface a forgotten fact.
        """
        # Check if ID is actually present (drives the return value and CLI feedback).
        if not any(fact["id"] == fact_id for fact in self.facts):
            return False # Return false if the fact ID is not found.

        # Remove 1: JSON (source of truth)
        # Filter out the fact with the given ID from the in-memory list
        self.facts = [fact for fact in self.facts if fact["id"] != fact_id]
        self._save()

        # Remove 2: Chroma (recall index)
        # Drop the matching vector by the same ID.
        self.collection.delete(ids=[fact_id])
        return True # Return true if the fact was found and deleted.
    
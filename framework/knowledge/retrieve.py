"""
Author: Nigel Navarro
Retrieval

Turns a question into the most relevant stored chunks.
This is the read path of the Retrieval-Augmented Generation (RAG) pipeline. 
We compose a model (to embed the question) and a vector store (to find the nearest chunks) into one call,
so callers just ask a question (retrieve) without touching embeddings or Chroma directly.

Dependencies are only on ModelClient and VectorStore interfaces. This should work with any model or storage backend.
"""
from __future__ import annotations

from framework.knowledge.ingest import Chunk
from framework.knowledge.vectorstore.base import VectorStore
from framework.models.base import ModelClient

class Retriever:
    """
    A Retriever composes a ModelClient and a VectorStore to embed a question and return the nearest stored chunks.
    """

    # The Retriever is initialized with a model and a store. It uses the model to embed questions and the store to find the nearest chunks.
    def __init__(self, model: ModelClient, store: VectorStore, top_k: int = 5): 
        # top_k is set to 5 because we want to return the 5 most relevant chunks by default. This can be overridden per call.
        self.model = model
        self.store = store
        self.top_k = top_k
    
    def retrieve(self, question: str, top_k: int | None = None) -> list [Chunk]:
        """
        Given a question, return the chunks that are most relevant to it, nearest-first.
        Pass `top_k` to override the default for a single call.
        """

        k = top_k if top_k is not None else self.top_k # Use the provided top_k or fall back to the default
        query_embedding = self.model.embed([question])[0] # text -> one vector. 
        #We wrap the question in a list because embed() expects a list of texts, and we take the first (and only) result.
        return self.store.query(query_embedding, k) # vector -> nearest chunks.
    
"""
Notes
- Retriever initialization: The Retriever is handed a model and a store when it is created. 
Crucially, the type hints are the interfaces (ModelClient & VectorStore) because we want to be able to swap in any model or store that implements those interfaces. 
This makes the Retriever flexible and backend-agnostic.
Using something like `retriever = Retriever(OllamaModelClient(cfg), ChromaVectorStore(cfg), cfg.top_k)`, the caller can compose any model and store that implement the required interfaces.
But, the retriever itself does not know or care about the specific model or store. It just calls embed() and query() on the interfaces.
- "Either an INT or nothing." - The `top_k: int | None = None — the int | None` is a Union type. It hints that top_k can either be an integer representing how many chunks to return or None, which means use the default (in other words, we chose not to specify).
- `query_embedding = self.model.embed([question])[0]` Embeds the question (where [question] wraps it in a list, and [0] takes the single vector back), then hands that vector to the store `store.query`.
This is the same behavior that we tested in smoke_test_vectorstore.py, but now it is wrapped in a convenient Retriever class that hides the embedding and querying details from the caller.
"""
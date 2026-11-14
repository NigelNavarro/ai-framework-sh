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
    def __init__(self, model: ModelClient, store: VectorStore, top_k: int = 5, min_score: float = 0.0): 
        # top_k is set to 5 because we want to return the 5 most relevant chunks by default. This can be overridden per call.
        self.model = model
        self.store = store
        self.top_k = top_k
        self.min_score = min_score  # cosine similarity floor. Chunks scoring below it count as "not relevant" by retrieve().

    def retrieve_with_scores(self, question: str, top_k: int | None = None) -> list[tuple[Chunk, float]]:
        """
        An evolution of the original retrieve(). 
        Unfiltered: returns every top_k result regardless of the threshold, so the raw evidence is always available
        (for evaluations and traces)
        Each chunk is paired with its relevance score (cosine similarity, higher value = more relevant)
        This is the evidence that the framework uses to detect the knowledge boundary. 
        If even the best score is low, nothing in the corpus applies to the question.
        """

        k = top_k if top_k is not None else self.top_k      # Use the provided top_k or fall back to the default
        query_embedding = self.model.embed([question])[0]   # text -> one vector. 
        # We wrap the question in a list because embed() expects a list of texts, and we take the first (and only) result [0].
        return self.store.query_with_scores(query_embedding, k)         # vector -> nearest chunks, each with its own score.
    
    def retrieve(self, question: str, top_k: int | None = None) -> list [Chunk]:
        """
        Given a question, return the chunks that score at or above the min_score, nearest-first. 
        If no chunks meet the threshold, then an empty list is returned. This signals that the question
        falls outside what the knowledge base covers.
        
        Pass `top_k` to override the default for a single call.
        """

        # Keep the chunks at or above the threshold.
        return [chunk for chunk, score in self.retrieve_with_scores(question, top_k) if score >= self.min_score]  

"""
Author: Nigel J. Navarro
The VectorStore Interface

Like the ModelClient for models, this abstraction makes the storage backend a replaceable component. The RAG pipeline depends on this interface.
Future backend implementations will be able to use this same interface.
"""
from __future__ import annotations

from abc import ABC, abstractmethod # ABC = Abstract Base Class, a Python tool for defining interfaces/contracts
from framework.knowledge.ingest import Chunk # We need the shape of a Chunk to define the contract for adding documents

class VectorStore(ABC):
    """
    The contract for storing chunk embeddings and retrieving them by similarity. 
    """
    @abstractmethod # This decorator means that any concrete subclass MUST implement this method, or Python will raise an error when you try to instantiate it.
    def add(self, chunks: list [Chunk], embeddings: list[list[float]]) -> None:
        """
        Store chunks alongside their embedding vectors.
        `embeddings[i]` is the vector for `chunks[i]` - They are the same length and in the same order.
        The method returns nothing; it just stores the data. 
        The storage mechanism is up to the implementation (e.g. in-memory, ChromaDB., etc.)
        """
    
    @abstractmethod
    def query(self, query_embedding: list[float], top_k: int) -> list[Chunk]:
        """
        Return the `top_k` stored chunks whose vectors are nearest to the query.
        
        Results come back as Chunk objects (text + source metadata), ordered nearest-first, so the caller can cite where each answer came from.
        """

# --- Future option: Return similarity scores alongside chunks ---
# Useful later for thresholds ("only use chunks above 0.7") or showing confidence. Each backend would return (chunk, score) pairs, nearest-first.
# Should we use this, every backend will be required to implement it too. Un-commenting an abstractmethod adds an obligation to all implementations.
    # @abstractmethod
    # def query_with_scores(
    #     self, query_embedding: list[float], top_k: int
    # ) -> list[tuple[Chunk, float]]:
    #     """
    #     Like query(), but each result is paired with its similarity score (higher = more similar).
    #     Ordered nearest-first.
    #     """


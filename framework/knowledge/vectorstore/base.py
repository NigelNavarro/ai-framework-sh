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
    def query_with_scores(self, query_embedding: list[float], top_k: int) -> list[tuple[Chunk, float]]:
        """
        Return the `top_k` nearest chunks, each paired with a relevance score (nearest-first).
        
        The score is a cosine similarity metric on a fixed scale between '-1.0' to '1.0'. Higher value = higher relevancy.
          - A score of '1.0' means that the same direction is the same meaning.
          - Score ranging around '0.0 to 0.5' means that the text is unrelated.
          - Small negative scores like '-0.05' just means really unrelated.
          - Scores outside of '-1.0' to '1.0' point to a possible bug. Investigate further.
        """

    def query(self, query_embedding: list[float], top_k: int) -> list[Chunk]:
        """
        Return the `top_k` nearest chunks (nearest-first) without scores.
        
        The query_with_scores() function performs the search and returns full results with scores.
        Then, query() returns a trimmed down version of the answer, consisting of a list of the 'top_k' stored Chunk objects 
        nearest to the question's vector, ordered nearest-first (each with its text, source, and chunk index) and dropping 
        the score with '_score' since the relevance score is not required in the response.
        """
        return [chunk for chunk, _score in self.query_with_scores(query_embedding, top_k)]
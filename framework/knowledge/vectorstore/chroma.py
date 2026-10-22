"""
Author: Nigel J. Navarro
ChromaVectorStore

Chroma implementation of the VectorStore interface. This stores chunk embeddings in an embedded on-disk Chroma database (PersistentClient, no server required)
This fulfills the VectorStore Contract: add() stores chunks & their vectors, query() returns the nearest chunks. The collection is stamped with the embedding model that built it, so a future model switch can be detected (the re-index guardrail).
"""
from __future__ import annotations
import chromadb # The ChromaDB Python Client Library

from framework.config import Config
from framework.knowledge.ingest import Chunk
from framework.knowledge.vectorstore.base import VectorStore


COLLECTION_NAME = "knowledge_base"
SPACE = "cosine"                    # Distance metric. Cosine ignores vector length so that scores are comparable across chunks.

class ChromaVectorStore(VectorStore):
    """
    A VectorStore backed by an embedded Chroma database.
    """

    # Initialize the Chroma client and collection. The collection is where we store our chunks and embeddings. 
    # We use the embedding model name as a stamp in the collection metadata to detect incompatible stores if the model changes later.
    def __init__(self, config: Config):
        self.embed_model = config.embed_model

        # Embedded, on-disk Chroma. Files live under data/chroma and survive restarts.
        chroma_path = config.data_dir / "chroma"
        chroma_path.mkdir(parents=True, exist_ok=True) # Make sure the directory exists. parents=True allows creating intermediate directories if needed, exist_ok=True avoids errors if it already exists.
        self.client = chromadb.PersistentClient(path=str(chroma_path)) # PersistentClient stores data.

        # Get or create the collection, stamped with the embedding model used.
        self.collection = self.client.get_or_create_collection(
            name=COLLECTION_NAME,
            metadata={"embed_model": config.embed_model}, # Store the embedding model in the collection metadata for later reference
            configuration ={"hnsw": {"space": SPACE}}       # Hierarchical Navigable Small World - Graph-based indexing algorithm for extremely fast approximate nearest-neighbor (ANN) searches across large collections of high-dimensional vectors.
            # The 'configuration' only applies when the collection is first-created. Existing collections keep their metric.
        )

        # Guardrail 1
        # If this store was built with a *different* embedding model, its vectors live in an incompatible coordinate system. 
        # Searching would therefore return garbage. Should this happen, prompt for a re-indexing with the desired new model.
        built_with = (self.collection.metadata or {}).get("embed_model") # Safely obtain the embed_model from metadata, defaulting to 'None' if metadata is missing
        if built_with != config.embed_model:
            raise ValueError(
                f"Vector store was built with embedding model '{built_with}', but "
                f"config uses '{config.embed_model}'. "
                f"Re-index required: delete the '{COLLECTION_NAME}' collection and re-ingest."
            )

        # Guardrail 2
        # Cosine distance is the angle between two vectors.
        # Relevance scores are computed as '1 - cosine distance'.
        # Therefore, the scores are only correct if the collection measures the right distance.
        # get_or_create_collection() silently keeps an old collection's metric, so a store built before this change would produce incorrectly-scaled scores.
        built_space = (self.collection.configuration.get("hnsw") or {}).get("space")
        if built_space != SPACE:
            raise ValueError(
                f"Vector store uses the '{built_space}' metric, but relevance scores need '{SPACE}'. "
                f"Re-index required: delete the '{COLLECTION_NAME}' collection and re-ingest."
            )
        
    def add(self, chunks: list[Chunk], embeddings: list[list[float]]) -> None:
        if not chunks:
            return # No chunks to add, then there is nothing to store.
        
        # Chroma wants parallel lists: ids, vectors documents (text), and metadata.
        ids = [f"{c.source}:{c.chunk_idx}" for c in chunks] # Unique ID for each chunk, e.g. "doc1.txt:0", "doc1.txt:1", etc.
        documents = [c.text for c in chunks] # The text of each chunk
        metadata = [{"source": c.source, "chunk_idx": c.chunk_idx} for c in chunks] # Formatted as a dict. For citations later.

        self.collection.add(
            ids = ids, # Unique string IDs for each chunk, e.g. "doc1.txt:0", "doc1.txt:1", etc.
            embeddings = embeddings, # The vectors for each chunk, in the same order. These are our (nomic-embed-text) vectors, not chromas.
            documents = documents, # The text of each chunk, in the same order
            metadatas = metadata, # The metadata for each chunk, in the same order 
        )

    # Given a query vector, return a list of top_k nearest chunks as Chunk objects each paired with its relevance score.
    def query_with_scores(self, query_embedding: list[float], top_k: int) -> list[tuple[Chunk, float]]:
        results = self.collection.query(
            query_embeddings = [query_embedding],   # Since Chroma supports batch queries, we wrap our single query_embedding in a list to make it a batch of one.
            n_results = top_k,                      # This carries the same meaning as top_k: how many results to return, ordered nearest-first.
        )

        # Chroma nests results one level per query. 
        # Since we only have one query, we take the first (and only) element of each list [0] to get the results for our single query.
        documents = results["documents"][0]
        metadatas = results["metadatas"][0]
        distances = results["distances"][0] # cosine distance = 1 - similarity. Possible ranges: 0 (identical) to 2 (opposite). Real text usually lands around 0.2 to 0.8
    
        # Rebuild each Chunk and pair it with its score. This is so the caller gets (Chunk, score) tuples (nearest-first).
        return [
            (
                Chunk(
                    text=text,
                    source=meta["source"],
                    chunk_idx=meta["chunk_idx"],
                ),
                1.0 - distance,
            )
            # Zip the three parallel lists together: one (text, meta, distance) triple per Chunk result.
            for text, meta, distance in zip(documents, metadatas, distances)
        ]


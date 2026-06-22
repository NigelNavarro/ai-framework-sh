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
            metadata={"embed_model": config.embed_model}, # Store the embedding model in the collection metadata for later reference}
        )

        # The following is a Guardrail.
        # If this store was built with a different embedding model, its vectors live in an incompatible coordinate system. Searching would therefore return garbage. 
        # With this we will ensure that we are prompting a re-index with the desired new model.
        built_with = (self.collection.metadata or {}).get("embed_model") # Safely get the embed_model from metadata, defaulting to None if metadata is missing
        if built_with != config.embed_model:
            raise ValueError(
                f"Vector store was built with embedding model '{built_with}', but "
                f"config uses '{config.embed_model}'. "
                f"Re-index Required. Delete data/chroma and re-ingest before querying."
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
    
    def query(self, query_embedding: list[float], top_k: int) -> list [Chunk]: # Given a query vector, return the top_k nearest chunks as Chunk objects
        results = self.collection.query(
            query_embeddings = [query_embedding], # Since Chroma supports batch queries, we wrap our single query_embedding in a list to make it a batch of one.
            n_results = top_k, # This carries the same meaning as top_k: how many results to return, ordered nearest-first.
        )

        # Chroma nests results one level per query. 
        # Since we only have one query, we take the first (and only) element of each list to get the results for our single query.
        documents = results["documents"][0]
        metadatas = results["metadatas"][0]

        # Rebuild Chunk objects so the caller gets the same type that went in
        return [
            Chunk(
                text=text,
                source=meta["source"],
                chunk_idx=meta["chunk_idx"]
                )
            # Zip together the documents and their corresponding metadata to reconstruct the Chunk objects.
            for text, meta in zip(documents, metadatas)
        ]

"""
Notes
__init___ setup and guardrail
- chromadb.PersistentClient(path=...) - This is the embedded, on-disk Chroma client. It writes to the database at data/chroma (relative to the project root) and survives restarts. No server required. Vectors under data/ won't get committed.
- get_or_create_collection(name, metadata={...}) - A collection is Chroma's equivalent of a table. 
- built_with guardrail - This is a safety check that reads back the collection's stored embed_model and compares it to the current config. Should they differ, it raises a clear error telling you to re-index. This was built for a possible future feature that allows model switching, and the switch gets caught here instead of silently returning garbage results.

add - store chunks + vectors
- Chroma's add() method takes four parallel lists of all the same length and order: ids, embeddings, documents, and metadatas. Index i across all four describes one chunk.
- ids must be unique strings, so we use `"{source}:{chunk_idx}"` (e.g. frankenstein.txt:10) which is unique per chunk.
- embeddings=embeddings - We pass our own vectors (made by nomic-embed-text via embed()) to Chroma. This tells Chroma to not compute the embeddings itself and instead use our own pre-computed vectors. This is important because this is how we keep control of which model encodes the meaning, keeping our two-model design intact.
- metadatas carries the source and chunk_idx for each chunk.

query - retrieve the nearest chunks given a query vector
- query_embeddings=[query_embedding] - Chroma's query() supports batch queries (answering many queries at once), but we only need to send one query vector, so we wrap it in a list to make it a batch of one.
- results["documents"][0] - Given that Chroma supports batch queries, it nests our results one level per query (a list per query). Since we only sent one query, we take the first (and only) element of the results lists to get our actual results.
- Rebuilding chunks - We reconstruct Chunk objects from the returned documents (text) and metadatas (source and chunk_idx) so that query returns the same type (chunk) that add() took in. This way the VectorStore interface is consistent, and the caller doesn't have to know about Chroma's internal structure.
"""
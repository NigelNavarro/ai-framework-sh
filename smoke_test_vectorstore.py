"""
Author: Nigel J. Navarro
Smoke Test: Vector store round-trip (ingest -> embedd -> add -> query)

This test proves out the Retrieval-Augmented Generation (RAG) storage layer end to end.
It ingests a small batch of book chunks, embeds them using nomic-embed-text, stores them in Chroma,
then queries the store with a real question, confirming the nearest chunks come back as Chunk objects.

Note: add() is not an UPSERT. Running it twice will error our on duplicate IDs.
To re-run cleanly, delete the data/chroma folder first.

Run: .venv/bin/python smoke_test_vectorstore.py
"""
from pathlib import Path

from framework.config import load_config
from framework.models.ollama import OllamaModelClient
from framework.knowledge.ingest import ingest_folder
from framework.knowledge.vectorstore.chroma import ChromaVectorStore

cfg = load_config()
model = OllamaModelClient(cfg)
store = ChromaVectorStore(cfg)

# Ingest - Read & Chunk th books, then take a small subset to keep the test fast.
books = Path("/Users/nigelnavarro/gitRepos/bookbot/books")
all_chunks = ingest_folder(books, cfg.chunk_size, cfg.chunk_overlap) # Ingest the books into chunks.
chunks = all_chunks[:20] # Take the first 20 chunks for a quick test. These are our "documents" that we want to store and retrieve.
print(f"Ingested {len(all_chunks)} chunks total. Using the first {len(chunks)} for this test.\n")

# Embed - Turn each chunk's text into a vector with the embedding model nomic-embed-text.
embeddings = model.embed([c.text for c in chunks]) # Embed the text of each chunk into a vector. 
print(f"Embedded {len(embeddings)} chunks.\n")

# Add - Store the chunks and their vectors in the Chroma vector store. Each chunk gets a unique ID based on its source and index.
store.add(chunks, embeddings)
print(f"Stored {len(chunks)} chunks in the vector store.\n")

# Query - Embed a question, retrieve the nearest chunks
question = "What is the narrator's expedition about?"
query_vector = model.embed([question])[0] # Embed the question into a vector. We only have one question, so we take the first (and only) vector from the list.
hits = store.query(query_vector, top_k=3) # Query the vector store for the top 3 nearest chunks to the question vector.

print(f"Query: {question!r}") # Print the question being asked, using !r to show it as a string with quotes.
print(f"Top {len(hits)} matching chunks:\n")
for rank, hit in enumerate(hits, start=1):
    source = hit.source.split("/")[-1]
    print(f"{rank}. {source} (chunk {hit.chunk_idx})")
    print(f"   {hit.text[:100]!r}\n")

""" 
The output of the test should look something like this:
---
Ingested 3018 chunks total. Using the first 20 for this test.

Embedded 20 chunks.

Stored 20 chunks in the vector store.

Query: "What is the narrator's expedition about?"
Top 3 matching chunks:

1. frankenstein.txt (chunk 10)
   'agitation with which I began my\nletter, and I feel my heart glow with an enthusiasm which elevates m'

2. frankenstein.txt (chunk 12)
   'to this great\nenterprise. I commenced by inuring my body to hardship. I accompanied\nthe whale-fisher'

3. frankenstein.txt (chunk 9)
   'world never before\nvisited, and may tread a land never before imprinted by the foot of man.\nThese ar'
---
Note: The `\n` in this case are coming from the original text of the book, which has line breaks. 
The !r in the f-string shows the string representation, including escape characters like `\n`.
"""
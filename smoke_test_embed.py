"""
Author: Nigel J. Navarro
Smoke Test: Embedding (embed)

Verifies OllamaModelClient.embed() works with the configured embedding model (nomic-embed-text): feeds a few short texts and prints each resulting vector's dimension and a small preview, confirming text -> vector encoding works.

Requires: ollama pull nomic-embed-text

Run: .venv/bin/python smoke_test_embed.py
"""
from framework.config import load_config
from framework.models.ollama import OllamaModelClient

client = OllamaModelClient(load_config()) # Create an instance of the OllamaModelClient using the loaded configuration. This client will be used to call the embed method.

texts = [
    "The cat sat on the mat.",
    "A dog rested on the rug.",
    "Quantum mechanics describes subatomic particles.",
]

print(f"Embedding {len(texts)} texts using {client.embed_model} ... \n")

vectors = client.embed(texts) # Call the embed method with the list of texts. This should return a list of embedding vectors corresponding to each input text.

# Pair each text with its vector, and show the shape + a tiny preview
for text, vector in zip(texts, vectors):
    print(f"- {text!r}") # Print the original text (repr for quotes)
    print(f"    dimensions: {len(vector)}") # Print the length of the vector, which is its dimensionality (e.g. 768, 1024, etc.)
    print(f"    first 5 numbers: {vector[:5]}") # Print a small preview of the vector (first 5 numbers) to confirm we got something that looks like an embedding. 
    print()
"""
Author: Nigel J. Navarro
Smoke Test: Semantic Similarity Search

Embeds three sentences and prints the cosine similarity between each pair.
We want to show that similar meanings produce closer vectors.
The two 'pet' sentences should score higher with each other than either does with the 'physics' sentence.

Requires: ollama pull nomic-embed-text
Run: .venv/bin/python smoke_test_similarity.py
"""
import math

from framework.config import load_config
from framework.models.ollama import OllamaModelClient

def cosine_similarity(a: list[float], b: list[float]) -> float:
    """
    How aligned two vectors are: ~1.0 = same direction (similar meaning), ~0.0 = unrelated.
    Formula: dot(a,b) / (length(a) * length(b))
    """
    dot =sum(x * y for x, y in zip(a, b))       # dot is the sum of the products of corresponding elements. It rewards matching coordinates.
    length_a = math.sqrt(sum(x * x for x in a)) # length is the square root of the sum of squares. It measures the magnitude of vector a.
    length_b = math.sqrt(sum(y * y for y in b)) # length is the square root of the sum of squares. It measures the magnitude of vector b.
    return dot / (length_a * length_b)          # Normalize -> -1.0 .. 1.0

client = OllamaModelClient(load_config())

sentences = [
    "The cat sat on the mat.",
    "A dog was lying on the rug.",
    "Quantum physics is a fundamental theory in physics."
]

# Embed all three -> one 768-dimensional vector per sentence (for the nomic-embed-text model).
vectors = client.embed(sentences)

# Compare every unique pair and print the cosine similarity.
# We expect the first two sentences to be more similar to each other.
print("Cosine Similarity per pair (higher - more similar meaning):\n")
for i in range(len(sentences)):             # Loop over each sentence index
    for j in range(i + 1, len(sentences)):  # Loop over each subsequent sentence index to avoid duplicate pairs and self-comparison
        score = cosine_similarity(vectors[i], vectors[j])
        # Print the similarity score rounded to 3 decimal places, along with the two sentences being compared. The !r in the f-string calls repr() on the sentence, which adds quotes around it for clarity.
        print(f"  {score:.3f}   {sentences[i]!r} vs   {sentences[j]!r}")

"""
Notes
- cosine_similarity(a, b) returns a float between -1.0 and 1.0 that indicates how similar the two vectors are in terms of direction. A score close to 1.0 means the vectors are pointing in the same direction (similar meaning), while a score close to 0.0 means they are orthogonal (unrelated meaning). Negative scores would indicate opposite meanings, but with text embeddings we usually expect non-negative scores.
- sum(x * y for x, y in zip(a, b)) computes the dot product of the two vectors a and b. It multiplies corresponding elements and sums them up. This rewards pairs of coordinates that are both high (e.g. both vectors have a strong presence in the same dimension). 
- zip() is a Python function that pairs up elements from two lists, so we can multiply them together.
- math.sqrt(sum(x * x for x in a)) computes the length (magnitude) of vector a. It sums the squares of the coordinates and takes the square root. This measures how long the vector is in high-dimensional space.
- The nested loop with range( i + 1, len(sentences)) ensures that we only compare each unique pair of sentences once, and we don't compare a sentence with itself. By starting j at `i + 1`, we compare (0,1), (0,2), and (1,2). We don't compare a sentence with itself (e.g. (0,0)), never the same pair twice, or repeat pairs in reverse order (e.g. (1,0)).
- {score:.3f} formats the similarity score to 3 decimal places for readability (0.612 instead of 0.6123456789).
Remember that what we are testing here are the embeddings: the semantic representations that the model produces for each sentence. Weights are the trained parameters that generate those embeddings.
"""
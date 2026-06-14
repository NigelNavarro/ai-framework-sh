"""
Author: Nigel J. Navarro
OllamaModelClient implementation of the ModelClient contract. Talks to a local Ollama server via REST API using the `requests` library.
"""
from __future__ import annotations

import requests

from framework.config import Config
from framework.models.base import ModelClient, Message

class OllamaModelClient(ModelClient):
    """
    A ModelClient backed by a local Ollama server.
    """

    def __init__(self, config: Config):
        # Pull what we need from the config. Never hardcode the model names or hosts.
        self.model = config.model
        self.embed_model = config.embed_model
        self.host = config.ollama_host

    def chat(self, messages: list[Message]) -> str:
        """
        Send a conversation to the Ollama server and return the model's reply text.
        """
        response = requests.post(
            f"{self.host}/api/chat",
            json={
                "model": self.model,
                "messages": messages,
                "stream": False, # Wait for the full response, don't stream tokens.
            }
        )
        response.raise_for_status() # Raise an error if the request failed (e.g HTTP 404, 500, etc.) as exceptions.
        return response.json()["message"]["content"] # Ollama nests the text under message -> content. The structure of the response is defined by Ollama's API.
    
    def embed(self, texts: list[str]) -> list[list[float]]:
        """
        Turn text into embedding vectors using the Ollama server (it's embeddings endpoint).

        One request per text; reeturns the vectors in the same order. Requires the embedding model to be pulled. 
        06-13-26: Today we are going to define the embed method to fulfill the contract.
        """
        vectors: list[list[float]] = []
        for text in texts:
            response = requests.post(
                f"{self.host}/api/embeddings",
                json={"model": self.embed_model, "prompt": text},
            )
            response.raise_for_status()
            vectors.append(response.json()["embedding"])
        return vectors

"""
Notes
A single embedding is a list[float]. When you "embed" a piece of text, The model converts its meaning into a long list of numbers (aka, a vector).
For example, the sentence "the cat sat" might become:
`[0.0123, -0.4567, 0.8901, ... ]   # often 768 or 1024 numbers long`
That list of floats is a vector: a point in high-dimensional space that represents the text's meaning. Texts with similar meaning will have vectors that are "close together" in that space, which is how we can search for relevant documents later on using RAG
"""
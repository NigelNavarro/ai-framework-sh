"""
Author: Nigel J. Navarro
OllamaModelClient implementation of the ModelClient contract. Talks to a local Ollama server via REST API using the `requests` library.
"""
from __future__ import annotations
import requests
import json
from framework.config import Config
from framework.models.base import ModelClient, Message
from typing import Iterator


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
    
    def chat_stream(self, messages: list[Message]) -> Iterator[str]:
        """
        Stream the model's reply piece by piece as it is generated.  This is like chat(), but instead
        of waitinf for the full response and returning a string, this `yields` each chunk of text as 
        Ollama produces it. This allows the application to display the response in real time, token by token, instead of waiting for the full answer. The method returns an iterator that produces strings (chunks of the reply) one at a time.
        """
        with requests.post(
            f"{self.host}/api/chat",
            json={
                "model": self.model,
                "messages": messages,
                "stream": True, # This tells Ollama to stream the response as it is generated.
            },
            stream=True, # This tells the `requests` library to receive the response in streaming mode, allowing us to iterate over it as it comes in.       
        ) as response:
            response.raise_for_status()
            # Ollama streams back many JSON lines, one per chunk. Read them as they arrive.
            for line in response.iter_lines():
                if not line:
                    continue # Skip empty lines
                chunk = json.loads(line) # Each line is a JSON object with the same structure as the full response, but with only a piece of the text in message -> content.
                content = chunk.get("message", {}).get("content", "") # Safely extract the content; if the structure is unexpected, default to an empty string.
                if content: # Only yield if there is content in this chunk
                    yield content # hand this piece to the caller.
        


"""
Author: Nigel J. Navarro

ModelClient
This is the interface that the rest of the framework uses to talk to the LLM. 
This is the abstraction that makes the model runtime pluggable. 
The rest of the framework depends on this interface.

To use a different backend later, write a new class that implements the same methods (e.g. `chat`) and swap it in here. Nothing else in the framework needs to change.
"""

from __future__ import annotations
from abc import ABC, abstractmethod

# A chat message is like a dict {"role": "user"|"assistant"|"system", "content": "text of the message"}
# We alias it here so the contract reads clearly, and the shape is defined in one place.
Message = dict[str,str]

class ModelClient(ABC):
    """
    Contract for talking to the LLM. Any concrete adapter (e.g. OllamaModelClient) subclasses this and implements the @abstractmethod below.
    Should you forget any part, Python will raise an error reminding you to implement the missing pieces. Python won't let the class be instantiated until all abstract methods are implemented.
    """

    @abstractmethod
    def chat(self, message: list[Message]) -> str:
        """
        Send a conversation and return the model's reply text.
        
        'message' is a list of {"role", "content"} dicts. This is the common format for chat-based LLM APIs (e.g. OpenAI, Ollama). The "role" indicates who is speaking (user, assistant, system), and "content" is the text of the message.
        Returns just the assistant's reply, as a string.
        """

        ...
    def embed(self, texts: list[str]) -> list[list[float]]:
        """
        Turn texts into embedding vectors (used later for RAG).
        
        This takes a list o strings, returns one vector (a list of floats) per string in the same order.
        This is how documents get sorted and searched by meaning.
        """

        ...
        
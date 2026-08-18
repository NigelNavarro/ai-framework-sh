"""
Author: Nigel J. Navarro
Tool Registry

This space holds the tools the agent can use. Each tool pairs a schema (what the model reads to decide whether and/or how to call it) with a handler, the Python function that runs it. 
The registry lets us a) advertise all tool schemas to the model and b) dispatch a tool call by name to its handler.

Backend-agnostic: No model or Ollama specifics here. 
"""
from __future__ import annotations # Store annotations as strings so modern hints (list[str], etc.) work without runtime cost.
from dataclasses import dataclass
from typing import Any, Callable

@dataclass                          # This decorator automatically generates special methods for the class, like __init__(), __repr__(), and __eq__(), based on the class attributes.
class Tool:                         # Represents a tool with a schema and a handler. The schema describes the tool for the model, and the handler is the Python function that executes the tool.
    name: str                       # This is a unique, single-word, lowercase identifier the model uses to call the tool.
    description: str                # A brief description of what the tool does. The model reads this to decide when to use it.
    parameters: dict[str, Any]      # A dictionary of parameters that the tool accepts in JSON Schema format. Describes the tool's inputs (arg names, required types).
    handler: Callable[..., str]     # The Python function that executes the tool. Returns a string result.

class ToolRegistry:
    """
    A registry for tools that the agent can use. Each tool has a schema and a handler.
    This class manages a collection of tools, allowing for registration and retrieval of tools by name.
    """

    def __init__(self):
        self._tools: dict[str, Tool] = {}  # Initializes an empty dictionary to hold the registered tools.

    def register(self, tool: Tool) -> None:
        """
        Registers a new tool in the registry. 
        Rejects duplicate names so two tools cannot shadow one another
        """
        if tool.name in self._tools:
            raise ValueError(f"Tool with name '{tool.name}' is already registered.")
        self._tools[tool.name] = tool  # Adds the tool to the registry.
    
    def get(self, name:str) -> Tool:
        """
        Look up a tool by name, or raise an error if it does not exist.
        """
        if name not in self._tools:
            raise KeyError(f"No tool named '{name}' is registered.")
        return self._tools[name]  # Returns the tool associated with the given name.

    def schemas(self) -> list[dict]:
        """
        The tool specifications to advertise to the model (name, description, input schema only).
        We deliberately omit the handler since the model never sees or runs our Python code.
        The shape follows the common 'function' tool specification used by other frameworks.
        """
        return [
            {
                "type": "function",
                "function" : {
                    "name": tool.name,
                    "description": tool.description,
                    "parameters": tool.parameters
                },
            }
            for tool in self._tools.values()
        ]
    
    def execute(self, name: str, arguments: dict) -> str:
        """
        Run the named tool's handler with the given arguments. Returns the result as a string.
        The 'arguments' dict is the one the model supplied (ex. {"query": "Frankenstein"})
        """
        tool = self.get(name)  # Retrieves the tool by name.
        return tool.handler(**arguments)  # Calls the tool's handler with the provided arguments and returns the result.
    


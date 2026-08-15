"""
Author: Nigel J. Navarro
Tests: ToolRegistry

Verifies the ToolRegistry contract from framework/tools/registry.py — the name->Tool mapping, the schemas() advertisement, and the execute() dispatch to the handler. 

Covers both the happy paths and the guardrails:

      - register + get      : a registered tool can be fetched back by the same name
      - duplicate register  : a second tool with an existing name raises ValueError
      - get unknown         : fetching an unregistered name raises KeyError
      - schemas()           : advertises name/description/parameters, and NEVER the handler
      - execute()           : looks up a tool by name and calls its handler with the arguments

Hermetic: pure logic, no ModelClient, no Ollama, no disk or network — so it runs in milliseconds and is fully deterministic. 
Uses a trivial dummy Tool whose handler just echoes its input, since these tests care about the registry's behavior, not any real tool.

Run:  .venv/bin/pytest tests/test_registry.py -v
"""
from __future__ import annotations

import pytest

from framework.tools.registry import Tool, ToolRegistry

def make_dummy_tool(name: str = "echo") -> Tool:
    """
    Handler echoes the argument back. This is a simple tool for testing the registry.
    """
    return Tool(
        name=name,
        description="Echoes the text back.",
        parameters={
            "type": "object",
            "properties": {"text": {"type": "string"}},
            "required": ["text"],    
        },
        handler=lambda text: f"echo: {text}",       # The handler is a simple lambda function that returns the text prefixed with "echo: ".
    )

def test_register_and_get_roundtrip():
    """
    A registered tool can be retrieved by name.
    """
    registry = ToolRegistry()
    tool = make_dummy_tool()
    registry.register(tool)
    assert registry.get("echo") is tool             # Check that the retrieved tool is the same object as the one we registered.

def test_duplicate_name_raises():
    """
    Registering two tools with the same name is rejected.
    """
    registry = ToolRegistry()
    registry.register(make_dummy_tool("echo"))
    with pytest.raises(ValueError):
        registry.register(make_dummy_tool("echo"))  # Attempting to register a tool with a duplicate name should raise a ValueError.

def test_get_unknown_raises():
    """
    Fetching a tool that was never registered raises an error.
    """
    registry = ToolRegistry()
    with pytest.raises(KeyError):
        registry.get("does_not_exist")              # Attempting to fetch a tool that was never registered should raise a KeyError.

def test_schemas_shape():
    """
    schemas() advertises name/description/parameters and omits the handler.
    """
    registry = ToolRegistry()
    registry.register(make_dummy_tool("echo"))
    schemas = registry.schemas()
    assert len(schemas) == 1                        # Check that there is exactly one schema in the list.
    fn = schemas[0]["function"]                     # Check that the schema contains the expected keys and values.
    assert fn["name"] == "echo"                     # Check that the name in the schema matches the registered tool's name.
    assert "parameters" in fn                       # Check that the schema includes the parameters key.
    assert "handler" not in fn                      # The handler should not be exposed to the model.

def test_execute_dispatches_to_handler():
    """
    execute() looks up the tool and calls its handler with the given arguments.
    """
    registry = ToolRegistry()
    registry.register(make_dummy_tool("echo"))
    result = registry.execute("echo", {"text": "hello"})
    assert result == "echo: hello"                  # Check that the execute method correctly calls the handler and returns the expected result.

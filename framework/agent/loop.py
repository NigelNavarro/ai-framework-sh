"""
Author: Nigel J. Navarro
Agent Loop

This is the control loop that let's the model decide when to use tools instead of always retrieving.
We describe the available tools to the model, and it will steer along these lines:

  1. Think - The model will read the question + tool list, then answers or asks for a tool
  2. Act - If it calls for a tool, the request will be routed through registry.execute() to the tool's handler.
  3. Observe - The model will be fed the tool's output back into the conversation as a new message.
  4. Repeat - The model will evaluate again now that it has new information. It will then decide whether to call another tool or to answer. There will be a configured hard limit for this roundtrip behavior (via config.max_tool_rounds).

We are specifically making it such that the tool calls are prompt-based and not native to whatever model functionality is available. By abstracting this behavior to be model agnostic, this ensures that the framework works with any instruction following model without having to call special APIs and or determine if the model supports any loop-like features. 
"""
from __future__ import annotations # Store annotations as strings so modern hints (list[str], etc.) work without runtime cost.

import json
from framework.models.base import ModelClient, Message
from framework.tools.registry import ToolRegistry

def build_system_prompt(registry: ToolRegistry) -> str:
    """
    Builds the system prompt that describes the agent's capabilities and the available tools to the model. 
    It tells the model what tools exist and the exact protocol for calling one. We serialize the registry schemas to JSON text and paste them in so the model reads the same tool specs we would send a native API. 
    """
    tools_json = json.dumps(registry.schemas(), indent=2) # Serialize the tool schemas to JSON text with indentation for readability.
    return (
        "You are an agent that can answer questions and use tools to retrieve information. "
        "You have access to the following tools:\n"
        f"{tools_json}\n\n"
        "When you want to use a tool, respond with ONLY a JSON object and nothing else in the following format:\n"
        "{\n"
        '  "tool": "<tool_name>",\n'
        '  "arguments": {\n'
        '    "<arg1>": "<value1>",\n'
        '    "<arg2>": "<value2>"\n'
        "  }\n"
        "}\n\n"
        "You will then receive the tool's result and may call another tool or answer.\n"
        "If you do not need a tool, respond with your answer in plain text, not JSON. "
        "Cite tool result by their [number] labels. If the results don't contain the answer, say so plainly. "
    )

def parse_tool_call(reply: str) -> dict | None:
    """
    Try to read the model's reply as a tool-call JSON object.
    It returns the parsed dict if it looks like a tool call, or None if it doesn't (e.g. plain text answer).
    None is the loop's signal that the mode is done.
    """
    text = reply.strip() # Remove leading/trailing whitespace to avoid parsing issues.

    # Models often wrap JSON in code blocks (```json ... ```). We strip those away to get the raw JSON text.
    if text.startswith("```"):
        text = text.strip("`") # Remove the code block markers completely in the arguement.
        if text.startswith("json"):
            # text = text[len("json"):].strip() # Remove the "json" language hint if present, then strip afterwards to clean up whitespace.
            # We don't use the above because then it makes the .strip() conditional. We want the strip() to run unconditionally.
            text = text[len("json"):]
        text = text.strip()

    try:
        data = json.loads(text) # Attempt to parse the text as JSON.
    except json.JSONDecodeError:
        return None # If parsing fails, return None to indicate it's not a valid tool call. Not JSON, so we assume it's a plain text answer.
    
    if isinstance(data, dict) and "tool" in data: # Check if the parsed data is a dictionary and contains the "tool" key, which indicates it's a tool call.
        return data
    return None # If the parsed data is not a valid tool call, return None.

def run_agent(client: ModelClient, registry: ToolRegistry, question: str, max_rounds: int = 5) -> str: 
    """
    Runs the agent loop. Think -> Act -> Observe. Returns the model's final answer.

    Args:
        client: The model client used to communicate with the language model.
        registry: The tool registry containing available tools.
        question: The initial question to ask the agent.
        max_rounds: The maximum number of interaction rounds with the agent.

    Each round asks the mode, checks if it wants a tool. If not, we are done. If so, execute the tool and feed the observation back.
    Stops after max_rounds is hit.
    """
    messages: list[Message] = [
        # System prompt describes the agent's capabilities and the available tools to the model, saying what tools exist and the exact protocol for calling one. 
        # We serialize the registry schemas to JSON text and paste them in so the model reads the same tool specs we would send a native API.
        {"role": "system", "content": build_system_prompt(registry)},
        {"role": "user", "content": question}, 
        # The first user message is the question we want the agent to answer.
    ]

    for _ in range(max_rounds):             # Loop for a maximum number of rounds to prevent infinite loops.
        reply = client.chat(messages)       # Think -> Get the model's reply based on the current conversation history.
        call = parse_tool_call(reply)  # Try to parse the model's reply as a tool call.

        if call is None:               # If the model's reply is not a tool call, we assume it's the final answer.
            return reply                    # Return the model's final answer.

        # Act -> If we have a valid tool call, extract the tool name and call.
        name = call.get("tool")
        arguments = call.get("arguments", {}) # Default to an empty dict if no arguments are provided.

        try:
            observation = registry.execute(name, arguments) # Execute the requested tool with the provided call.
        except (KeyError, TypeError) as error:
            observation = f"Tool Error: {error}" # Handle any errors during tool execution.

        # Observe -> Append the model's request and the tool's result to the conversation history for context in the next round.
        messages.append({"role": "assistant", "content": reply}) # Add the model's request to the conversation history.
        messages.append({"role": "user", "content": f"Tool result:\n{observation}"}) # Add the tool's observation to the conversation history.

    # Return a default message if max rounds are reached
    return "I couldn't complete the request within the allowed number of tool rounds"
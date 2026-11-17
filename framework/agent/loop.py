"""
Author: Nigel J. Navarro
Agent Loop

This is the control loop that lets the model decide when to use tools instead of always retrieving.
We describe the available tools to the model, and it will steer along these lines:

  1. Think - The model will read the question + tool list, then answers or asks for a tool
  2. Act - If it calls for a tool, the request will be routed through registry.execute() to the tool's handler.
  3. Observe - The model will be fed the tool's output back into the conversation as a new message.
  4. Repeat - The model will evaluate again now that it has new information. It will then decide whether to call another tool or to answer. There will be a configured hard limit for this roundtrip behavior (via config.max_tool_rounds).

We are specifically making it such that the tool calls are prompt-based and not native to whatever model functionality is available. By abstracting this behavior to be model agnostic, this ensures that the framework works with any instruction following model without having to call special APIs and or determine if the model supports any loop-like features. 
"""
from __future__ import annotations                      # Store annotations as strings so modern hints (list[str], etc.) work without runtime cost.

import json
import time
import uuid
from dataclasses import dataclass, field                # Using for ToolCall and AgentRun
from framework.models.base import ModelClient, Message
from framework.tools.registry import ToolRegistry
from framework.observability.trace import Tracer

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
        "\nThe user's questions are usually about their own documents. Before answering any question that the "
        "documents might cover call 'search_knowledge' first, even if you think you already know the answer. "
        "Answer from general knowledge only after a search finds nothing relevant, and say that the answer did "
        "not come from the documents. "
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
        text = text.strip("`")          # Remove the code block markers completely in the argument.
        if text.startswith("json"):
            text = text[len("json"):]   # Remove the "json" language hint if present.
        text = text.strip()             # Strip whitespace unconditionally after removing code block markers and language hint.

    try:
        data = json.loads(text)         # Attempt to parse the text as JSON.
    except json.JSONDecodeError:
        return None                     # If parsing fails, return None to indicate that it is not a valid tool call. 
                                        # Not JSON, so we assume it's a plain text answer.
    
    if isinstance(data, dict) and "tool" in data: # Check if the parsed data is a dictionary and contains the "tool" key, which indicates it's a tool call.
        return data
    return None # If the parsed data is not a valid tool call, return None.

# In cases where the model replies with nothing at all (see the empty-reply check in the loop):
EMPTY_REPLY_NUDGE = (
    "Your last reply was empty. Either call a tool by responding with only the JSON object or answer the question in plain text."
)

@dataclass
class ToolCall:
    """
    One tool call the agent made during a run, and what came back.
    """
    name: str                   # The tool the model asked for (ex. 'search_knowledge')
    arguments: dict             # The arguments the model passed (ex. {"query": "Who is Ahab?"}) 
    observation: str            # The text the tool returned (what the model was shown)
    error: str | None = None    # None on success, otherwise provide the error message.

@dataclass
class AgentRun:
    """
    The record of one agent run: the model's final answer plus every tool call the loop made and what each returned. 
    Provenance labeling reads this record to decide where the answer came from. In other words, the label rests on what actually
    happened in the loop as opposed to whatever the model claims happened in the event window.
    """
    answer: str                                                 # The final answer (or cap-reached fallback message)
    run_id: str                                                 # The same short ID stamped on this run's trace events.
    tool_calls: list[ToolCall] = field(default_factory=list)    # Every tool call, in order.
    rounds_used: int = 0                                        # How many tool rounds ran before the answer.
    capped: bool = False                                        # True when 'max_rounds' was hit without a final answer.

def run_agent(client: ModelClient, registry: ToolRegistry, question: str, max_rounds: int = 5, tracer: Tracer | None = None) -> str:
    """
    Runs the agent loop and returns only the final answer text.
    This is a thin wrapper over run_agent_with_record() for callers who do not need the agent record.
    """
    return run_agent_with_record(client, registry, question, max_rounds, tracer).answer

def run_agent_with_record(client: ModelClient, registry: ToolRegistry, question: str, max_rounds: int = 5, tracer: Tracer | None = None) -> AgentRun: 
    """
    Runs the agent loop. Think -> Act -> Observe. Returns the model's final answer and every tool call made.

    Each round asks the model, then checks if it wants a tool. If not, the reply is the final answer.
    If yes, then we execute the tool, and feed the result back for the next round.
    The loop stops when the model answers, or after the 'max_rounds' limit has been reached.

    Args:
        client: The model client used to communicate with the language model.
        registry: The tool registry containing available tools the model may call.
        question: The initial question to ask the agent.
        max_rounds: The maximum number of interaction rounds with the agent. Prevents runaway loops.
        tracer (optional): If provided, the run is traced for observability. If None, the loop behaves identically with no recordings.
    """
    run_id = uuid.uuid4().hex[:8]       # A short run ID (first 8 hex chars of a UUID) stamped on every event so the run can be filtered and replayed.
    tools_available = [schema["function"]["name"] for schema in registry.schemas()]
    run_start = time.perf_counter()
    tool_calls: list[ToolCall] = []     # Every tool call this run makes, kept for provenance labeling.

    messages: list[Message] = [
        {"role": "system", "content": build_system_prompt(registry)},
        {"role": "user", "content": question}, 
    ]

    if tracer:
        tracer.record({
            "type": "agent_start",
            "command": "agent",
            "run_id": run_id,
            "model" : client.model,
            "question": question,
            "tools_available": tools_available,
        }) 

    for round_num in range(1, max_rounds + 1):  # Loop for each interaction round with the agent.
        reply = client.chat(messages)           # Think -> Get the model's reply based on the current conversation history.
        call = parse_tool_call(reply)           # Try to parse the model's reply as a tool call.

        # An empty reply is neither a tool call nor an answer. Thinking models occasionally spend the whole turn on hidden reasoning and return no visible text. 
        # Instead of returning nothing (ex. ""), we are going to ask again. 
        # This does use up a round, so a model that keeps replying with nothing will still stop at 'max_rounds'.
        if not reply.strip():
            if tracer:
                tracer.record({"type": "empty_reply", "run_id": run_id, "round": round_num})
            messages.append({"role": "user", "content": EMPTY_REPLY_NUDGE})
            continue

        if call is None:               # If the model's reply is not a tool call, we assume it's the final answer.
            if tracer:
                tracer.record({
                    "type": "agent_end",
                    "run_id": run_id,
                    "rounds_used": round_num - 1,
                    "answer_chars": len(reply),
                    "latency_s": round(time.perf_counter() - run_start, 3),
                })
            return AgentRun(answer=reply, run_id=run_id, tool_calls=tool_calls, rounds_used=(round_num - 1))

        # Act -> If we have a valid tool call, extract the tool name and call.
        name = call.get("tool")
        arguments = call.get("arguments", {})   # Default to an empty dict if no arguments are provided.

        tool_start = time.perf_counter()
        error_msg = None                        # None means the tool executed successfully.
        try:
            observation = registry.execute(name, arguments) # Execute the requested tool with the provided call.
        except (KeyError, TypeError) as error:
            error_msg = str(error)                          # Record the error message for tracing.
            observation = f"Tool Error: {error}"            # Handle any errors during tool execution.
        tool_latency = round(time.perf_counter() - tool_start, 3)
        tool_calls.append(ToolCall(name=name, arguments=arguments, observation=observation, error=error_msg))

        if tracer:
            tracer.record({
                "type": "tool_call",
                "run_id": run_id,
                "round": round_num,
                "tool": name,
                "arguments": arguments,
                "observation": observation[:200],           # Record the first 200 characters of the observation.
                "error": error_msg,                         # None on success, message on failure
                "latency_s": tool_latency,
            })

        # Observe -> Append the model's request and the tool's result to the conversation history for context in the next round.
        messages.append({"role": "assistant", "content": reply}) # Add the model's request to the conversation history.
        messages.append({"role": "user", "content": f"Tool result:\n{observation}"}) # Add the tool's observation to the conversation history.

    # If the loop finished all the rounds without the model answering, then invoke the safety trace:
    if tracer:
        tracer.record({
            "type": "cap_reached",
            "run_id": run_id,
            "rounds": max_rounds,
            "latency_s": round(time.perf_counter() - run_start, 3),
        })
    return AgentRun(
        answer="I couldn't complete the request within the allowed number of tool rounds.",
        run_id=run_id, tool_calls=tool_calls, rounds_used=max_rounds, capped=True,
    )
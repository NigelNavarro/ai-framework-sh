"""
Author: Nigel J. Navarro
Chat REPL (Read-Eval-Print Loop) for testing the model runtime. This is a simple CLI (Command-Line Interface) that lets you have a conversation with the model. You can type messages, and the model will reply. This is useful for quick experiments and debugging.

Run: .venv/bin/python apps/cli/main.py
"""
import time
from framework.config import load_config
from framework.models.ollama import OllamaModelClient
from framework.observability.trace import Tracer

# Build the model client once, from our configuration.
config = load_config() 
client = OllamaModelClient(config)
tracer = Tracer(config) # The Tracer wraps the model client to record all interactions for observability.

print(f"You are now chatting with {client.model}. Type 'exit' or 'quit' to stop.\n")

# Conversation History: A list of messages that the model expects.
# Each message is a dict with "role" (e.g. "user", "assistant") and "content" (the text). This is the chat's memory. 
# We resend the whole thing every turn so that the model has context.
messages = []

# Loop: Keep chatting until the user types "exit" or "quit".
while True:
    # Read: Wait for the user to type a line and press Enter.
    user_input = input(">you ")

    # Exit Condition: If the user types "exit" or "quit", break the loop and end the program.
    if user_input.strip().lower() in ("exit", "quit"):
        print("Goodbye!")
        break # exits the while loop and ends the program

    # Record the user's message before sending.
    messages.append({"role": "user", "content": user_input})

    # Evaluate and Print:
    # Send the full conversation history to the model and stream its reply.
    print(f"{client.model}> ", end="", flush=True)
    reply = ""
    start = time.perf_counter() # Start the Stopwatch
    # Stream the model's reply piece by piece as it is generated. This allows us to see the response in real time, token by token, instead of waiting for the full answer. The method returns an iterator that produces strings (chunks of the reply) one at a time.
    for chunk in client.chat_stream(messages):
        print(chunk, end="", flush=True) # Print each chunk of the model's reply as it arrives, without adding a newline, and flush to show it immediately.
        reply += chunk # Accumulate the full reply in case we want to use it later. 
    latency_s = time.perf_counter() - start # Stop the Stopwatch; calculate latency in seconds.
    print("\n")

    # Record what just happened in the Tracer for observability. This includes the model name, the user's message, the length of the model's reply, and the latency of the response.
    tracer.record (
        {
            "model": client.model,
            "user_msg": user_input,
            "reply_chars": len(reply),
            "latency_s": round(latency_s,3,) # Record latency rounded to 3 decimal places for readability.
        }
    )

    # Record the model's fully reply for the next turn's context.
    messages.append({"role": "assistant", "content": reply})
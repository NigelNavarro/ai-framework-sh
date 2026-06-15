"""
Author: Nigel J. Navarro
Chat REPL (Read-Eval-Print Loop) for testing the model runtime. This is a simple CLI (Command-Line Interface) that lets you have a conversation with the model. You can type messages, and the model will reply. This is useful for quick experiments and debugging.

Run: .venv/bin/python apps/cli/main.py
"""
from framework.config import load_config
from framework.models.ollama import OllamaModelClient

# Build the model client once, from our configuration.
client = OllamaModelClient(load_config())

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
    print(f"{client.model}> ", end="", flush=True) # Print the model's name as a prompt, and flush to ensure it appears immediately.
    reply = ""
    for chunk in client.chat_stream(messages):
        print(chunk, end="", flush=True)
        reply += chunk
    print("\n")

    # Record the model's fully reply for the next turn's context.
    messages.append({"role": "assistant", "content": reply})
"""
Author: Nigel J. Navarro
Smoke Test: Streaming Chat (chat_stream)

This is a quick test to verify that the new `chat_stream` method in `OllamaModelClient` works as expected. It sends a simple conversation to the model and prints each chunk of the response as it is received. This allows us to see the response in real time, token by token, instead of waiting for the full answer.

Run: .venv/bin/python smoke_test_streaming.py
"""
import sys

from framework.config import load_config
from framework.models.ollama import OllamaModelClient

client = OllamaModelClient(load_config())

messages = [{'role': 'user', 'content': " Count slowly from 1 to 10, one numnber per line."}]

print(f"Streaming reply from {client.model} ... \n")

# Consume the generator; print each chunnk the instant it arrives.
# flush() forces the output to appear immediately (no buffering), so we can see it streaming in real time.
for chunk in client.chat_stream(messages):
    sys.stdout.write(chunk)
    sys.stdout.flush()

print()
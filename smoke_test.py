"""
Day 1: Smoke Test
The smoke test is a simple script that imports the framework and prints the configuration. Let's see if we can reach the LLM (Ollama) and get a response.

Throwaway connectivity check. This is not a part of the framework.
Uses only the standard libary (no `ollama` package yet). Currently dependency free.

Run: .venv/bin/python smoke_test.py
"""

import json

import urllib
import urllib.request

from framework.config import load_config

# Load our configuration (the same model name and host the real framework will use)
cfg = load_config()

# Build the request Ollama's API chat endpoint expects.
payload = {
    "model": cfg.model,
    "messages": [
        {"role": "user", "content": "In one short sentence, say hello and confirm you're working."},
        {"role": "user", "content": "Then answer: What is the airspeed of a laden swallow?"}
    ],
    "stream": False, # Wait for the full response, don't stream tokens.
    # Trailing comma is fine in Python dicts, and makes it easier to add fields later.
}

url = f"{cfg.ollama_host}/api/chat" # e.g. http://localhost:11434/api/chat
data = json.dumps(payload).encode("utf-8") # JSON text -> bytes for the HTTP body
request = urllib.request.Request(url, data=data, headers={"Content-Type": "application/json"}) # Build the HTTP POST request with the JSON body and content type header

print(f"Asking model {cfg.model} at {cfg.ollama_host} ... \n")

# Send the request and print the response. Close the connection after reading the response to avoid resource leaks.
with urllib.request.urlopen(request) as response:
    result = json.loads(response.read()) # Read the response body and parse it from JSON text into a Python dict

# Ollama nests the text under message -> content. The structure of the response is defined by Ollama's API.
print("Reply:", result["message"]["content"])

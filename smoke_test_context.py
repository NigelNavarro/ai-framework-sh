"""
Author: Nigel J. Navarro
Smoke Test: Context

Verifies that the context assembly works as expected: 
Given a question and a few chunks, it builds the system and user messages that instruct the model to answer only from those chunks and to cite them.

This is a preview of the future "ask" command: full cited-Q&A round-trip (retrieve -> build_messages -> chat).

Run: .venv/bin/python smoke_test_context.py
"""
from framework.config import load_config
from framework.models.ollama import OllamaModelClient
from framework.knowledge.vectorstore.chroma import ChromaVectorStore
from framework.knowledge.retrieve import Retriever
from framework.agent.context import build_messages

cfg = load_config()
model = OllamaModelClient(cfg)
store = ChromaVectorStore(cfg)
retriever = Retriever(model, store, cfg.top_k)

question = "What is the narrator's expedition about?"

# 1. RETRIEVE the most relevant chunks for the question.
chunks = retriever.retrieve(question, top_k=3)

# 2. ASSEMBLE the grounded, cite-instructed prompt.
messages = build_messages(question, chunks)

# 3. ASK the model and print its (hopefully cited) answer.
answer = model.chat(messages)

print(f"Q: {question}\n")
print(f"A: {answer}\n")
print("Retrieved chunks labeled:",
    [f"[{i}] {c.source.split('/')[-1]}:{c.chunk_idx}" for i, c in enumerate(chunks, start=1)])

# Now test "I don't know"
question = "What is the airspeed velocity of an unladen swallow?"

# 1. RETRIEVE the most relevant chunks for the question.
chunks = retriever.retrieve(question, top_k=3)

# 2. ASSEMBLE the grounded, cite-instructed prompt.
messages = build_messages(question, chunks)

# 3. ASK the model and print its (hopefully cited) answer.
answer = model.chat(messages)

print(f"Q: {question}\n")
print(f"A: {answer}\n")
print("Retrieved chunks labeled:",
    [f"[{i}] {c.source.split('/')[-1]}:{c.chunk_idx}" for i, c in enumerate(chunks, start=1)])
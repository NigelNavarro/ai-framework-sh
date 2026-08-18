"""
Author: Nigel J. Navarro
Context Assembly: Building a grounded, cite-instructed prompt from a question and relevant context chunks.

The bridge between the Retrieval and Generation stages of the RAG pipeline. It takes in the user's question
and the chunks retrieved for it. Then, it assembles the message that instruct the model to answer
only from those chunks, and to cite their sources. 
This turns raw retrieval of data into trustworthy and cited Q&A.
"""
from __future__ import annotations

from framework.knowledge.ingest import Chunk
from framework.models.base import Message

# The system prompt sets the model's rules before it sees the question.
# This is what keeps answers grounded (with no hallucination) and cited.
SYSTEM_PROMPT = (
    "You are a careful assistant. Answer the user's question using only the "
    "sources provided in their message. Rules: \n"
    "- Use only the given sources; do not rely on outside knowledge. \n"
    "- If the sources don't contain the answer, say so plainly. \n"
    "- Cite the sources you used by their [number] labels \n"
    "- Be concise and accurate."
)

def format_sources(chunks: list[Chunk]) -> str:
    """
    Turns retrieved chunks into a numbered and labeled block for the prompt.

    Each chunk becomes `[n] (source: filename, chunk i)` followed by its text.
    This is so the model can cite the specific [n] back to the real document.
    """
    blocks = []
    for number, chunk in enumerate(chunks, start=1):
        source = chunk.source.split("/")[-1] # Get the filename from the full path
        blocks.append(f"[{number}] (source: {source}, chunk {chunk.chunk_idx})\n{chunk.text}")
    return "\n\n".join(blocks)

def build_messages(question: str, chunks: list[Chunk]) -> list[Message]:
    """
    Assemble the system and user messages for a grounded, cite-instructed prompt.
    """
    sources = format_sources(chunks) # Format the retrieved chunks into a numbered block for the prompt
    user_content = (
        f"Sources: \n{sources} \n\n"
        f"Question: {question}\n\n"
        "Answer using only the sources above. Cite them by [number]."
    )
    return [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": user_content}
    ]


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

"""
Notes
SYSTEM_PROMPT: This is a system message that sets the model's standing rules before it ever sees the question. 
  Its four (4) rules are designed to keep the model grounded in the sources and to cite them properly. Those are:
  1. "Use only the given sources" -> This provides grounding and stops the model from answering outside of its training memory (aka avoiding hallucinations). Answers will stay within the bounds of your documents.
  2. "If the sources don't contain the answer, say so" -> This is an additional safety net that allows the model to freely admit that "it doesn't know" which is critical because a model with no escape hatch will invent an answer to please the end user.
  3. "Cite the sources you used by their [number]" -> This ensures that the models trustworthiness by having answers that you can trace the sources back to.
  4. "Be concise" -> This ensures that the responses are tight and to the point, avoiding unnecessary verbosity.
format_sources(): This function takes the retrieved chunks and formats them into a numbered block that the model can reference ([1], [2], [3], etc.). Two reasons for this:
  1. It gives the model a short, clean label to cite instead of quoting long filenames.
  2. Because each block also records the chunk index, you will be able to map the citation back to the real originating document. 
build_messages(): This assembles the final two-message list into 
  1. The system rules, and
  2. The user messages that contains the numbered Sources, the Question, and the Reminder to Cite. 
  It returns the the desired shape that client.chat(...) expects, which is a list of messages with the role and content. This allows the caller (the ask command) to simply perform client.chat(build_messages(q, chunks)) without worrying about the details of how to format the prompt.
return [ { ... } ] -> The return value is formatted in this method for a reason:
  1. The outer square brackets (`[]`) indicates that this is a list. This is because we are returning a sequence of messages, and the order matters (the system message must come before the user message).
  2. The inner curly braces (`{}`) indicate that every single message is a dictionary/object/'named data': "role" and "content". Curly braces with `key: value` (KV) pairs are how we represent "this thing has named fields". A message is a labeled record.
  Therefore, the outer container is a ordered sequence of messages (a list), and each message inside is a named recored (a dict). 
  list[Messsage] -> a list of Message
  models/base.py -> Message = dict[str, str]
  Therefore -> list[ dict[str, str] ]
"""
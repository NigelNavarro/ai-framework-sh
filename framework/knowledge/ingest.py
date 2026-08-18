"""
Author: Nigel J. Navarro
Knowledge Ingestion

This module reads documents and splits them into overlapping chunks.

This is the first stage of the RAG pipeline. It walks a folder, reads the supported text files, and splits each into fixed size character chunks with overlap so that they can later be embedded and retreived. Each chunk carries source metadata so answers can cite it.
"""
from __future__ import annotations # Defers evaluation of type annotations: Python stores them as strings instead of evaluating them at definition time. This lets us reference not-yet-defined types (forward references) and use modern syntax like list[str] freely, without runtime errors.

from dataclasses import dataclass
from pathlib import Path

# Identifying which file types we currently support (know how to read)
# TODO: Add PDF support later
SUPPORTED_EXTENSIONS = {".md",".txt"}

# A simple data struct to hold a chunk of text and its source metadata. The dataclass decorator automatically generates an __init__ method and other utility methods.
@dataclass
class Chunk:
    """
    A piece of a document, with enough metadata to trace/cite it later.
    """
    text: str
    source: str     # which file this came from
    chunk_idx: int  # position within that file (e.g. 0, 1, 2, ...)

def chunk_text(text: str, chunk_size: int, chunk_overlap: int) -> list[str]:
    """
    Split the text into overlapping pieces of about `chunk_size` characters.
    Each chunk repeats the last `chunk_overlap` characters of the previous one, so ideas that fall on a boundary sruvive intact in an least one chunk.
    """
    chunks: list[str] = [] # chunks is a list of strings that will hold the resulting text chunks.
    start = 0
    while start < len(text):
        end = start + chunk_size
        chunks.append(text[start:end]) # Take a slice of the text from 'start' to 'end' and add it to the chunks list.
        if end >= len(text):
            break # If we have reached the end of the text, exit the loop.
        start = end - chunk_overlap # step back by the overlap for the next chunk.
    return chunks

def ingest_file(path: Path, chunk_size: int, chunk_overlap: int) -> list[Chunk]:
    """
    Read a single file and split it into overlapping chunks, each tagged with its source.
    """
    text = path.read_text(encoding="utf-8", errors="ignore")
    return [
        Chunk(text=piece, source=str(path), chunk_idx=index)                        # Create a Chunk object for each piece of text, including the source file path and the index of the chunk.
        for index, piece in enumerate(chunk_text(text, chunk_size, chunk_overlap))  # Use enumerate to get both the index and the piece of text from chunk_text, which splits the text into overlapping chunks.
    ]

def ingest_folder(folder: Path, chunk_size: int, chunk_overlap: int) -> list[Chunk]:
    """
    Read every supported file under the `folder`, and return a flat list of chunks.
    """
    chunks: list[Chunk] =[] # Provides one flast list of all chunks from all files within a folder.
    for path in sorted(folder.rglob("*")): # As we recursively walk the folder, we sort the paths to ensure a consistent order (for deterministic ingestion)
        if path.suffix.lower() not in SUPPORTED_EXTENSIONS:
            continue
        chunks.extend(ingest_file(path, chunk_size, chunk_overlap)) # For each supported file, we call `ingest_file` to get its chunks and extend our main list of chunks with the results. We use `extend` instead of `append` to flatten the list of chunks from each file into a single list.
    return chunks


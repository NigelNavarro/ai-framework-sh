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

def ingest_folder(folder: Path, chunk_size: int, chunk_overlap: int) -> list[Chunk]:
    """
    Read every supported file under the `folder`, and return a flat list of chunks.
    """
    chunks: list[Chunk] =[] # chunks is a list that will hold the resulting Chunk objects.
    for path in sorted(folder.rglob("*")): # As we recursively walk the folder, we sort the paths to ensure a consistent order (for deterministic ingestion)
        if path.suffix.lower() not in SUPPORTED_EXTENSIONS:
            continue # if the file extension is not in our supported list, skip it.
        text = path.read_text(encoding="utf-8", errors="ignore") # The `text` is the content of the file read as a string. We specify UTF-8 encoding and ignore errors to handle a wide range of text files without crashing on encoding issues.
        for index, piece in enumerate(chunk_text(text, chunk_size, chunk_overlap)):
            chunks.append(Chunk(text=piece, source=str(path), chunk_idx=index)) # For each chunk piece we get from `chunk_text`, we create a `Chunk` object with the text, the source file path (converted to string), and the chunk index, then add it to our list of chunks. 
    return chunks

"""
Notes
The `ingest_folder` function is the main entry point. It walks through all files in the given folder (and subfolders) using `rglob("*")`, checks if they have a supported extension, reads their content, and then splits that content into chunks using the `chunk_text` function. Each chunk is wrapped in a `Chunk` dataclass instance that includes the text and metadata about its source.
"""
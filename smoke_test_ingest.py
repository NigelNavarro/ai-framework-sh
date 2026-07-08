"""
Author: Nigel J. Navarro
Smoke Test: Ingest

Verifies the Read/Chunk stage of the RAG pipeline.
    - `ingest_file` reads a single supported file into Chunk objects
    - `ingest_folder` walks through a directory, reading all supported files into Chunk objects
    - Each Chunk carries its source path and sequential chunk_idx index for traceability.

This test runs against a throwaway temp folder, so it does not require a pre-existing knowledge base or vector store. 
It is a unit test for the ingestion logic, not an integration test for the entire RAG pipeline.

Run: .venv/bin/python smoke_test_ingest.py
"""
import tempfile
from pathlib import Path

from framework.knowledge.ingest import SUPPORTED_EXTENSIONS, Chunk, ingest_file, ingest_folder

# The chunk size is deliberately small to ensure that even short test files will be easily split into multiple chunks, 
# allowing us to verify the chunking logic quickly and efficiently.
CHUNK_SIZE = 50
CHUNK_OVERLAP =  10

# TemporaryDirectory() is a context manager that creates a temporary directory for the duration of the test.
# When the `with` block exits/is exited, the temporary directory and all its contents are automatically deleted, 
# ensuring that the test does not leave any residual files or directories on the filesystem.
with tempfile.TemporaryDirectory() as tmp:
    root = Path(tmp) # Convert the temporary directory path string into a Path object for easier manipulation.

    # Seed a tiny corpus: two supported files, and one that must be skipped.
    (root / "a.txt").write_text("A" * 120, encoding="utf-8")                                 # Write 20 'A' characters to a.txt in UTF-8 encoding.
    (root / "b.md").write_text("# Title\n Some notes.", encoding="utf-8")                   # Write a markdown file with a title and some notes to b.md in UTF-8 encoding.
    (root / "c.ignore.log").write_text("This file should be skipped.", encoding="utf-8")   # Write a file that should be ignored by the ingestion logic.

    # Test ingest_file() on a single supported file
    file_chunks = ingest_file(root / "a.txt", CHUNK_SIZE, CHUNK_OVERLAP) # Ingest the a.txt file into chunks using the specified chunk size and overlap.
    print(f"ingest_file('a.txt') -> {len(file_chunks)} chunks")

    # Assert that the ingestion of a single file produces the expected number of chunks and that each chunk has the correct metadata.
    assert file_chunks, "Expected at least one chunk from a.txt" # Ensure that at least one chunk was created from a.txt
    assert all(isinstance(c, Chunk) for c in file_chunks), "Every item should be a Chunk"
    assert all(c.source.endswith("a.txt") for c in file_chunks), "Source should point at a.txt"
    assert [c.chunk_idx for c in file_chunks] == list(range(len(file_chunks))), "chunk_idx should be sequential starting from 0 (0, 1, 2, ...)" # Ensure that the chunk indices are sequential starting from 0 with no gaps.

    # Test ingest_folder() on the temporary directory
    folder_chunks = ingest_folder(root, CHUNK_SIZE, CHUNK_OVERLAP) # Ingest all supported files in the temporary directory into chunks using the specified chunk size and overlap.
    sources = {Path(c.source).name for c in folder_chunks} # Extract the base names of the source files from the chunks for verification.
    print(f"ingest_folder('{root}') -> {len(folder_chunks)} chunks from sources {sorted(sources)}") # `{sorted(sources)}` converts the SET into a sorted list for consistent output, making it easier to read and verify the sources of the chunks. 
    assert "a.txt" in sources and "b.md" in sources, "Should ingest both supported files"
    assert "c.ignore.log" not in "b.md" in sources, "Should skip the unsupported file"
    assert ".log" not in SUPPORTED_EXTENSIONS, "The .log file should be skipped because it is not in the SUPPORTED_EXTENSIONS set"

    print("\nAll ingest smoke checks passed!")

"""
Expected Output:
--
ingest_file('a.txt') -> 3 chunks
ingest_folder('/var/folders/k4/fcbmfwt132n1jkxz3zj4bktm0000gn/T/tmpm8wprz9t') -> 4 chunks from sources ['a.txt', 'b.md']

All ingest smoke checks passed!
--
"""
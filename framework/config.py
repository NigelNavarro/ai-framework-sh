"""
Author: Nigel J. Navarro
Config for the framework.

Edit June 11, 2026 -- 04:50 p.m.
Only adding one typed object and sensible defaults, optionally overridden by `config.toml` in the project root. 
Everything tunable lives here so the rest of the framework reads settings from one place. Notably, the model name is configurable, making the model runtime flexible.
"""
from __future__ import annotations
import tomllib
from dataclasses import dataclass, fields
from pathlib import Path

# Project Root
# Setting to the folder above this files package, (…/ai-framework-sh/)
# framework/config.py  ->  framework/  ->  ROOT
ROOT = Path(__file__).resolve().parent.parent

@dataclass
class Config:
    """All framework settings, with defaults"""

    # --- Model runtime ---
    model: str = "gemma4:latest"                # Default reasoning model, named by `ollama list`
    embed_model: str = "nomic-embed-text"       # Embeddings for RAG
    ollama_host: str = "http://localhost:11434" # Ollama server URL (Local listening)

    # --- Knowledge/RAG ---
    chunk_size: int = 1000                      # characters per chunk when splitting documents for RAG
    chunk_overlap: int = 200                    # characters shared betweeen chunks to preserve context
    top_k: int = 5                              # how many relevant chunks to retrieve for RAG per question
    relevance_threshold: float = 0.60           # cosine similarity floor. Chunks scoring below it count as "not relevant". 
                                                # MODEL SPECIFIC. Re-calibrate when embed_model changes.
                                                # Current: 0.60 calibrated for nomic-embed-text


    # -- Agent loop ---
    max_tool_rounds: int = 5                    # hard cap on tool-call rounds (preventing runaway loops)

    # -- Storage paths ---
    data_dir: Path = ROOT / "data"                      # where to store all data (knowledge, memory, etc.). ChromaDB, traces, memory, drafts, etc.
    knowledge_base_dir: Path = ROOT / "knowledge_base"  # where to store source documents for RAG (PDFs, text files, etc.)

    def ensure_dirs(self):
        """
        Creates the data/knowledge directories if they don't exist.
        Called explicitly by the app at startup (not at import time) so that merely importing the framework module doesn't have side effects like creating folders."""
        self.data_dir.mkdir(parents=True, exist_ok=True)
        self.knowledge_base_dir.mkdir(parents=True, exist_ok=True)

def load_config(path: str | Path | None = None) -> Config:
    """
    Build a Config from defaults, then overlay the values from a TOML file if present.

    Look up order:
      1. Start with the dataclass defaults above.
      2. If `config.toml` exists (or the path we pass), read it and override any matching keys. Unknown keys are ignored.
    
    Returns are a ready-to-use Config. It never raises if the file is missing. Defaults are always a valid configuraiton.
    """
    cfg = Config()                                          # Assign defaults from the dataclass

    toml_path = Path(path) if path else ROOT / "config.toml"
    if not toml_path.exists():
        return cfg                                          # No file, just return defaults
    
    with open(toml_path, "rb") as f:
        data = tomllib.load(f)                              # Load the TOML file into a dict; requires binary mode

    valid_keys = {field.name for field in fields(Config)}   # Get valid config keys from the dataclass fields
    for key, value in data.items():
        if key not in valid_keys:
            continue                                        # Silently skip settings we do not recognize. If the field is a Path, convert the string from TOML into a Path.
        if isinstance(getattr(cfg, key), Path):             # If the default value for this key is a Path, we expect the user to provide a string path in the TOML, which we convert to a Path object.
            value = Path(value) 
        setattr(cfg, key, value)                            # Override the default with the value from the TOML file
    return cfg
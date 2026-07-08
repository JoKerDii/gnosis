"""Central configuration loaded from environment / .env file."""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

from dotenv import load_dotenv

# Load a local .env if present; real environment variables take precedence.
load_dotenv()


@dataclass(frozen=True)
class Settings:
    """Runtime settings resolved from environment variables."""

    knowledge_dir: Path
    lancedb_path: Path
    lancedb_table: str
    embedding_model: str
    chunk_tokens: int
    chunk_overlap: int
    top_k: int
    server_host: str
    server_port: int
    docs_dir: Path


def get_settings() -> Settings:
    """Build a Settings object from the current environment."""
    return Settings(
        knowledge_dir=Path(os.getenv("KNOWLEDGE_DIR", "./notes")).expanduser(),
        lancedb_path=Path(os.getenv("LANCEDB_PATH", "./.lancedb")).expanduser(),
        lancedb_table=os.getenv("LANCEDB_TABLE", "notes"),
        embedding_model=os.getenv("EMBEDDING_MODEL", "all-MiniLM-L6-v2"),
        chunk_tokens=int(os.getenv("CHUNK_TOKENS", "256")),
        chunk_overlap=int(os.getenv("CHUNK_OVERLAP", "50")),
        top_k=int(os.getenv("TOP_K", "5")),
        server_host=os.getenv("SERVER_HOST", "127.0.0.1"),
        server_port=int(os.getenv("SERVER_PORT", "8000")),
        docs_dir=Path(os.getenv("DOCS_DIR", "./docs")).expanduser(),
    )

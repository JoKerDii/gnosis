"""Static keyword-search index generator for the standalone (GitHub Pages) mode.

Produces ``docs/search_index.json`` — a self-contained document set the frontend
loads into MiniSearch when no local vector server is reachable. This reuses the
same Markdown parsing + chunking as the vector pipeline, but performs **no
embedding** (uses the default whitespace token counter), so it is fast and has
no torch / model dependency.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List

from .parser import find_markdown_files, iter_chunks

# Schema version for the emitted JSON (lets the frontend adapt if it changes).
SCHEMA_VERSION = 1


def build_documents(root: Path, max_tokens: int = 256, overlap: int = 50) -> List[Dict[str, Any]]:
    """Parse every Markdown file under ``root`` into MiniSearch documents.

    Each chunk becomes one document with a stable ``id`` of ``"<doc_id>#<n>"``.
    """
    documents: List[Dict[str, Any]] = []
    for chunk in iter_chunks(root, max_tokens=max_tokens, overlap=overlap):
        documents.append(
            {
                "id": f"{chunk.doc_id}#{chunk.chunk_index}",
                "title": chunk.title,
                "doc_id": chunk.doc_id,
                "path": chunk.doc_id,
                "chunk_index": chunk.chunk_index,
                "tags": list(chunk.tags),
                "date": chunk.date or "",
                "text": chunk.text,
            }
        )
    return documents


def build_search_index(
    root: Path,
    out_path: Path,
    max_tokens: int = 256,
    overlap: int = 50,
) -> Dict[str, int]:
    """Build the static index and write it to ``out_path`` as JSON.

    Returns a small stats dict: ``{"files": N, "documents": M}``. Raises
    :class:`FileNotFoundError` if ``root`` does not exist.
    """
    root = Path(root)
    out_path = Path(out_path)

    files = find_markdown_files(root)  # raises FileNotFoundError if missing
    documents = build_documents(root, max_tokens=max_tokens, overlap=overlap)

    payload = {
        "schema_version": SCHEMA_VERSION,
        "model": "static-keyword",
        "count": len(documents),
        "documents": documents,
    }

    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")

    return {"files": len(files), "documents": len(documents)}

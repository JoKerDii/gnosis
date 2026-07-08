"""Tests for the static search-index builder (src.static_index)."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from src.static_index import build_documents, build_search_index


def write(path: Path, text: str) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")
    return path


def test_build_documents_fields(tmp_path: Path):
    write(
        tmp_path / "a.md",
        "---\ntitle: Alpha\ntags: [x, y]\ndate: 2024-01-15\n---\nHello world content.",
    )
    docs = build_documents(tmp_path)
    assert len(docs) == 1
    d = docs[0]
    assert d["id"] == "a.md#0"
    assert d["title"] == "Alpha"
    assert d["doc_id"] == "a.md"
    assert d["path"] == "a.md"
    assert d["chunk_index"] == 0
    assert d["tags"] == ["x", "y"]
    assert d["date"] == "2024-01-15"
    assert "Hello world" in d["text"]


def test_build_search_index_writes_valid_json(tmp_path: Path):
    notes = tmp_path / "notes"
    write(notes / "a.md", "---\ntitle: A\n---\nAlpha body text.")
    write(notes / "sub" / "b.md", "---\ntitle: B\n---\nBeta body text.")
    out = tmp_path / "docs" / "search_index.json"

    stats = build_search_index(notes, out)
    assert stats["files"] == 2
    assert stats["documents"] == 2
    assert out.exists()

    payload = json.loads(out.read_text(encoding="utf-8"))
    assert payload["schema_version"] == 1
    assert payload["model"] == "static-keyword"
    assert payload["count"] == 2
    assert isinstance(payload["documents"], list)
    ids = {d["id"] for d in payload["documents"]}
    assert ids == {"a.md#0", str(Path("sub") / "b.md") + "#0"}


def test_build_search_index_creates_parent_dir(tmp_path: Path):
    notes = tmp_path / "notes"
    write(notes / "a.md", "content")
    out = tmp_path / "deeply" / "nested" / "search_index.json"
    build_search_index(notes, out)
    assert out.exists()


def test_missing_directory_raises(tmp_path: Path):
    with pytest.raises(FileNotFoundError):
        build_search_index(tmp_path / "nope", tmp_path / "out.json")


def test_empty_directory_writes_empty_document_set(tmp_path: Path):
    notes = tmp_path / "notes"
    notes.mkdir()
    out = tmp_path / "search_index.json"
    stats = build_search_index(notes, out)
    assert stats["documents"] == 0
    payload = json.loads(out.read_text(encoding="utf-8"))
    assert payload["documents"] == []

"""Tests for the FastAPI backend (src.server).

These use a fake VectorStore injected via ``app.dependency_overrides`` so they
run fast and never load torch / sentence-transformers / LanceDB.
"""

from __future__ import annotations

from typing import Any, Dict, List

import pytest
from fastapi.testclient import TestClient

from src.server import create_app, get_store


class FakeStore:
    """Minimal stand-in for VectorStore used in server tests."""

    def __init__(self, rows: List[Dict[str, Any]], count: int = 0, raise_on_search: bool = False):
        self._rows = rows
        self._count = count
        self._raise = raise_on_search
        self.last_query = None
        self.last_top_k = None

    def count(self) -> int:
        return self._count

    def search(self, query: str, top_k: int = 5) -> List[Dict[str, Any]]:
        self.last_query = query
        self.last_top_k = top_k
        if self._raise:
            raise RuntimeError("boom")
        return self._rows[:top_k]


def _row(**kw) -> Dict[str, Any]:
    base = {
        "vector": [0.1, 0.2, 0.3],  # must be stripped from the response
        "text": "Vector databases store embeddings.",
        "doc_id": "notes/a.md",
        "chunk_index": 0,
        "title": "Alpha",
        "source": "/abs/notes/a.md",
        "tags": ["demo"],
        "date": "2024-01-01",
        "_distance": 0.2,
    }
    base.update(kw)
    return base


def make_client(store: FakeStore) -> TestClient:
    app = create_app()
    app.dependency_overrides[get_store] = lambda: store
    return TestClient(app)


# --------------------------------------------------------------------------- #
# /status
# --------------------------------------------------------------------------- #
def test_status_shape():
    client = make_client(FakeStore(rows=[], count=42))
    res = client.get("/status")
    assert res.status_code == 200
    body = res.json()
    assert body["status"] == "online"
    assert body["mode"] == "vector"
    assert body["indexed_chunks"] == 42


def test_status_survives_store_error():
    class BadCount(FakeStore):
        def count(self):
            raise RuntimeError("db down")

    client = make_client(BadCount(rows=[]))
    res = client.get("/status")
    assert res.status_code == 200
    assert res.json()["status"] == "online"
    assert res.json()["indexed_chunks"] is None


# --------------------------------------------------------------------------- #
# /search
# --------------------------------------------------------------------------- #
def test_search_returns_hits_with_score_and_no_vector():
    client = make_client(FakeStore(rows=[_row()]))
    res = client.post("/search", json={"query": "how are embeddings stored"})
    assert res.status_code == 200
    body = res.json()
    assert body["mode"] == "vector"
    assert body["count"] == 1
    hit = body["results"][0]
    assert hit["title"] == "Alpha"
    assert hit["doc_id"] == "notes/a.md"
    assert hit["tags"] == ["demo"]
    # score = 1 - distance(0.2) = 0.8
    assert hit["score"] == pytest.approx(0.8)
    # The raw embedding vector must never be leaked to clients.
    assert "vector" not in hit


def test_search_empty_query_is_400():
    client = make_client(FakeStore(rows=[_row()]))
    # Whitespace-only and empty strings both fail our non-empty check with 400.
    assert client.post("/search", json={"query": "   "}).status_code == 400
    assert client.post("/search", json={"query": ""}).status_code == 400


def test_search_missing_query_field_is_422():
    client = make_client(FakeStore(rows=[_row()]))
    # Absent required field -> Pydantic validation error.
    assert client.post("/search", json={}).status_code == 422


def test_search_respects_top_k():
    rows = [_row(doc_id=f"n{i}.md", chunk_index=i) for i in range(10)]
    store = FakeStore(rows=rows)
    client = make_client(store)
    res = client.post("/search", json={"query": "x", "top_k": 3})
    assert res.status_code == 200
    assert res.json()["count"] == 3
    assert store.last_top_k == 3


def test_search_empty_index_returns_empty_list():
    client = make_client(FakeStore(rows=[]))
    res = client.post("/search", json={"query": "anything"})
    assert res.status_code == 200
    assert res.json()["results"] == []
    assert res.json()["count"] == 0


def test_search_backend_error_is_500():
    client = make_client(FakeStore(rows=[], raise_on_search=True))
    res = client.post("/search", json={"query": "boom"})
    assert res.status_code == 500
    assert "search failed" in res.json()["detail"]


def test_search_missing_distance_yields_null_score():
    row = _row()
    del row["_distance"]
    client = make_client(FakeStore(rows=[row]))
    res = client.post("/search", json={"query": "x"})
    assert res.json()["results"][0]["score"] is None


# --------------------------------------------------------------------------- #
# CORS
# --------------------------------------------------------------------------- #
def test_cors_header_present_on_actual_request():
    client = make_client(FakeStore(rows=[], count=1))
    res = client.get("/status", headers={"Origin": "https://example.github.io"})
    assert res.status_code == 200
    assert res.headers.get("access-control-allow-origin") == "*"


def test_cors_preflight_allowed():
    client = make_client(FakeStore(rows=[]))
    res = client.options(
        "/search",
        headers={
            "Origin": "https://example.github.io",
            "Access-Control-Request-Method": "POST",
            "Access-Control-Request-Headers": "content-type",
        },
    )
    assert res.status_code in (200, 204)
    assert res.headers.get("access-control-allow-origin") == "*"

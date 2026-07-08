"""FastAPI backend for the hybrid Markdown search agent.

This is the "Local Vector Mode" backend: it embeds a natural-language query with
the local sentence-transformers model, searches LanceDB, and returns ranked
snippets as JSON. The browser frontend (``docs/``) probes ``/status`` on load and
uses ``/search`` when this server is reachable; otherwise it falls back to a
static client-side keyword index.

Run it with::

    python -m src.cli serve
    # or: uvicorn src.server:app --reload
"""

from __future__ import annotations

from functools import lru_cache
from typing import Any, Dict, List, Optional

from fastapi import Depends, FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

from .config import Settings, get_settings

# --------------------------------------------------------------------------- #
# Response / request models
# --------------------------------------------------------------------------- #
class StatusResponse(BaseModel):
    status: str = "online"
    mode: str = "vector"
    indexed_chunks: Optional[int] = None
    model: Optional[str] = None


class SearchRequest(BaseModel):
    query: str = Field(..., description="Natural-language search query.")
    top_k: Optional[int] = Field(
        None, ge=1, le=50, description="Number of results to return."
    )


class SearchHit(BaseModel):
    title: str
    doc_id: str
    chunk_index: int
    text: str
    tags: List[str] = []
    date: Optional[str] = None
    source: Optional[str] = None
    score: Optional[float] = None


class SearchResponse(BaseModel):
    query: str
    mode: str = "vector"
    count: int
    results: List[SearchHit]


# --------------------------------------------------------------------------- #
# Store dependency (overridable in tests via app.dependency_overrides)
# --------------------------------------------------------------------------- #
@lru_cache(maxsize=1)
def _cached_store():
    """Build one VectorStore for the process. Heavy imports happen lazily."""
    from .db import VectorStore

    s = get_settings()
    return VectorStore(s.lancedb_path, s.lancedb_table, s.embedding_model)


def get_store():
    """FastAPI dependency returning the shared VectorStore.

    Tests override this with ``app.dependency_overrides[get_store]`` so they can
    inject a fake and avoid loading torch / LanceDB.
    """
    return _cached_store()


# --------------------------------------------------------------------------- #
# Row -> hit mapping
# --------------------------------------------------------------------------- #
def _score_from_distance(distance: Optional[float]) -> Optional[float]:
    """Convert a cosine distance into a 0..1 similarity score."""
    if distance is None:
        return None
    return max(0.0, 1.0 - float(distance))


def _hit_from_row(row: Dict[str, Any]) -> SearchHit:
    """Map a raw LanceDB row into a JSON-safe SearchHit (drops the vector)."""
    return SearchHit(
        title=str(row.get("title") or "(untitled)"),
        doc_id=str(row.get("doc_id") or row.get("source") or "?"),
        chunk_index=int(row.get("chunk_index") or 0),
        text=str(row.get("text") or ""),
        tags=list(row.get("tags") or []),
        date=(str(row.get("date")).strip() or None) if row.get("date") else None,
        source=str(row.get("source")) if row.get("source") else None,
        score=_score_from_distance(row.get("_distance")),
    )


# --------------------------------------------------------------------------- #
# App factory
# --------------------------------------------------------------------------- #
def create_app(settings: Optional[Settings] = None) -> FastAPI:
    settings = settings or get_settings()
    app = FastAPI(
        title="Markdown Knowledge Search",
        description="Local vector-search backend for the hybrid Markdown agent.",
        version="0.1.0",
    )

    # Allow the frontend to call us from any origin (incl. GitHub Pages).
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=False,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    @app.get("/status", response_model=StatusResponse)
    def status(store=Depends(get_store)) -> StatusResponse:
        """Health check used by the frontend to detect Local Vector Mode."""
        try:
            n = store.count()
        except Exception:  # noqa: BLE001 - status must never hard-fail
            n = None
        return StatusResponse(
            status="online",
            mode="vector",
            indexed_chunks=n,
            model=settings.embedding_model,
        )

    @app.post("/search", response_model=SearchResponse)
    def search(req: SearchRequest, store=Depends(get_store)) -> SearchResponse:
        """Embed the query, search LanceDB, and return ranked snippets."""
        query = (req.query or "").strip()
        if not query:
            raise HTTPException(status_code=400, detail="query must not be empty")

        top_k = req.top_k or settings.top_k
        try:
            rows = store.search(query, top_k=top_k)
        except Exception as exc:  # noqa: BLE001 - surface as a clean 500
            raise HTTPException(status_code=500, detail=f"search failed: {exc}")

        hits = [_hit_from_row(r) for r in rows]
        return SearchResponse(query=query, mode="vector", count=len(hits), results=hits)

    # Optionally serve the static frontend so `serve` can host the UI locally
    # at http://<host>:<port>/app/ (same-origin). API routes above win because
    # they are registered before this catch-all mount.
    docs_dir = settings.docs_dir
    if docs_dir.exists() and docs_dir.is_dir():
        from fastapi.staticfiles import StaticFiles

        app.mount("/app", StaticFiles(directory=str(docs_dir), html=True), name="app")

    return app


# Module-level app for `uvicorn src.server:app`.
app = create_app()

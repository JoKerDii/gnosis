#!/usr/bin/env python3
"""Quick verification that LanceDB and the embedding model load correctly.

Run after installing dependencies:

    python verify_setup.py

It performs a tiny end-to-end round trip (embed -> store -> search) in a
temporary directory so nothing touches your real index.
"""

from __future__ import annotations

import sys
import tempfile
from pathlib import Path


def check_imports() -> bool:
    print("1. Checking imports ...")
    try:
        import lancedb  # noqa: F401
        import pyarrow  # noqa: F401
        import frontmatter  # noqa: F401
        import typer  # noqa: F401
        from sentence_transformers import SentenceTransformer  # noqa: F401

        print("   ✓ All core packages import cleanly.")
        return True
    except Exception as exc:  # pragma: no cover - diagnostic path
        print(f"   ✗ Import failed: {exc}")
        return False


def check_embedding_model(model_name: str = "all-MiniLM-L6-v2"):
    print(f"2. Loading embedding model '{model_name}' (may download on first run) ...")
    from sentence_transformers import SentenceTransformer

    model = SentenceTransformer(model_name)
    dim = model.get_sentence_embedding_dimension()
    vec = model.encode(["hello world"], normalize_embeddings=True)[0]
    print(f"   ✓ Model loaded. Embedding dimension = {dim}, sample len = {len(vec)}.")
    return model, dim


def check_lancedb(model, dim: int) -> bool:
    print("3. Testing LanceDB round trip (embed -> store -> search) ...")
    import lancedb
    import pyarrow as pa

    with tempfile.TemporaryDirectory() as tmp:
        db = lancedb.connect(tmp)
        schema = pa.schema(
            [
                pa.field("vector", pa.list_(pa.float32(), dim)),
                pa.field("text", pa.string()),
            ]
        )
        table = db.create_table("verify", schema=schema)

        docs = [
            "LanceDB is an embedded vector database.",
            "Sentence transformers create text embeddings.",
            "Markdown files store notes and metadata.",
        ]
        vectors = model.encode(docs, normalize_embeddings=True).tolist()
        table.add([{"vector": v, "text": t} for v, t in zip(vectors, docs)])

        q = model.encode(["what is a vector database?"], normalize_embeddings=True)[0]
        hits = table.search(q).metric("cosine").limit(1).to_list()

        top = hits[0]["text"] if hits else "(none)"
        print(f"   ✓ Stored {table.count_rows()} rows. Top match: {top!r}")
        return bool(hits)


def main() -> int:
    print("=== Verifying Local Markdown Knowledge Graph setup ===\n")
    if not check_imports():
        return 1
    try:
        model, dim = check_embedding_model()
        ok = check_lancedb(model, dim)
    except Exception as exc:  # pragma: no cover - diagnostic path
        print(f"\n✗ Verification failed: {exc}")
        return 1

    if ok:
        print("\n✅ All checks passed. Environment is ready.")
        return 0
    print("\n✗ Search returned no results.")
    return 1


if __name__ == "__main__":
    sys.exit(main())

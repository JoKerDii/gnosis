"""Vector storage backed by LanceDB + a local Sentence-Transformers model."""

from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Sequence

import lancedb
import pyarrow as pa

from .parser import (
    Chunk,
    ParsedDocument,
    chunk_document,
    find_markdown_files,
    parse_file,
)

# Treat mtimes within this many seconds as unchanged (filesystem granularity).
_MTIME_EPSILON = 1e-4


@dataclass
class IndexStats:
    """Summary of a single incremental sync run."""

    scanned: int = 0        # markdown files seen on disk
    added: int = 0          # new documents indexed
    updated: int = 0        # changed documents re-indexed
    skipped: int = 0        # unchanged documents left as-is
    removed: int = 0        # documents deleted because the file is gone
    chunks: int = 0         # chunks written this run
    errors: int = 0         # files that failed to parse/index

    def as_dict(self) -> Dict[str, int]:
        return {
            "scanned": self.scanned,
            "added": self.added,
            "updated": self.updated,
            "skipped": self.skipped,
            "removed": self.removed,
            "chunks": self.chunks,
            "errors": self.errors,
        }


@lru_cache(maxsize=4)
def _load_model(model_name: str):
    """Load (and cache) a SentenceTransformer model.

    Imported lazily so lightweight commands don't pay the torch/transformers
    import cost until embeddings are actually needed.
    """
    from sentence_transformers import SentenceTransformer

    return SentenceTransformer(model_name)


def make_token_counter(model_name: str) -> Callable[[str], int]:
    """Return a token counter that uses the embedding model's own tokenizer.

    This makes chunk sizes align with what the model actually consumes, rather
    than a whitespace approximation.
    """
    model = _load_model(model_name)
    tokenizer = model.tokenizer

    def count(text: str) -> int:
        return len(tokenizer.encode(text, add_special_tokens=False))

    return count


def model_max_tokens(model_name: str) -> int:
    """Maximum input sequence length (in tokens) the model will embed."""
    return int(_load_model(model_name).get_max_seq_length())


def embed(texts: Sequence[str], model_name: str) -> List[List[float]]:
    """Embed a batch of texts into normalized vectors."""
    model = _load_model(model_name)
    vectors = model.encode(
        list(texts),
        convert_to_numpy=True,
        show_progress_bar=False,
        normalize_embeddings=True,
    )
    return vectors.tolist()


def embedding_dim(model_name: str) -> int:
    """Return the output dimensionality of the embedding model."""
    return int(_load_model(model_name).get_sentence_embedding_dimension())


def _schema(dim: int) -> pa.Schema:
    """Arrow schema for the notes table."""
    return pa.schema(
        [
            pa.field("vector", pa.list_(pa.float32(), dim)),
            pa.field("text", pa.string()),
            pa.field("doc_id", pa.string()),
            pa.field("chunk_index", pa.int64()),
            pa.field("title", pa.string()),
            pa.field("source", pa.string()),
            pa.field("tags", pa.list_(pa.string())),
            pa.field("date", pa.string()),
            pa.field("mtime", pa.float64()),
        ]
    )


def _sql_quote(value: str) -> str:
    """Escape a string for use inside a LanceDB SQL predicate literal."""
    return "'" + value.replace("'", "''") + "'"


class VectorStore:
    """Thin wrapper over a LanceDB table for note chunks."""

    def __init__(self, db_path: Path, table_name: str, model_name: str):
        self.db_path = Path(db_path)
        self.table_name = table_name
        self.model_name = model_name
        self.db_path.mkdir(parents=True, exist_ok=True)
        self._db = lancedb.connect(str(self.db_path))

    # -- table lifecycle ---------------------------------------------------- #
    def _has_table(self) -> bool:
        return self.table_name in self._db.table_names()

    def _open_or_create(self) -> Any:
        if self._has_table():
            return self._db.open_table(self.table_name)
        dim = embedding_dim(self.model_name)
        return self._db.create_table(self.table_name, schema=_schema(dim))

    def reset(self) -> None:
        """Drop the table so it can be rebuilt from scratch."""
        if self._has_table():
            self._db.drop_table(self.table_name)

    def count(self) -> int:
        """Number of chunks currently stored."""
        if not self._has_table():
            return 0
        return self._db.open_table(self.table_name).count_rows()

    # -- writes ------------------------------------------------------------- #
    def add_chunks(self, chunks: Sequence[Chunk], batch_size: int = 64) -> int:
        """Embed and insert chunks. Returns the number added."""
        if not chunks:
            return 0
        table = self._open_or_create()
        added = 0
        for start in range(0, len(chunks), batch_size):
            batch = chunks[start : start + batch_size]
            vectors = embed([c.text for c in batch], self.model_name)
            rows: List[Dict[str, Any]] = [
                {
                    "vector": vec,
                    "text": c.text,
                    "doc_id": c.doc_id,
                    "chunk_index": c.chunk_index,
                    "title": c.title,
                    "source": c.source,
                    "tags": c.tags,
                    "date": c.date or "",
                    "mtime": c.mtime,
                }
                for c, vec in zip(batch, vectors)
            ]
            table.add(rows)
            added += len(rows)
        return added

    def delete_doc(self, doc_id: str) -> None:
        """Remove all chunks belonging to a single document."""
        if not self._has_table():
            return
        table = self._db.open_table(self.table_name)
        table.delete(f"doc_id = {_sql_quote(doc_id)}")

    # -- reads -------------------------------------------------------------- #
    def indexed_mtimes(self) -> Dict[str, float]:
        """Map each indexed ``doc_id`` to the mtime it was indexed at."""
        if not self._has_table():
            return {}
        table = self._db.open_table(self.table_name)
        arrow = table.to_arrow()
        if arrow.num_rows == 0:
            return {}
        doc_ids = arrow.column("doc_id").to_pylist()
        mtimes = arrow.column("mtime").to_pylist()
        result: Dict[str, float] = {}
        for doc_id, mtime in zip(doc_ids, mtimes):
            # All chunks of a document share an mtime; keep the largest seen.
            if doc_id not in result or (mtime or 0.0) > result[doc_id]:
                result[doc_id] = mtime or 0.0
        return result

    def search(self, query: str, top_k: int = 5) -> List[Dict[str, Any]]:
        """Return the top-k most similar chunks for a text query."""
        if not self._has_table():
            return []
        table = self._db.open_table(self.table_name)
        query_vec = embed([query], self.model_name)[0]
        return table.search(query_vec).metric("cosine").limit(top_k).to_list()

    # -- incremental indexing ---------------------------------------------- #
    def sync_directory(
        self,
        root: Path,
        max_tokens: int,
        overlap: int,
        use_model_tokenizer: bool = True,
        prune_missing: bool = True,
        on_error: Optional[Callable[[Path, Exception], None]] = None,
    ) -> IndexStats:
        """Index ``root`` incrementally, only touching changed files.

        A document is re-indexed when its file mtime differs from the mtime
        recorded in the table. Unchanged documents are skipped. When
        ``prune_missing`` is true, documents whose files no longer exist are
        deleted from the index.

        A single unreadable / malformed file does not abort the run: it is
        counted in ``stats.errors`` and, if provided, reported via ``on_error``.
        Files that fail to parse are never pruned from an existing index.

        Raises :class:`FileNotFoundError` if ``root`` itself does not exist.
        """
        root = Path(root)
        stats = IndexStats()

        count_tokens: Optional[Callable[[str], int]] = (
            make_token_counter(self.model_name) if use_model_tokenizer else None
        )

        existing = self.indexed_mtimes()
        seen: set[str] = set()

        for path in find_markdown_files(root):
            stats.scanned += 1

            # Compute the doc_id up front (without parsing) so a file that
            # still exists is never pruned just because it failed to parse.
            try:
                doc_id = str(path.resolve().relative_to(root.resolve()))
            except ValueError:
                doc_id = path.name
            seen.add(doc_id)

            try:
                doc: ParsedDocument = parse_file(path, root)

                prev = existing.get(doc.doc_id)
                if prev is not None and abs(prev - doc.mtime) <= _MTIME_EPSILON:
                    stats.skipped += 1
                    continue

                if prev is not None:
                    self.delete_doc(doc.doc_id)
                    stats.updated += 1
                else:
                    stats.added += 1

                doc_chunks = chunk_document(doc, max_tokens, overlap, count_tokens)
                stats.chunks += self.add_chunks(doc_chunks)
            except Exception as exc:  # noqa: BLE001 - one bad file must not abort
                stats.errors += 1
                if on_error is not None:
                    on_error(path, exc)

        if prune_missing:
            for doc_id in existing:
                if doc_id not in seen:
                    self.delete_doc(doc_id)
                    stats.removed += 1

        return stats

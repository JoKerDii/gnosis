"""Markdown ingestion: recursive discovery, front-matter parsing, chunking.

This module is intentionally free of heavy dependencies (no torch / lancedb) so
it stays fast to import and trivial to unit-test. Token counting is pluggable:
the default counter approximates tokens by whitespace-splitting, and the real
model tokenizer can be injected at call time (see ``db.make_token_counter``).
"""

from __future__ import annotations

import datetime as _dt
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Dict, Iterable, Iterator, List, Optional

import frontmatter

# A token counter maps a string to an integer token count.
TokenCounter = Callable[[str], int]

# Recognised Markdown file extensions.
MARKDOWN_SUFFIXES = (".md", ".markdown", ".mdown", ".mkd")


@dataclass
class ParsedDocument:
    """A Markdown file after front-matter extraction."""

    doc_id: str                       # path relative to the scan root
    source: str                       # absolute path to the source file
    title: str                        # front-matter title or filename stem
    content: str                      # raw body (front matter stripped)
    tags: List[str] = field(default_factory=list)
    date: Optional[str] = None        # ISO-8601 string if present
    mtime: float = 0.0                # file modification time (epoch seconds)
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class Chunk:
    """A single embeddable slice of a document."""

    doc_id: str
    chunk_index: int
    text: str
    title: str
    source: str
    tags: List[str] = field(default_factory=list)
    date: Optional[str] = None
    mtime: float = 0.0
    metadata: Dict[str, Any] = field(default_factory=dict)


# --------------------------------------------------------------------------- #
# Discovery
# --------------------------------------------------------------------------- #
def find_markdown_files(root: Path) -> List[Path]:
    """Recursively collect Markdown files under ``root``, sorted by path."""
    root = Path(root)
    if not root.exists():
        raise FileNotFoundError(f"Knowledge directory does not exist: {root}")
    if root.is_file():
        return [root] if root.suffix.lower() in MARKDOWN_SUFFIXES else []

    files = [
        p
        for p in root.rglob("*")
        if p.is_file() and p.suffix.lower() in MARKDOWN_SUFFIXES
    ]
    return sorted(files)


# --------------------------------------------------------------------------- #
# Front-matter parsing
# --------------------------------------------------------------------------- #
def _normalize_tags(raw: Any) -> List[str]:
    """Coerce a front-matter ``tags`` value into a clean list of strings."""
    if raw is None:
        return []
    if isinstance(raw, str):
        # Allow comma- or space-separated strings: "a, b c".
        parts = re.split(r"[,\n]", raw)
        return [t.strip() for t in parts if t.strip()]
    if isinstance(raw, Iterable):
        return [str(t).strip() for t in raw if str(t).strip()]
    return [str(raw).strip()]


def _normalize_date(raw: Any) -> Optional[str]:
    """Normalize a front-matter date to an ISO-8601 string (or None)."""
    if raw is None:
        return None
    if isinstance(raw, (_dt.datetime, _dt.date)):
        return raw.isoformat()
    text = str(raw).strip()
    return text or None


def parse_file(path: Path, root: Optional[Path] = None) -> ParsedDocument:
    """Parse one Markdown file into a :class:`ParsedDocument`.

    ``root`` is used to compute a stable, relative ``doc_id``. When omitted the
    file's own name is used.
    """
    path = Path(path)
    post = frontmatter.load(str(path))
    meta: Dict[str, Any] = dict(post.metadata)

    if root is not None:
        try:
            doc_id = str(path.resolve().relative_to(Path(root).resolve()))
        except ValueError:
            doc_id = path.name
    else:
        doc_id = path.name

    title = str(meta.get("title") or path.stem)
    tags = _normalize_tags(meta.get("tags"))
    date = _normalize_date(meta.get("date"))

    try:
        mtime = path.stat().st_mtime
    except OSError:
        mtime = 0.0

    return ParsedDocument(
        doc_id=doc_id,
        source=str(path.resolve()),
        title=title,
        content=post.content,
        tags=tags,
        date=date,
        mtime=mtime,
        metadata=meta,
    )


# --------------------------------------------------------------------------- #
# Chunking
# --------------------------------------------------------------------------- #
def _default_token_count(text: str) -> int:
    """Approximate a token count by counting whitespace-separated words."""
    return len(text.split())


def split_paragraphs(text: str) -> List[str]:
    """Split ``text`` into paragraphs on blank lines."""
    normalized = text.replace("\r\n", "\n").replace("\r", "\n")
    parts = re.split(r"\n[ \t]*\n", normalized.strip())
    return [p.strip() for p in parts if p.strip()]


def _window_words(
    text: str,
    max_tokens: int,
    overlap: int,
    count_tokens: TokenCounter,
) -> List[str]:
    """Split an oversized paragraph into overlapping word windows.

    Windowing is done on words (a stable, deterministic unit); ``count_tokens``
    decides how many words fit under the token budget, so the result honours a
    real tokenizer when one is supplied.
    """
    words = text.split()
    if not words:
        return []

    chunks: List[str] = []
    start = 0
    n = len(words)
    while start < n:
        # Grow the window until adding another word would exceed max_tokens.
        end = start
        while end < n and count_tokens(" ".join(words[start : end + 1])) <= max_tokens:
            end += 1
        if end == start:
            # A single word already exceeds the budget; emit it alone so we
            # always make forward progress.
            end = start + 1
        chunks.append(" ".join(words[start:end]))
        if end >= n:
            break
        # Step back by `overlap` words to preserve context across the boundary.
        start = max(end - overlap, start + 1)
    return chunks


def chunk_text(
    text: str,
    max_tokens: int = 256,
    overlap: int = 50,
    count_tokens: Optional[TokenCounter] = None,
) -> List[str]:
    """Split ``text`` into paragraph-aware chunks of at most ``max_tokens``.

    Whole paragraphs are packed together until the budget is reached. Paragraphs
    larger than the budget are split into overlapping word windows. When a new
    chunk starts, trailing paragraphs up to ``overlap`` tokens are carried over
    so semantic context spans chunk boundaries.
    """
    if max_tokens <= 0:
        raise ValueError("max_tokens must be positive")
    if overlap < 0 or overlap >= max_tokens:
        raise ValueError("overlap must be >= 0 and < max_tokens")

    count = count_tokens or _default_token_count
    paragraphs = split_paragraphs(text)
    if not paragraphs:
        return []

    chunks: List[str] = []
    current: List[str] = []
    current_tokens = 0

    def flush() -> None:
        nonlocal current, current_tokens
        if current:
            chunks.append("\n\n".join(current))
            current = []
            current_tokens = 0

    for para in paragraphs:
        ptokens = count(para)

        if ptokens > max_tokens:
            # Paragraph too big to ever fit: flush, then window it directly.
            flush()
            chunks.extend(_window_words(para, max_tokens, overlap, count))
            continue

        if current and current_tokens + ptokens > max_tokens:
            # Close the current chunk, then seed the next one with overlap.
            chunks.append("\n\n".join(current))
            carry: List[str] = []
            carry_tokens = 0
            for prev in reversed(current):
                t = count(prev)
                if carry_tokens + t > overlap:
                    break
                carry.insert(0, prev)
                carry_tokens += t
            current = carry
            current_tokens = carry_tokens

        current.append(para)
        current_tokens += ptokens

    flush()
    return chunks


def chunk_document(
    doc: ParsedDocument,
    max_tokens: int = 256,
    overlap: int = 50,
    count_tokens: Optional[TokenCounter] = None,
) -> List[Chunk]:
    """Turn a :class:`ParsedDocument` into a list of :class:`Chunk` objects."""
    chunks: List[Chunk] = []
    for i, body in enumerate(chunk_text(doc.content, max_tokens, overlap, count_tokens)):
        chunks.append(
            Chunk(
                doc_id=doc.doc_id,
                chunk_index=i,
                text=body,
                title=doc.title,
                source=doc.source,
                tags=doc.tags,
                date=doc.date,
                mtime=doc.mtime,
                metadata=doc.metadata,
            )
        )
    return chunks


def iter_chunks(
    root: Path,
    max_tokens: int = 256,
    overlap: int = 50,
    count_tokens: Optional[TokenCounter] = None,
) -> Iterator[Chunk]:
    """Yield chunks for every Markdown file under ``root``."""
    for path in find_markdown_files(root):
        doc = parse_file(path, root)
        yield from chunk_document(doc, max_tokens, overlap, count_tokens)

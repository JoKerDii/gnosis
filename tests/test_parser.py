"""Comprehensive unit tests for src.parser.

These tests deliberately avoid the embedding model and LanceDB so they run fast
and hermetically. Token counting uses the default whitespace counter unless a
custom counter is injected.
"""

from __future__ import annotations

import datetime
import os
from pathlib import Path

import pytest

from src.parser import (
    Chunk,
    ParsedDocument,
    _normalize_date,
    _normalize_tags,
    chunk_document,
    chunk_text,
    find_markdown_files,
    iter_chunks,
    parse_file,
    split_paragraphs,
)


# --------------------------------------------------------------------------- #
# Helpers
# --------------------------------------------------------------------------- #
def write(path: Path, text: str, mtime: float | None = None) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")
    if mtime is not None:
        os.utime(path, (mtime, mtime))
    return path


# --------------------------------------------------------------------------- #
# Discovery: find_markdown_files
# --------------------------------------------------------------------------- #
class TestFindMarkdownFiles:
    def test_recursive_and_sorted(self, tmp_path: Path):
        write(tmp_path / "b.md", "b")
        write(tmp_path / "a.md", "a")
        write(tmp_path / "sub" / "c.md", "c")
        write(tmp_path / "sub" / "deep" / "d.markdown", "d")

        found = find_markdown_files(tmp_path)
        names = [p.name for p in found]
        assert names == sorted(names)  # sorted by full path
        assert {"a.md", "b.md", "c.md", "d.markdown"} == set(names)

    def test_ignores_non_markdown(self, tmp_path: Path):
        write(tmp_path / "note.md", "keep")
        write(tmp_path / "data.txt", "skip")
        write(tmp_path / "image.png", "skip")
        found = find_markdown_files(tmp_path)
        assert [p.name for p in found] == ["note.md"]

    def test_case_insensitive_suffix(self, tmp_path: Path):
        write(tmp_path / "UPPER.MD", "x")
        found = find_markdown_files(tmp_path)
        assert [p.name for p in found] == ["UPPER.MD"]

    def test_missing_directory_raises(self, tmp_path: Path):
        with pytest.raises(FileNotFoundError):
            find_markdown_files(tmp_path / "does-not-exist")

    def test_single_file_path(self, tmp_path: Path):
        f = write(tmp_path / "solo.md", "x")
        assert find_markdown_files(f) == [f]

    def test_single_non_markdown_file(self, tmp_path: Path):
        f = write(tmp_path / "solo.txt", "x")
        assert find_markdown_files(f) == []

    def test_empty_directory(self, tmp_path: Path):
        assert find_markdown_files(tmp_path) == []


# --------------------------------------------------------------------------- #
# Front-matter parsing: parse_file
# --------------------------------------------------------------------------- #
class TestParseFile:
    def test_title_from_frontmatter(self, tmp_path: Path):
        f = write(tmp_path / "note.md", "---\ntitle: My Title\n---\nBody text")
        doc = parse_file(f, tmp_path)
        assert doc.title == "My Title"
        assert doc.content.strip() == "Body text"

    def test_title_falls_back_to_stem(self, tmp_path: Path):
        f = write(tmp_path / "my-note.md", "No front matter here")
        doc = parse_file(f, tmp_path)
        assert doc.title == "my-note"
        assert doc.content.strip() == "No front matter here"

    def test_tags_as_yaml_list(self, tmp_path: Path):
        f = write(tmp_path / "n.md", "---\ntags:\n  - alpha\n  - beta\n---\nx")
        doc = parse_file(f, tmp_path)
        assert doc.tags == ["alpha", "beta"]

    def test_tags_as_comma_string(self, tmp_path: Path):
        f = write(tmp_path / "n.md", "---\ntags: alpha, beta, gamma\n---\nx")
        doc = parse_file(f, tmp_path)
        assert doc.tags == ["alpha", "beta", "gamma"]

    def test_date_from_yaml_date(self, tmp_path: Path):
        # Unquoted YAML dates are parsed into datetime.date objects.
        f = write(tmp_path / "n.md", "---\ndate: 2024-01-15\n---\nx")
        doc = parse_file(f, tmp_path)
        assert doc.date == "2024-01-15"

    def test_date_from_string(self, tmp_path: Path):
        f = write(tmp_path / "n.md", "---\ndate: 'Jan 2024'\n---\nx")
        doc = parse_file(f, tmp_path)
        assert doc.date == "Jan 2024"

    def test_missing_date_is_none(self, tmp_path: Path):
        f = write(tmp_path / "n.md", "---\ntitle: t\n---\nx")
        doc = parse_file(f, tmp_path)
        assert doc.date is None

    def test_doc_id_is_relative_to_root(self, tmp_path: Path):
        f = write(tmp_path / "sub" / "deep" / "n.md", "x")
        doc = parse_file(f, tmp_path)
        assert doc.doc_id == str(Path("sub") / "deep" / "n.md")

    def test_doc_id_without_root_is_name(self, tmp_path: Path):
        f = write(tmp_path / "n.md", "x")
        doc = parse_file(f)
        assert doc.doc_id == "n.md"

    def test_source_is_absolute(self, tmp_path: Path):
        f = write(tmp_path / "n.md", "x")
        doc = parse_file(f, tmp_path)
        assert Path(doc.source).is_absolute()
        assert Path(doc.source) == f.resolve()

    def test_mtime_matches_file(self, tmp_path: Path):
        f = write(tmp_path / "n.md", "x", mtime=1_700_000_000)
        doc = parse_file(f, tmp_path)
        assert doc.mtime == pytest.approx(1_700_000_000, abs=1)

    def test_metadata_preserved(self, tmp_path: Path):
        f = write(tmp_path / "n.md", "---\ntitle: t\nauthor: Ada\n---\nx")
        doc = parse_file(f, tmp_path)
        assert doc.metadata["author"] == "Ada"
        assert doc.metadata["title"] == "t"

    def test_no_frontmatter_empty_metadata(self, tmp_path: Path):
        f = write(tmp_path / "n.md", "just content")
        doc = parse_file(f, tmp_path)
        assert doc.metadata == {}
        assert doc.tags == []


# --------------------------------------------------------------------------- #
# Normalizers
# --------------------------------------------------------------------------- #
class TestNormalizeTags:
    def test_none(self):
        assert _normalize_tags(None) == []

    def test_list(self):
        assert _normalize_tags(["a", " b ", "c"]) == ["a", "b", "c"]

    def test_comma_string(self):
        assert _normalize_tags("a, b,c") == ["a", "b", "c"]

    def test_newline_string(self):
        assert _normalize_tags("a\nb") == ["a", "b"]

    def test_scalar_number(self):
        assert _normalize_tags(2024) == ["2024"]

    def test_drops_empty_entries(self):
        assert _normalize_tags(["a", "", "  ", "b"]) == ["a", "b"]


class TestNormalizeDate:
    def test_date_object(self):
        assert _normalize_date(datetime.date(2024, 1, 15)) == "2024-01-15"

    def test_datetime_object(self):
        got = _normalize_date(datetime.datetime(2024, 1, 15, 9, 30))
        assert got.startswith("2024-01-15T09:30")

    def test_string(self):
        assert _normalize_date("  2024-01  ") == "2024-01"

    def test_none(self):
        assert _normalize_date(None) is None

    def test_empty_string(self):
        assert _normalize_date("   ") is None


# --------------------------------------------------------------------------- #
# Paragraph splitting
# --------------------------------------------------------------------------- #
class TestSplitParagraphs:
    def test_basic(self):
        assert split_paragraphs("a\n\nb\n\nc") == ["a", "b", "c"]

    def test_multiple_blank_lines(self):
        assert split_paragraphs("a\n\n\n\nb") == ["a", "b"]

    def test_crlf(self):
        assert split_paragraphs("a\r\n\r\nb") == ["a", "b"]

    def test_whitespace_only_between(self):
        assert split_paragraphs("a\n   \nb") == ["a", "b"]

    def test_strips_and_drops_empty(self):
        assert split_paragraphs("\n\n  a  \n\n\n") == ["a"]

    def test_empty(self):
        assert split_paragraphs("") == []
        assert split_paragraphs("   \n  \n") == []


# --------------------------------------------------------------------------- #
# Chunking: chunk_text
# --------------------------------------------------------------------------- #
class TestChunkText:
    def test_empty(self):
        assert chunk_text("", 10, 2) == []

    def test_single_small_paragraph(self):
        assert chunk_text("one two three", 10, 2) == ["one two three"]

    def test_packs_paragraphs_under_budget(self):
        text = "one two\n\nthree four"
        assert chunk_text(text, max_tokens=10, overlap=2) == ["one two\n\nthree four"]

    def test_splits_with_paragraph_overlap(self):
        text = "one two\n\nthree four\n\nfive six\n\nseven eight"
        chunks = chunk_text(text, max_tokens=5, overlap=2)
        assert chunks == [
            "one two\n\nthree four",
            "three four\n\nfive six",
            "five six\n\nseven eight",
        ]

    def test_no_overlap_carries_nothing(self):
        text = "one two\n\nthree four\n\nfive six"
        chunks = chunk_text(text, max_tokens=4, overlap=0)
        assert chunks == ["one two\n\nthree four", "five six"]

    def test_long_paragraph_is_windowed(self):
        words = " ".join(f"w{i}" for i in range(10))
        chunks = chunk_text(words, max_tokens=4, overlap=1)
        assert chunks == ["w0 w1 w2 w3", "w3 w4 w5 w6", "w6 w7 w8 w9"]

    def test_custom_token_counter_forces_progress(self):
        # Char-length counter: the word "aaaa" (4) alone exceeds max_tokens=3,
        # but it must still be emitted so chunking always advances.
        counter = lambda s: sum(len(w) for w in s.split())  # noqa: E731
        chunks = chunk_text("aaaa bb cc", max_tokens=3, overlap=0, count_tokens=counter)
        assert chunks[0] == "aaaa"
        assert "bb" in " ".join(chunks) and "cc" in " ".join(chunks)

    def test_custom_counter_changes_packing(self):
        # With a char counter, two 3-char words (6) exceed a budget of 5.
        counter = lambda s: sum(len(w) for w in s.split())  # noqa: E731
        chunks = chunk_text("foo\n\nbar", max_tokens=5, overlap=0, count_tokens=counter)
        assert chunks == ["foo", "bar"]

    @pytest.mark.parametrize(
        "max_tokens, overlap",
        [(0, 0), (-1, 0), (10, 10), (10, 11), (10, -1)],
    )
    def test_invalid_params_raise(self, max_tokens, overlap):
        with pytest.raises(ValueError):
            chunk_text("some text", max_tokens=max_tokens, overlap=overlap)

    def test_result_never_exceeds_budget_default_counter(self):
        text = "\n\n".join(f"para {i} word word word" for i in range(20))
        chunks = chunk_text(text, max_tokens=8, overlap=2)
        for c in chunks:
            assert len(c.split()) <= 8


# --------------------------------------------------------------------------- #
# Chunking: chunk_document
# --------------------------------------------------------------------------- #
class TestChunkDocument:
    def _doc(self, content: str) -> ParsedDocument:
        return ParsedDocument(
            doc_id="notes/a.md",
            source="/abs/notes/a.md",
            title="A",
            content=content,
            tags=["t1", "t2"],
            date="2024-01-01",
            mtime=123.0,
            metadata={"k": "v"},
        )

    def test_sequential_indices_and_propagation(self):
        doc = self._doc("one two\n\nthree four\n\nfive six")
        chunks = chunk_document(doc, max_tokens=4, overlap=0)
        assert [c.chunk_index for c in chunks] == list(range(len(chunks)))
        for c in chunks:
            assert isinstance(c, Chunk)
            assert c.doc_id == "notes/a.md"
            assert c.title == "A"
            assert c.tags == ["t1", "t2"]
            assert c.date == "2024-01-01"
            assert c.mtime == 123.0
            assert c.source == "/abs/notes/a.md"

    def test_empty_content_yields_no_chunks(self):
        assert chunk_document(self._doc("   \n\n  "), 10, 2) == []


# --------------------------------------------------------------------------- #
# End-to-end iteration: iter_chunks
# --------------------------------------------------------------------------- #
class TestIterChunks:
    def test_iterates_all_files(self, tmp_path: Path):
        write(tmp_path / "a.md", "---\ntitle: A\n---\nalpha beta")
        write(tmp_path / "sub" / "b.md", "---\ntitle: B\n---\ngamma delta")
        chunks = list(iter_chunks(tmp_path, max_tokens=10, overlap=2))
        titles = {c.title for c in chunks}
        assert titles == {"A", "B"}
        assert all(c.text for c in chunks)

    def test_respects_chunking(self, tmp_path: Path):
        write(tmp_path / "a.md", "one two\n\nthree four\n\nfive six")
        chunks = list(iter_chunks(tmp_path, max_tokens=4, overlap=0))
        assert len(chunks) == 2
        assert chunks[0].chunk_index == 0
        assert chunks[1].chunk_index == 1

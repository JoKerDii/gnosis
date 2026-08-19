"""Tests for the period summary-report pipeline (src/reports.py)."""

import json
from pathlib import Path

import pytest

from src import reports
from src.reports import (
    OpenAIReportGenerator,
    ReportSpec,
    SourcePost,
    StubReportGenerator,
    build_reports,
    discover_posts,
    fill_prompt,
    plan_reports,
    resolve_generator,
)


def _write_note(dir_path: Path, name: str, title: str, date: str, tags: str, body: str):
    (dir_path / name).write_text(
        f"---\ntitle: {title}\ndate: {date}\ntags: {tags}\n---\n\n{body}\n",
        encoding="utf-8",
    )


def _make_notes(root: Path):
    root.mkdir(parents=True, exist_ok=True)
    _write_note(root, "2026-January.md", "Jan post", "2026-01-15 10:00:00", "readings", "Body J")
    _write_note(root, "2026-February.md", "Feb post", "2026-02-15 10:00:00", "readings", "Body F")
    _write_note(root, "2026-March.md", "Mar post", "2026-03-15 10:00:00", "readings", "Body M")
    _write_note(root, "2026-April.md", "Apr post", "2026-04-15 10:00:00", "readings", "Body A")
    # A non-journal book note (different tag) must be excluded.
    _write_note(root, "Some-Book.md", "A Book", "2026-01-20 10:00:00", "book", "Not a journal")
    # A journal post with no parseable date must be skipped.
    _write_note(root, "Undated.md", "No date", "", "readings", "Skipped")


def test_discover_filters_by_tag_and_date(tmp_path):
    root = tmp_path / "notes"
    _make_notes(root)
    posts = discover_posts(root)
    titles = {p.title for p in posts}
    assert titles == {"Jan post", "Feb post", "Mar post", "Apr post"}
    assert all(isinstance(p, SourcePost) for p in posts)
    jan = next(p for p in posts if p.title == "Jan post")
    assert (jan.year, jan.month) == (2026, 1)
    assert jan.month_label == "January 2026"


def test_plan_groups_into_periods(tmp_path):
    root = tmp_path / "notes"
    _make_notes(root)
    plan = plan_reports(discover_posts(root))
    assert len(plan["month"]) == 4
    # Jan-Mar -> Q1, Apr -> Q2.
    quarter_slugs = {s.slug for s in plan["quarter"]}
    assert quarter_slugs == {"2026-Q1", "2026-Q2"}
    assert len(plan["year"]) == 1
    assert plan["year"][0].slug == "2026"
    # Newest-first ordering for months.
    assert plan["month"][0].slug == "2026-04"


def test_build_reports_writes_files_and_manifest(tmp_path):
    root = tmp_path / "notes"
    out = tmp_path / "out"
    _make_notes(root)
    result = build_reports(root, out, generator=StubReportGenerator())

    assert result == {"posts": 4, "month": 4, "quarter": 2, "year": 1}

    manifest = json.loads((out / "index.json").read_text(encoding="utf-8"))
    assert manifest["generator"] == "stub"
    assert {e["slug"] for e in manifest["levels"]["month"]} == {
        "2026-01", "2026-02", "2026-03", "2026-04",
    }

    # Every manifest entry points at a file that exists and is nonempty.
    for entries in manifest["levels"].values():
        for e in entries:
            body = (out / e["file"]).read_text(encoding="utf-8")
            assert body.strip()
            assert "Reading Report" in body


class _RecordingGenerator:
    """Captures the source material handed to each report for assertions."""

    name = "recording"

    def __init__(self):
        self.materials = {}

    def generate(self, spec, source_material, prompt_template):
        self.materials[spec.slug] = source_material
        return f"# {spec.period} ({spec.level})\n\nbody for {spec.slug}\n"


def test_build_only_month_preserves_other_levels(tmp_path):
    root = tmp_path / "notes"
    out = tmp_path / "out"
    _make_notes(root)
    build_reports(root, out, generator=StubReportGenerator())  # full build first

    result = build_reports(
        root, out, generator=StubReportGenerator(), only={"month"}
    )
    # Only months regenerated this run.
    assert result == {"posts": 4, "month": 4, "quarter": 0, "year": 0}

    # But the manifest still lists the previously-built quarter/year reports.
    manifest = json.loads((out / "index.json").read_text(encoding="utf-8"))
    assert len(manifest["levels"]["month"]) == 4
    assert len(manifest["levels"]["quarter"]) == 2
    assert len(manifest["levels"]["year"]) == 1


def test_build_only_quarter_reads_month_children_from_disk(tmp_path):
    root = tmp_path / "notes"
    out = tmp_path / "out"
    _make_notes(root)
    # Seed month reports on disk with a recognizable body.
    build_reports(root, out, generator=_RecordingGenerator())

    rec = _RecordingGenerator()
    build_reports(root, out, generator=rec, only={"quarter"})

    # The Q1 roll-up should have been fed its month reports read back from disk.
    assert "2026-Q1" in rec.materials
    assert "body for 2026-01" in rec.materials["2026-Q1"]
    # It did not regenerate months or years — only quarter slugs were touched.
    assert rec.materials and all("Q" in k for k in rec.materials)


def test_stub_report_lists_sources_and_coverage():
    spec = ReportSpec(
        level="month",
        period="January 2026",
        slug="2026-01",
        year=2026,
        unit=1,
        sources=[SourcePost("Jan post", "2026-January.md", 2026, 1, "x", ["readings"])],
    )
    md = StubReportGenerator().generate(spec, "material", "template")
    assert "# January 2026 Reading Report (month)" in md
    assert "Placeholder report" in md
    assert "2026-January.md" in md
    assert "1 post(s)" in md


def test_fill_prompt_substitutes_placeholders():
    template = (
        "REPORT_LEVEL: {{month / quarter / year}}\n"
        'PERIOD: {{e.g., "March 2026" / "Q1 2026" / "2026"}}\n'
        "<journal> {{PASTE SOURCE MATERIAL HERE}} </journal>\n"
        "{{PERIOD}} Reading Report ({{REPORT_LEVEL}})"
    )
    spec = ReportSpec("quarter", "Q1 2026", "2026-Q1", 2026, 1, [])
    out = fill_prompt(template, spec, "SRC")
    assert "REPORT_LEVEL: quarter" in out
    assert "PERIOD: Q1 2026" in out
    assert "<journal> SRC </journal>" in out
    assert "Q1 2026 Reading Report (quarter)" in out
    assert "{{" not in out


def test_load_generator_defaults_to_stub(monkeypatch):
    monkeypatch.delenv("REPORT_GENERATOR", raising=False)
    gen = reports.load_generator()
    assert isinstance(gen, StubReportGenerator)


def test_resolve_generator_aliases():
    assert isinstance(resolve_generator(""), StubReportGenerator)
    assert isinstance(resolve_generator("stub"), StubReportGenerator)


def test_resolve_generator_module_path():
    gen = resolve_generator("src.reports:StubReportGenerator")
    assert isinstance(gen, StubReportGenerator)


class _FakeMessage:
    def __init__(self, content):
        self.content = content


class _FakeChoice:
    def __init__(self, content):
        self.message = _FakeMessage(content)


class _FakeResponse:
    def __init__(self, content):
        self.choices = [_FakeChoice(content)]


class _FakeCompletions:
    def __init__(self, content):
        self._content = content
        self.calls = []

    def create(self, **kwargs):
        self.calls.append(kwargs)
        return _FakeResponse(self._content)


class _FakeChat:
    def __init__(self, content):
        self.completions = _FakeCompletions(content)


class _FakeClient:
    def __init__(self, content):
        self.chat = _FakeChat(content)


def test_openai_generator_uses_client_and_prompt():
    client = _FakeClient("# Real report\n\nsynthesised.")
    gen = OpenAIReportGenerator(model="gpt-4o", client=client)
    spec = ReportSpec("month", "January 2026", "2026-01", 2026, 1, [])
    out = gen.generate(spec, "SOURCE", "PROMPT {{PERIOD}} {{PASTE SOURCE MATERIAL HERE}}")

    assert out.startswith("# Real report")
    # The filled prompt (period + source) reached the API as the user message.
    call = client.chat.completions.calls[0]
    assert call["model"] == "gpt-4o"
    user_msg = call["messages"][-1]["content"]
    assert "January 2026" in user_msg
    assert "SOURCE" in user_msg


def test_openai_generator_requires_key(monkeypatch):
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    with pytest.raises(RuntimeError, match="OPENAI_API_KEY"):
        OpenAIReportGenerator()

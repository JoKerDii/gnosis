"""Period summary reports for the reading journal.

The journal is a set of monthly Markdown posts (front-matter ``tags: readings``).
This module distils one period — a month, quarter, or year — into a structured
report following the synthesis prompt in ``PROMPT.md``.

Because a *quality* report requires an LLM to extract, cluster, rank, and write
prose, generation is **pluggable**: :class:`ReportGenerator` is the interface, and
the shipped :class:`StubReportGenerator` produces a clearly-marked placeholder so
the UI works end-to-end with no model. To wire a real model, implement the
interface and either pass it to :func:`build_reports` or point the
``REPORT_GENERATOR`` env var at ``"your_module:YourGenerator"``.

Reports are *pre-generated* to static Markdown files under ``docs/reports/`` plus
an ``index.json`` manifest, so the frontend can display them offline (both in the
local-server and GitHub-Pages modes) with no runtime LLM call.
"""

from __future__ import annotations

import importlib
import json
import os
import re
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Protocol

from .parser import ParsedDocument, find_markdown_files, parse_file

# Manifest schema version (lets the frontend adapt if the shape changes).
SCHEMA_VERSION = 1

# A post is treated as a journal entry when it carries this tag.
JOURNAL_TAG = "readings"

_MONTH_NAMES = [
    "January", "February", "March", "April", "May", "June",
    "July", "August", "September", "October", "November", "December",
]


# --------------------------------------------------------------------------- #
# Period model
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class SourcePost:
    """A single journal post assigned to a calendar month."""

    title: str
    doc_id: str
    year: int
    month: int  # 1..12
    content: str = ""
    tags: List[str] = field(default_factory=list)

    @property
    def month_label(self) -> str:
        return f"{_MONTH_NAMES[self.month - 1]} {self.year}"


@dataclass
class ReportSpec:
    """Everything a generator needs to write one report."""

    level: str            # "month" | "quarter" | "year"
    period: str           # e.g. "June 2026" / "Q2 2026" / "2026"
    slug: str             # e.g. "2026-06" / "2026-Q2" / "2026"
    year: int
    # 1..12 for month level, 1..4 for quarter level, None for year level.
    unit: Optional[int] = None
    sources: List[SourcePost] = field(default_factory=list)


def _parse_year_month(date: Optional[str]) -> Optional[tuple[int, int]]:
    """Pull (year, month) out of an ISO-ish date string, or return None."""
    if not date:
        return None
    m = re.match(r"\s*(\d{4})-(\d{2})", str(date))
    if not m:
        return None
    year, month = int(m.group(1)), int(m.group(2))
    if 1 <= month <= 12:
        return year, month
    return None


def _is_journal(doc: ParsedDocument) -> bool:
    return any(t.strip().lower() == JOURNAL_TAG for t in doc.tags)


def discover_posts(root: Path) -> List[SourcePost]:
    """Find every journal post under ``root`` that has a usable date."""
    posts: List[SourcePost] = []
    for path in find_markdown_files(root):
        doc = parse_file(path, root)
        if not _is_journal(doc):
            continue
        ym = _parse_year_month(doc.date)
        if ym is None:
            continue
        year, month = ym
        posts.append(
            SourcePost(
                title=doc.title,
                doc_id=doc.doc_id,
                year=year,
                month=month,
                content=doc.content,
                tags=doc.tags,
            )
        )
    return posts


def _quarter_of(month: int) -> int:
    return (month - 1) // 3 + 1


def plan_reports(posts: List[SourcePost]) -> Dict[str, List[ReportSpec]]:
    """Group posts into month/quarter/year report specs (newest first)."""
    months: Dict[tuple[int, int], List[SourcePost]] = {}
    for p in posts:
        months.setdefault((p.year, p.month), []).append(p)

    month_specs: List[ReportSpec] = []
    for (year, month), items in months.items():
        items_sorted = sorted(items, key=lambda s: s.doc_id)
        month_specs.append(
            ReportSpec(
                level="month",
                period=f"{_MONTH_NAMES[month - 1]} {year}",
                slug=f"{year:04d}-{month:02d}",
                year=year,
                unit=month,
                sources=items_sorted,
            )
        )

    quarter_specs: List[ReportSpec] = []
    quarters: Dict[tuple[int, int], List[SourcePost]] = {}
    for p in posts:
        quarters.setdefault((p.year, _quarter_of(p.month)), []).append(p)
    for (year, q), items in quarters.items():
        quarter_specs.append(
            ReportSpec(
                level="quarter",
                period=f"Q{q} {year}",
                slug=f"{year:04d}-Q{q}",
                year=year,
                unit=q,
                sources=sorted(items, key=lambda s: (s.month, s.doc_id)),
            )
        )

    year_specs: List[ReportSpec] = []
    years: Dict[int, List[SourcePost]] = {}
    for p in posts:
        years.setdefault(p.year, []).append(p)
    for year, items in years.items():
        year_specs.append(
            ReportSpec(
                level="year",
                period=f"{year}",
                slug=f"{year:04d}",
                year=year,
                unit=None,
                sources=sorted(items, key=lambda s: (s.month, s.doc_id)),
            )
        )

    # Newest period first so the frontend dropdowns default to recent work.
    month_specs.sort(key=lambda s: (s.year, s.unit or 0), reverse=True)
    quarter_specs.sort(key=lambda s: (s.year, s.unit or 0), reverse=True)
    year_specs.sort(key=lambda s: s.year, reverse=True)
    return {"month": month_specs, "quarter": quarter_specs, "year": year_specs}


# --------------------------------------------------------------------------- #
# Prompt assembly
# --------------------------------------------------------------------------- #
def load_prompt_template(project_root: Optional[Path] = None) -> str:
    """Read ``PROMPT.md`` from the project root (empty string if absent)."""
    root = project_root or Path(__file__).resolve().parent.parent
    prompt_path = root / "PROMPT.md"
    try:
        return prompt_path.read_text(encoding="utf-8")
    except OSError:
        return ""


def fill_prompt(template: str, spec: ReportSpec, source_material: str) -> str:
    """Substitute report parameters and source text into the prompt template.

    A real :class:`ReportGenerator` sends the return value to its model. The
    placeholders mirror the ``{{...}}`` slots authored in ``PROMPT.md``.
    """
    filled = template
    filled = filled.replace("{{month / quarter / year}}", spec.level)
    filled = filled.replace('{{e.g., "March 2026" / "Q1 2026" / "2026"}}', spec.period)
    filled = filled.replace("{{PASTE SOURCE MATERIAL HERE}}", source_material)
    filled = filled.replace("{{PERIOD}}", spec.period)
    filled = filled.replace("{{REPORT_LEVEL}}", spec.level)
    return filled


# --------------------------------------------------------------------------- #
# Generator interface + built-in stub
# --------------------------------------------------------------------------- #
class ReportGenerator(Protocol):
    """Turns assembled source material into a finished Markdown report.

    Implement this to plug in an LLM. ``source_material`` is the text to
    synthesise (raw posts for a month; lower-level reports for quarter/year).
    ``prompt_template`` is the raw ``PROMPT.md`` — use :func:`fill_prompt` to
    build the final model prompt.
    """

    def generate(
        self, spec: ReportSpec, source_material: str, prompt_template: str
    ) -> str:
        ...

    # Short identifier recorded in the manifest (e.g. "stub", "claude").
    name: str


class StubReportGenerator:
    """Placeholder generator: no LLM, honest about it.

    Emits a report in the ``PROMPT.md`` output format with real coverage counts
    and the list of source posts, but leaves the three synthesis lists pending.
    Replace it with a real generator and re-run ``build-reports``.
    """

    name = "stub"

    def generate(
        self, spec: ReportSpec, source_material: str, prompt_template: str
    ) -> str:
        posts = spec.sources
        n_posts = len(posts)

        # "dominant topics": most common non-journal tags across the posts.
        tag_counts: Dict[str, int] = {}
        for p in posts:
            for t in p.tags:
                key = t.strip().lower()
                if key and key != JOURNAL_TAG:
                    tag_counts[key] = tag_counts.get(key, 0) + 1
        top_tags = [
            t for t, _ in sorted(tag_counts.items(), key=lambda kv: kv[1], reverse=True)
        ][:5]
        topics = ", ".join(top_tags) if top_tags else "—"

        if spec.level == "month":
            basis = "raw journal post(s)"
        elif spec.level == "quarter":
            basis = "the month reports in this quarter"
        else:
            basis = "the quarter reports in this year"

        lines: List[str] = []
        lines.append(f"# {spec.period} Reading Report ({spec.level})")
        lines.append("")
        lines.append(
            "> ⚠️ **Placeholder report.** Generated by the built-in stub, which does "
            "not run any LLM synthesis. Wire a real generator (`ReportGenerator` in "
            "`src/reports.py`, or set the `REPORT_GENERATOR` env var) and re-run "
            "`python -m src.cli build-reports` to replace this with a real report."
        )
        lines.append("")
        lines.append(
            f"**Coverage:** {n_posts} post(s) · dominant topics: {topics} · "
            f"source basis: {basis} · pipeline: stub — no synthesis performed"
        )
        lines.append("")
        lines.append("## Source material included")
        if posts:
            for p in posts:
                lines.append(f"- **{p.title}** — {p.month_label} · `{p.doc_id}`")
        else:
            lines.append("- _No source posts found for this period._")
        lines.append("")
        for heading in ("Top 10 Learnings", "Top 10 Trends", "Top 10 Concepts"):
            lines.append(f"## {heading}")
            lines.append("")
            lines.append("_Pending LLM generation._")
            lines.append("")
        return "\n".join(lines).rstrip() + "\n"


_SYSTEM_PROMPT = (
    "You are a rigorous knowledge-synthesis analyst. Follow the user's "
    "instructions exactly. Output only the final report in Markdown — no preamble, "
    "no explanation of your process, no code fences around the whole document.\n\n"
    "CRITICAL: Each of the three sections (Learnings, Trends, Concepts) must "
    "contain a FULL list of 10 distinct items — 10 Learnings, 10 Trends, and 10 "
    "Concepts. Do not stop at 5. Only produce fewer than 10 in a section if the "
    "source material genuinely cannot support 10 items that meet the quality bar, "
    "and in that case explicitly write 'Only N items met the threshold this "
    "period.' before that section's items. Prefer digging deeper into the source "
    "to reach 10 rather than truncating."
)


class OpenAIReportGenerator:
    """Generates reports with the OpenAI Chat Completions API.

    Reads the API key from the ``OPENAI_API_KEY`` environment variable (handled
    by the OpenAI SDK itself — the key is never passed through this code). The
    model defaults to ``OPENAI_MODEL`` or ``gpt-4o``.
    """

    name = "openai"

    def __init__(
        self,
        model: Optional[str] = None,
        temperature: float = 0.3,
        max_tokens: int = 8000,
        client=None,
    ) -> None:
        self.model = model or os.getenv("OPENAI_MODEL", "gpt-4o-mini")
        self.temperature = temperature
        self.max_tokens = max_tokens
        if client is not None:
            self._client = client
            return
        try:
            from openai import OpenAI
        except ImportError as exc:  # pragma: no cover - import-guard
            raise RuntimeError(
                "The 'openai' package is not installed. Run: "
                "pip install -r requirements.txt"
            ) from exc
        if not os.getenv("OPENAI_API_KEY"):
            raise RuntimeError(
                "OPENAI_API_KEY is not set. Export it in your shell or add it to "
                ".env before running build-reports with the OpenAI generator."
            )
        # Let the SDK retry transient errors, and cap each request's wall time.
        self._client = OpenAI(max_retries=5, timeout=120.0)

    def generate(
        self, spec: ReportSpec, source_material: str, prompt_template: str
    ) -> str:
        prompt = fill_prompt(prompt_template, spec, source_material)
        messages = [
            {"role": "system", "content": _SYSTEM_PROMPT},
            {"role": "user", "content": prompt},
        ]
        # Outer retry with backoff on top of the SDK's own retries, so a single
        # flaky connection doesn't abort a long multi-report batch.
        last_exc: Optional[Exception] = None
        attempts = 6
        for attempt in range(attempts):
            try:
                resp = self._client.chat.completions.create(
                    model=self.model,
                    temperature=self.temperature,
                    max_tokens=self.max_tokens,
                    messages=messages,
                )
                content = (resp.choices[0].message.content or "").strip()
                return content + "\n" if content else content
            except Exception as exc:  # noqa: BLE001 - retry any transient failure
                last_exc = exc
                if attempt < attempts - 1:
                    time.sleep(min(4 * 2 ** attempt, 60))
        raise RuntimeError(
            f"OpenAI request failed for {spec.slug} after retries: {last_exc}"
        )


# Short aliases usable via --generator or the REPORT_GENERATOR env var.
_BUILTIN_GENERATORS = {
    "stub": StubReportGenerator,
    "openai": OpenAIReportGenerator,
}


def resolve_generator(spec: str) -> ReportGenerator:
    """Resolve a generator from a name/alias or a ``"module:Class"`` path.

    Accepts a built-in alias (``"stub"``, ``"openai"``) or a dotted import path
    ``"package.module:ClassName"``. The class is instantiated with no arguments.
    """
    spec = (spec or "").strip()
    if not spec:
        return StubReportGenerator()
    if spec in _BUILTIN_GENERATORS:
        return _BUILTIN_GENERATORS[spec]()
    module_name, _, attr = spec.partition(":")
    if not module_name or not attr:
        return StubReportGenerator()
    module = importlib.import_module(module_name)
    return getattr(module, attr)()


def load_generator() -> ReportGenerator:
    """Resolve the generator named by ``REPORT_GENERATOR`` env, else the stub."""
    return resolve_generator(os.getenv("REPORT_GENERATOR", ""))


# --------------------------------------------------------------------------- #
# Source-material assembly
# --------------------------------------------------------------------------- #
def _raw_material(posts: List[SourcePost]) -> str:
    """Concatenate raw posts with headers for the month level."""
    blocks = []
    for p in posts:
        blocks.append(f"## {p.title}  (article/podcast journal — {p.month_label})\n\n{p.content}")
    return "\n\n---\n\n".join(blocks)


def _rollup_material(reports: List[str]) -> str:
    """Concatenate already-generated lower-level reports (quarter/year)."""
    return "\n\n---\n\n".join(reports)


# --------------------------------------------------------------------------- #
# Orchestration
# --------------------------------------------------------------------------- #
def _write_report(out_dir: Path, spec: ReportSpec, markdown: str) -> str:
    """Write one report file and return its manifest-relative path."""
    rel = f"{spec.level}/{spec.slug}.md"
    dest = out_dir / rel
    dest.parent.mkdir(parents=True, exist_ok=True)
    dest.write_text(markdown, encoding="utf-8")
    return rel


def _manifest_entry(spec: ReportSpec, rel_path: str) -> Dict:
    return {
        "level": spec.level,
        "period": spec.period,
        "slug": spec.slug,
        "file": rel_path,
        "year": spec.year,
        "unit": spec.unit,
        "post_count": len(spec.sources),
    }


LEVELS = ("month", "quarter", "year")


def build_reports(
    knowledge_dir: Path,
    out_dir: Path,
    generator: Optional[ReportGenerator] = None,
    prompt_template: Optional[str] = None,
    only: Optional[Iterable[str]] = None,
) -> Dict[str, int]:
    """Generate month/quarter/year reports and write them under ``out_dir``.

    Reports are produced in dependency order: months first (from raw posts),
    then quarters (from their month reports), then years (from their quarter
    reports), matching ``PROMPT.md``'s roll-up rules. Writes one Markdown file
    per report plus ``index.json``. Returns per-level counts of what was generated.

    ``only`` restricts generation to the given level(s) (e.g. ``{"month"}``) for
    a cheaper partial run. Skipped levels are left untouched on disk but kept in
    the manifest if their files already exist, and a regenerated roll-up level
    reads its children from previously-written files when they weren't part of
    this run.
    """
    knowledge_dir = Path(knowledge_dir)
    out_dir = Path(out_dir)
    gen = generator or load_generator()
    template = prompt_template if prompt_template is not None else load_prompt_template()
    targets = set(only) if only else set(LEVELS)

    posts = discover_posts(knowledge_dir)
    plan = plan_reports(posts)

    # In-memory store of markdown generated this run so roll-ups can read children.
    generated: Dict[str, Dict[str, str]] = {lvl: {} for lvl in LEVELS}
    manifest_levels: Dict[str, List[Dict]] = {lvl: [] for lvl in LEVELS}
    counts: Dict[str, int] = {lvl: 0 for lvl in LEVELS}

    def child_markdown(level: str, slug: str) -> Optional[str]:
        """Child report text: from this run if available, else from disk."""
        if slug in generated[level]:
            return generated[level][slug]
        path = out_dir / level / f"{slug}.md"
        if path.exists():
            return path.read_text(encoding="utf-8")
        return None

    def child_slugs(spec: ReportSpec) -> List[str]:
        if spec.level == "quarter":
            return [
                f"{spec.year:04d}-{m:02d}"
                for m in range((spec.unit - 1) * 3 + 1, spec.unit * 3 + 1)
            ]
        if spec.level == "year":
            return [f"{spec.year:04d}-Q{q}" for q in range(1, 5)]
        return []

    for level in LEVELS:
        for spec in plan[level]:
            if level not in targets:
                # Not regenerating: keep an existing report visible in the manifest.
                rel = f"{level}/{spec.slug}.md"
                if (out_dir / rel).exists():
                    manifest_levels[level].append(_manifest_entry(spec, rel))
                continue

            if level == "month":
                material = _raw_material(spec.sources)
            else:
                children = [c for c in (child_markdown(
                    "month" if level == "quarter" else "quarter", s
                ) for s in child_slugs(spec)) if c]
                material = _rollup_material(children) if children else _raw_material(spec.sources)

            md = gen.generate(spec, material, template)
            generated[level][spec.slug] = md
            rel = _write_report(out_dir, spec, md)
            manifest_levels[level].append(_manifest_entry(spec, rel))
            counts[level] += 1

    manifest = {
        "schema_version": SCHEMA_VERSION,
        "generator": getattr(gen, "name", "unknown"),
        "levels": manifest_levels,
    }
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "index.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    return {"posts": len(posts), **counts}

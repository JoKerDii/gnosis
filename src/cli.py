"""Command-line interface for the Markdown knowledge graph & search agent.

Commands
--------
scan   : (re)build the local LanceDB index from a directory of Markdown notes.
query  : run a single natural-language semantic search.
chat   : interactive REPL for running many queries without restarting.
stats  : show information about the current index.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Optional

import typer
from rich.console import Console, Group
from rich.panel import Panel
from rich.rule import Rule
from rich.table import Table
from rich.text import Text

from .config import get_settings

app = typer.Typer(
    add_completion=False,
    no_args_is_help=True,
    help="Local Markdown Knowledge Graph & Search Agent.",
)

console = Console()
err_console = Console(stderr=True)

# Words that end the interactive chat loop.
_EXIT_WORDS = {"exit", "quit", ":q", ":quit", "\\q"}


# --------------------------------------------------------------------------- #
# Shared helpers
# --------------------------------------------------------------------------- #
def _store():
    """Build a VectorStore from current settings (lazy import of heavy deps)."""
    from .db import VectorStore

    s = get_settings()
    return VectorStore(s.lancedb_path, s.lancedb_table, s.embedding_model), s


def _fail(message: str) -> "typer.Exit":
    """Print an error to stderr and return an Exit(1) to raise."""
    err_console.print(f"[bold red]Error:[/] {message}")
    return typer.Exit(code=1)


def _similarity(distance: Optional[float]) -> Optional[float]:
    """Convert a cosine distance into a 0..1 similarity score."""
    if distance is None:
        return None
    return max(0.0, 1.0 - float(distance))


def _score_style(sim: Optional[float]) -> str:
    if sim is None:
        return "dim"
    if sim >= 0.5:
        return "bold green"
    if sim >= 0.3:
        return "yellow"
    return "red"


def _clean_snippet(text: str, limit: int = 500) -> str:
    """Collapse whitespace and truncate a chunk for display."""
    collapsed = " ".join((text or "").split())
    if len(collapsed) > limit:
        collapsed = collapsed[:limit].rstrip() + " …"
    return collapsed


def _render_result(index: int, r: Dict[str, Any]) -> Panel:
    """Render a single search hit as a styled panel."""
    title = str(r.get("title") or "(untitled)")
    doc_id = str(r.get("doc_id") or r.get("source") or "?")
    chunk_index = r.get("chunk_index", 0)
    tags: List[str] = list(r.get("tags") or [])
    date = str(r.get("date") or "").strip()
    sim = _similarity(r.get("_distance"))

    # Header shown as the panel title: rank badge + document title.
    header = Text()
    header.append(f" {index} ", style="bold white on blue")
    header.append(f"  {title}", style="bold")

    # Metadata line: path · chunk · date.
    meta = Text()
    meta.append("path  ", style="dim")
    meta.append(doc_id, style="cyan")
    meta.append(f"   chunk #{chunk_index}", style="dim")
    if date:
        meta.append(f"   {date}", style="dim")

    lines: List[Any] = [meta]

    # Tags line, if any.
    if tags:
        tag_line = Text("tags  ", style="dim")
        for t in tags:
            tag_line.append(f"#{t} ", style="magenta")
        lines.append(tag_line)

    # Snippet block.
    lines.append(Text(""))
    lines.append(Text(_clean_snippet(str(r.get("text") or ""))))

    score_text = "n/a" if sim is None else f"{sim:.3f}"
    subtitle = Text("relevance ", style="dim")
    subtitle.append(score_text, style=_score_style(sim))

    return Panel(
        Group(*lines),
        title=header,
        title_align="left",
        subtitle=subtitle,
        subtitle_align="right",
        border_style="blue",
        padding=(1, 2),
    )


def _render_results(query: str, results: List[Dict[str, Any]]) -> None:
    console.print()
    console.print(Rule(Text(f"{len(results)} results for “{query}”", style="bold")))
    for i, r in enumerate(results, 1):
        console.print(_render_result(i, r))


def _run_query(store, query: str, top_k: int) -> None:
    """Embed a query, search, and render results (with model-load spinner)."""
    query = query.strip()
    if not query:
        return
    with console.status("[dim]Embedding query and searching…[/]", spinner="dots"):
        results = store.search(query, top_k=top_k)
    if not results:
        console.print("[yellow]No matching chunks found.[/]")
        return
    _render_results(query, results)


def _warn_if_empty(store) -> bool:
    """Return True (and warn) if the index has no rows yet."""
    if store.count() == 0:
        console.print(
            "[yellow]The index is empty.[/] Run "
            "[bold cyan]python -m src.cli scan[/] first."
        )
        return True
    return False


# --------------------------------------------------------------------------- #
# scan
# --------------------------------------------------------------------------- #
@app.command()
def scan(
    directory: Optional[Path] = typer.Option(
        None, "--dir", "-d", help="Directory of Markdown notes (overrides KNOWLEDGE_DIR)."
    ),
    rebuild: bool = typer.Option(
        False, "--rebuild", help="Drop the existing table and re-index everything."
    ),
    no_prune: bool = typer.Option(
        False, "--no-prune", help="Keep index entries for files that were deleted."
    ),
) -> None:
    """Parse a Markdown directory and update the local LanceDB index.

    Incremental by default: only files whose modification time changed since the
    last run are re-embedded, and files deleted from disk are pruned.
    """
    store, s = _store()
    root = Path(directory or s.knowledge_dir)

    if not root.exists():
        raise _fail(
            f"directory does not exist: {root}\n"
            "Set KNOWLEDGE_DIR in your .env or pass --dir /path/to/notes."
        )
    if root.is_file():
        raise _fail(f"expected a directory but got a file: {root}")

    if rebuild:
        store.reset()
        console.print("[dim]Dropped existing table.[/]")

    console.print(f"Scanning [cyan]{root}[/] …")

    def _on_error(path: Path, exc: Exception) -> None:
        err_console.print(f"[yellow]skipped[/] {path}: {exc}")

    try:
        with console.status("[dim]Parsing and embedding…[/]", spinner="dots"):
            stats = store.sync_directory(
                root,
                max_tokens=s.chunk_tokens,
                overlap=s.chunk_overlap,
                prune_missing=not no_prune,
                on_error=_on_error,
            )
    except FileNotFoundError as exc:
        raise _fail(str(exc))

    if stats.scanned == 0:
        console.print(f"[yellow]No Markdown files found under {root}.[/]")
        return

    # Clean summary table.
    table = Table(show_header=False, box=None, padding=(0, 2, 0, 0))
    table.add_column(style="dim")
    table.add_column(justify="right", style="bold")
    table.add_row("Files scanned", str(stats.scanned))
    table.add_row("New", str(stats.added))
    table.add_row("Updated", str(stats.updated))
    table.add_row("Unchanged", str(stats.skipped))
    table.add_row("Removed", str(stats.removed))
    if stats.errors:
        table.add_row("[yellow]Errors[/]", f"[yellow]{stats.errors}[/]")
    table.add_row("Chunks written", str(stats.chunks))
    table.add_row("Total chunks in index", str(store.count()))

    console.print(Panel(table, title="[bold green]Scan complete[/]", border_style="green"))


# --------------------------------------------------------------------------- #
# query
# --------------------------------------------------------------------------- #
@app.command()
def query(
    text: List[str] = typer.Argument(..., help="Natural-language search query."),
    top_k: Optional[int] = typer.Option(
        None, "--top-k", "-k", min=1, max=50, help="Number of results (default from TOP_K)."
    ),
) -> None:
    """Search the index for the chunks most relevant to a query."""
    store, s = _store()
    if _warn_if_empty(store):
        raise typer.Exit(code=0)
    _run_query(store, " ".join(text), top_k or s.top_k)


# --------------------------------------------------------------------------- #
# chat
# --------------------------------------------------------------------------- #
@app.command()
def chat(
    top_k: Optional[int] = typer.Option(
        None, "--top-k", "-k", min=1, max=50, help="Number of results per query."
    ),
) -> None:
    """Interactive loop: type queries continuously; 'exit' or Ctrl-D to quit."""
    store, s = _store()
    k = top_k or s.top_k

    banner = Text()
    banner.append("Knowledge Chat\n", style="bold")
    banner.append("Type a question and press Enter. ", style="dim")
    banner.append("Commands: ", style="dim")
    banner.append("exit", style="bold cyan")
    banner.append("/", style="dim")
    banner.append("quit", style="bold cyan")
    banner.append(" or Ctrl-D to leave.", style="dim")
    console.print(Panel(banner, border_style="cyan"))

    if _warn_if_empty(store):
        return

    while True:
        try:
            line = console.input("[bold cyan]knowledge ›[/] ")
        except (EOFError, KeyboardInterrupt):
            console.print("\n[dim]Goodbye.[/]")
            break

        q = line.strip()
        if not q:
            continue
        if q.lower() in _EXIT_WORDS:
            console.print("[dim]Goodbye.[/]")
            break

        try:
            _run_query(store, q, k)
        except Exception as exc:  # noqa: BLE001 - keep the REPL alive
            err_console.print(f"[red]Query failed:[/] {exc}")


# --------------------------------------------------------------------------- #
# stats
# --------------------------------------------------------------------------- #
@app.command()
def stats() -> None:
    """Show information about the current index and configuration."""
    store, s = _store()
    table = Table(show_header=False, box=None, padding=(0, 2, 0, 0))
    table.add_column(style="dim")
    table.add_column(style="bold")
    table.add_row("Knowledge dir", str(s.knowledge_dir))
    table.add_row("LanceDB path", str(s.lancedb_path))
    table.add_row("Table", s.lancedb_table)
    table.add_row("Model", s.embedding_model)
    table.add_row("Chunk tokens / overlap", f"{s.chunk_tokens} / {s.chunk_overlap}")
    table.add_row("Indexed chunks", str(store.count()))
    console.print(Panel(table, title="[bold]Index status[/]", border_style="blue"))


# --------------------------------------------------------------------------- #
# build-web  (static keyword index for GitHub Pages / offline mode)
# --------------------------------------------------------------------------- #
@app.command(name="build-web")
def build_web(
    directory: Optional[Path] = typer.Option(
        None, "--dir", "-d", help="Directory of Markdown notes (overrides KNOWLEDGE_DIR)."
    ),
    out: Optional[Path] = typer.Option(
        None, "--out", "-o", help="Output JSON path (default: <DOCS_DIR>/search_index.json)."
    ),
) -> None:
    """Generate the static search_index.json used by the standalone frontend.

    This parses the notes (no embeddings) and writes a MiniSearch-ready document
    set into the docs/ folder for GitHub Pages / offline keyword search.
    """
    from .static_index import build_search_index

    s = get_settings()
    root = Path(directory or s.knowledge_dir)
    out_path = Path(out) if out else s.docs_dir / "search_index.json"

    if not root.exists():
        raise _fail(
            f"directory does not exist: {root}\n"
            "Set KNOWLEDGE_DIR in your .env or pass --dir /path/to/notes."
        )
    if root.is_file():
        raise _fail(f"expected a directory but got a file: {root}")

    console.print(f"Building static index from [cyan]{root}[/] …")
    try:
        result = build_search_index(
            root, out_path, max_tokens=s.chunk_tokens, overlap=s.chunk_overlap
        )
    except FileNotFoundError as exc:
        raise _fail(str(exc))

    table = Table(show_header=False, box=None, padding=(0, 2, 0, 0))
    table.add_column(style="dim")
    table.add_column(justify="right", style="bold")
    table.add_row("Files parsed", str(result["files"]))
    table.add_row("Documents written", str(result["documents"]))
    table.add_row("Output", str(out_path))
    console.print(
        Panel(table, title="[bold green]Static index built[/]", border_style="green")
    )


# --------------------------------------------------------------------------- #
# build-reports  (pre-generate period summary reports for the web UI)
# --------------------------------------------------------------------------- #
@app.command(name="build-reports")
def build_reports_cmd(
    directory: Optional[Path] = typer.Option(
        None, "--dir", "-d", help="Directory of Markdown notes (overrides KNOWLEDGE_DIR)."
    ),
    out: Optional[Path] = typer.Option(
        None, "--out", "-o", help="Output directory (default: <DOCS_DIR>/reports)."
    ),
    generator: Optional[str] = typer.Option(
        None,
        "--generator",
        "-g",
        help="Generator: 'stub', 'openai', or 'module:Class' "
        "(default: REPORT_GENERATOR env, else 'stub').",
    ),
    only: Optional[str] = typer.Option(
        None,
        "--only",
        help="Generate only one level: 'month', 'quarter', or 'year'. "
        "Other levels are left as-is (roll-ups reuse existing child reports).",
    ),
) -> None:
    """Pre-generate month/quarter/year reading reports for the web UI.

    Discovers journal posts (front-matter ``tags: readings``), groups them into
    periods, and writes one Markdown report per period plus an ``index.json``
    manifest. Choose the synthesis engine with ``--generator`` (e.g. ``openai``)
    or the ``REPORT_GENERATOR`` env var; the default stub writes placeholders.
    """
    from .reports import build_reports, load_generator, resolve_generator

    s = get_settings()
    root = Path(directory or s.knowledge_dir)
    out_dir = Path(out) if out else s.docs_dir / "reports"

    if not root.exists():
        raise _fail(
            f"directory does not exist: {root}\n"
            "Set KNOWLEDGE_DIR in your .env or pass --dir /path/to/notes."
        )
    if root.is_file():
        raise _fail(f"expected a directory but got a file: {root}")

    only_levels = None
    if only:
        level = only.strip().lower()
        if level not in ("month", "quarter", "year"):
            raise _fail("--only must be one of: month, quarter, year")
        only_levels = {level}

    try:
        gen = resolve_generator(generator) if generator else load_generator()
    except Exception as exc:  # noqa: BLE001 - surface config errors cleanly
        raise _fail(str(exc))
    gen_name = getattr(gen, "name", "unknown")
    scope = f" ([bold]{only_levels and next(iter(only_levels))}[/] only)" if only_levels else ""
    console.print(
        f"Building reports{scope} from [cyan]{root}[/] "
        f"using generator [bold]{gen_name}[/] …"
    )
    if gen_name != "stub":
        console.print("[dim]This calls an external LLM API and may take a while.[/]")
    try:
        with console.status("[dim]Generating reports…[/]", spinner="dots"):
            result = build_reports(root, out_dir, generator=gen, only=only_levels)
    except FileNotFoundError as exc:
        raise _fail(str(exc))
    except Exception as exc:  # noqa: BLE001 - surface generator/API errors cleanly
        raise _fail(f"report generation failed: {exc}")

    if result["posts"] == 0:
        console.print(
            f"[yellow]No journal posts (tags: readings) with dates found under {root}.[/]"
        )
        return

    table = Table(show_header=False, box=None, padding=(0, 2, 0, 0))
    table.add_column(style="dim")
    table.add_column(justify="right", style="bold")
    table.add_row("Journal posts", str(result["posts"]))
    table.add_row("Month reports", str(result["month"]))
    table.add_row("Quarter reports", str(result["quarter"]))
    table.add_row("Year reports", str(result["year"]))
    table.add_row("Output", str(out_dir))
    console.print(
        Panel(table, title="[bold green]Reports built[/]", border_style="green")
    )


# --------------------------------------------------------------------------- #
# serve  (launch the local FastAPI vector-search backend)
# --------------------------------------------------------------------------- #
@app.command()
def serve(
    host: Optional[str] = typer.Option(None, "--host", help="Bind host (default from SERVER_HOST)."),
    port: Optional[int] = typer.Option(None, "--port", "-p", help="Bind port (default from SERVER_PORT)."),
    reload: bool = typer.Option(False, "--reload", help="Auto-reload on code changes (dev)."),
) -> None:
    """Launch the local web server for AI-powered vector search."""
    try:
        import uvicorn
    except ImportError:
        raise _fail("uvicorn is not installed. Run: pip install -r requirements.txt")

    s = get_settings()
    bind_host = host or s.server_host
    bind_port = port or s.server_port

    console.print(
        Panel(
            f"[bold]Local Vector Mode[/] backend starting\n"
            f"API      [cyan]http://{bind_host}:{bind_port}[/]\n"
            f"Status   [cyan]http://{bind_host}:{bind_port}/status[/]\n"
            f"UI       [cyan]http://{bind_host}:{bind_port}/app/[/]  (if docs/ exists)\n"
            f"[dim]Press Ctrl-C to stop.[/]",
            border_style="green",
        )
    )

    if reload:
        # Reload requires an import string rather than an app instance.
        uvicorn.run("src.server:app", host=bind_host, port=bind_port, reload=True)
    else:
        from .server import create_app

        uvicorn.run(create_app(s), host=bind_host, port=bind_port)


def main() -> None:
    app()


if __name__ == "__main__":
    main()

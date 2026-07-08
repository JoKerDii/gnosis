# Development Log

A comprehensive, step-by-step record of how the **Local Markdown Knowledge Graph
& Search Agent** was built — the decisions made, the problems hit, and how each
was solved. Read this top-to-bottom to understand *why* the code looks the way it
does, or jump to a phase to reproduce a specific slice of the work.

- **Platform:** macOS (Apple Silicon), Python **3.9.6** (system interpreter)
- **Core stack:** LanceDB · sentence-transformers · Typer + Rich · FastAPI · MiniSearch
- **Guiding principle:** *local-first & offline* — no API keys, no network at query time.

The project was built in four phases:

1. [Project scaffolding](#phase-1--project-scaffolding)
2. [Core ingestion engine](#phase-2--core-ingestion-engine)
3. [User-facing CLI](#phase-3--user-facing-cli)
4. [Hybrid architecture (server + static fallback)](#phase-4--hybrid-architecture)

---

## Phase 1 — Project scaffolding

**Goal:** a modern Python project skeleton with dependencies installed in a virtual
environment, plus a script to prove the heavy libraries actually load.

### Steps

1. **Checked the environment.** Only Python `3.9.6` (the macOS system Python) was
   available — no `python3.10+`. This became a recurring constraint (see the
   frontmatter fix below).
2. **Created the structure:**
   ```
   src/__init__.py      # package marker
   src/config.py        # settings from .env / environment
   src/parser.py        # markdown parsing (stub)
   src/db.py            # vector storage (stub)
   src/cli.py           # command-line interface (stub)
   requirements.txt
   .env.example
   .gitignore
   README.md
   verify_setup.py      # standalone sanity check
   notes/example.md     # a sample note to index
   ```
3. **Declared dependencies** in `requirements.txt`: `lancedb`, `pyarrow`,
   `sentence-transformers`, `typer` (chosen over `click`), `python-frontmatter`,
   `python-dotenv`.
4. **Set up `.env.example`** with `KNOWLEDGE_DIR`, `LANCEDB_PATH`, `LANCEDB_TABLE`,
   `EMBEDDING_MODEL`, chunking params, and `TOP_K`.
5. **Created the venv and installed:**
   ```bash
   python3 -m venv .venv
   .venv/bin/python -m pip install --upgrade pip
   .venv/bin/python -m pip install -r requirements.txt   # pulls in torch — slow
   ```
6. **Ran `verify_setup.py`** — a tiny embed → store → search round trip in a temp
   directory.

### Problem & fix — `python-frontmatter` on Python 3.9

The verify script failed on import:

```
ImportError: cannot import name 'TypeGuard' from 'typing'
```

`python-frontmatter` **1.1.0+** uses `typing.TypeGuard`, which only exists in
Python **3.10+**. Everything else installed fine on 3.9.

**Fix:** pin to the last 3.9-compatible release and document why:

```
# requirements.txt
python-frontmatter==1.0.1   # 1.1.0+ needs Python 3.10+ (typing.TypeGuard)
```

Re-running `verify_setup.py` then passed all checks:
- ✓ all packages import
- ✓ model loads (`all-MiniLM-L6-v2`, 384-dim)
- ✓ LanceDB round trip returns the expected top match

**Outcome:** a working, installable project skeleton, verified end-to-end.

---

## Phase 2 — Core ingestion engine

**Goal:** turn Markdown files into searchable vectors, with smart chunking and an
incremental (only-changed-files) indexer. Plus comprehensive parser tests.

### `src/parser.py` — dependency-light parsing

Kept deliberately free of `torch`/`lancedb` so it imports fast and unit-tests
hermetically.

- **`find_markdown_files(root)`** — recursive, sorted, case-insensitive suffix
  match (`.md`, `.markdown`, `.mdown`, `.mkd`). Raises `FileNotFoundError` for a
  missing directory; handles a single-file path.
- **`parse_file(path, root)`** — uses `python-frontmatter` to extract:
  - **title** (front-matter `title`, else the filename stem)
  - **tags** (YAML list *or* comma/newline-separated string → normalized list)
  - **date** (`datetime.date`/`datetime` objects and strings → ISO-8601)
  - full metadata dict, and the file's **mtime** (the key to incremental indexing).
- **`chunk_text(...)` / `chunk_document(...)`** — **paragraph-aware, token-based**
  chunking: whole paragraphs are packed to a token budget; oversized paragraphs
  fall back to overlapping word windows; a token overlap is carried across chunk
  boundaries to preserve context. Token counting is **pluggable** — a whitespace
  approximation by default (fast, testable), the real model tokenizer in production.

### `src/db.py` — embeddings + LanceDB

- **Schema:** `vector, text, doc_id, chunk_index, title, source, tags, date, mtime`.
- **`VectorStore`** wraps a LanceDB table: `add_chunks`, `search` (cosine),
  `count`, `reset`, `delete_doc`.
- **`make_token_counter(model)`** — token counts from the model's own tokenizer, so
  "N tokens" means what the model actually consumes.
- **`sync_directory(...)`** — the incremental indexer:
  - Compares each file's mtime to the mtime stored in the table.
  - **New / changed** → re-embed; **unchanged** → skip; **deleted** → prune.
  - Returns an `IndexStats` summary (`scanned/added/updated/skipped/removed/chunks/errors`).

### Key decision — chunk size of **256**, not 500

The request suggested ~500-token chunks, but `all-MiniLM-L6-v2` **truncates inputs
at 256 tokens** (confirmed at runtime via `model_max_tokens`). Chunking to 500
would silently discard the second half of every chunk before embedding, hurting
search precision. So the default is **`CHUNK_TOKENS=256`** — configurable, with a
comment in `.env.example` explaining to raise it only for a longer-context model.

### Tests

Wrote `tests/test_parser.py` — **55 tests** covering discovery, front-matter,
normalizers, paragraph splitting, chunking (packing / overlap / windowing / custom
counters / invalid params / budget bounds), field propagation, and `iter_chunks`.
Added `pytest.ini` and a root `conftest.py` so `import src...` resolves.

```bash
.venv/bin/python -m pytest        # 55 passed
```

Also ran a **live smoke test** of the incremental engine: first run indexed 2 docs
→ second run skipped both → editing one showed `updated=1 skipped=1` → deleting one
showed `removed=1`. 

**Outcome:** a robust ingestion pipeline that re-embeds only what changed.

---

## Phase 3 — User-facing CLI

**Goal:** a beautiful, robust terminal interface built on Typer + Rich.

### Commands

| Command | What it does |
| --- | --- |
| `scan` | Parse the notes dir and incrementally update the LanceDB index. Flags: `--dir`, `--rebuild`, `--no-prune`. Prints a summary panel. |
| `query "..."` | One-off semantic search; renders each hit as a Rich panel (title, path, `#tags`, snippet, color-coded relevance score). `-k` sets result count. |
| `chat` | Interactive REPL (`knowledge ›` prompt) to run many queries without restarting. Exits on `exit`/`quit`/`:q` or Ctrl-D. |
| `stats` | Shows config + index status. |

### Robustness hardening (reviewed across the codebase)

- **Missing directory** → clean `Error:` to stderr, exit code **1**.
- **Malformed YAML front-matter** → that one file is skipped with a warning and
  counted in `stats.errors`; the scan continues. (Verified: a file with broken YAML
  did not abort the run.)
- **Empty / body-less Markdown** → produces zero chunks, no crash.
- **Empty index** on `query`/`chat` → friendly "run scan first" message.
- Files that fail to parse are **never pruned** — the `doc_id` is computed before
  parsing, so a transient error can't wipe good data.

### Verification

Ran the full `scan → query → chat` flow live, plus every error path above. All 55
parser tests stayed green. (Note: this phase renamed the earlier `index`/`search`
commands to **`scan`/`query`**.)

**Outcome:** a polished CLI that fails gracefully and reads clearly.

---

## Phase 4 — Hybrid architecture

**Goal:** make the same knowledge base work in **two modes**, chosen automatically
by the browser:

| Mode | Badge | Backend | Search |
| --- | --- | --- | --- |
| **Local Vector Mode** | 🟢 `● Local AI Connected` | FastAPI + LanceDB | AI semantic |
| **Static Cloud Mode** | 🔵 `● Static Cloud Mode` | `search_index.json` + MiniSearch (client-side) | keyword |

### Planning first

Before coding, an exploration pass confirmed a critical gap: **`docs/` did not
exist**, and there was **no `search_index.json` generator**. The request said
"update `docs/app.js`", but the whole frontend + static-index pipeline was
greenfield. Three decisions were confirmed with the user:

1. **MiniSearch, vendored** into `docs/vendor/` (offline-capable, no CDN reliance).
2. **A separate `build-web` command** to generate the static index (decoupled from
   the vector `scan`).
3. **FastAPI + uvicorn** for the backend.

### What was built

**Dependencies** — added `fastapi`, `uvicorn` to `requirements.txt`; `httpx` to
`requirements-dev.txt` (needed by FastAPI's `TestClient`).

**`src/config.py`** — extended `Settings` with `SERVER_HOST`, `SERVER_PORT`,
`DOCS_DIR`.

**`src/server.py`** (FastAPI):
- `GET /status` → `{"status":"online","mode":"vector", "indexed_chunks": N, "model": ...}`.
- `POST /search` → embeds the query (reusing `VectorStore.search`), **strips the raw
  vector**, converts cosine distance → a 0–1 `score`, returns ranked JSON.
  Empty query → **400**, backend error → **500**, empty index → `[]` (200).
- **CORS `*`** so any origin (including GitHub Pages) can call it.
- An **injectable `get_store()` dependency** so tests can swap in a fake store and
  avoid loading torch. Also mounts the UI at `/app/` for same-origin local use.

**`src/static_index.py` + `build-web` command** — parses notes with the torch-free
`iter_chunks` and writes `docs/search_index.json`
(`{schema_version, model, count, documents:[...]}`), one document per chunk.

**`serve` command** — launches uvicorn (`--host/--port/--reload`).

**`docs/` frontend (all new):**
- `index.html` — search bar + a corner **connection badge**.
- `style.css` — dark theme; green/blue/amber badge states; result cards.
- `app.js` — the hybrid logic:
  1. On load, `fetch(localhost:8000/status)` with a **1.5 s `AbortController` timeout**.
  2. Success → **vector mode**, green badge, queries hit `POST /search`.
  3. Any failure → `console.info` fallback notice → **static mode**: fetch
     `search_index.json`, initialize MiniSearch, blue badge.
  4. A dropped connection mid-session catches and **degrades to static** instead of
     crashing. Every path is wrapped so nothing throws uncaught.
- `vendor/minisearch.min.js` — vendored UMD build (~19 KB).
- `.nojekyll` — let GitHub Pages serve all assets.

### Tests

- `tests/test_server.py` — via `TestClient` + `dependency_overrides` with a
  `FakeStore` (no torch): `/status` shape, `/search` returns hits with computed
  `score` and **no `vector`**, empty query → 400, missing field → 422, `top_k`
  respected, empty index → `[]`, backend error → 500, **CORS header + preflight**.
- `tests/test_static_index.py` — builds from a temp dir, asserts valid JSON with the
  expected fields/counts.

```bash
.venv/bin/python -m pytest        # 71 passed (55 parser + 16 new)
```

### Live verification (real server + real browser)

1. **`curl`** the running server: `/status` returned the right shape; `/search`
   returned real semantic results with a `score` and no leaked vector; empty → 400;
   CORS preflight → `access-control-allow-origin: *`.
2. **Browser, Local Vector Mode** (via Chrome automation): loaded
   `http://localhost:8000/app/` → **green** badge → a query returned an AI result
   (relevance 0.418) with highlighted snippet and tags.
3. **Browser, graceful failure:** stopped the API server, served `docs/` as plain
   static files, reloaded → **blue** badge, keyword results. Injected trackers
   confirmed the refused connection is **caught** (`TypeError: Failed to fetch`) with
   **`uncaughtErrors: []`**. (The `[EXCEPTION]` lines Chrome prints are its automatic
   logging of the refused socket — not JS crashes.)

**Outcome:** one knowledge base, two search backends, auto-selected, with a frontend
that never crashes when the server is down.

---

## Final architecture

```
first-claude-code-app/
├── src/
│   ├── config.py        # env-driven settings
│   ├── parser.py        # markdown parse + paragraph/token chunking (no torch)
│   ├── db.py            # embeddings + LanceDB store + incremental sync
│   ├── static_index.py  # builds docs/search_index.json (no torch)
│   ├── server.py        # FastAPI: GET /status, POST /search (CORS *)
│   └── cli.py           # Typer CLI: scan/query/chat/stats/build-web/serve
├── docs/                # static frontend (GitHub Pages root)
│   ├── index.html · app.js · style.css
│   ├── search_index.json          # generated by `build-web`
│   └── vendor/minisearch.min.js    # vendored
├── notes/               # your Markdown files live here (KNOWLEDGE_DIR)
├── tests/               # test_parser.py · test_server.py · test_static_index.py
├── verify_setup.py
├── requirements.txt · requirements-dev.txt
└── .env.example
```

**Data flow:** `notes/*.md` → `parser` (parse + chunk) → either
`db` (embed → LanceDB → `/search`) **or** `static_index` (→ `search_index.json` →
MiniSearch). The frontend picks the backend at runtime.

---

## Reproduce from scratch

```bash
# 1. Environment
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt -r requirements-dev.txt
cp .env.example .env

# 2. Sanity check
python verify_setup.py

# 3. Add notes, then build both indexes
#    (drop .md files into notes/ or set KNOWLEDGE_DIR)
python -m src.cli scan          # vector index (LanceDB)
python -m src.cli build-web     # static index (docs/search_index.json)

# 4. Search from the terminal
python -m src.cli query "how do I get started?"
python -m src.cli chat

# 5. Or use the web UI
python -m src.cli serve         # http://localhost:8000/app/

# 6. Tests
python -m pytest                # 71 passed
```

## Key decisions & gotchas (quick reference)

| Topic | Decision | Why |
| --- | --- | --- |
| `python-frontmatter` | Pinned `==1.0.1` | 1.1.0+ needs Python 3.10+ (`typing.TypeGuard`); only 3.9 available. |
| Chunk size | `256` tokens, not 500 | `all-MiniLM-L6-v2` truncates at 256; larger chunks lose their tail. |
| Incremental indexing | Compare file **mtime** vs stored mtime | Re-embed only changed files; prune deleted ones. |
| CLI framework | Typer + Rich | Type-safe commands + beautiful panels. |
| Static search lib | MiniSearch, **vendored** | Fully offline GitHub Pages, no CDN dependency. |
| Static index build | Separate `build-web` command | Keeps the vector scan and static export decoupled. |
| Backend | FastAPI + uvicorn | Async, built-in CORS, Pydantic models, easy TestClient tests. |
| Server routes | UI mounted at `/app/` | FastAPI reserves `/docs` for its own Swagger UI. |
| Frontend failure | Catch every fetch, fall back | Server-down must never crash the UI. |

/*
 * Hybrid frontend for the Markdown Knowledge Search agent.
 *
 * On load it probes the local FastAPI server:
 *   - reachable  -> "Local Vector Mode"  (AI semantic search via POST /search)
 *   - unreachable -> "Static Cloud Mode" (client-side MiniSearch over search_index.json)
 *
 * Everything is wrapped in try/catch so a missing server or a missing index
 * never throws an uncaught error or breaks the UI.
 */
(function () {
  "use strict";

  const API_BASE = "http://localhost:8000";
  const STATUS_TIMEOUT_MS = 1500;
  const DEFAULT_TOP_K = 5;
  const SNIPPET_LIMIT = 320;

  // App state.
  const state = {
    mode: "connecting", // "vector" | "static" | "offline"
    miniSearch: null, // MiniSearch instance (static mode)
    docsById: new Map(), // id -> full document (static mode, for rendering)
  };

  // DOM refs (resolved on DOMContentLoaded).
  let els = {};

  // ---------------------------------------------------------------------- //
  // Badge / status helpers
  // ---------------------------------------------------------------------- //
  function setBadge(mode) {
    const badge = els.badge;
    const label = els.modeLabel;
    if (!badge || !label) return;
    badge.classList.remove(
      "badge--connecting",
      "badge--local",
      "badge--static",
      "badge--offline"
    );
    if (mode === "vector") {
      badge.classList.add("badge--local");
      label.textContent = "Local AI Connected";
    } else if (mode === "static") {
      badge.classList.add("badge--static");
      label.textContent = "Static Cloud Mode";
    } else if (mode === "offline") {
      badge.classList.add("badge--offline");
      label.textContent = "Search Unavailable";
    } else {
      badge.classList.add("badge--connecting");
      label.textContent = "Connecting…";
    }
  }

  function setStatus(message, isError) {
    if (!els.statusLine) return;
    els.statusLine.textContent = message || "";
    els.statusLine.classList.toggle("error", Boolean(isError));
  }

  // ---------------------------------------------------------------------- //
  // Mode detection
  // ---------------------------------------------------------------------- //
  async function fetchWithTimeout(url, options, timeoutMs) {
    const controller = new AbortController();
    const timer = setTimeout(() => controller.abort(), timeoutMs);
    try {
      return await fetch(url, Object.assign({ signal: controller.signal }, options));
    } finally {
      clearTimeout(timer);
    }
  }

  async function detectMode() {
    setBadge("connecting");
    setStatus("Detecting search backend…");
    try {
      const res = await fetchWithTimeout(
        API_BASE + "/status",
        { method: "GET" },
        STATUS_TIMEOUT_MS
      );
      if (!res.ok) throw new Error("status " + res.status);
      const data = await res.json();
      if (data && data.status === "online") {
        state.mode = "vector";
        setBadge("vector");
        const n = typeof data.indexed_chunks === "number" ? data.indexed_chunks : "?";
        setStatus("Local AI ready — semantic search over " + n + " indexed chunks.");
        return;
      }
      throw new Error("unexpected status payload");
    } catch (err) {
      // Expected whenever the local server is down (e.g. GitHub Pages).
      console.info(
        "[hybrid] Local vector server not reachable — falling back to Static Cloud Mode.",
        err && err.message ? "(" + err.message + ")" : ""
      );
      await initStaticMode();
    }
  }

  async function initStaticMode() {
    try {
      const res = await fetch("search_index.json", { cache: "no-cache" });
      if (!res.ok) throw new Error("search_index.json HTTP " + res.status);
      const payload = await res.json();
      const documents = Array.isArray(payload) ? payload : payload.documents || [];

      if (typeof MiniSearch === "undefined") {
        throw new Error("MiniSearch library failed to load");
      }

      const miniSearch = new MiniSearch({
        idField: "id",
        fields: ["title", "text", "tags"],
        storeFields: ["title", "doc_id", "path", "chunk_index", "tags", "date", "text"],
        searchOptions: { boost: { title: 2 }, prefix: true, fuzzy: 0.2 },
      });
      // Tags is an array; MiniSearch indexes it fine, but join for safety.
      const prepared = documents.map((d) =>
        Object.assign({}, d, { tags: Array.isArray(d.tags) ? d.tags.join(" ") : d.tags })
      );
      miniSearch.addAll(prepared);

      state.miniSearch = miniSearch;
      state.docsById = new Map(documents.map((d) => [String(d.id), d]));
      state.mode = "static";
      setBadge("static");
      setStatus(
        "Static keyword mode — " + documents.length + " chunks searchable offline."
      );
    } catch (err) {
      console.warn("[hybrid] Static index unavailable:", err);
      state.mode = "offline";
      setBadge("offline");
      setStatus(
        "No search backend available. Start the local server or run `build-web`.",
        true
      );
    }
  }

  // ---------------------------------------------------------------------- //
  // Searching
  // ---------------------------------------------------------------------- //
  async function runSearch(query) {
    query = (query || "").trim();
    if (!query) return;

    if (state.mode === "vector") {
      try {
        await runVectorSearch(query);
        return;
      } catch (err) {
        // Server went away mid-session: degrade gracefully to static mode.
        console.warn("[hybrid] Vector search failed, re-detecting backend:", err);
        setStatus("Local server unreachable — switching to static mode…", true);
        await initStaticMode();
        // fall through to whatever mode we ended up in
      }
    }

    if (state.mode === "static") {
      runStaticSearch(query);
    } else if (state.mode === "offline") {
      setStatus(
        "No search backend available. Start the local server or run `build-web`.",
        true
      );
      renderEmpty("Search is unavailable in this deployment.");
    }
  }

  async function runVectorSearch(query) {
    setStatus("Searching (AI semantic)…");
    const res = await fetchWithTimeout(
      API_BASE + "/search",
      {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ query: query, top_k: DEFAULT_TOP_K }),
      },
      8000
    );
    if (!res.ok) throw new Error("search HTTP " + res.status);
    const data = await res.json();
    const results = (data.results || []).map((r) => ({
      title: r.title,
      path: r.doc_id,
      chunk_index: r.chunk_index,
      tags: r.tags || [],
      text: r.text || "",
      score: typeof r.score === "number" ? r.score : null,
    }));
    setStatus(
      results.length
        ? "Local AI — " + results.length + " result(s) for “" + query + "”."
        : "No matches for “" + query + "”."
    );
    renderResults(results, query, true);
  }

  function runStaticSearch(query) {
    if (!state.miniSearch) {
      renderEmpty("Static index not loaded.");
      return;
    }
    setStatus("Searching (keyword)…");
    const hits = state.miniSearch.search(query).slice(0, DEFAULT_TOP_K);
    const results = hits.map((h) => {
      const full = state.docsById.get(String(h.id)) || {};
      return {
        title: h.title || full.title || "(untitled)",
        path: full.path || full.doc_id || h.id,
        chunk_index: full.chunk_index != null ? full.chunk_index : "",
        tags: Array.isArray(full.tags) ? full.tags : [],
        text: full.text || "",
        score: null, // keyword mode has no AI relevance score
      };
    });
    setStatus(
      results.length
        ? "Static keyword — " + results.length + " result(s) for “" + query + "”."
        : "No matches for “" + query + "”."
    );
    renderResults(results, query, false);
  }

  // ---------------------------------------------------------------------- //
  // Rendering
  // ---------------------------------------------------------------------- //
  function truncate(text) {
    const clean = String(text || "").replace(/\s+/g, " ").trim();
    return clean.length > SNIPPET_LIMIT ? clean.slice(0, SNIPPET_LIMIT).trimEnd() + " …" : clean;
  }

  function escapeHtml(s) {
    return String(s)
      .replace(/&/g, "&amp;")
      .replace(/</g, "&lt;")
      .replace(/>/g, "&gt;");
  }

  function highlight(text, query) {
    const safe = escapeHtml(text);
    const terms = query.trim().split(/\s+/).filter((t) => t.length >= 2);
    if (!terms.length) return safe;
    const pattern = new RegExp(
      "(" + terms.map((t) => t.replace(/[.*+?^${}()|[\]\\]/g, "\\$&")).join("|") + ")",
      "gi"
    );
    return safe.replace(pattern, "<mark>$1</mark>");
  }

  function renderEmpty(message) {
    els.results.innerHTML = '<p class="empty">' + escapeHtml(message) + "</p>";
  }

  function renderResults(results, query, isVector) {
    if (!results.length) {
      renderEmpty("No results found.");
      return;
    }
    const html = results
      .map(function (r) {
        const scoreHtml =
          isVector && r.score != null
            ? '<span class="result-score">relevance ' + r.score.toFixed(3) + "</span>"
            : "";
        const chunkLabel = r.chunk_index !== "" ? " · chunk #" + r.chunk_index : "";
        const tagsHtml = (r.tags || []).length
          ? '<div class="result-tags">' +
            r.tags.map((t) => '<span class="tag">#' + escapeHtml(t) + "</span>").join("") +
            "</div>"
          : "";
        return (
          '<article class="result-card">' +
          '<div class="result-head">' +
          '<h3 class="result-title">' +
          escapeHtml(r.title) +
          "</h3>" +
          scoreHtml +
          "</div>" +
          '<div class="result-path">' +
          escapeHtml(r.path) +
          chunkLabel +
          "</div>" +
          '<p class="result-snippet">' +
          highlight(truncate(r.text), query) +
          "</p>" +
          tagsHtml +
          "</article>"
        );
      })
      .join("");
    els.results.innerHTML = html;
  }

  // ---------------------------------------------------------------------- //
  // Minimal, safe Markdown -> HTML (headings, lists, quotes, hr, inline)
  // ---------------------------------------------------------------------- //
  function renderInline(text) {
    let s = escapeHtml(text);
    s = s.replace(/`([^`]+)`/g, "<code>$1</code>");
    s = s.replace(/\*\*([^*]+)\*\*/g, "<strong>$1</strong>");
    s = s.replace(/(^|[^*])\*([^*]+)\*/g, "$1<em>$2</em>");
    s = s.replace(/(^|[^_])_([^_]+)_/g, "$1<em>$2</em>");
    s = s.replace(
      /\[([^\]]+)\]\((https?:[^)\s]+)\)/g,
      '<a href="$2" target="_blank" rel="noopener">$1</a>'
    );
    // Dim the trailing source citation, e.g. "(Source: … )" / "(Sources: …)".
    s = s.replace(/(\(Sources?:[^)]*\))\s*$/, '<span class="cite">$1</span>');
    // Item headlines render on their own line, so drop the "— " that separated
    // a leading "**Headline** — body" from its body text.
    s = s.replace(/^(<strong>[^<]*<\/strong>)\s*[—–-]\s+/, "$1 ");
    return s;
  }

  // Reports follow PROMPT.md's plain-text layout (no #/## markers). Promote the
  // known structural lines to Markdown headings so they render as headings.
  function reportToMarkdown(raw) {
    const lines = String(raw || "").replace(/\r\n/g, "\n").split("\n");
    let titled = false;
    return lines
      .map((line) => {
        const t = line.trim();
        if (!t || /^#{1,6}\s/.test(t)) return line;
        if (/^Top\s+\d+\s+(Learnings|Trends|Concepts)\b/i.test(t)) return "## " + t;
        if (/^(Threads and tensions|Watchlist for next period)\b/i.test(t)) return "## " + t;
        if (!titled && /Reading Report\b/.test(t)) {
          titled = true;
          return "# " + t;
        }
        return line;
      })
      .join("\n");
  }

  function renderMarkdown(md) {
    const lines = String(md || "").replace(/\r\n/g, "\n").split("\n");
    const out = [];
    let listType = null; // "ul" | "ol" | null
    let para = [];
    let quote = [];

    function closeList() {
      if (listType) {
        out.push("</" + listType + ">");
        listType = null;
      }
    }
    function closePara() {
      if (para.length) {
        const joined = para.join(" ");
        const cls = /^Coverage:/i.test(joined.trim()) ? ' class="coverage"' : "";
        out.push("<p" + cls + ">" + renderInline(joined) + "</p>");
        para = [];
      }
    }
    function closeQuote() {
      if (quote.length) {
        out.push("<blockquote><p>" + renderInline(quote.join(" ")) + "</p></blockquote>");
        quote = [];
      }
    }
    function closeAll() {
      closePara();
      closeList();
      closeQuote();
    }

    for (const raw of lines) {
      const line = raw.replace(/\s+$/, "");
      if (!line.trim()) {
        closeAll();
        continue;
      }
      let m;
      if (/^\s*(-{3,}|\*{3,}|_{3,})\s*$/.test(line)) {
        closeAll();
        out.push("<hr />");
      } else if ((m = line.match(/^(#{1,6})\s+(.*)$/))) {
        closeAll();
        const level = Math.min(m[1].length, 6);
        out.push("<h" + level + ">" + renderInline(m[2]) + "</h" + level + ">");
      } else if ((m = line.match(/^\s*>\s?(.*)$/))) {
        closePara();
        closeList();
        quote.push(m[1]);
      } else if ((m = line.match(/^\s*[-*]\s+(.*)$/))) {
        closePara();
        closeQuote();
        if (listType !== "ul") {
          closeList();
          out.push("<ul>");
          listType = "ul";
        }
        out.push("<li>" + renderInline(m[1]) + "</li>");
      } else if ((m = line.match(/^\s*\d+[.)]\s+(.*)$/))) {
        closePara();
        closeQuote();
        if (listType !== "ol") {
          closeList();
          out.push("<ol>");
          listType = "ol";
        }
        out.push("<li>" + renderInline(m[1]) + "</li>");
      } else {
        closeList();
        closeQuote();
        para.push(line.trim());
      }
    }
    closeAll();
    return out.join("\n");
  }

  // ---------------------------------------------------------------------- //
  // Summary report
  // ---------------------------------------------------------------------- //
  const reportState = {
    manifest: null, // { levels: { month:[], quarter:[], year:[] } }
    cache: new Map(), // file path -> markdown string
  };

  function setReportStatus(message, isError) {
    if (!els.reportStatus) return;
    els.reportStatus.textContent = message || "";
    els.reportStatus.classList.toggle("error", Boolean(isError));
  }

  function showReportPlaceholder(message) {
    if (!els.reportBody) return;
    els.reportBody.classList.add("empty-state");
    els.reportBody.textContent = message;
  }

  async function initReport() {
    if (!els.reportLevel || !els.reportPeriod || !els.reportBody) return;
    try {
      const res = await fetch("reports/index.json", { cache: "no-cache" });
      if (!res.ok) throw new Error("index.json HTTP " + res.status);
      const manifest = await res.json();
      const levels = (manifest && manifest.levels) || {};
      const hasAny = ["month", "quarter", "year"].some(
        (k) => Array.isArray(levels[k]) && levels[k].length
      );
      if (!hasAny) throw new Error("no reports in manifest");
      reportState.manifest = manifest;
      els.reportLevel.addEventListener("change", onLevelChange);
      els.reportPeriod.addEventListener("change", onPeriodChange);
      populatePeriods();
    } catch (err) {
      console.info("[report] No pre-generated reports available.", err && err.message);
      if (els.report) {
        setReportStatus("No reports yet — run `python -m src.cli build-reports`.", true);
        showReportPlaceholder(
          "Summary reports have not been generated for this deployment."
        );
      }
    }
  }

  function currentLevelEntries() {
    const levels = (reportState.manifest && reportState.manifest.levels) || {};
    return Array.isArray(levels[els.reportLevel.value]) ? levels[els.reportLevel.value] : [];
  }

  function populatePeriods() {
    const entries = currentLevelEntries();
    els.reportPeriod.innerHTML = entries
      .map((e, i) => '<option value="' + i + '">' + escapeHtml(e.period) + "</option>")
      .join("");
    if (!entries.length) {
      showReportPlaceholder("No reports at this level.");
      setReportStatus("");
      return;
    }
    loadReport(entries[0]);
  }

  function onLevelChange() {
    populatePeriods();
  }

  function onPeriodChange() {
    const entries = currentLevelEntries();
    const entry = entries[Number(els.reportPeriod.value)];
    if (entry) loadReport(entry);
  }

  async function loadReport(entry) {
    setReportStatus("Loading " + entry.period + "…");
    try {
      let md = reportState.cache.get(entry.file);
      if (md == null) {
        const res = await fetch("reports/" + entry.file, { cache: "no-cache" });
        if (!res.ok) throw new Error("report HTTP " + res.status);
        md = await res.text();
        reportState.cache.set(entry.file, md);
      }
      els.reportBody.classList.remove("empty-state");
      els.reportBody.innerHTML = renderMarkdown(reportToMarkdown(md));
      const posts = typeof entry.post_count === "number" ? entry.post_count : "?";
      setReportStatus(entry.period + " · " + posts + " journal post(s)");
    } catch (err) {
      console.warn("[report] Failed to load report:", err);
      setReportStatus("Could not load this report.", true);
      showReportPlaceholder("Failed to load " + entry.period + ".");
    }
  }

  // ---------------------------------------------------------------------- //
  // Bootstrap
  // ---------------------------------------------------------------------- //
  function onSubmit(e) {
    e.preventDefault();
    runSearch(els.input.value);
  }

  document.addEventListener("DOMContentLoaded", function () {
    els = {
      badge: document.getElementById("mode-badge"),
      modeLabel: document.getElementById("mode-label"),
      statusLine: document.getElementById("status-line"),
      results: document.getElementById("results"),
      form: document.getElementById("search-form"),
      input: document.getElementById("search-input"),
      button: document.getElementById("search-button"),
      report: document.getElementById("report"),
      reportLevel: document.getElementById("report-level"),
      reportPeriod: document.getElementById("report-period"),
      reportStatus: document.getElementById("report-status"),
      reportBody: document.getElementById("report-body"),
    };
    if (els.form) els.form.addEventListener("submit", onSubmit);
    // Load pre-generated reports; independent of the search backend.
    initReport().catch(function (err) {
      console.error("[report] Unexpected report init error:", err);
    });
    // Detect backend; never let a failure bubble up uncaught.
    detectMode().catch(function (err) {
      console.error("[hybrid] Unexpected detection error:", err);
      state.mode = "offline";
      setBadge("offline");
      setStatus("Initialization error — see console.", true);
    });
  });
})();

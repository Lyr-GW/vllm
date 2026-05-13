/* vLLM Wiki — search.js
 * Tiny front-end search using a static JSON index.
 * Loads relative to the current page; works for index.html and pages/*.html.
 */
(function () {
  "use strict";

  function indexUrl() {
    // Pages live under pages/, index.html at root. Detect and resolve.
    const path = location.pathname;
    if (path.includes("/pages/")) return "../assets/search-index.json";
    return "assets/search-index.json";
  }

  function relUrl(target) {
    // target is a path relative to wiki root, e.g. "pages/01-introduction.html"
    const path = location.pathname;
    if (path.includes("/pages/")) {
      if (target.startsWith("pages/")) return "../" + target;
      return "../" + target;
    }
    return target;
  }

  function tokenize(s) {
    return (s || "")
      .toLowerCase()
      .split(/[\s,/.()\[\]<>"'`：，。、；！？「」]+/u)
      .filter((t) => t.length > 0);
  }

  function score(entry, qTokens) {
    let s = 0;
    const titleL = (entry.title || "").toLowerCase();
    const sumL = (entry.summary || "").toLowerCase();
    const kw = (entry.keywords || []).map((k) => k.toLowerCase());
    qTokens.forEach((qt) => {
      if (titleL.includes(qt)) s += 6;
      if (kw.some((k) => k.includes(qt))) s += 4;
      if (sumL.includes(qt)) s += 2;
    });
    return s;
  }

  let INDEX = null;
  function ensureIndex() {
    if (INDEX) return Promise.resolve(INDEX);
    return fetch(indexUrl())
      .then((r) => r.json())
      .then((data) => {
        INDEX = data;
        return data;
      })
      .catch(() => {
        INDEX = [];
        return [];
      });
  }

  function render(results, box) {
    if (!results.length) {
      box.innerHTML = '<div class="sr-empty">未找到匹配结果</div>';
      box.classList.add("open");
      return;
    }
    box.innerHTML = results
      .slice(0, 8)
      .map((r) => {
        const url = relUrl(r.url);
        return (
          '<a href="' +
          url +
          '">' +
          '<div class="sr-title">' +
          escapeHtml(r.title) +
          "</div>" +
          (r.summary
            ? '<div class="sr-summary">' + escapeHtml(r.summary) + "</div>"
            : "") +
          "</a>"
        );
      })
      .join("");
    box.classList.add("open");
  }

  function escapeHtml(s) {
    return (s || "").replace(/[&<>"']/g, function (c) {
      return {
        "&": "&amp;",
        "<": "&lt;",
        ">": "&gt;",
        '"': "&quot;",
        "'": "&#39;",
      }[c];
    });
  }

  document.addEventListener("DOMContentLoaded", function () {
    const input = document.querySelector(".search-box input");
    const box = document.querySelector(".search-results");
    if (!input || !box) return;

    let timer = null;
    function run() {
      const q = input.value.trim();
      if (!q) {
        box.classList.remove("open");
        return;
      }
      ensureIndex().then((idx) => {
        const tokens = tokenize(q);
        const ranked = idx
          .map((e) => ({ e, s: score(e, tokens) }))
          .filter((x) => x.s > 0)
          .sort((a, b) => b.s - a.s)
          .map((x) => x.e);
        render(ranked, box);
      });
    }
    input.addEventListener("input", () => {
      clearTimeout(timer);
      timer = setTimeout(run, 80);
    });
    input.addEventListener("focus", run);
    document.addEventListener("click", (e) => {
      if (!box.contains(e.target) && e.target !== input)
        box.classList.remove("open");
    });
    input.addEventListener("keydown", (e) => {
      if (e.key === "Escape") box.classList.remove("open");
    });
  });
})();

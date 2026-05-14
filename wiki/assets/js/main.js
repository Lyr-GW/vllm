/* vLLM Wiki — main.js
 * - Inject shared header / sidebar / footer / prev-next (so each HTML file
 *   only carries its unique content).
 * - Theme toggle (persisted in localStorage)
 * - Mobile sidebar toggle
 * - TOC scrollspy
 * - highlight.js / Mermaid initialization
 */
(function () {
  "use strict";

  // -------------------- Page registry --------------------
  // Order here is the canonical order shown in the sidebar and used by prev/next.
  const PAGES = [
    { id: "introduction",          file: "01-introduction.html",          title: "项目简介",                group: "入门" },
    { id: "architecture",          file: "02-architecture.html",          title: "整体架构",                group: "入门" },
    { id: "installation",          file: "03-installation.html",          title: "安装与环境",              group: "入门" },
    { id: "quickstart",            file: "04-quickstart.html",            title: "快速上手",                group: "入门" },

    { id: "paged-attention",       file: "05-paged-attention.html",       title: "PagedAttention 原理",     group: "核心原理" },
    { id: "kv-cache",              file: "06-kv-cache.html",              title: "KV Cache 与 Prefix Cache",group: "核心原理" },
    { id: "rope",                  file: "23-rope.html",                  title: "RoPE 旋转位置编码",        group: "核心原理" },
    { id: "continuous-batching",   file: "07-continuous-batching.html",   title: "连续批处理 & Chunked",    group: "核心原理" },
    { id: "scheduler",             file: "08-scheduler.html",             title: "调度器与执行循环",        group: "核心原理" },

    { id: "quantization",          file: "09-quantization.html",          title: "量化",                    group: "关键特性" },
    { id: "parallelism",           file: "10-parallelism.html",           title: "分布式与并行",            group: "关键特性" },
    { id: "speculative-decoding",  file: "11-speculative-decoding.html",  title: "投机解码",                group: "关键特性" },
    { id: "lora",                  file: "12-lora.html",                  title: "LoRA 多适配器",           group: "关键特性" },
    { id: "multimodal",            file: "13-multimodal.html",            title: "多模态",                  group: "关键特性" },
    { id: "api-server",            file: "14-api-server.html",            title: "OpenAI 兼容 API",         group: "使用面" },
    { id: "sampling",              file: "15-sampling.html",              title: "采样与结构化输出",        group: "使用面" },
    { id: "cuda-graph-compile",    file: "16-cuda-graph-compile.html",    title: "CUDA Graph & torch.compile", group: "性能" },

    { id: "hardware-support",      file: "17-hardware-support.html",      title: "硬件后端支持",            group: "性能" },
    { id: "benchmarking",          file: "18-benchmarking.html",          title: "基准与调优",              group: "性能" },

    { id: "source-tree",           file: "19-source-tree.html",           title: "源码目录速查",            group: "参考" },
    { id: "contributing",          file: "20-contributing.html",          title: "贡献指南",                group: "参考" },
    { id: "faq",                   file: "21-faq.html",                   title: "常见问答 FAQ",            group: "参考" },
    { id: "glossary",              file: "22-glossary.html",              title: "术语表",                  group: "参考" },
  ];

  function isUnderPages() {
    return location.pathname.replace(/\\/g, "/").includes("/pages/");
  }
  function rootHref(p) {
    // p is relative to wiki root, e.g. "index.html" or "pages/xxx.html" or "assets/img/logo.svg"
    return isUnderPages() ? "../" + p : p;
  }
  function pageHref(file) {
    return isUnderPages() ? file : "pages/" + file;
  }

  // -------------------- Theme --------------------
  const THEME_KEY = "vllm-wiki-theme";
  function applyTheme(t) {
    document.documentElement.setAttribute("data-theme", t);
  }
  function initTheme() {
    let t = localStorage.getItem(THEME_KEY);
    if (!t) {
      t = window.matchMedia("(prefers-color-scheme: dark)").matches
        ? "dark"
        : "light";
    }
    applyTheme(t);
  }
  function toggleTheme() {
    const cur = document.documentElement.getAttribute("data-theme") || "light";
    const next = cur === "dark" ? "light" : "dark";
    applyTheme(next);
    localStorage.setItem(THEME_KEY, next);
    if (window.mermaid && window.mermaid.initialize) {
      try {
        window.mermaid.initialize({
          startOnLoad: false,
          theme: next === "dark" ? "dark" : "default",
          securityLevel: "loose",
        });
        document.querySelectorAll(".mermaid").forEach((el) => {
          if (el.dataset.src) el.innerHTML = el.dataset.src;
          el.removeAttribute("data-processed");
        });
        window.mermaid.run({ querySelector: ".mermaid" });
      } catch (e) {}
    }
  }
  initTheme();

  // -------------------- Header --------------------
  function injectHeader() {
    const slot = document.getElementById("header");
    if (!slot) return;
    slot.outerHTML = `
<header class="site-header">
  <button class="menu-toggle" aria-label="切换侧边栏">☰</button>
  <a class="brand" href="${rootHref("index.html")}">
    <img src="${rootHref("assets/img/logo.svg")}" alt="vLLM Wiki"/>
    <span>vLLM Wiki <small>· 中文</small></span>
  </a>
  <nav class="top-links">
    <div class="search-box">
      <input type="search" placeholder="搜索 wiki…" aria-label="搜索"/>
      <div class="search-results"></div>
    </div>
    <a href="https://github.com/vllm-project/vllm" target="_blank" rel="noopener">GitHub ↗</a>
    <button class="theme-toggle" title="切换明暗主题" aria-label="切换主题">◐</button>
  </nav>
</header>`;
  }

  // -------------------- Sidebar --------------------
  function injectSidebar() {
    const slot = document.getElementById("sidebar");
    if (!slot) return;
    const groups = {};
    PAGES.forEach((p) => {
      groups[p.group] = groups[p.group] || [];
      groups[p.group].push(p);
    });
    let html = `<aside class="sidebar"><h4>导航</h4><ul><li><a href="${rootHref(
      "index.html"
    )}">🏠 首页</a></li></ul>`;
    Object.keys(groups).forEach((g) => {
      html += `<h4>${g}</h4><ul>`;
      groups[g].forEach((p) => {
        html += `<li><a href="${pageHref(p.file)}">${p.title}</a></li>`;
      });
      html += `</ul>`;
    });
    html += `</aside>`;
    slot.outerHTML = html;
  }

  // -------------------- Footer --------------------
  function injectFooter() {
    const slot = document.getElementById("footer");
    if (!slot) return;
    slot.outerHTML = `
<footer class="site-footer">
  <div>vLLM Wiki · 中文社区学习资料 · 内容基于 <a href="https://github.com/vllm-project/vllm" target="_blank" rel="noopener">vllm-project/vllm</a> 整理</div>
  <div style="margin-top:6px;font-size:0.82rem;">非官方文档，仅供学习参考；vLLM 代码遵循 Apache-2.0 协议</div>
</footer>`;
  }

  // -------------------- Prev / Next --------------------
  function injectPrevNext() {
    const slot = document.getElementById("prev-next");
    if (!slot) return;
    const cur = document.body.dataset.page;
    const idx = PAGES.findIndex((p) => p.id === cur);
    if (idx === -1) {
      slot.style.display = "none";
      return;
    }
    const prev = idx > 0 ? PAGES[idx - 1] : null;
    const next = idx < PAGES.length - 1 ? PAGES[idx + 1] : null;
    let html = '<nav class="prev-next">';
    if (prev) {
      html += `<a class="prev" href="${pageHref(prev.file)}"><span class="label">← 上一页</span>${prev.title}</a>`;
    } else {
      html += `<a class="prev" href="${rootHref("index.html")}"><span class="label">← 上一页</span>首页</a>`;
    }
    if (next) {
      html += `<a class="next" href="${pageHref(next.file)}"><span class="label">下一页 →</span>${next.title}</a>`;
    } else {
      html += `<span class="next" style="opacity:0.4"></span>`;
    }
    html += "</nav>";
    slot.outerHTML = html;
  }

  // -------------------- Active link highlight --------------------
  function highlightSidebar() {
    const here = location.pathname.split("/").pop() || "index.html";
    document.querySelectorAll(".sidebar a").forEach((a) => {
      const href = (a.getAttribute("href") || "").split("/").pop();
      if (href === here) a.classList.add("active");
    });
  }

  // -------------------- TOC --------------------
  function buildToc() {
    const tocEl = document.querySelector("aside.toc ul");
    if (!tocEl) return;
    const content = document.querySelector("main.content");
    if (!content) return;
    const headings = content.querySelectorAll("h2, h3");
    if (!headings.length) {
      const aside = document.querySelector("aside.toc");
      if (aside) aside.style.display = "none";
      const wrap = document.querySelector(".main-wrap");
      if (wrap) wrap.classList.add("no-toc");
      return;
    }
    headings.forEach((h, i) => {
      if (!h.id) {
        let base = h.textContent
          .trim()
          .toLowerCase()
          .replace(/\s+/g, "-")
          .replace(/[^a-z0-9\-_\u4e00-\u9fff]/g, "");
        if (!base) base = "h-" + i;
        h.id = base;
      }
      const li = document.createElement("li");
      if (h.tagName === "H3") li.className = "h3";
      const a = document.createElement("a");
      a.href = "#" + h.id;
      a.textContent = h.textContent;
      li.appendChild(a);
      tocEl.appendChild(li);
    });

    const links = tocEl.querySelectorAll("a");
    const map = new Map();
    links.forEach((l) => map.set(l.getAttribute("href").slice(1), l));
    const io = new IntersectionObserver(
      (entries) => {
        entries.forEach((e) => {
          const link = map.get(e.target.id);
          if (!link) return;
          if (e.isIntersecting) {
            links.forEach((l) => l.classList.remove("active"));
            link.classList.add("active");
          }
        });
      },
      { rootMargin: "-30% 0px -60% 0px", threshold: 0 }
    );
    headings.forEach((h) => io.observe(h));
  }

  // -------------------- Mobile sidebar --------------------
  function initMobileSidebar() {
    const btn = document.querySelector(".menu-toggle");
    const sb = document.querySelector(".sidebar");
    if (!btn || !sb) return;
    let bd = document.createElement("div");
    bd.className = "sidebar-backdrop";
    document.body.appendChild(bd);
    function close() {
      sb.classList.remove("open");
      bd.classList.remove("open");
    }
    btn.addEventListener("click", (e) => {
      e.preventDefault();
      sb.classList.toggle("open");
      bd.classList.toggle("open");
    });
    bd.addEventListener("click", close);
  }

  // -------------------- Theme toggle button --------------------
  function bindThemeToggle() {
    const tgl = document.querySelector(".theme-toggle");
    if (tgl) tgl.addEventListener("click", toggleTheme);
  }

  // -------------------- Init --------------------
  document.addEventListener("DOMContentLoaded", () => {
    injectHeader();
    injectSidebar();
    injectFooter();
    injectPrevNext();
    highlightSidebar();
    buildToc();
    initMobileSidebar();
    bindThemeToggle();

    if (window.hljs) {
      try { window.hljs.highlightAll(); } catch (e) {}
    }
    if (window.mermaid && window.mermaid.initialize) {
      try {
        const isDark =
          document.documentElement.getAttribute("data-theme") === "dark";
        window.mermaid.initialize({
          startOnLoad: false,
          theme: isDark ? "dark" : "default",
          securityLevel: "loose",
        });
        document.querySelectorAll(".mermaid").forEach((el) => {
          el.dataset.src = el.textContent;
        });
        window.mermaid.run({ querySelector: ".mermaid" });
      } catch (e) {}
    }
  });
})();

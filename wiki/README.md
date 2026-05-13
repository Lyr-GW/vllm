# vLLM 中文 Wiki

本目录是 [vLLM](https://github.com/vllm-project/vllm) 的中文学习与速查 Wiki，
**纯静态 HTML**，无需任何构建工具，双击 `index.html` 即可在浏览器打开。

> 内容为社区学习材料，并非官方文档。权威信息请以 <https://docs.vllm.ai> 为准。

## 目录结构

```
wiki/
├── index.html              # 首页
├── pages/                  # 22 个内容页（01-introduction.html … 22-glossary.html）
├── assets/
│   ├── css/style.css       # 共享样式（明/暗双主题，响应式）
│   ├── js/main.js          # 注入 header/sidebar/footer、主题切换、TOC scrollspy
│   ├── js/search.js        # 简单前端搜索
│   ├── search-index.json   # 搜索索引
│   └── img/logo.svg
└── .nojekyll               # 告知 GitHub Pages 不要做 Jekyll 处理
```

## 本地预览

```bash
# 方式 1：直接打开
xdg-open index.html        # Linux
open index.html            # macOS

# 方式 2：本地静态服务器（搜索功能需要这种方式才能 fetch JSON）
python3 -m http.server 8000 -d .
# 然后访问 http://localhost:8000/
```

## 在 GitHub Pages 上发布

仓库已自带一个 workflow：`.github/workflows/wiki-pages.yml`。
它使用 `actions/configure-pages@v5` 的 `enablement: true` 在首次运行时
**自动启用 Pages**（Source 自动设为 GitHub Actions），无需手动点 Settings。

唯一可能需要你手动调一次的地方：
**Settings → Environments → `github-pages` → "Deployment branches and tags"**

- 默认仅允许从默认分支（`main`）部署；
- 如果想在合并前先用本分支（`cursor/llm-wiki-a6e3`）预览，把它加进允许列表，
  或临时选 "All branches"。

之后只要修改 `wiki/**` 下任何文件并推送到允许的分支，
Action 就会自动把整个 `wiki/` 部署到 Pages。

- 站点 URL 模板：`https://<owner>.github.io/<repo>/`
  - 例如 fork 在 `Lyr-GW/vllm`，URL 即 `https://lyr-gw.github.io/vllm/`。
- 也可以在 Actions 标签页用 "Run workflow" 手动触发。

## 修改/扩展

- 新增页面：在 `pages/` 下加 HTML 文件，并在 `assets/js/main.js` 的 `PAGES`
  数组与 `assets/search-index.json` 中各加一条。
- 修改样式：编辑 `assets/css/style.css` 即可，所有页面共享。
- 改主题色：调整 `:root` 和 `[data-theme="dark"]` 下的 CSS 变量。

## 协议

Wiki 文字内容采用 [CC BY-SA 4.0](https://creativecommons.org/licenses/by-sa/4.0/)，
站点代码（HTML/CSS/JS）随主仓库一同采用 Apache-2.0。

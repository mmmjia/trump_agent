# agents/agent_info_collector.py
"""
Agent 1: 信息采集器
职责：从维基百科和 Google 搜索收集公众人物的原始传记文本
输出：合并后的原始文本字符串，供 Agent 2（事件提取器）使用
无 LLM 依赖，纯网络 I/O。
"""

import re
import requests
from typing import Optional, List


class InfoCollectorAgent:
    """
    从公开网络来源（Wikipedia、Google）采集人物传记文本。
    返回合并的原始文本，不做任何 LLM 处理。
    所有网络请求均在失败时优雅降级，不抛出异常。
    """

    WIKI_ACTION_API = "https://en.wikipedia.org/w/api.php"
    WIKI_SUMMARY_API = "https://en.wikipedia.org/api/rest_v1/page/summary/{name}"
    HEADERS = {
        "User-Agent": (
            "Mozilla/5.0 (compatible; PsychoBiographyAgent/1.0)"
        )
    }

    def collect(self, figure_name: str, use_google: bool = True) -> str:
        """
        主入口：收集人物信息并返回合并文本。
        优先使用 Wikipedia 完整条目，再附加 Google 搜索结果。

        Args:
            figure_name: 公众人物姓名（英文）
            use_google: 是否尝试 Google 搜索补充（需要 gsearchpy）
        Returns:
            合并的原始文本字符串
        """
        parts = []

        wiki_text = self._fetch_wikipedia_full(figure_name)
        if wiki_text:
            parts.append(wiki_text)
        else:
            summary = self._fetch_wikipedia_summary(figure_name)
            if summary:
                parts.append(summary)

        if use_google:
            google_text = self._fetch_google(figure_name)
            if google_text:
                parts.append(google_text)

        combined = "\n\n---\n\n".join(parts)
        print(f"[InfoCollector] 采集完成：{len(combined)} 字符")
        return combined

    def collect_supplemental(self, figure_name: str, query: str) -> str:
        """
        补充搜索：用于事件验证循环中针对特定主题的精准搜索。
        优先 Google 抓取，Google 无结果时自动回退到 Wikipedia 搜索 API。

        Args:
            figure_name: 人物姓名（仅用于日志）
            query: 具体搜索关键词
        Returns:
            搜索结果文本（空字符串表示无结果）
        """
        # ── 1. 优先 Google ──
        text = self._fetch_google(query)
        if text:
            return text

        # ── 2. 回退：Wikipedia 全文搜索 ──
        print(f"[InfoCollector] Google 无结果，回退 Wikipedia 搜索：{query}")
        text = self._fetch_wikipedia_search(query)
        if text:
            return text

        print(f"[InfoCollector] 补充搜索无结果：{query}")
        return ""

    def _fetch_wikipedia_search(self, query: str, max_results: int = 3) -> str:
        """
        Wikipedia Action API 全文搜索：搜索 query，抓取 top-N 命中条目的完整正文。
        适合关键词不是精确标题的情况（比直接按名称查效果更好）。
        """
        search_params = {
            "action": "query",
            "list": "search",
            "srsearch": query,
            "srlimit": max_results,
            "srwhat": "text",
            "format": "json",
        }
        try:
            resp = requests.get(
                self.WIKI_ACTION_API,
                params=search_params,
                headers=self.HEADERS,
                timeout=15,
            )
            resp.raise_for_status()
            hits = resp.json().get("query", {}).get("search", [])
            if not hits:
                return ""

            parts = []
            for hit in hits:
                title = hit.get("title", "")
                if not title:
                    continue
                extract = self._fetch_wikipedia_full(title)
                if extract:
                    parts.append(extract)
                    print(f"[InfoCollector] Wikipedia 补充条目: {title} ({len(extract)} 字符)")

            return "\n\n---\n\n".join(parts) if parts else ""
        except Exception as e:
            print(f"[InfoCollector] Wikipedia 搜索失败: {e}")
            return ""

    def _fetch_wikipedia_full(self, figure_name: str) -> str:
        """调用 Wikipedia Action API 获取完整条目纯文本"""
        params = {
            "action": "query",
            "titles": figure_name.replace(" ", "_"),
            "prop": "extracts",
            "explaintext": True,
            "exsectionformat": "plain",
            "format": "json",
        }
        try:
            resp = requests.get(
                self.WIKI_ACTION_API,
                params=params,
                headers=self.HEADERS,
                timeout=15,
            )
            resp.raise_for_status()
            pages = resp.json().get("query", {}).get("pages", {})
            for page in pages.values():
                extract = page.get("extract", "")
                if extract:
                    return f"Wikipedia ({figure_name}):\n{extract}"
        except Exception as e:
            print(f"[InfoCollector] Wikipedia 完整文章抓取失败: {e}")
        return ""

    def _fetch_wikipedia_summary(self, figure_name: str) -> str:
        """降级：Wikipedia REST summary 接口"""
        url = self.WIKI_SUMMARY_API.format(name=figure_name.replace(" ", "_"))
        try:
            resp = requests.get(url, headers=self.HEADERS, timeout=10)
            resp.raise_for_status()
            extract = resp.json().get("extract", "")
            if extract:
                return f"Wikipedia Summary ({figure_name}):\n{extract}"
        except Exception as e:
            print(f"[InfoCollector] Wikipedia 摘要抓取失败: {e}")
        return ""

    def _fetch_google(self, query: str, num_results: int = 5, chars_per_page: int = 10000) -> str:
        """
        Web 搜索 + 逐页抓取正文。
        使用 DuckDuckGo（Google 在无头环境中被 CAPTCHA 拦截，DDG 更可靠）。
        依赖：pip install ddgs
        """
        urls = self._ddg_search_urls(query, num_results)
        if not urls:
            return ""

        print(f"[InfoCollector] DDG 找到 {len(urls)} 个 URL，开始抓取...")
        parts = []
        for url in urls:
            text = self._fetch_page_text(url, max_chars=chars_per_page)
            if text:
                parts.append(f"[来源: {url}]\n{text}")
                print(f"[InfoCollector] 已抓取页面: {url} ({len(text)} 字符)")

        if not parts:
            return ""

        return f"Web Search ({query}):\n\n" + "\n\n---\n\n".join(parts)

    def _ddg_search_urls(self, query: str, num_results: int) -> List[str]:
        """
        用 DuckDuckGo 搜索，返回英文结果 URL 列表。
        优先用新包名 ddgs，回退到旧包名 duckduckgo_search。
        """
        # ── 方案 A：ddgs（新包名，pip install ddgs）──
        for module_name, cls_name in [("ddgs", "DDGS"), ("duckduckgo_search", "DDGS")]:
            try:
                mod = __import__(module_name, fromlist=[cls_name])
                DDGS = getattr(mod, cls_name)
                with DDGS() as ddgs:
                    hits = list(ddgs.text(
                        query,
                        max_results=num_results,
                        region="wt-wt",   # 全球英文结果，不受本地 locale 影响
                    ))
                urls = [h["href"] for h in hits if h.get("href")]
                print(f"[InfoCollector] [DDG] 搜索完成，获得 {len(urls)} 个 URL")
                return urls
            except ImportError:
                continue
            except Exception as e:
                print(f"[InfoCollector] [DDG/{module_name}] 搜索失败: {e}")
                return []

        print("[InfoCollector] 未安装 DDG 搜索库，运行: pip install ddgs")
        return []

    def _extract_urls_from_html(self, html: str, max_urls: int = 3) -> List[str]:
        """从 Google 搜索结果 HTML 中提取结果页面 URL（过滤 Google 自身链接）"""
        try:
            from bs4 import BeautifulSoup
            soup = BeautifulSoup(html, "html.parser")
            seen, urls = set(), []
            for a in soup.find_all("a", href=True):
                href = a["href"]
                # 跳过 Google 内部链接和常见非文章链接
                if not href.startswith("http"):
                    continue
                if any(skip in href for skip in ("google.", "youtube.", "javascript:", "#")):
                    continue
                if href not in seen:
                    seen.add(href)
                    urls.append(href)
                if len(urls) >= max_urls:
                    break
            return urls
        except ImportError:
            # beautifulsoup4 未安装，用正则简单提取
            return re.findall(r'href=["\']?(https?://(?!(?:www\.)?google\.)[^\s"\'<>]+)', html)[:max_urls]

    def _fetch_page_text(self, url: str, max_chars: int = 3000) -> str:
        """抓取目标页面并提取纯文本正文"""
        try:
            resp = requests.get(url, headers=self.HEADERS, timeout=10)
            resp.raise_for_status()
            try:
                from bs4 import BeautifulSoup
                soup = BeautifulSoup(resp.text, "html.parser")
                for tag in soup(["script", "style", "nav", "footer", "header", "aside"]):
                    tag.decompose()
                text = soup.get_text(separator=" ", strip=True)
            except ImportError:
                # 无 BeautifulSoup：简单去除 HTML 标签
                text = re.sub(r"<[^>]+>", " ", resp.text)
                text = re.sub(r"\s+", " ", text).strip()
            return text[:max_chars]
        except Exception as e:
            print(f"[InfoCollector] 页面抓取失败 ({url}): {e}")
            return ""

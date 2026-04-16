#!/usr/bin/env python
"""
诊断脚本：测试 Google 搜索是否可用
运行方法：python scripts/test_google_search.py
"""

import sys

QUERY = "Donald Trump early life Queens New York"

print("=" * 60)
print(f"测试查询: {QUERY}")
print("=" * 60)

# ── 1. googlesearch-python ──────────────────────────────────────
print("\n[1] googlesearch-python")
try:
    from googlesearch import search
    results = list(search(QUERY, num_results=3, lang="en"))
    if results:
        print(f"  ✓ 返回 {len(results)} 个 URL:")
        for r in results:
            print(f"    {r}")
    else:
        print("  ✗ 返回 0 个结果（可能被 Google CAPTCHA 拦截）")
except ImportError:
    print("  ✗ 未安装，运行: pip install googlesearch-python")
except Exception as e:
    print(f"  ✗ 异常: {type(e).__name__}: {e}")

# ── 2. ddgs（新包名）/ duckduckgo-search（旧包名）────────────────
print("\n[2] ddgs / duckduckgo-search  (region=wt-wt 强制英文结果)")
found = False
for mod_name in ("ddgs", "duckduckgo_search"):
    try:
        mod = __import__(mod_name, fromlist=["DDGS"])
        DDGS = getattr(mod, "DDGS")
        with DDGS() as ddgs:
            results = list(ddgs.text(QUERY, max_results=3, region="wt-wt"))
        if results:
            print(f"  ✓ [{mod_name}] 返回 {len(results)} 个结果:")
            for r in results:
                print(f"    [{r.get('title','')[:50]}]")
                print(f"     {r.get('href','')}")
        else:
            print(f"  ✗ [{mod_name}] 返回 0 个结果")
        found = True
        break
    except ImportError:
        print(f"  – [{mod_name}] 未安装")
    except Exception as e:
        print(f"  ✗ [{mod_name}] 异常: {type(e).__name__}: {e}")
        found = True
        break
if not found:
    print("  ✗ 两个包都未安装，运行: pip install ddgs")

# ── 3. gsearchpy ─────────────────────────────────────────────────
print("\n[3] gsearchpy")
try:
    from gsearchpy.google import GoogleScraper
    scraper = GoogleScraper()
    html = scraper.google_search(QUERY)
    if html:
        import re
        urls = re.findall(r'href=["\']?(https?://(?!(?:www\.)?google\.)[^\s"\'<>]+)', html)
        print(f"  HTML 长度: {len(html)} 字符，提取到 {len(urls)} 个 URL")
        for u in urls[:3]:
            print(f"    {u}")
        if not urls:
            print("  ✗ HTML 内无外部 URL（可能被拦截，返回了 consent/CAPTCHA 页）")
            # 打印 HTML 片段帮助诊断
            snippet = html[:500].replace('\n', ' ')
            print(f"  HTML 片段: {snippet}")
    else:
        print("  ✗ 返回空 HTML")
except ImportError:
    print("  ✗ 未安装")
except Exception as e:
    print(f"  ✗ 异常: {type(e).__name__}: {e}")

# ── 4. 直接 requests 测试 Google 连通性 ─────────────────────────
print("\n[4] 直接 requests 访问 Google")
try:
    import requests
    headers = {"User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) Chrome/120.0.0.0 Safari/537.36"}
    resp = requests.get("https://www.google.com/search?q=test&num=3&hl=en", headers=headers, timeout=10)
    print(f"  HTTP 状态码: {resp.status_code}  内容长度: {len(resp.text)} 字符")
    if "captcha" in resp.text.lower() or "unusual traffic" in resp.text.lower():
        print("  ✗ Google 检测到异常流量（CAPTCHA）")
    elif "enablejs" in resp.text or len(resp.text) < 5000:
        print("  ✗ Google 要求启用 JavaScript（返回了 consent 页，非搜索结果）")
    else:
        from bs4 import BeautifulSoup
        soup = BeautifulSoup(resp.text, "html.parser")
        links = [a["href"] for a in soup.find_all("a", href=True)
                 if a["href"].startswith("http") and "google" not in a["href"]]
        print(f"  找到 {len(links)} 个外部链接")
        for l in links[:3]:
            print(f"    {l}")
except Exception as e:
    print(f"  ✗ 异常: {type(e).__name__}: {e}")

print("\n" + "=" * 60)
print("Python:", sys.executable)
print("=" * 60)

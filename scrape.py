#!/usr/bin/env python3
"""
纯抓取：把页面拿下来 → 存档 → 切成行 → 存 CSV
不做任何分类、日期解析或统计。

用法:
    python scrape.py https://www.tradecomplianceresourcehub.com/2026/09/02/trump-2-0-tariff-tracker/
    python scrape.py tracker.html                 # 浏览器存下来的本地文件
    python scrape.py <url> --out data/             # 指定输出目录

产出:
    data/raw/<timestamp>_<slug>.html   原始 HTML 存档（研究可复现的前提）
    data/rows.csv                      切好的行：source, tag, path, text
"""

import argparse
import hashlib
import re
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from urllib.parse import urlparse

import pandas as pd
import requests
from bs4 import BeautifulSoup

# ========================================================================
# 配置 CONFIG —— 要抓哪个网址就改这里
# 不改也行：运行时用 `python scrape.py <网址或文件>` 传参会覆盖这个默认值
# ========================================================================
DEFAULT_URL = "https://www.tradecomplianceresourcehub.com/2026/09/02/trump-2-0-tariff-tracker/"

BROWSER_HEADERS = {
    "User-Agent": ("Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
                   "AppleWebKit/537.36 (KHTML, like Gecko) "
                   "Chrome/122.0.0.0 Safari/537.36"),
    "Accept": ("text/html,application/xhtml+xml,application/xml;q=0.9,"
               "image/avif,image/webp,*/*;q=0.8"),
    "Accept-Language": "en-US,en;q=0.9",
    "Accept-Encoding": "gzip, deflate, br",
    "Connection": "keep-alive",
    "Upgrade-Insecure-Requests": "1",
    "Sec-Fetch-Dest": "document",
    "Sec-Fetch-Mode": "navigate",
    "Sec-Fetch-Site": "none",
}

BLOCKED = {403, 429, 503, 521, 522, 523, 525}


# ========================================================================
# 抓取 FETCH —— 拿网页 / 本地文件，带重试和存档
# ========================================================================

def slugify(url: str) -> str:
    p = urlparse(url)
    s = re.sub(r"[^a-z0-9]+", "-", (p.netloc + p.path).lower()).strip("-")
    return s[:80] or hashlib.md5(url.encode()).hexdigest()[:12]


def fetch(url: str, tries: int = 3, wait: float = 2.0) -> str:
    """带重试的请求。被拦就明确告诉用户怎么绕。"""
    sess = requests.Session()
    sess.headers.update(BROWSER_HEADERS)

    last = None
    for i in range(1, tries + 1):
        try:
            r = sess.get(url, timeout=25)
            last = r.status_code
            print(f"  尝试 {i}/{tries}: HTTP {r.status_code}  ({len(r.content):,} bytes)")
            if r.status_code == 200:
                if not r.encoding or r.encoding.lower() == "iso-8859-1":
                    r.encoding = r.apparent_encoding
                return r.text
            if r.status_code not in BLOCKED:
                break
        except requests.RequestException as e:
            print(f"  尝试 {i}/{tries}: {type(e).__name__}")
            last = None
        if i < tries:
            time.sleep(wait * i)          # 退避

    print(f"\n✗ 抓不下来（最后状态 {last}）。这站在 Cloudflare 后面。")
    print("  按成本从低到高的三条路：")
    print("   1. 浏览器打开页面 → Ctrl+S(Cmd+S) 存成 tracker.html → python scrape.py tracker.html")
    print("   2. 装 Playwright 用真浏览器：pip install playwright && playwright install chromium")
    print("   3. 找该站有没有 RSS / sitemap.xml，往往不设防")
    sys.exit(1)


def load(src: str) -> tuple[str, str]:
    """返回 (html, 来源标识)"""
    p = Path(src)
    if p.exists():
        print(f"→ 读本地文件 {p}")
        return p.read_text(encoding="utf-8", errors="ignore"), p.name
    print(f"→ 请求 {src}")
    return fetch(src), src


def archive(html: str, src: str, outdir: Path) -> Path:
    """把原始 HTML 存下来。抓取的东西会变，存档是复现的唯一保证。"""
    raw = outdir / "raw"
    raw.mkdir(parents=True, exist_ok=True)
    ts = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    path = raw / f"{ts}_{slugify(src)}.html"
    path.write_text(html, encoding="utf-8")
    return path


# ========================================================================
# 解析 PARSE —— 把 HTML 切成一行行 (source, tag, path, text)
# ========================================================================

def dom_path(el) -> str:
    """给每行记一个 DOM 路径，方便回头核对是从页面哪块抓的"""
    parts = []
    for p in list(el.parents)[:4][::-1]:
        if p.name in (None, "[document]", "html", "body"):
            continue
        cls = p.get("class")
        parts.append(p.name + ("." + cls[0] if cls else ""))
    parts.append(el.name)
    return ">".join(parts)


def to_rows(html: str, source: str) -> pd.DataFrame:
    soup = BeautifulSoup(html, "html.parser")
    for t in soup(["script", "style", "noscript", "svg"]):
        t.decompose()

    rows, seen = [], set()

    # 优先按表格行取——tracker 类页面基本都是表格
    for tr in soup.find_all("tr"):
        cells = [td.get_text(" ", strip=True) for td in tr.find_all(["td", "th"])]
        text = " | ".join(c for c in cells if c)
        if len(text) > 15 and text not in seen:
            seen.add(text)
            rows.append({"source": source, "tag": "tr", "n_cells": len(cells),
                         "path": dom_path(tr), "text": text})

    # 再补列表项和段落
    for el in soup.find_all(["li", "p"]):
        if el.find_parent("tr"):          # 已经在表格里取过了
            continue
        text = el.get_text(" ", strip=True)
        if len(text) > 15 and text not in seen:
            seen.add(text)
            rows.append({"source": source, "tag": el.name, "n_cells": 0,
                         "path": dom_path(el), "text": text})

    return pd.DataFrame(rows)


# ========================================================================
# 诊断与预览 DIAGNOSE / PREVIEW —— 抓完之后打印出来看看抓得对不对
# ========================================================================

def diagnose(html: str, df: pd.DataFrame) -> None:
    soup = BeautifulSoup(html, "html.parser")
    for t in soup(["script", "style"]):
        t.decompose()
    visible = len(soup.get_text(strip=True))

    print(f"\n{'='*50}")
    print(f"  HTML {len(html):,} 字符 → 可见文字 {visible:,} 字符")
    print(f"  切出 {len(df):,} 行")
    if len(df):
        print(f"  行长中位数 {int(df.text.str.len().median())} 字符")
        print("\n  按标签：")
        for tag, n in df.tag.value_counts().items():
            print(f"    {tag:<6}{n:>6}")
        print("\n  行数最多的 DOM 位置：")
        for path, n in df.path.value_counts().head(5).items():
            print(f"    {n:>5}  {path}")
    print(f"{'='*50}")

    if visible < 1000:
        print("\n⚠ 可见文字很少 —— 内容可能是 JS 动态加载的。")
        print("  F12 → Network → Fetch/XHR，看有没有直接返回 JSON 的接口。")
    if len(df) < 10:
        print("\n⚠ 切出来的行太少 —— 页面结构可能不是表格/列表。")
        print("  看看上面的 DOM 位置，再调 to_rows 里的标签选择。")


def preview(df: pd.DataFrame, n: int = 5) -> None:
    print(f"\n前 {n} 行：\n")
    for _, r in df.head(n).iterrows():
        t = r.text if len(r.text) <= 150 else r.text[:150] + "…"
        print(f"  [{r.tag}] {t}\n")


# ========================================================================
# 主程序 MAIN —— 命令行入口
# 不传参数就用上面 DEFAULT_URL；传了参数（网址或文件）就用参数
# ========================================================================

def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("src", nargs="?", default=DEFAULT_URL,
                     help="网址或本地 HTML 文件（不填就用 DEFAULT_URL）")
    ap.add_argument("--out", default="data", help="输出目录（默认 data/）")
    ap.add_argument("--no-archive", action="store_true", help="不存原始 HTML")
    a = ap.parse_args()

    outdir = Path(a.out)
    outdir.mkdir(parents=True, exist_ok=True)

    html, source = load(a.src)

    if not a.no_archive:
        p = archive(html, a.src, outdir)
        print(f"✓ 原始 HTML 存档 → {p}")

    df = to_rows(html, source)
    diagnose(html, df)

    if df.empty:
        sys.exit("\n✗ 没切出任何行。")

    csv = outdir / "rows.csv"
    df.to_csv(csv, index=False, encoding="utf-8-sig")
    preview(df)
    print(f"✓ {len(df):,} 行 → {csv}")


if __name__ == "__main__":
    main()

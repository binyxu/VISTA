#!/usr/bin/env python3
"""Official-style GAIA MCP tools: web, attachments, and Python.

No ground-truth answers or annotator metadata are loaded here.
"""

import argparse
import csv
import html
import json
import os
import re
import subprocess
import tempfile
import textwrap
import urllib.parse
import zipfile
from pathlib import Path
from typing import Any

import requests
from fastmcp import FastMCP

os.environ["FASTMCP_SHOW_CLI_BANNER"] = "false"


def _clip(text: str, max_chars: int) -> str:
    text = text or ""
    return text if len(text) <= max_chars else text[:max_chars] + f"\n...[truncated {len(text) - max_chars} chars]"


def _strip_html(raw: str) -> str:
    raw = re.sub(r"(?is)<script.*?</script>|<style.*?</style>", " ", raw)
    raw = re.sub(r"(?s)<[^>]+>", " ", raw)
    raw = html.unescape(raw)
    raw = re.sub(r"\s+", " ", raw)
    return raw.strip()


def _xlsx_text(path: Path, max_chars: int) -> str:
    out = []
    try:
        with zipfile.ZipFile(path) as z:
            shared = []
            if "xl/sharedStrings.xml" in z.namelist():
                xml = z.read("xl/sharedStrings.xml").decode("utf-8", "ignore")
                shared = [re.sub(r"<[^>]+>", "", s) for s in re.findall(r"<si.*?</si>", xml, re.S)]
            sheets = [n for n in z.namelist() if re.match(r"xl/worksheets/sheet\d+\.xml", n)]
            for sheet in sheets:
                out.append(f"=== {sheet} ===")
                xml = z.read(sheet).decode("utf-8", "ignore")
                for row_xml in re.findall(r"<row.*?</row>", xml, re.S):
                    vals = []
                    for c in re.findall(r"<c([^>]*)>(.*?)</c>", row_xml, re.S):
                        attrs, body = c
                        m = re.search(r"<v>(.*?)</v>", body, re.S)
                        if not m:
                            vals.append("")
                            continue
                        v = html.unescape(m.group(1))
                        if 't="s"' in attrs and v.isdigit() and int(v) < len(shared):
                            v = shared[int(v)]
                        vals.append(v)
                    if any(vals):
                        out.append("\t".join(vals))
                    if sum(len(x) for x in out) > max_chars:
                        return _clip("\n".join(out), max_chars)
    except Exception as exc:
        return f"Could not parse xlsx {path.name}: {exc}"
    return _clip("\n".join(out), max_chars)


def _read_file(path: Path, max_chars: int) -> str:
    suffix = path.suffix.lower()
    if suffix in {".txt", ".md", ".json", ".jsonl", ".csv", ".tsv", ".html", ".xml"}:
        raw = path.read_text(encoding="utf-8", errors="ignore")
        if suffix in {".html", ".xml"}:
            raw = _strip_html(raw)
        return _clip(raw, max_chars)
    if suffix in {".xlsx", ".xlsm"}:
        return _xlsx_text(path, max_chars)
    if suffix == ".csv":
        with path.open(newline="", encoding="utf-8", errors="ignore") as f:
            return _clip("\n".join("\t".join(r) for r in csv.reader(f)), max_chars)
    return (
        f"Attachment {path.name} is a binary file ({suffix or 'no extension'}), "
        f"{path.stat().st_size} bytes. Use python_execute for custom processing."
    )


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--attachments-dir", default="data/attachments")
    ap.add_argument("--work-dir", default="runs/official_tool_workspace")
    ap.add_argument("--port", type=int, default=8111)
    ap.add_argument("--timeout", type=int, default=30)
    args = ap.parse_args()

    attachments_dir = Path(args.attachments_dir).resolve()
    work_dir = Path(args.work_dir).resolve()
    work_dir.mkdir(parents=True, exist_ok=True)

    mcp = FastMCP(name="gaia-official-tools")

    @mcp.tool(
        name="web_search",
        description="Search the live web using DuckDuckGo Lite. Returns titles, URLs, and snippets.",
    )
    def web_search(query: str, k: int = 5) -> list[dict[str, Any]]:
        headers = {
            "User-Agent": (
                "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
                "AppleWebKit/537.36 (KHTML, like Gecko) Chrome/126 Safari/537.36"
            )
        }
        errors = []
        text = ""
        for base in ("https://lite.duckduckgo.com/lite/?", "https://html.duckduckgo.com/html/?"):
            url = base + urllib.parse.urlencode({"q": query})
            try:
                r = requests.get(url, headers=headers, timeout=args.timeout)
                if r.status_code >= 400:
                    errors.append(f"{base} HTTP {r.status_code}")
                    continue
                text = r.text
                break
            except Exception as exc:
                errors.append(f"{base} {type(exc).__name__}: {exc}")
        if not text:
            return [{
                "title": "SEARCH_ERROR",
                "url": "",
                "snippet": "Search provider failed: " + "; ".join(errors),
            }]
        rows = []
        for m in re.finditer(r'<a rel="nofollow" href="(?P<href>.*?)".*?>(?P<title>.*?)</a>', text, re.S):
            href = html.unescape(re.sub(r"<[^>]+>", "", m.group("href")))
            title = html.unescape(re.sub(r"<[^>]+>", "", m.group("title")))
            if "uddg=" in href:
                qs = urllib.parse.parse_qs(urllib.parse.urlparse(href).query)
                href = qs.get("uddg", [href])[0]
            if href and title:
                rows.append({"title": title, "url": href, "snippet": ""})
            if len(rows) >= max(1, min(k, 10)):
                break
        if not rows and errors:
            return [{"title": "SEARCH_ERROR", "url": "", "snippet": "; ".join(errors)}]
        return rows

    @mcp.tool(
        name="fetch_url",
        description="Fetch a URL and return extracted visible text. Use after web_search.",
    )
    def fetch_url(url: str, max_chars: int = 12000) -> dict[str, str]:
        headers = {
            "User-Agent": (
                "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
                "AppleWebKit/537.36 (KHTML, like Gecko) Chrome/126 Safari/537.36"
            )
        }
        try:
            r = requests.get(url, headers=headers, timeout=args.timeout)
            if r.status_code >= 400:
                return {
                    "url": url,
                    "content_type": r.headers.get("content-type", ""),
                    "error": f"HTTP {r.status_code}",
                    "text": "",
                }
        except Exception as exc:
            return {
                "url": url,
                "content_type": "",
                "error": f"{type(exc).__name__}: {exc}",
                "text": "",
            }
        ctype = r.headers.get("content-type", "")
        text = r.text if "text" in ctype or "html" in ctype or not r.content else r.text
        if "html" in ctype or "<html" in text[:500].lower():
            text = _strip_html(text)
        return {"url": url, "content_type": ctype, "text": _clip(text, max_chars)}

    @mcp.tool(
        name="list_attachments",
        description="List local GAIA attached files available to this run.",
    )
    def list_attachments() -> list[dict[str, Any]]:
        if not attachments_dir.exists():
            return []
        return [
            {"file_name": p.name, "size": p.stat().st_size}
            for p in sorted(attachments_dir.iterdir())
            if p.is_file() and p.name != "manifest.json"
        ]

    @mcp.tool(
        name="read_attachment",
        description="Read an attached GAIA file by file_name. Supports text/csv/json/html and basic xlsx extraction.",
    )
    def read_attachment(file_name: str, max_chars: int = 20000) -> dict[str, str]:
        path = (attachments_dir / Path(file_name).name).resolve()
        if not str(path).startswith(str(attachments_dir)) or not path.exists():
            return {"file_name": file_name, "error": "attachment not found"}
        return {"file_name": path.name, "text": _read_file(path, max_chars)}

    @mcp.tool(
        name="python_execute",
        description="Run Python code for calculations or local attachment analysis. Prints stdout/stderr.",
    )
    def python_execute(code: str, timeout: int = 20) -> dict[str, Any]:
        timeout = max(1, min(timeout, args.timeout))
        with tempfile.NamedTemporaryFile("w", suffix=".py", dir=work_dir, delete=False, encoding="utf-8") as f:
            f.write(code)
            script = f.name
        env = dict(os.environ)
        env["GAIA_ATTACHMENTS_DIR"] = str(attachments_dir)
        try:
            p = subprocess.run(
                [os.sys.executable, script],
                cwd=work_dir,
                env=env,
                text=True,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                timeout=timeout,
            )
            return {
                "returncode": p.returncode,
                "stdout": _clip(p.stdout, 20000),
                "stderr": _clip(p.stderr, 12000),
            }
        except subprocess.TimeoutExpired:
            return {"returncode": 124, "stdout": "", "stderr": f"Timed out after {timeout}s"}

    print(f"GAIA official MCP listening on http://127.0.0.1:{args.port}/mcp")
    print(f"attachments_dir={attachments_dir}")
    mcp.run(transport="sse", path="/mcp", port=args.port)


if __name__ == "__main__":
    main()

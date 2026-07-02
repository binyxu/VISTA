#!/usr/bin/env python3
"""Local MCP search/get_document server for GAIA stress corpus."""

import argparse
import json
import os
import re
from pathlib import Path
from typing import Any

os.environ["FASTMCP_SHOW_CLI_BANNER"] = "false"

from fastmcp import FastMCP


def load_corpus(path: Path) -> list[dict[str, Any]]:
    docs = []
    with path.open(encoding="utf-8") as f:
        for line in f:
            if line.strip():
                docs.append(json.loads(line))
    return docs


def terms(text: str) -> set[str]:
    return set(re.findall(r"[A-Za-z0-9_-]{3,}", text.lower()))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--corpus", default="data/stress/corpus.jsonl")
    parser.add_argument("--port", type=int, default=8091)
    parser.add_argument("--k", type=int, default=4)
    parser.add_argument("--snippet-chars", type=int, default=240)
    args = parser.parse_args()

    docs = load_corpus(Path(args.corpus))
    by_id = {d["docid"]: d for d in docs}
    doc_terms = {d["docid"]: terms(" ".join([d.get("docid", ""), d.get("qid", ""), d.get("title", ""), d.get("text", "")[:2000]])) for d in docs}

    mcp = FastMCP(name="gaia-local-search")

    @mcp.tool(
        name="search",
        description=(
            "Search the fixed local GAIA evidence corpus. Query with the task id "
            "for best recall. Returns top evidence shards with docid, score, and snippet."
        ),
    )
    def search(query: str) -> list[dict[str, Any]]:
        qterms = terms(query)
        scored = []
        for d in docs:
            score = len(qterms & doc_terms[d["docid"]])
            if str(d.get("qid", "")).lower() in query.lower():
                score += 1000
            scored.append((score, d))
        scored.sort(key=lambda x: (-x[0], str(x[1].get("docid", ""))))
        out = []
        for score, d in scored[: args.k]:
            text = (
                f"GAIA evidence shard {d.get('shard')} for task {d.get('qid')}. "
                "Full evidence is intentionally withheld from search results; call "
                f"get_document(docid={d['docid']!r}) to inspect this shard."
            )
            out.append({
                "docid": d["docid"],
                "score": float(score),
                "snippet": text[: args.snippet_chars],
            })
        return out

    @mcp.tool(
        name="get_document",
        description="Retrieve a full GAIA evidence shard by docid.",
    )
    def get_document(docid: str) -> dict[str, Any] | None:
        d = by_id.get(docid)
        if not d:
            return None
        return {"docid": d["docid"], "title": d.get("title", ""), "text": d.get("text", "")}

    print(f"GAIA MCP listening on http://127.0.0.1:{args.port}/mcp corpus={args.corpus} n={len(docs)}")
    mcp.run(transport="sse", path="/mcp", port=args.port)


if __name__ == "__main__":
    main()

"""GPU-free dense searcher MCP server for BrowseComp-Plus.

Uses the prebuilt Qwen3-Embedding-8B document index (doc vectors already
computed) and encodes only the QUERY at runtime via an OpenAI-compatible
/v1/embeddings endpoint. No local embedding model, torch, tevatron, or GPU.

Compatibility note: the query vector must live in the same space as the
prebuilt index. We replicate the official setup: prepend the Qwen3 retrieval
instruction and L2-normalize (the qwen3-embedding-8b index is built normalized).
If the endpoint's pooling differs from the index build, recall will drop, so
validate retrieval recall on a few queries before trusting results.

Serves the same tools as the BM25 searcher: search(query) and get_document(docid).
"""
import argparse, glob, json, os, pickle, sys
import numpy as np
import faiss
from fastmcp import FastMCP
from openai import OpenAI

TASK_PREFIX = "Instruct: Given a web search query, retrieve relevant passages that answer the query\nQuery:"


def _load_index(pattern):
    files = sorted(glob.glob(pattern))
    if not files:
        raise SystemExit(f"No index shards match: {pattern}")
    reps_all, lookup = [], []
    for fp in files:
        with open(fp, "rb") as f:
            reps, lk = pickle.load(f)
        reps_all.append(np.asarray(reps, dtype=np.float32))
        lookup += list(lk)
    reps = np.vstack(reps_all)
    faiss.normalize_L2(reps)  # index built normalized -> cosine via inner product
    index = faiss.IndexFlatIP(reps.shape[1])
    index.add(reps)
    print(f"[dense] index: {reps.shape[0]} docs, dim={reps.shape[1]} from {len(files)} shard(s)")
    return index, lookup


def _load_corpus(name):
    from datasets import load_dataset
    ds = load_dataset(name, split="train")
    # fields are typically docid + text (and maybe title)
    d2t = {}
    for r in ds:
        docid = str(r.get("docid") or r.get("id") or r.get("_id"))
        text = r.get("text") or r.get("contents") or r.get("body") or ""
        title = r.get("title") or ""
        d2t[docid] = (title + "\n" + text).strip() if title else text
    print(f"[dense] corpus: {len(d2t)} docs from {name}")
    return d2t


def build_server(args):
    index, lookup = _load_index(args.index_path)
    d2t = _load_corpus(args.corpus)
    # Official snippet truncation: Qwen3-0.6B tokenizer, 512 tokens (matches
    # searcher/tools.py so dense scores stay comparable to the BM25 / official setup).
    tok = None
    if args.snippet_max_tokens and args.snippet_max_tokens > 0:
        from transformers import AutoTokenizer
        tok = AutoTokenizer.from_pretrained("Qwen/Qwen3-0.6B")
    client = OpenAI(
        base_url=os.getenv("OPENAI_BASE_URL") or os.getenv("LOCA_OPENAI_BASE_URL"),
        api_key=os.getenv("OPENAI_API_KEY") or os.getenv("LOCA_OPENAI_API_KEY"),
    )

    def embed_query(q):
        import time
        last = None
        for attempt in range(6):
            try:
                r = client.embeddings.create(model=args.embed_model, input=TASK_PREFIX + " " + q)
                v = np.asarray(r.data[0].embedding, dtype=np.float32)[None, :]
                faiss.normalize_L2(v)
                return v
            except Exception as e:  # qpm 120 rate limit -> backoff and retry
                last = e
                time.sleep(min(30, 2 ** attempt))
        raise last

    def snippet(text):
        if tok is None:
            return text
        ids = tok.encode(text, add_special_tokens=False)
        if len(ids) <= args.snippet_max_tokens:
            return text
        return tok.decode(ids[: args.snippet_max_tokens], skip_special_tokens=True)

    mcp = FastMCP(name="dense-search-server")

    @mcp.tool(name="search", description=f"Dense search (Qwen3-Embedding-8B); returns top-{args.k} {{docid, score, snippet}}.")
    def search(query: str):
        v = embed_query(query)
        scores, idx = index.search(v, args.k)
        out = []
        for s, i in zip(scores[0], idx[0]):
            if i < 0:
                continue
            docid = str(lookup[i])
            out.append({"docid": docid, "score": float(s), "snippet": snippet(d2t.get(docid, ""))})
        return out

    @mcp.tool(name="get_document", description="Get the full text of a document by docid.")
    def get_document(docid: str):
        t = d2t.get(str(docid))
        return None if t is None else {"docid": str(docid), "text": t}

    return mcp


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--index-path", required=True, help='Glob for index pickles, e.g. "indexes/qwen3-embedding-8b/corpus.shard*.pkl"')
    ap.add_argument("--corpus", default="Tevatron/browsecomp-plus-corpus")
    ap.add_argument("--embed-model", default="Qwen/Qwen3-Embedding-8B")
    ap.add_argument("--k", type=int, default=5)
    ap.add_argument("--snippet-max-tokens", type=int, default=512,
                    help="Official setting: truncate each snippet to 512 Qwen3-0.6B tokens. -1 disables.")
    ap.add_argument("--port", type=int, default=8081)
    args = ap.parse_args()
    build_server(args).run(transport="sse", path="/mcp", port=args.port)

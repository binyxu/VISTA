# VISTA on BrowseComp-Plus

This integrates the LOCAbench `strict_lc_better_dashboard` method (VISTA: a
training-free self-managed context layer) as an agent for BrowseComp-Plus.

The method is unchanged. Every turn the agent sees a `<context_workspace_status>`
dashboard (per-block id / size / age / status) and manages its own context with
`context_workspace_archive` (move a block out of the prompt, leaving an
`[ARCHIVED:Bx Ln]` placeholder) and `context_workspace_recover` (read the exact
payload back). Only the surrounding environment is BrowseComp-Plus: the agent
acts with the `search` / `get_document` tools served by the BrowseComp-Plus
searcher, and the backbone is driven through an OpenAI-compatible endpoint.

## What is reused vs new

- Reused, unmodified, from the LOCAbench checkout (`$LOCABENCH_ROOT`, default
  `../LOCAbench`):
  - `gem/.../context_workspace/workspace_manager.py` — `WorkspaceManager`
    (register / assemble / dashboard / preflight offload).
  - `gem/.../context_workspace/server.py` — the `context_workspace_*` tools, run
    in-process against the same `workspace_state.json`.
  - One faithful addition was made to that `server.py`: a `context_workspace_recover`
    tool. In LOCAbench recovery was done with the agent's file/python tools; a
    BrowseComp-Plus agent has only `search`/`get_document`, so recover is exposed
    as an explicit tool with the same effect (read the exact archived payload).
- New, in this repo:
  - `search_agent/gemini_vista_client.py` — the manual VISTA tool loop.
  - `run_vista_bm25.sh` — end-to-end BM25 runner.

## GPU requirement

With BM25, only the final judging step needs a GPU.

| Stage | GPU |
|---|---|
| Dataset/corpus download | no (HF token) |
| BM25 index build (pyserini + Java 21) | no |
| Searcher MCP (BM25) | no |
| VISTA answer generation (Gemini via Venus) | no |
| `evaluate_run.py` (Qwen3-32B judge, vLLM) | yes |

So run everything locally, then ship `runs/...` to a GPU host for judging.

## Backbone note (important)

The paper backbone is gemini-3-flash. Through an OpenAI-compatible proxy that
maps to the native Gemini API (e.g. the skynet/Venus proxy), **Gemini-3.x models
cannot run as multi-turn tool agents**: Gemini 3 requires a `thought_signature`
to be echoed back with each prior function call, and the OpenAI-compatible layer
neither exposes nor replays it, so the second turn fails with
`function call ... missing a thought_signature`. This is a proxy limitation, not
a client bug.

Workable options:
- Run on this proxy with a non-Gemini backbone. Verified working multi-turn tool
  calling: `deepseek-v4-flash`, `gpt-5.4-mini` (and other non-Gemini ids). This
  is the default in `run_vista_bm25.sh`.
- For paper-faithful gemini-3-flash, point the client at a Gemini-native endpoint
  (the google-genai API, which manages thought_signature internally), or a proxy
  that replays it. Set `--model` / `LOCA_OPENAI_BASE_URL` accordingly.

## 1. Environment

Two processes need deps: the searcher (pyserini/Java) and the client
(openai/fastmcp/tiktoken). The simplest path is the repo's `uv` env:

```bash
cd external/BrowseComp-Plus
uv venv --python 3.10
source .venv/bin/activate
# Full env (heavy; includes vllm for the judge):
uv sync
# Or a slim local env that skips the GPU judge stack:
uv pip install "pyserini>=1.2.0" "fastmcp==2.9.2" openai tiktoken python-dotenv tqdm rich datasets
```

Java 21 is required by pyserini for BM25:

```bash
conda install -c conda-forge openjdk=21      # or: brew install openjdk@21
export JAVA_HOME=/path/to/jdk21              # e.g. ~/miniconda3/lib/jvm
export PATH="$JAVA_HOME/bin:$PATH"
java -version                                # confirm 21
```

BM25-only gotchas (handled by `run_vista_bm25.sh`):
- The searcher also needs `pyngrok` (`uv pip install pyngrok`).
- pyserini instantiates an OpenAI client at import, so the searcher process needs
  a non-empty `OPENAI_API_KEY` (any value; unused for BM25). Keep it scoped to the
  searcher so it does not override the model key.
- `searcher/searchers/__init__.py` was patched to import the dense (faiss/tevatron)
  searchers lazily, so a BM25-only run does not require torch/tevatron.

## 2. Dataset + corpus

```bash
huggingface-cli login        # needs an HF token
python scripts_build_index/decrypt_dataset.py \
  --output data/browsecomp_plus_decrypted.jsonl \
  --generate-tsv topics-qrels/queries.tsv
```

`topics-qrels/queries.tsv` (qid<TAB>query) is the client input;
`data/browsecomp_plus_decrypted.jsonl` (query_id, query, answer) is the judge
ground truth. The corpus is public:

```python
from datasets import load_dataset
ds = load_dataset("Tevatron/browsecomp-plus-corpus", split="train")
```

## 3. BM25 index

Build a Lucene/pyserini BM25 index over the corpus (see
`scripts_build_index/` and the main README). Output to `indexes/bm25`.

## 4. Run VISTA

```bash
export LOCA_OPENAI_API_KEY=...          # Venus key
# optional overrides: MODEL, INDEX_PATH, PORT, QUERIES, OUTPUT_DIR, NUM_THREADS
./run_vista_bm25.sh
```

The script launches the BM25 searcher MCP, then runs the client. Or run the two
manually:

```bash
python searcher/mcp_server.py --searcher-type bm25 --index-path indexes/bm25 --port 8080 &

export LOCA_OPENAI_BASE_URL=https://api.example.com/v1
export LOCA_OPENAI_API_KEY=...
export SM_STRICT_LONG_CONTEXT=1 SM_BETTER_DASHBOARD=1
export LOCABENCH_ROOT=$(cd .. && pwd)/LOCAbench
python search_agent/gemini_vista_client.py \
  --model gemini-3-flash \
  --mcp-url http://127.0.0.1:8080/mcp \
  --query topics-qrels/queries.tsv \
  --output-dir runs/bm25/vista_gemini_3_flash \
  --max-context-size 128000 --num-threads 4
```

Records are written as `runs/.../run_<ts>.json`, the same schema the other
BrowseComp-Plus clients produce (`status`, `result[]` ending in an `output_text`,
`retrieved_docids`, `usage`), so evaluation is unchanged. Each query keeps its
own context workspace under `runs/.../_workspaces/<qid>/`.

Useful client flags: `--max-context-size` (VISTA token budget), `--max-steps`,
`--enable-delete`, `--query-template`
(`QUERY_TEMPLATE` keeps `get_document` available; the default).

## 5. Judge (GPU host)

Copy the run dir to a GPU machine with the full env and judge:

```bash
python scripts_evaluation/evaluate_run.py \
  --input_dir runs/bm25/vista_gemini_3_flash \
  --ground_truth data/browsecomp_plus_decrypted.jsonl \
  --tensor_parallel_size <num_gpus>
```

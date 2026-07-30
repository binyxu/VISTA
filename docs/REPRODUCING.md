# Reproducing VISTA experiments

This guide separates a fast implementation check from the paper benchmark
runs. Commands are launched from the repository root unless stated otherwise.

Python 3.10 or 3.11 is the safest common environment across the reference LOCA
runtime and the benchmark adapters.

## 1. Credentials and shared setup

```bash
cp .env.example .env
# Fill in the endpoint variables required by your model provider.
set -a
. ./.env
set +a
```

The main launchers accept an OpenAI-compatible base URL and API key through
`LOCA_OPENAI_BASE_URL` / `LOCA_OPENAI_API_KEY`. Some benchmark-native clients
also use `OPENAI_BASE_URL` / `OPENAI_API_KEY`; the wrappers bridge these names
where possible.

Do not commit `.env`, downloaded datasets, workspace payloads, or output
trajectories. Archived payloads contain the original tool evidence and can be
just as sensitive as the visible transcript.

## 2. Fast implementation check

```bash
python3 prototype/test_context_workspace.py
python3 prototype/workspace_demo.py
```

This checks stable block IDs, archive visibility, dashboard metadata, and
archive reads in the small prototype. It does not exercise the LOCA reference
runtime or reproduce a benchmark number.

For the reference runtime, a lightweight syntax/import check is:

```bash
python3 -m py_compile \
  benchmarks/LOCAbench/gem/tools/mcp_server/context_workspace/workspace_manager.py \
  benchmarks/LOCAbench/gem/tools/mcp_server/context_workspace/server.py \
  benchmarks/BrowseComp-Plus/search_agent/gemini_vista_client.py \
  benchmarks/AMA-Bench/src/method/self_managed_agentic.py
```

## 3. LOCA-Bench: primary online result

Install the benchmark environment first:

```bash
cd benchmarks/LOCAbench
bash install.sh
cd ../..
```

Download/place the required data as described in [`../DATA.md`](../DATA.md),
then run the full method:

```bash
CONTEXT_K=128 \
MAX_CONTEXT_SIZE=128000 \
MAX_WORKERS=8 \
bash run_loca_self_managed.sh
```

The launcher sets the two defining full-method flags:

```text
SM_STRICT_LONG_CONTEXT=1
SM_BETTER_DASHBOARD=1
```

It invokes `loca run -s self_managed` with `gemini-3-flash` by default, a
128,000-token active budget, a 1,800-second task timeout, and the selected
`task-configs/final_128k_set_config.json` suite.

### LOCA component ablations

Run these from `benchmarks/LOCAbench/` after setup:

| Variant | Command | Capability removed or changed |
|---|---|---|
| Full VISTA | `bash run_strict_lc_better_dashboard.sh` | Dashboard + agent archive + exact recovery. |
| No dashboard | `bash run_strict_lc_no_dashboard.sh` | Hides the agent-facing state ledger while keeping context tools. |
| No recovery | `bash run_strict_lc_no_recover.sh` | Removes the payload-read path. |
| No archive | `bash run_strict_lc_no_archive.sh` | Removes recoverable archive and enables irreversible delete for the matched capability ablation. |
| Fixed archive policy | `bash run_strict_lc_fixed_archive_policy.sh` | Replaces agent target choice with a deterministic policy. |
| State-board variant | `bash run_strict_lc_state_board.sh` | Adds an agent-authored status board; this is an interface variant, not the full-method definition. |

Generated task configs have `.generated.json` names. Outputs are written by the
LOCA harness; keep the generated config, command environment, model endpoint,
and repository commit with every reported result.

## 4. BrowseComp-Plus: deep-research transfer

See the benchmark-specific guide at
[`../benchmarks/BrowseComp-Plus/docs/vista.md`](../benchmarks/BrowseComp-Plus/docs/vista.md)
for Java, `uv`, corpus, and index setup. The dense W/B sweep used by the root
convenience script expects a Qwen3 embedding index under:

```text
benchmarks/BrowseComp-Plus/indexes/qwen3-embedding-8b/
```

Run a VISTA/ReAct comparison:

```bash
W=12288 \
B=163840 \
MODEL=deepseek-v4-pro-qcloud \
THREADS=5 \
bash run_browsecomp_wb_sweep.sh
```

Here `W` is the active context window and `B` is the total trajectory token
budget. Results are placed under `benchmarks/BrowseComp-Plus/runs/` and judged
by `scripts_evaluation/judge_api.py`.

For a paper-faithful Gemini run, use a Gemini-native endpoint or a proxy that
preserves function-call thought signatures across turns. The benchmark guide
documents this endpoint constraint.

## 5. GAIA: general-assistant transfer

Accept the gated GAIA dataset terms and download the validation data as
described in [`../DATA.md`](../DATA.md). Then run the official-data comparison:

```bash
LIMIT=165 \
W=12288 \
B=80000 \
MODEL=deepseek-v4-pro-qcloud \
bash benchmarks/GAIA/run_gaia_official_compare.sh
```

The script starts the GAIA tool server, calls the shared BrowseComp VISTA client
in both `vista` and truncated-`react` modes, and scores both output directories.
The paper uses a fixed 165-question validation subset; preserve `SEED`, selected
task IDs, model identifier, and generated dataset directory when reproducing
the exact split.

## 6. AMA-Bench: offline trajectory-memory transfer

Download the AMA-Bench dataset:

```bash
cd benchmarks/AMA-Bench
huggingface-cli download AMA-bench/AMA-bench \
  --repo-type dataset --local-dir ./dataset
cd ../..
```

Run the agentic self-managed replay:

```bash
bash run_ama_self_managed.sh
```

The launcher ships with credential-free example configs and reads endpoint
credentials from the environment. Override `LLM_CONFIG`, `JUDGE_CONFIG`, or
`METHOD_CONFIG` if your provider or budget differs.

The paper-aligned implementation is
`src/method/self_managed_agentic.py`: it replays trajectory steps incrementally,
keeps future questions hidden during construction, and invokes the LOCA archive
implementation when the workspace crosses its high-water mark.

`src/method/self_managed_context.py` is a deterministic early adapter and should
not be substituted when reproducing the paper's agentic replay result.

## 7. SWE-Bench exploratory adapter

```bash
bash run_swe_self_managed.sh --limit 1
```

This adapter generates standard SWE-Bench prediction JSONL and leaves official
Docker evaluation unchanged. It is not used in the VISTA paper tables and does
not yet reuse the full reference archive/recovery runtime; report it separately
from the LOCA/BrowseComp/AMA reference-core results.

## 8. What to save with a result

For an auditable run, record:

- repository commit SHA and any working-tree diff;
- model name, endpoint type, reasoning setting, and sampling parameters;
- active context budget and total trajectory budget;
- all `SM_*` flags;
- dataset version, split, task IDs, and random seed;
- per-task timeout and worker count;
- generated configuration file;
- raw trajectory, `workspace_state.json`, archive/recovery event log, and final
  evaluator output (after removing sensitive payload content).

The dashboard is regenerated runtime state, so the final answer alone is not
enough to diagnose a reproduction mismatch.

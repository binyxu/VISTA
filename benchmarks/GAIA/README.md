# GAIA VISTA vs ReAct

This directory adapts GAIA validation tasks to the same VISTA/ReAct harness used
by `../BrowseComp-Plus/search_agent/gemini_vista_client.py`.

The VISTA method is not reimplemented here. The runner calls the existing client:

- `--mode vista`: strict long-context VISTA with dashboard + archive/recover.
- `--mode react --react-cm truncate`: append-only ReAct with hard-window
  truncation.

## Data

GAIA is gated on Hugging Face. After accepting access on the dataset page, run:

```bash
export HF_TOKEN=...
./scripts/download_gaia.py --config 2023_all --split validation
```

Then build a fixed local evidence corpus:

```bash
./scripts/build_stress_corpus.py --limit 20 --window 12288 --shard-tokens 5200
```

The builder keeps real GAIA `Question` and `Final answer` as the evaluated task
and ground truth. It creates long local evidence shards from the GAIA metadata.
The gold answer is split into two fragments: shard 1 contains
`answer_fragment_1`, shard 4 contains `answer_fragment_2`, and intermediate
shards carry verification metadata/steps. Default evidence span is about
`4 * 5200` tokens per task, larger than the default `W=12288`.

## Run

```bash
set -a; . ../BrowseComp-Plus/.vista_env; set +a
./run_gaia_compare.sh
```

Useful overrides:

```bash
MODEL=deepseek-v4-pro-qcloud W=12288 B=196608 LIMIT=20 THREADS=2 ./run_gaia_compare.sh
```

Results are written under `runs/<RUN_ID>_{vista,react}` and judged by
`scripts/judge_gaia_api.py`.

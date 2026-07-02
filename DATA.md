# Data and Index Downloads

This repository keeps code, configs, and small metadata in git. Large datasets,
retrieval indexes, and benchmark artifacts should be downloaded by users.

## AMA-Bench

Official dataset:
https://huggingface.co/datasets/AMA-bench/AMA-bench

```bash
cd benchmarks/AMA-Bench
huggingface-cli download AMA-bench/AMA-bench --repo-type dataset --local-dir ./dataset
```

The default evaluation path is `dataset/test/`.

## BrowseComp-Plus

Download/decrypt queries, answers, and relevance judgements:

```bash
cd benchmarks/BrowseComp-Plus
pip install datasets
python scripts_build_index/decrypt_dataset.py \
  --output data/browsecomp_plus_decrypted.jsonl \
  --generate-tsv topics-qrels/queries.tsv
```

Download the corpus through Hugging Face datasets:

```python
from datasets import load_dataset
ds = load_dataset("Tevatron/browsecomp-plus-corpus", split="train")
```

Download prebuilt indexes:

```bash
cd benchmarks/BrowseComp-Plus
bash scripts_build_index/download_indexes.sh
```

For our W/B scripts, the Qwen3 embedding index is expected under
`indexes/qwen3-embedding-8b/`.

## SWE-Bench

SWE-Bench datasets are loaded from Hugging Face:

```python
from datasets import load_dataset
swebench = load_dataset("SWE-bench/SWE-bench_Lite", split="test")
```

SWE-Bench also requires Docker for evaluation.

## LoCoBench-Agent

The upstream README provides a Google Drive download for `data.zip`:

```bash
cd benchmarks/LoCoBench-Agent
pip install gdown
gdown https://drive.google.com/uc?id=1HwPztd0bipUUi8zs7Pxo3StZCOnJBwVR
unzip data.zip
```

This creates the `data/` directory.

## GAIA

GAIA data is not vendored here. Use the upstream GAIA/Harbor dataset source used
by your evaluation harness, or place downloaded GAIA task files under:

```text
benchmarks/GAIA/data/
```

## LOCA-Bench

LOCA task configurations and local mock environment code are included. Generated
outputs are intentionally excluded.


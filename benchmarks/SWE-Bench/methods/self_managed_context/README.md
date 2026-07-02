# Self-Managed Context Method for SWE-Bench

This directory adds a SWE-Bench inference method corresponding to the LOCAbench
`strict_lc_better_dashboard` setup.

LOCAbench used:

- `SM_STRICT_LONG_CONTEXT=1`
- `SM_BETTER_DASHBOARD=1`
- strategy `self_managed`
- `MAX_CONTEXT_SIZE=128000`

SWE-Bench expects model predictions as JSONL records with `instance_id`,
`model_name_or_path`, and `model_patch`. The adapter here keeps the official
SWE-Bench evaluation harness unchanged and only adds an inference method that
produces that predictions file.

## Run Inference

From the SWE-Bench root:

```bash
export OPENAI_API_KEY=...
export OPENAI_BASE_URL=http://your-openai-compatible-endpoint/v1

MODEL=gemini-3-flash \
DATASET_NAME=SWE-bench/SWE-bench_Lite \
SPLIT=test \
./methods/self_managed_context/run_strict_lc_better_dashboard.sh \
  --limit 1
```

The wrapper also accepts `LOCA_OPENAI_API_KEY` and `LOCA_OPENAI_BASE_URL` for
compatibility with the LOCAbench environment.

## Evaluate

After inference, run the command printed by the wrapper. On Apple Silicon, add
`--namespace ''` so SWE-Bench builds local Docker images instead of pulling
Linux images.

## Using Preprocessed Text

If you already created a SWE-Bench text dataset with
`swebench.inference.make_datasets`, point `--dataset_name` to that local dataset
or JSONL file. When a record has a `text` field, this method uses it as the main
problem context. Otherwise it falls back to the raw SWE-Bench issue fields.

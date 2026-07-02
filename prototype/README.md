# Minimal Context Squeeze Prototype

This prototype is intentionally small and synthetic. It tests a narrow hypothesis:

> Long-horizon agents can fail not only because they forget useful information, but because stale or unrelated context remains visible.

It includes two runners: a deterministic contamination-sensitive reader for fast smoke tests, and a real LLM runner using an OpenAI-compatible chat completions endpoint.

## Agentic Context Workspace

The full workspace implementation is in `context_workspace.py`.

It provides:

- addressable context blocks (`ContextBlock`),
- hierarchical episode/block tracking,
- mini/full dashboards,
- archive and visible workspace,
- validated `ABSTRACT`, `HIDE`, `RECALL`, and `ABSTRACT_AND_HIDE` actions,
- simple extractive summaries for smoke testing.

Run the workspace smoke test and demo:

```bash
python3 prototype/test_context_workspace.py
python3 prototype/workspace_demo.py
```

The demo output is saved at `results/workspace_demo_output.txt` when run through the test command.

## Strategies

- `reset_per_task`: each task starts clean.
- `full_squeeze`: all task histories remain visible forever.
- `sliding_window`: only recent events remain visible.
- `periodic_summary`: old context is periodically summarized, mixing stale and valid facts.
- `living_state_hide`: maintains a compact living state and hides invalidated/tool-log details.

## Run deterministic smoke test

```bash
python3 prototype/context_squeeze.py --n-tasks 30 --seed 7 --budget-tokens 900
```

## Run real LLM test

Set credentials through environment variables rather than committing them to files:

```bash
export LLM_API_KEY='your_key'
export LLM_BASE_URL='https://api.example.com/v1/chat/completions'
python3 prototype/context_squeeze_llm.py --n-tasks 6 --model gpt-5.1-mini
```

Outputs are written to `results/`:

- `prototype_summary.json`, `prototype_summary.md`, `prototype_details.csv`
- `llm_prototype_summary.json`, `llm_prototype_summary.md`, `llm_prototype_details.csv`

## Important Caveat

This is a smoke-test harness, not evidence for a paper. The next step is to replace the simulated reader with a real LLM agent and use LoCoBench-Agent or an adapted ContextLifecycle benchmark.

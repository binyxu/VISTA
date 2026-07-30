# Code provenance and modification boundaries

VISTA vendors several benchmark trees to make experimental entry points
self-contained. This document records which areas are method code, which are
VISTA adapters, and which originate from upstream projects.

## Upstream projects

| Vendored directory | Upstream project | VISTA-specific work in this repository |
|---|---|---|
| `benchmarks/LOCAbench/` | [hkust-nlp/LOCA-bench](https://github.com/hkust-nlp/LOCA-bench) | New context-workspace server/manager, deep changes to the ReAct loop, strategy wiring, run scripts, ablations, and result instrumentation. |
| `benchmarks/BrowseComp-Plus/` | [texttron/BrowseComp-Plus](https://github.com/texttron/BrowseComp-Plus) | VISTA client, VISTA/React sweep scripts, exact-recovery bridge, API judge helpers, and long-context experiment plumbing. |
| `benchmarks/AMA-Bench/` | [AMA-Bench/AMA-Bench](https://github.com/AMA-Bench/AMA-Bench) | Self-managed agentic replay method, registration/model-client changes, usage accounting, runners, and evaluation helpers. |
| `benchmarks/SWE-Bench/` | [SWE-bench/SWE-bench](https://github.com/SWE-bench/SWE-bench) | Added `methods/self_managed_context/`; upstream evaluation remains the authority for patch scoring. |
| `benchmarks/LoCoBench-Agent/` | [SalesforceAIResearch/LoCoBench-Agent](https://github.com/SalesforceAIResearch/LoCoBench-Agent) | Earlier context-workspace integration and local configuration support; not the main LOCA-Bench paper path. |
| `benchmarks/GAIA/` | [GAIA benchmark](https://huggingface.co/datasets/gaia-benchmark/GAIA) plus local adapter code | Dataset preparation, tool server, VISTA/React launchers, and scoring integration. The VISTA runtime itself is imported from the LOCA/BrowseComp path. |

Upstream license files remain inside the relevant benchmark directories. When
updating a vendored tree, preserve its history/license and reapply only the
adapter surface described below.

## VISTA-owned method code

The following files or directories are primarily VISTA code rather than
benchmark content:

- `benchmarks/LOCAbench/gem/tools/mcp_server/context_workspace/workspace_manager.py`
- `benchmarks/LOCAbench/gem/tools/mcp_server/context_workspace/server.py`
- VISTA-specific branches in `benchmarks/LOCAbench/inference/run_react.py`
- `benchmarks/LOCAbench/run_strict_lc*.sh`
- `benchmarks/BrowseComp-Plus/search_agent/gemini_vista_client.py`
- VISTA-specific `run_wb*.sh` and evaluation helpers in BrowseComp-Plus
- `benchmarks/AMA-Bench/src/method/self_managed_agentic.py`
- `benchmarks/AMA-Bench/src/method/self_managed_context.py`
- VISTA registration/runners/evaluation glue in AMA-Bench
- `benchmarks/GAIA/`
- `benchmarks/SWE-Bench/methods/self_managed_context/`
- `prototype/`
- `openclaw-context-workspace/`
- `analysis/`

The most important distinction is that `workspace_manager.py` and `server.py`
implement the method, while large parts of `benchmarks/*` provide tasks,
environment tools, model clients, and evaluators.

## Modification boundaries by benchmark

### LOCA-Bench

The VISTA integration changes the runner at message lifecycle boundaries:
workspace initialization, per-message registration, prompt assembly, dashboard
injection, context-tool registration, oversized-result handling, and strict
budget gating. Task definitions, environment state transitions, and success
checks remain benchmark concerns.

### BrowseComp-Plus

The custom client imports the LOCA workspace core instead of maintaining a
forked second implementation. Search and document retrieval remain
BrowseComp-Plus tools. The VISTA layer controls only what interaction evidence
is visible to the model and how archived evidence is recovered.

### AMA-Bench

The paper adapter must fit AMA-Bench's construction/retrieval memory API. It
therefore replays trajectory steps through VISTA during construction and emits
the assembled workspace during retrieval. This is an adapter of the online
method to offline memory evaluation, not a claim that AMA-Bench itself is an
online agent benchmark.

### GAIA

The GAIA directory supplies tasks, attachments, tools, and scoring. Its launcher
calls the existing BrowseComp VISTA client, which in turn imports the LOCA core.
There is no third independent implementation to keep in sync.

### SWE-Bench

The current method directory is a simplified dashboard/block adapter. It should
not be described as an exact port of the reference archive/recovery runtime
until it imports or faithfully reimplements the adapter contract in
[`ARCHITECTURE.md`](ARCHITECTURE.md#adapter-contract).

## Citation boundaries

When publishing results:

- Cite the VISTA paper for the workspace/dashboard/archive method.
- Cite each benchmark used for tasks and evaluation.
- Do not describe local EMem-style or Mem0-style AMA adapters as official
  reproductions of those systems.
- Distinguish exact reference-core runs from simplified or exploratory adapters.

## Release checklist

Before calling the combined repository fully open source:

- [ ] Choose and add a top-level license for VISTA-owned code. The repository
  currently has no root license; an upstream benchmark license does not
  automatically cover new VISTA files.
- [ ] Confirm that the chosen top-level license is compatible with every
  vendored subtree and clearly state exceptions.
- [ ] Add a `NOTICE` file if required by the selected license or upstream
  dependencies.
- [ ] Keep API keys, `.env`, run payloads, downloaded datasets, and model outputs
  untracked.
- [ ] Preserve upstream `LICENSE`, attribution, and citation files when
  refreshing benchmark copies.
- [ ] Record upstream commit SHAs for future releases so a reader can recreate
  the base trees and inspect the VISTA patch independently.
- [ ] Tag the exact code revision used for each paper result table.

The license choice is a maintainer/legal decision and is intentionally not
guessed by this documentation update.

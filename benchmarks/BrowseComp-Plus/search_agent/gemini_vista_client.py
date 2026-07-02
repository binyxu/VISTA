"""VISTA agent client for BrowseComp-Plus.

Faithful port of the LOCAbench ``strict_lc_better_dashboard`` (self-managed
context) method onto BrowseComp-Plus. The method is unchanged: a manual tool
loop that, every turn, assembles the visible context (archived blocks become
``[ARCHIVED:Bx Ln]`` placeholders) and injects a ``<context_workspace_status>``
dashboard, while the agent manages its own context with archive / recover tools.

It reuses the exact LOCAbench components rather than reimplementing them:
  - ``WorkspaceManager`` (harness side: register / assemble / dashboard / offload)
  - the ``context_workspace`` MCP tools (agent side: archive / delete / recover),
    run in-process against the same ``workspace_state.json``.

Only the environment-specific pieces differ from LOCAbench:
  - tools the agent acts with are BrowseComp-Plus ``search`` / ``get_document``,
    served by the BrowseComp-Plus searcher MCP server (``--mcp-url``);
  - the backbone is driven through an OpenAI-compatible endpoint (Venus proxy),
    which is how LOCAbench drives gemini-3-flash for VISTA;
  - input is ``topics-qrels/queries.tsv`` and output is the BrowseComp-Plus
    ``run_<ts>.json`` record, so ``scripts_evaluation/evaluate_run.py`` works
    unchanged.

Run the searcher MCP server first (see docs/vista.md), then this client.
"""

import argparse
import asyncio
import csv
import html
import json
import os
import re
import sys
import time
import uuid
from datetime import datetime
from pathlib import Path
from types import SimpleNamespace

from dotenv import load_dotenv
from fastmcp import Client
from fastmcp.client.transports import SSETransport
from openai import AsyncOpenAI
from tqdm import tqdm

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from prompts import format_query  # noqa: E402
from utils import extract_retrieved_docids_from_result  # noqa: E402

load_dotenv()


# ── Locate and import the reused LOCAbench components ────────────────────────────
def _locabench_root() -> Path:
    env = os.environ.get("LOCABENCH_ROOT", "").strip()
    if env:
        return Path(env).expanduser().resolve()
    # Default: sibling of the BrowseComp-Plus repo, i.e. external/LOCAbench.
    repo_root = Path(__file__).resolve().parents[1]
    return (repo_root.parent / "LOCAbench").resolve()


LOCABENCH_ROOT = _locabench_root()
_CW_DIR = LOCABENCH_ROOT / "gem" / "tools" / "mcp_server" / "context_workspace"
if not _CW_DIR.is_dir():
    raise SystemExit(
        f"Could not find the context_workspace module under {LOCABENCH_ROOT}. "
        "Set LOCABENCH_ROOT to the LOCAbench checkout that contains gem/."
    )


# VISTA method flags. These select the exact strict_lc_better_dashboard variant.
# They must be set before importing the context_workspace modules, since the
# dashboard renderer and payload writer read them at import/call time.
os.environ.setdefault("SM_STRICT_LONG_CONTEXT", "1")
os.environ.setdefault("SM_BETTER_DASHBOARD", "1")
# Compact, de-duplicated archive representation: single-line in-place [ARCHIVED]
# markers (drop role/recoverable/label redundancy) + folded archived rows in the
# dashboard (details live in the inline markers). Position/order preserved.
os.environ.setdefault("SM_COMPACT_ARCHIVE", "1")
os.environ.setdefault("LOCA_QUIET", "1")


def _load_standalone(name: str, path: Path):
    """Load a single LOCAbench module by file path.

    This deliberately avoids importing the ``gem`` package, whose __init__ pulls
    in heavy deps (numpy, the full env stack). The context_workspace modules are
    self-contained (stdlib + fastmcp), so loading them standalone is exact reuse
    without the rest of LOCAbench.
    """
    import importlib.util

    spec = importlib.util.spec_from_file_location(name, str(path))
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


_wm = _load_standalone("_vista_workspace_manager", _CW_DIR / "workspace_manager.py")
cw_server = _load_standalone("_vista_cw_server", _CW_DIR / "server.py")
WorkspaceManager = _wm.WorkspaceManager
count_msg_tokens = _wm.count_msg_tokens

try:
    import tiktoken

    _TKT_ENC = tiktoken.get_encoding("cl100k_base")
except Exception:
    _TKT_ENC = None


# ── CONTEXT MANAGEMENT PROTOCOL (LOCAbench better_dashboard + task goal) ───────
CONTEXT_PROTOCOL = (
    "\n\n"
    "CONTEXT MANAGEMENT PROTOCOL:\n"
    "Main goal: solve the task; manage context only as needed, and answer immediately when evidence is sufficient.\n"
    "A <context_workspace_status> dashboard is shown every turn as a compact map of context blocks. "
    "Use context tools only when clearly needed. "
    "Do not archive, delete, or offload content solely because it is old, large, or listed in context metadata; "
    "leave content visible when the context budget is sufficient. "
    "Archive a block only when you are certain it has no further value — a near-duplicate search that "
    "returned nothing new, or results clearly unrelated to every part of the task; keep everything else, "
    "including anything you are not sure about. "
    "Large payloads may be represented by placeholders; inspect originals only when needed. "
    "Use ordinary source tools, source metadata, and any in-context payload placeholders "
    "to inspect external evidence when details are needed. "
    "For structured data or calculations, use the source tool or query directly. "
    "Do not copy table, CSV, or JSON rows from the conversation into code."
)


# ── Context-workspace tools executed in-process (the reused MCP tools) ───────────
def _call_cw_tool(tool, *args):
    """Call a FastMCP tool across versions.

    Some FastMCP releases leave @app.tool() as the original function; others
    return a FunctionTool wrapper with .fn. Keep the BrowseComp harness agnostic.
    """
    fn = getattr(tool, "fn", tool)
    return fn(*args)


def _context_tool_specs(cw, enable_delete: bool):
    """Build the in-process context-management tool specs bound to a specific
    cw_server module instance ``cw`` (per query, so concurrent queries do not
    share module state or the global workspace-dir env var)."""
    specs = [
        {
            "name": "context_workspace_archive",
            "description": (
                "Replace one or more blocks with compact indexes.\n\n"
                "The original content is stored externally as a payload file. Operations\n"
                "are block-level: only listed block IDs are archived. Multiple IDs and ranges\n"
                "are accepted. Mixed-level batches archive the lowest-level listed blocks and\n"
                "skip higher levels.\n\n"
                "Args:\n"
                "    block_id: Block IDs, ranges, or group IDs, e.g. \"B3\", \"B3,B4\",\n"
                "              \"B10-B20\", \"B10 to B20\", or \"G2\".\n"
                "    replacement: Short index text for the archived block(s)."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "block_id": {"type": "string", "description": "Block IDs, ranges, or group IDs."},
                    "replacement": {"type": "string", "description": "Short index text for the archived block(s)."},
                },
                "required": ["block_id"],
            },
            "fn": lambda a: _call_cw_tool(
                cw.context_workspace_archive, a.get("block_id", ""), a.get("replacement", "")
            ),
        },
        {
            "name": "context_workspace_recover",
            "description": (
                "Read back the exact original content of an archived block or group "
                "(shown as [ARCHIVED:Bx Ln] in the dashboard). Args: block_id."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "block_id": {"type": "string", "description": "Block IDs, ranges, or group IDs."},
                },
                "required": ["block_id"],
            },
            "fn": lambda a: _call_cw_tool(cw.context_workspace_recover, a.get("block_id", "")),
        },
    ]
    if enable_delete:
        specs.append(
            {
                "name": "context_workspace_delete",
                "description": (
                    "Permanently remove one or more blocks. Deleted content cannot be "
                    "recovered. Args: block_id, reason."
                ),
                "parameters": {
                    "type": "object",
                    "properties": {
                        "block_id": {"type": "string"},
                        "reason": {"type": "string"},
                    },
                    "required": ["block_id", "reason"],
                },
                "fn": lambda a: _call_cw_tool(
                    cw.context_workspace_delete, a.get("block_id", ""), a.get("reason", "")
                ),
            }
        )
    return specs


def _mcp_tools_to_openai(mcp_tools):
    """Convert fastmcp tool descriptors to OpenAI function-tool schemas."""
    out = []
    for t in mcp_tools:
        schema = getattr(t, "inputSchema", None) or {"type": "object", "properties": {}}
        out.append(
            {
                "type": "function",
                "function": {
                    "name": t.name,
                    "description": (getattr(t, "description", "") or "")[:1024],
                    "parameters": schema,
                },
            }
        )
    return out


def _context_tools_to_openai(specs):
    return [
        {
            "type": "function",
            "function": {
                "name": s["name"],
                "description": s["description"],
                "parameters": s["parameters"],
            },
        }
        for s in specs
    ]


def _extract_mcp_text(result) -> str:
    """Pull a plain-text payload out of a fastmcp call_tool result."""
    data = getattr(result, "data", None)
    if isinstance(data, str) and data:
        return data
    content = getattr(result, "content", None)
    if content:
        texts = [getattr(c, "text", None) for c in content]
        texts = [t for t in texts if t]
        if texts:
            return "\n".join(texts)
    if data is not None:
        try:
            return json.dumps(data, ensure_ascii=False)
        except Exception:
            return str(data)
    return ""


async def _call_mcp_tool_with_retries(
    mcp_client,
    name: str,
    args_: dict,
    *,
    timeout_seconds: int = 0,
    retries: int = 0,
):
    """Call an MCP tool with bounded per-attempt latency and retry transport glitches."""
    attempts = max(1, int(retries) + 1)
    last_exc = None
    for attempt in range(attempts):
        try:
            coro = mcp_client.call_tool(name, args_)
            if timeout_seconds and timeout_seconds > 0:
                return await asyncio.wait_for(coro, timeout=timeout_seconds)
            return await coro
        except Exception as exc:
            last_exc = exc
            if attempt >= attempts - 1:
                break
            delay = min(10.0, 1.0 * (2 ** attempt))
            print(
                f"[tool retry] {name} attempt {attempt + 1}/{attempts} failed: "
                f"{type(exc).__name__}: {exc}; retrying in {delay:.1f}s",
                flush=True,
            )
            await asyncio.sleep(delay)
    raise last_exc


def _is_mcp_transport_error(exc: Exception) -> bool:
    name = type(exc).__name__
    msg = str(exc).lower()
    return (
        name in {"ClosedResourceError", "BrokenResourceError", "EndOfStream", "RemoteProtocolError"}
        or "closedresource" in msg
        or "server disconnected" in msg
        or "connection reset" in msg
    )


def _format_context_reject(
    name: str,
    args: dict,
    raw_tokens: int,
    conv_now: int,
    overhead_tok: int,
    hard_available: int,
) -> str:
    query = args.get("query", "") if isinstance(args, dict) else ""
    docid = args.get("docid", "") if isinstance(args, dict) else ""
    lines = [
        f"TOOL_RESULT {name}",
        "status: context_limit_rejected",
    ]
    if query:
        lines.append(f"query: {query}")
    if docid:
        lines.append(f"docid: {docid}")
    retry_target = "this tool call" if name.startswith("context_workspace_") else "search/get_document"
    lines.extend(
        [
            f"raw_result_tokens_not_added: ~{raw_tokens:,}",
            f"assembled_context_tokens: ~{conv_now:,}",
            f"overhead_tokens: ~{overhead_tok:,}",
            f"hard_context_limit: ~{hard_available:,}",
            f"remaining_context_tokens: ~{max(0, hard_available - overhead_tok - conv_now):,}",
            "content_not_added: true",
            f"required_next_action: call context_workspace_archive before retrying {retry_target}.",
            "CONTEXT_LIMIT_REJECTED: Free space with context_workspace_archive(block_id, replacement), then retry.",
        ]
    )
    return "\n".join(lines)


def _format_result_too_large(
    name: str,
    args: dict,
    raw_tokens: int,
    hard_available: int,
    max_fraction: float,
) -> str:
    query = args.get("query", "") if isinstance(args, dict) else ""
    docid = args.get("docid", "") if isinstance(args, dict) else ""
    limit_tokens = int(hard_available * max_fraction)
    lines = [
        f"TOOL_RESULT {name}",
        "status: result_too_large_rejected",
    ]
    if query:
        lines.append(f"query: {query}")
    if docid:
        lines.append(f"docid: {docid}")
    lines.extend(
        [
            f"raw_result_tokens_not_added: ~{raw_tokens:,}",
            f"hard_context_limit: ~{hard_available:,}",
            f"single_result_limit: ~{limit_tokens:,} ({max_fraction:.0%} of context window)",
            "content_not_added: true",
            "reason: This one result is too large to be useful in the active context.",
            f"required_next_action: do not retry this exact {name} call; use snippets, a narrower query, or a different source.",
        ]
    )
    return "\n".join(lines)


def _format_protocol_violation(name: str, args: dict) -> str:
    query = args.get("query", "") if isinstance(args, dict) else ""
    docid = args.get("docid", "") if isinstance(args, dict) else ""
    lines = [
        f"TOOL_RESULT {name}",
        "status: protocol_violation_not_executed",
    ]
    if query:
        lines.append(f"query_not_executed: {query}")
    if docid:
        lines.append(f"docid_not_executed: {docid}")
    lines.extend(
        [
            "content_not_added: true",
            "reason: A previous search/get_document was rejected by the hard context limit.",
            "required_next_action: call context_workspace_archive before any more search/get_document.",
        ]
    )
    return "\n".join(lines)


_BUDGET_CLOSE_INSTRUCTION = (
    "Your token budget is now exhausted, so you can no longer call any tools. "
    "Based on all the evidence you have already gathered, commit to your single "
    "best final answer now and state it directly."
)


def _is_retryable_model_error(exc: Exception) -> bool:
    status = getattr(exc, "status_code", None)
    msg = str(exc).lower()
    if status in {408, 409, 429, 500, 502, 503, 504}:
        return True
    # The Venus/QCloud proxy intermittently reports auth/not-found for valid
    # model aliases; a retry usually recovers. Persistent auth errors still fail
    # after the bounded retry budget.
    if status == 404 and ("authentication_error" in msg or "not found" in msg):
        return True
    return any(x in msg for x in ("timeout", "temporarily", "connection reset", "rate limit"))


async def _chat_completion_with_retries(oai, **kwargs):
    retries = int(os.getenv("SM_MODEL_RETRIES", "5"))
    delay = float(os.getenv("SM_MODEL_RETRY_DELAY", "2"))
    last_exc = None
    for attempt in range(retries + 1):
        try:
            return await oai.chat.completions.create(**kwargs)
        except Exception as exc:
            last_exc = exc
            if attempt >= retries or not _is_retryable_model_error(exc):
                raise
            sleep_s = delay * (2 ** attempt)
            print(
                f"[model_retry] attempt {attempt + 1}/{retries} failed: {exc}; "
                f"sleeping {sleep_s:.1f}s",
                flush=True,
            )
            await asyncio.sleep(sleep_s)
    raise last_exc


async def _forced_final_answer(oai, model, api_messages, max_tokens, temperature):
    """Closing turn once the budget is exhausted: no tools are offered, so the
    model must commit a final answer from the evidence it already gathered.
    Returns the answer text ("" if the model still produces nothing). Applied to
    both vista and react so the two are held to the same commit-on-close standard.
    """
    msgs = list(api_messages) + [{"role": "user", "content": _BUDGET_CLOSE_INSTRUCTION}]
    try:
        resp = await _chat_completion_with_retries(
            oai,
            model=model,
            messages=msgs,
            max_tokens=max_tokens,
            temperature=temperature,
        )
        return (resp.choices[0].message.content or "").strip()
    except Exception:
        return ""


# Control notices (reject / protocol-violation / invalid-arg / budget) are
# immediate tool responses required for API protocol validity, but they carry no
# task evidence. Without a lifecycle they pile up as uncompressible squatters
# (archiving a short notice reclaims ~nothing) and drive the assembled-context
# token count negative, which then auto-rejects every further external call.
# Give them a short TTL: keep them full-size for a few turns so the agent can
# read and react, then shrink in place to a tiny stub. Keyed by tool_call_id so
# no private fields leak into the API payload; only content is edited (never
# removed), so registered-block indices and assistant tool_call pairing stay
# valid and assemble()/conv_tokens immediately reflect the smaller size.
_EXPIRED_CONTROL_STUB = "[expired control notice]"


def _control_notice_ttl() -> int:
    try:
        return int(os.getenv("SM_CONTROL_NOTICE_TTL", "10"))
    except ValueError:
        return 10


def _expire_control_notices(messages, control_notice_steps, current_step, ttl) -> None:
    if ttl is None or ttl < 0 or not control_notice_steps:
        return
    for m in messages:
        if not isinstance(m, dict) or m.get("role") != "tool":
            continue
        created = control_notice_steps.get(m.get("tool_call_id"))
        if created is None:
            continue
        if current_step - created >= ttl and m.get("content") != _EXPIRED_CONTROL_STUB:
            m["content"] = _EXPIRED_CONTROL_STUB


def _is_workspace_id(value: str) -> bool:
    return bool(re.fullmatch(r"\s*[BG]\d+\s*", str(value or "")))


def _format_invalid_workspace_id_tool_arg(name: str, args: dict) -> str:
    docid = args.get("docid", "") if isinstance(args, dict) else ""
    return "\n".join(
        [
            f"TOOL_RESULT {name}",
            "status: invalid_tool_argument_not_executed",
            f"docid_not_executed: {docid}",
            "content_not_added: true",
            "reason: This is a workspace ID for an archived conversation block, not a source document ID.",
            "Use source document IDs returned by search results with get_document.",
        ]
    )


def _tool_overhead_tokens(openai_tools) -> int:
    if _TKT_ENC is None:
        return 0
    total = 0
    for t in openai_tools:
        fn = t.get("function") or {}
        s = (
            (fn.get("name") or "")
            + (fn.get("description") or "")
            + json.dumps(fn.get("parameters") or {}, ensure_ascii=False)
        )
        total += len(_TKT_ENC.encode(s, disallowed_special=()))
    return total


def _msg_tokens(m) -> int:
    s = json.dumps(m, ensure_ascii=False)
    if _TKT_ENC is None:
        return max(1, len(s) // 4)
    return len(_TKT_ENC.encode(s, disallowed_special=()))


def _truncate_to_budget(messages, budget_tokens):
    """Fair ReAct truncation under a HARD window. The pinned task message (the
    query) is never trimmed; every other round is. Drop whole tool-rounds from the
    OLDEST first until the task message + what remains fits ``budget_tokens``.
    No round is exempt except the task message: if the newest round itself cannot
    fit, the window becomes head-only and the ReAct agent continues from that
    truncated state. Rounds are kept whole (assistant tool_calls + their tool
    results) so the API payload stays valid.
    """
    if not messages:
        return messages, 0, False
    head, rest = messages[0], messages[1:]
    groups, i = [], 0
    while i < len(rest):
        m = rest[i]
        if m.get("role") == "assistant" and m.get("tool_calls"):
            grp = [m]; j = i + 1
            while j < len(rest) and rest[j].get("role") == "tool":
                grp.append(rest[j]); j += 1
            groups.append(grp); i = j
        else:
            groups.append([m]); i += 1
    n_groups = len(groups)
    head_tok = _msg_tokens(head)
    # While over the window, drop the oldest whole round. No round is exempt
    # except the task message, so this can drain to head-only if the newest round
    # alone does not fit.
    while groups and head_tok + sum(_msg_tokens(x) for grp in groups for x in grp) > budget_tokens:
        groups.pop(0)
    out = [head] + [m for grp in groups for m in grp]
    # For a pure truncation baseline, draining to the task-only message is valid:
    # it is exactly what a hard append-only window does when all observations fall
    # out of scope. Do not label this as overflow; continue with the trimmed
    # context and let the total trajectory budget stop repeated re-fetch loops.
    return out, n_groups - len(groups), False


_CM_STUB_PREFIX = "[context-managed:"


def _cm_stub(kind: str, tool_msg: dict, orig_tok: int) -> str:
    """Fixed-policy replacement content for one tool observation. All variants are
    lossy and irreversible (no handle, no recovery), matching the fixed external
    policies in the literature. They differ only in what they leave behind."""
    name = tool_msg.get("name") or "tool"
    if kind == "clear":
        # Tool-result Clearing: drop the payload, leave a tiny token-count stub.
        return f"{_CM_STUB_PREFIX} cleared {name} result, ~{max(1, orig_tok // 1000)}K tokens removed]"
    if kind == "mask":
        # Stale-observation Masking: hide the observation by age; note only that
        # an earlier observation was masked, no size/content.
        return f"{_CM_STUB_PREFIX} earlier {name} observation masked as stale]"
    if kind == "skeleton":
        # Skeleton Compression: keep a compact skeleton (head slice + size), drop
        # the bulk payload. Retains schema/first lines, not the full transcript.
        raw = str(tool_msg.get("content") or "")
        head = raw[:240].replace("\n", " ")
        return (f"{_CM_STUB_PREFIX} {name} skeleton] {head} "
                f"[... {max(0, orig_tok - 60)} more tokens omitted]")
    return str(tool_msg.get("content") or "")


def _rewrite_to_fit(messages, budget_tokens, kind):
    """Fixed-policy context reduction for the ReAct-family baselines. Unlike
    ``_truncate_to_budget`` (which drops whole rounds), this keeps every message in
    place and rewrites the OLDEST tool observations' content to a lossy stub until
    the pinned task message + the rest fits ``budget_tokens``. Assistant turns and
    tool-call/result pairing are untouched, so the API payload stays valid and the
    trajectory skeleton survives. Returns ``(out, n_rewritten, starved)``; starved
    means even after stubbing every observation the context still overflows, so the
    caller must answer in place."""
    if not messages:
        return messages, 0, False
    out = [dict(m) for m in messages]

    def total():
        return sum(_msg_tokens(m) for m in out)

    if total() <= budget_tokens:
        return out, 0, False
    n_mod = 0
    for m in out[1:]:  # never rewrite the pinned task message (out[0])
        if total() <= budget_tokens:
            break
        if m.get("role") != "tool":
            continue
        if str(m.get("content") or "").startswith(_CM_STUB_PREFIX):
            continue
        orig_tok = _msg_tokens(m)
        m["content"] = _cm_stub(kind, m, orig_tok)
        n_mod += 1
    if total() <= budget_tokens:
        return out, n_mod, False

    # A fixed rewrite policy can still exceed W when assistant/tool-call
    # scaffolding alone becomes too large. The fair hard-window fallback is to
    # trim oldest whole rounds, matching ReAct's final safety net instead of
    # failing the episode with context_overflow.
    trimmed, n_drop, starved = _truncate_to_budget(out, budget_tokens)
    return trimmed, n_mod + n_drop, starved


def _group_rounds(rest):
    """Split the non-head messages into whole rounds (assistant tool_call turn +
    its tool results, or a lone message)."""
    groups, i = [], 0
    while i < len(rest):
        m = rest[i]
        if m.get("role") == "assistant" and m.get("tool_calls"):
            grp = [m]; j = i + 1
            while j < len(rest) and rest[j].get("role") == "tool":
                grp.append(rest[j]); j += 1
            groups.append(grp); i = j
        else:
            groups.append([m]); i += 1
    return groups


_SLIM_SUMMARY_PROMPT = (
    "You are compressing the earlier part of an ongoing research agent's transcript "
    "to save context. Write a faithful, compact summary of the messages below. "
    "Preserve: established facts and evidence with their source doc IDs, constraints "
    "from the task, intermediate conclusions, candidate answers considered, and open "
    "questions still to resolve. Drop verbatim tool output and chatter. Be terse."
)


async def _slim_summarize(oai, model, messages, budget_tokens, max_tokens, temperature, keep_recent=2):
    """SLIM-style periodic summarization. When the transcript exceeds the window,
    summarize all older rounds (keeping the pinned task message and the newest
    ``keep_recent`` rounds verbatim) into one lossy note via an LLM call, and splice
    it back in place of those rounds. Irreversible. Returns ``(new_messages,
    summary_completion_tokens)``; a no-op returns the input and 0."""
    if not messages or len(messages) < 2:
        return messages, 0
    head, rest = messages[0], messages[1:]
    groups = _group_rounds(rest)
    head_tok = _msg_tokens(head)
    if head_tok + sum(_msg_tokens(x) for grp in groups for x in grp) <= budget_tokens:
        return messages, 0
    if len(groups) <= keep_recent:
        return messages, 0  # nothing old enough to summarize; truncation handles it
    old = groups[:-keep_recent] if keep_recent > 0 else groups
    recent = groups[-keep_recent:] if keep_recent > 0 else []
    # Render the old rounds as text for the summarizer.
    old_msgs = [m for grp in old for m in grp]
    rendered = _gemini_safe_messages([head] + old_msgs)
    transcript = "\n\n".join(
        f"[{m.get('role')}] {m.get('content','')}" for m in rendered if m.get("content"))
    try:
        resp = await _chat_completion_with_retries(
            oai,
            model=model,
            messages=[{"role": "system", "content": _SLIM_SUMMARY_PROMPT},
                      {"role": "user", "content": transcript[:120000]}],
            max_tokens=max_tokens, temperature=temperature,
        )
        summary = (resp.choices[0].message.content or "").strip()
        u = getattr(resp, "usage", None)
        sum_tok = (getattr(u, "completion_tokens", 0) or 0) if u else _msg_tokens({"content": summary})
    except Exception:
        return messages, 0
    if not summary:
        return messages, 0
    note = {"role": "user",
            "content": f"[SLIM summary of {len(old)} earlier step(s)]\n{summary}"}
    new_messages = [head, note] + [m for grp in recent for m in grp]
    return new_messages, sum_tok


def _apply_active_compress(messages, summary, keep_recent=1):
    """Active Context Compression: the agent authored ``summary`` as a knowledge
    block; drop the older raw rounds it summarized (irreversible, no recovery) and
    splice the knowledge block in their place, keeping the pinned task and the
    newest ``keep_recent`` rounds. No extra LLM call (unlike SLIM the agent wrote
    the summary). Returns ``(new_messages, n_dropped_rounds)``."""
    if not messages:
        return messages, 0
    head, rest = messages[0], messages[1:]
    groups = _group_rounds(rest)
    note = {"role": "user",
            "content": f"[KNOWLEDGE BLOCK — agent-compressed]\n{summary}"}
    if len(groups) <= keep_recent:
        return [head, note] + [m for g in groups for m in g], 0
    old = groups[:-keep_recent] if keep_recent > 0 else groups
    recent = groups[-keep_recent:] if keep_recent > 0 else []
    note["content"] = f"[KNOWLEDGE BLOCK — agent-compressed from {len(old)} step(s)]\n{summary}"
    return [head, note] + [m for g in recent for m in g], len(old)


_ACC_TOOL = {
    "type": "function",
    "function": {
        "name": "compress_context",
        "description": ("Consolidate what you have confirmed so far into one knowledge block: the verified "
                        "facts and evidence (each with its source doc ID), the task constraints, and the "
                        "questions still open. The bulky raw search/read observations behind it are then "
                        "replaced by this block — your verified findings are kept in your own words; only the "
                        "redundant raw transcripts are dropped (those cannot be re-read from context). On a "
                        "multi-clue question this is the normal way to advance, not a last resort: each time "
                        "you confirm a clue, lock it into the knowledge block and clear its raw results, so "
                        "your working context grows with conclusions instead of a pile of transcripts. "
                        "Consolidating early and often keeps your reasoning sharp and your evidence intact."),
        "parameters": {
            "type": "object",
            "properties": {"summary": {"type": "string", "description": "The knowledge block to keep."}},
            "required": ["summary"],
        },
    },
}

_FOLD_BRANCH_TOOL = {
    "type": "function",
    "function": {
        "name": "branch",
        "description": ("Delegate one identifying clue or sub-question to a sub-agent that researches it in "
                        "its own fresh context (it can search and read documents). When it finishes, only "
                        "its summary returns to you; the full search trace is folded away. This is the "
                        "preferred way to investigate a multi-clue question: each independent clue is a "
                        "natural branch, so you can verify candidates lead by lead without crowding your "
                        "main context with raw search results."),
        "parameters": {
            "type": "object",
            "properties": {
                "description": {"type": "string", "description": "Short label for the sub-task."},
                "prompt": {"type": "string", "description": "Full instructions for the sub-agent."},
            },
            "required": ["description", "prompt"],
        },
    },
}

_FOLD_RETURN_TOOL = {
    "type": "function",
    "function": {
        "name": "return_to_main",
        "description": "Finish this sub-task and return a summary of your findings to the main agent.",
        "parameters": {
            "type": "object",
            "properties": {"summary": {"type": "string", "description": "Findings to return."}},
            "required": ["summary"],
        },
    },
}


async def _run_branch(oai, mcp_client, model, search_tools, description, prompt,
                      max_tokens, temperature, sub_budget, window_budget, sub_step_cap=8,
                      tool_timeout_seconds=0, tool_retries=0, mcp_url=""):
    """Context-Folding sub-agent. Solves ``prompt`` in a fresh sub-context with the
    same retrieval tools plus return_to_main; only the returned summary survives to
    the main thread (the sub-trajectory is folded away).

    Fairness controls: the sub-agent may spend at most ``sub_budget`` new-content
    tokens (so one branch cannot drain the whole run budget), and its own context is
    truncated to ``window_budget`` each turn — the SAME hard window as the main
    agent, so a branch is not a way to exceed the context limit. Returns
    ``(summary, tokens_spent, (usage_in, usage_out, usage_total))`` so the caller
    charges both the shared budget and the real API usage."""
    tools = list(search_tools) + [_FOLD_RETURN_TOOL]
    sub = [{"role": "user", "content": (
        f"SUBTASK: {description}\n\n{prompt}\n\nWork autonomously. When finished, call "
        "return_to_main(summary) with your findings and source doc IDs. You cannot branch again.")}]
    spent = 0
    u_in = u_out = u_tot = 0
    summary = ""
    for _ in range(max(1, sub_step_cap)):
        if spent >= sub_budget:
            break
        api_sub, _, _ = _truncate_to_budget(sub, window_budget)  # align sub window to main W
        try:
            resp = await _chat_completion_with_retries(
                oai,
                model=model, messages=api_sub, tools=tools, tool_choice="auto",
                max_tokens=max_tokens, temperature=temperature)
        except Exception as exc:
            return f"(branch failed: {exc})", spent, (u_in, u_out, u_tot)
        u = getattr(resp, "usage", None)
        if u:
            u_in += getattr(u, "prompt_tokens", 0) or 0
            u_out += getattr(u, "completion_tokens", 0) or 0
            u_tot += getattr(u, "total_tokens", 0) or 0
            spent += getattr(u, "completion_tokens", 0) or 0
        m = resp.choices[0].message
        tcs = m.tool_calls or []
        rec = {"role": "assistant", "content": m.content or ""}
        if tcs:
            rec["tool_calls"] = [{"id": t.id, "type": "function",
                                  "function": {"name": t.function.name, "arguments": t.function.arguments}}
                                 for t in tcs]
        sub.append(rec)
        if not tcs:
            summary = m.content or ""
            break
        done = False
        for t in tcs:
            if t.function.name == "return_to_main":
                try:
                    summary = (json.loads(t.function.arguments or "{}") or {}).get("summary", "")
                except Exception:
                    summary = ""
                done = True
                sub.append({"role": "tool", "tool_call_id": t.id, "name": t.function.name,
                            "content": "ok"})
                break
            try:
                args_ = json.loads(t.function.arguments or "{}")
            except Exception:
                args_ = {}
            try:
                res = await _call_mcp_tool_with_retries(
                    mcp_client,
                    t.function.name,
                    args_,
                    timeout_seconds=tool_timeout_seconds,
                    retries=tool_retries,
                )
                out = _extract_mcp_text(res)
            except Exception as exc:
                if _is_mcp_transport_error(exc):
                    raise
                out = f"Error: tool {t.function.name} failed: {exc}"
            tmsg = {"role": "tool", "tool_call_id": t.id, "name": t.function.name,
                    "content": out if isinstance(out, str) else json.dumps(out, ensure_ascii=False)}
            sub.append(tmsg)
            spent += _msg_tokens(tmsg)
        if done:
            break
    return (summary or "(branch produced no summary)"), spent, (u_in, u_out, u_tot)


def _is_gemini(model: str) -> bool:
    return "gemini" in (model or "").lower()


_TEXT_TOOLCALL_RE = re.compile(
    r"^\s*(\[tool_call\]|search\s*\(|get_document\s*\(|context_workspace_\w+\s*\(|\{?\s*\"?(tool_call|name)\"?\s*[:=]|<｜DSML｜tool_calls>)",
    re.I,
)
_DSML_TOOLCALL_RE = re.compile(
    r"<｜DSML｜invoke\s+name=\"(?P<name>[^\"]+)\">\s*(?P<body>.*?)\s*</｜DSML｜invoke>",
    re.S,
)
_DSML_PARAM_RE = re.compile(
    r"<｜DSML｜parameter\s+name=\"(?P<name>[^\"]+)\"(?:\s+[^>]*)?>(?P<value>.*?)</｜DSML｜parameter>",
    re.S,
)
_DSML_BLOCK_RE = re.compile(
    r"<｜DSML｜tool_calls>.*?(?:</｜DSML｜tool_calls>|</｜DSML｜tool_runs>)",
    re.S,
)
_DRIFT_CORRECTION = (
    "You wrote a tool call as plain text. Tools can ONLY be invoked through the "
    "function-calling interface, never by typing the call in your message. If you "
    "still need information, call the tool properly now; if you already have enough, "
    "give your final answer in prose with no tool-call syntax."
)


def _looks_like_text_tool_call(text: str) -> bool:
    """Detect a tool call accidentally emitted as plain text (Gemini drift), so it
    is not mistaken for a final answer."""
    return bool(text) and bool(_TEXT_TOOLCALL_RE.match(text.strip()))


def _parse_dsml_tool_calls(text: str):
    """Venus may return function calls as DSML text instead of OpenAI tool_calls.
    Convert the markup into the minimal object shape used by the loop."""
    if not text or "<｜DSML｜invoke" not in text:
        return [], text
    calls = []
    for i, m in enumerate(_DSML_TOOLCALL_RE.finditer(text)):
        name = html.unescape(m.group("name")).strip()
        args = {}
        for pm in _DSML_PARAM_RE.finditer(m.group("body") or ""):
            key = html.unescape(pm.group("name")).strip()
            value = html.unescape(re.sub(r"<[^>]+>", "", pm.group("value"))).strip()
            args[key] = value
        calls.append(SimpleNamespace(
            id=f"dsml_{uuid.uuid4().hex}_{i}",
            type="function",
            function=SimpleNamespace(name=name, arguments=json.dumps(args, ensure_ascii=False)),
        ))
    cleaned = _DSML_BLOCK_RE.sub("", text)
    cleaned = cleaned.replace("<｜end▁of▁sentence｜>", "").strip()
    return calls, cleaned


def _gemini_safe_messages(api_messages):
    """Gemini 3 (Vertex) strictly validates history: every structured
    ``function_call`` part replayed in the conversation must carry a
    ``thought_signature``. This OpenAI-compatible proxy neither exposes that
    signature in responses nor accepts a placeholder, so a multi-turn tool agent
    400s on the second turn ("missing a thought_signature").

    Fix: never replay a structured tool call. Fold prior tool interactions into
    text so the request carries no ``function_call`` part to validate, while the
    model keeps emitting a *fresh* structured tool call each turn via the normal
    function-calling interface.

    Critical detail: do NOT render the assistant's past tool call as assistant
    text (e.g. ``[tool_call] search(...)``) — the model imitates that pattern and
    starts emitting tool calls as plain text in ``content``, which the loop then
    mistakes for a final answer. Instead, the assistant turn keeps only its
    genuine natural-language content, and the tool call + its result are shown
    from the *observation* (user) side. The model thus never sees itself issuing
    a textual tool call, so it keeps using the structured interface.
    """
    # Map tool_call_id -> (name, arguments) so results can name their query.
    call_meta = {}
    for m in api_messages:
        if m.get("role") == "assistant" and m.get("tool_calls"):
            for tc in m["tool_calls"]:
                cid = tc["id"] if isinstance(tc, dict) else tc.id
                fn = tc["function"] if isinstance(tc, dict) else tc.function
                name = fn["name"] if isinstance(fn, dict) else fn.name
                argz = fn["arguments"] if isinstance(fn, dict) else fn.arguments
                call_meta[cid] = (name, argz)

    flat = []
    for m in api_messages:
        role = m.get("role")
        if role == "assistant" and m.get("tool_calls"):
            # Keep only real reasoning/content; drop the structured calls. If the
            # assistant said nothing, omit the turn entirely.
            content = (m.get("content") or "").strip()
            if content:
                flat.append({"role": "assistant", "content": content})
        elif role == "tool":
            cid = m.get("tool_call_id")
            name, argz = call_meta.get(cid, (m.get("name") or "tool", ""))
            header = [
                "TOOL_CALL",
                f"name: {name}",
            ]
            if argz:
                header.append(f"arguments: {argz}")
            header.append("TOOL_RESULT_FOLLOWS:")
            flat.append({"role": "user", "content": "\n".join(header) + f"\n{m.get('content', '')}"})
        else:
            flat.append({"role": role, "content": m.get("content", "")})

    merged = []
    for m in flat:
        if (merged and merged[-1]["role"] == m["role"]
                and isinstance(merged[-1].get("content"), str)
                and isinstance(m.get("content"), str)):
            merged[-1]["content"] += "\n" + m["content"]
        else:
            merged.append(dict(m))
    return merged


async def _run_one_query_react(
    *, oai: AsyncOpenAI, mcp_client: Client, searcher_openai_tools, args, qid, qtext, out_dir
):
    """Fair ReAct baseline: same loop / retriever / tools as VISTA, but no
    dashboard and no context tools. When the prompt exceeds the budget, drop the
    oldest tool-rounds (truncation)."""
    all_tools = list(searcher_openai_tools)
    _cm0 = getattr(args, "react_cm", "truncate")
    if _cm0 == "active_compress":
        all_tools = all_tools + [_ACC_TOOL]          # agent-controlled lossy compression
    elif _cm0 == "fold":
        all_tools = all_tools + [_FOLD_BRANCH_TOOL]  # Context-Folding: branch/return
    overhead = _tool_overhead_tokens(all_tools)
    initial_user = format_query(qtext, args.query_template)
    if _cm0 == "active_compress":
        initial_user += ("\n\nStrategy: this question is defined by several largely independent clues. Work "
                         "clue by clue: search to confirm a clue, then immediately call compress_context("
                         "summary) to lock its verified facts and evidence (with source doc IDs) into your "
                         "knowledge block and clear the raw search results behind it. Treat 'confirm a clue "
                         "-> consolidate it' as one repeating step you take after each clue, so your working "
                         "context holds your accumulated conclusions rather than a growing pile of "
                         "transcripts. Consolidating keeps your verified findings; only the raw results are "
                         "dropped, so compress early and often rather than waiting.")
    elif _cm0 == "fold":
        initial_user += ("\n\nStrategy: this question is defined by several largely independent identifying "
                         "clues. Rather than searching them all in one growing context, work clue by clue — "
                         "for each clue, branch(description, prompt) a sub-agent to research it and report "
                         "back a short summary, so candidates are verified in isolation and your main "
                         "context stays small. Branching per clue is encouraged; gather the returned "
                         "summaries and reconcile them into the final answer.")
    messages = [{"role": "user", "content": initial_user}]
    result_entries, tool_counts = [], {}
    usage_in = usage_out = usage_total = 0
    budget_used = 0  # new-content meter: generated tokens + newly ingested tool results
    final_text, status, n_trunc = "", "incomplete", 0
    try:
        ratio = float(os.getenv("SM_PREFLIGHT_TARGET_RATIO", "0.98"))
    except ValueError:
        ratio = 0.98
    budget = max(1, int(args.max_context_size * ratio) - overhead)

    cm = getattr(args, "react_cm", "truncate")

    def _reduce_ctx(msgs):
        # summary mode persistently compacts `messages` before this; truncate is its
        # safety net. clear/mask/skeleton rewrite oldest observations in place.
        # active_compress/fold are agent-tool modes (the model itself compresses or
        # branches): when it chooses not to, fall back to whole-round truncation —
        # the SAME safety net react gets — so the context still fits. Without this
        # the no-op stub rewrite leaves the context over-window and the episode dies
        # at "starved" after 3-4 steps (active_compress) instead of running normally.
        if cm in ("truncate", "summary", "active_compress", "fold"):
            return _truncate_to_budget(msgs, budget)
        return _rewrite_to_fit(msgs, budget, cm)

    step = 0
    while budget_used < args.max_total_tokens:
        step += 1
        if cm == "summary":
            messages, _sum_tok = await _slim_summarize(
                oai, args.model, messages, budget, args.max_tokens, args.temperature)
            if _sum_tok:
                n_trunc += 1
                budget_used += _sum_tok  # the summarization call costs new tokens
        api_messages, dropped, starved = _reduce_ctx(messages)
        if dropped:
            n_trunc += 1
        if starved:
            # Trimmed every round but the query and it still will not fit the
            # window. Stop searching and commit an answer in place from whatever
            # fits (the forced-close block below runs on the truncated context).
            status = "context_overflow"
            break
        if _is_gemini(args.model) and args.gemini_fold_tool_history:
            api_messages = _gemini_safe_messages(api_messages)
        try:
            resp = await _chat_completion_with_retries(
                oai,
                model=args.model, messages=api_messages, tools=all_tools,
                tool_choice="auto", max_tokens=args.max_tokens, temperature=args.temperature,
                extra_headers={"X-TraceId": uuid.uuid4().hex},
            )
        except Exception as exc:
            print(f"[Error] query={qid} step={step} model call failed: {exc}")
            status = "model_error"; break
        u = getattr(resp, "usage", None)
        if u:
            usage_in += getattr(u, "prompt_tokens", 0) or 0
            usage_out += getattr(u, "completion_tokens", 0) or 0
            usage_total += getattr(u, "total_tokens", 0) or 0
            budget_used += getattr(u, "completion_tokens", 0) or 0  # new generation counts
        msg = resp.choices[0].message
        assistant_text = msg.content or ""
        tool_calls = msg.tool_calls or []
        if not tool_calls:
            dsml_calls, cleaned_text = _parse_dsml_tool_calls(assistant_text)
            if dsml_calls:
                tool_calls = dsml_calls
                assistant_text = cleaned_text
        if not tool_calls and _is_gemini(args.model) and _looks_like_text_tool_call(assistant_text):
            # Gemini drifted to writing a tool call as text. Correct without
            # polluting history with the textual call, and retry.
            messages.append({"role": "user", "content": _DRIFT_CORRECTION})
            continue
        assistant_record = {"role": "assistant", "content": assistant_text}
        if tool_calls:
            assistant_record["tool_calls"] = [
                {"id": tc.id, "type": "function",
                 "function": {"name": tc.function.name, "arguments": tc.function.arguments}}
                for tc in tool_calls
            ]
        messages.append(assistant_record)
        if not tool_calls:
            final_text = assistant_text; status = "completed"; break
        for tc_i, tc in enumerate(tool_calls):
            name = tc.function.name
            try:
                tool_args = json.loads(tc.function.arguments or "{}")
            except Exception:
                tool_args = {}
            tool_counts[name] = tool_counts.get(name, 0) + 1
            if status == "budget_exhausted":
                budget_left = max(0, args.max_total_tokens - budget_used)
                budget_notice = (
                    "[TRAJECTORY_BUDGET_EXHAUSTED]\n"
                    f"The result of {name} was NOT executed because the per-query "
                    f"new-content budget is already exhausted.\n"
                    f"Budget used: ~{budget_used:,} tok | budget left: ~{budget_left:,} tok | "
                    f"budget limit: ~{args.max_total_tokens:,} tok."
                )
                result_entries.append(
                    {"type": "tool_call", "tool_name": name, "arguments": tool_args, "output": budget_notice}
                )
                messages.append({
                    "role": "tool",
                    "tool_call_id": tc.id,
                    "name": name,
                    "content": budget_notice,
                })
                continue
            # ── Active Context Compression: agent-driven lossy compression (no MCP) ──
            if name == "compress_context" and _cm0 == "active_compress":
                summ = tool_args.get("summary", "") if isinstance(tool_args, dict) else ""
                # compress the history BEFORE this assistant turn; keep the turn + result
                head_and_old, current = messages[:-1], messages[-1]
                compressed, n_drop = _apply_active_compress(head_and_old, summ)
                messages = compressed + [current]
                note = f"Compressed {n_drop} earlier round(s) into a knowledge block (not recoverable)."
                tmsg = {"role": "tool", "tool_call_id": tc.id, "name": name, "content": note}
                result_entries.append({"type": "tool_call", "tool_name": name, "arguments": tool_args, "output": note})
                messages.append(tmsg)
                budget_used += _msg_tokens({"content": summ})
                continue
            # ── Context-Folding: run a sub-agent, only its summary returns (no fold trace) ──
            if name == "branch" and _cm0 == "fold":
                remaining = max(0, args.max_total_tokens - budget_used)
                branch_cap = max(1, args.max_total_tokens // 4)   # no single branch drains the run
                sub_budget = min(remaining, branch_cap)
                summ, spent, (b_in, b_out, b_tot) = await _run_branch(
                    oai, mcp_client, args.model, searcher_openai_tools,
                    tool_args.get("description", "") if isinstance(tool_args, dict) else "",
                    tool_args.get("prompt", "") if isinstance(tool_args, dict) else "",
                    args.max_tokens, args.temperature, sub_budget, budget,
                    tool_timeout_seconds=args.tool_timeout_seconds,
                    tool_retries=args.tool_retries)  # budget = main hard window
                usage_in += b_in; usage_out += b_out; usage_total += b_tot  # count sub-agent API cost
                ret = f"[branch result]\n{summ}"
                tmsg = {"role": "tool", "tool_call_id": tc.id, "name": name, "content": ret}
                result_entries.append({"type": "tool_call", "tool_name": name, "arguments": tool_args, "output": ret})
                messages.append(tmsg)
                budget_used += spent + _msg_tokens(tmsg)  # branch work + returned summary
                continue
            try:
                res = await _call_mcp_tool_with_retries(
                    mcp_client,
                    name,
                    tool_args,
                    timeout_seconds=args.tool_timeout_seconds,
                    retries=args.tool_retries,
                )
                output = _extract_mcp_text(res)
            except Exception as exc:
                if _is_mcp_transport_error(exc):
                    raise
                output = f"Error: tool {name} failed: {exc}"
            tool_content = output if isinstance(output, str) else json.dumps(output, ensure_ascii=False)
            try:
                _parsed = json.loads(tool_content)
                if isinstance(_parsed, list):
                    tool_content = json.dumps({"results": _parsed}, ensure_ascii=False)
            except Exception:
                pass
            tool_msg = {"role": "tool", "tool_call_id": tc.id, "name": name, "content": tool_content}
            tok_est = _msg_tokens(tool_msg)
            if budget_used + tok_est > args.max_total_tokens:
                budget_left = max(0, args.max_total_tokens - budget_used)
                budget_notice = (
                    "[TRAJECTORY_BUDGET_EXHAUSTED]\n"
                    f"The result of {name} (~{tok_est:,} tok) was NOT added because it would "
                    f"exceed the per-query new-content budget.\n"
                    f"Budget used: ~{budget_used:,} tok | budget left: ~{budget_left:,} tok | "
                    f"budget limit: ~{args.max_total_tokens:,} tok."
                )
                result_entries.append(
                    {"type": "tool_call", "tool_name": name, "arguments": tool_args, "output": budget_notice}
                )
                messages.append({
                    "role": "tool",
                    "tool_call_id": tc.id,
                    "name": name,
                    "content": budget_notice,
                })
                status = "budget_exhausted"
                continue
            result_entries.append(
                {"type": "tool_call", "tool_name": name, "arguments": tool_args, "output": output}
            )
            messages.append(tool_msg)
            budget_used += tok_est  # newly ingested content counts

        if status != "incomplete":
            break

    if status == "incomplete":
        status = "budget_exhausted"

    # No committed answer (budget/context exhaustion, OR the model emitted a blank
    # no-tool turn — common when a lossy stub policy degrades the context enough
    # that the model gives up mid-search): one final no-tools turn so the baseline
    # commits an answer from the evidence it already has (same standard as vista).
    # Keeps its status label; the answer is still scored.
    if status != "model_error" and not final_text.strip():
        # At close, the in-loop reduction policy no longer matters — a lossy stub
        # rewrite can still leave the context "starved" (over the window), which
        # makes the closing API call fail and yields a blank answer. Drop whole
        # oldest rounds here so the close context reliably fits and the model can
        # commit an answer from its most recent evidence.
        close_msgs, _, _ = _truncate_to_budget(messages, budget)
        if _is_gemini(args.model) and args.gemini_fold_tool_history:
            close_msgs = _gemini_safe_messages(close_msgs)
        final_text = await _forced_final_answer(
            oai, args.model, close_msgs, args.max_tokens, args.temperature)

    result_entries.append({"type": "output_text", "tool_name": None, "arguments": None, "output": final_text})
    record = {
        "metadata": {"model": args.model, "method": f"react_{cm}",
                     "max_output_tokens": args.max_tokens, "max_context_size": args.max_context_size,
                     "max_total_tokens": args.max_total_tokens, "budget_used": budget_used,
                     "truncations": n_trunc, "steps": step, "output_dir": str(out_dir)},
        "query_id": qid, "tool_call_counts": tool_counts,
        "usage": {"input_tokens": usage_in, "output_tokens": usage_out, "total_tokens": usage_total},
        "status": status,
        "retrieved_docids": extract_retrieved_docids_from_result(result_entries),
        "result": result_entries,
    }
    out_dir.mkdir(parents=True, exist_ok=True)
    ts = datetime.utcnow().strftime("%Y%m%dT%H%M%S%fZ")
    fname = out_dir / f"run_{ts}.json"
    with open(fname, "w", encoding="utf-8") as f:
        json.dump(record, f, indent=2, default=str)
    print(f"[{qid or 'single'}] react status={status} steps={step} trunc={n_trunc} new_tok={budget_used:,} billed={usage_total:,} tools={tool_counts} -> {fname.name}")


async def _run_one_query(
    *,
    oai: AsyncOpenAI,
    mcp_client: Client,
    searcher_openai_tools,
    args,
    qid: str,
    qtext: str,
    out_dir: Path,
):
    if args.archive_placeholder_style == "locabench":
        os.environ["SM_ARCHIVE_ORIGINAL_PLACEHOLDER"] = "1"
    else:
        os.environ.pop("SM_ARCHIVE_ORIGINAL_PLACEHOLDER", None)

    # Per-query workspace + payload dirs.
    work_root = out_dir / "_workspaces" / (qid or "single")
    workspace_dir = work_root / "context_workspace"
    payload_dir = work_root / "payloads_public"
    workspace_dir.mkdir(parents=True, exist_ok=True)
    payload_dir.mkdir(parents=True, exist_ok=True)
    # Fresh state for this query.
    state_file = workspace_dir / "workspace_state.json"
    if state_file.exists():
        state_file.unlink()
    event_file = work_root / "events.jsonl"
    if event_file.exists():
        event_file.unlink()

    # Per-query ISOLATED cw_server instance. Concurrent queries must NOT share the
    # cw_server module or a global CONTEXT_WORKSPACE_DIR env var: under asyncio the
    # env var set at turn start is read much later (after the model-call await), by
    # which time another task has overwritten it, so archive/recover operate on the
    # wrong workspace (and the shared module's tool objects race). We load a fresh
    # module per query and patch its path resolvers to this query's dirs, so the
    # tools never touch global state.
    _q_cw = _load_standalone(f"_vista_cw_{(qid or 'single')}_{uuid.uuid4().hex[:8]}",
                             _CW_DIR / "server.py")
    _q_cw.get_workspace_dir = lambda _d=workspace_dir: _d
    _q_cw.public_payload_dir = lambda _d=payload_dir: _d

    manager = WorkspaceManager(
        workspace_dir,
        token_budget=args.max_context_size,
        public_payload_dir=payload_dir,
    )

    # ── VISTA ablations (default none = full method, all gates below are no-ops) ──
    ablate = getattr(args, "vista_ablate", "none")
    show_dashboard = ablate not in ("no_dashboard", "auto_archive")
    allow_recover = ablate != "no_recovery"
    allow_agent_archive = ablate != "auto_archive"

    context_specs = _context_tool_specs(_q_cw, args.enable_delete)
    context_openai_tools = _context_tools_to_openai(context_specs)
    # Filter the tools EXPOSED to the agent per ablation. context_fns keeps every
    # function (so the harness auto-archive can still archive internally even when
    # the agent is not given the archive tool).
    def _tname(t):
        return (t.get("function") or {}).get("name")
    if not allow_recover:
        context_openai_tools = [t for t in context_openai_tools if _tname(t) != "context_workspace_recover"]
    if not allow_agent_archive:
        context_openai_tools = [t for t in context_openai_tools if _tname(t) != "context_workspace_archive"]
    all_tools = list(searcher_openai_tools) + context_openai_tools
    recovery_tools = [
        t for t in context_openai_tools
        if (t.get("function") or {}).get("name") in {"context_workspace_archive", "context_workspace_delete"}
    ]
    context_fns = {s["name"]: s["fn"] for s in context_specs}
    context_bypass = {"context_workspace_archive"}

    # Initial user message = formatted query + the context-management protocol.
    protocol = CONTEXT_PROTOCOL
    if not show_dashboard:
        # No dashboard is shown this run; do not tell the agent to read one.
        protocol = protocol.replace(
            "A <context_workspace_status> dashboard is shown every turn as a compact map of context blocks. ",
            "")
    if not allow_agent_archive:
        protocol = protocol.replace(
            "Archive a block only when you are certain it has no further value — a near-duplicate search that "
            "returned nothing new, or results clearly unrelated to every part of the task; keep everything else, "
            "including anything you are not sure about. ",
            "When context is full the system automatically sets aside the oldest block; you cannot archive "
            "yourself. ")
    initial_user = format_query(qtext, args.query_template) + protocol
    messages = [{"role": "user", "content": initial_user}]
    manager.register_message(messages[0], 0)
    manager.set_overhead(_tool_overhead_tokens(all_tools))

    result_entries = []
    tool_counts = {}
    usage_in = usage_out = usage_total = 0
    budget_used = 0  # new-content meter: generated tokens + newly ingested tool results
    n_rejected = 0   # tool results rejected by the hard-limit gate (agent must archive)
    n_protocol_violations = 0
    n_auto_archived = 0  # oldest blocks the harness archived because the model
                         # stayed in must_archive without archiving (gate defied)
    search_admitted_counts = {}
    must_archive_before_external = False
    control_notice_steps = {}            # tool_call_id -> step the control notice was created
    control_notice_ttl = _control_notice_ttl()
    empty_retry_count = 0                 # consecutive empty (no tool, no text) responses
    max_empty_retries = int(os.getenv("SM_EMPTY_OUTPUT_RETRIES", "3"))
    # External actions (search/get_document) the model requested but that were not
    # executed because the hard context limit was hit. After the model frees space
    # with an archive, the harness replays these itself (no extra LLM call) — the
    # model already issued them, so this just executes the calls it already made.
    auto_retry = os.getenv("SM_AUTO_RETRY_AFTER_ARCHIVE", "1") != "0"
    pending_external_retry = []           # list of (name, args) to replay after archive
    must_archive_at_turn_start = False
    final_text = ""
    status = "incomplete"

    try:
        gate_ratio = float(os.getenv("SM_GATE_HARD_RATIO", "1.0"))
    except ValueError:
        gate_ratio = 1.0
    gate_ratio = max(0.50, min(1.00, gate_ratio))
    hard_available = int(args.max_context_size * gate_ratio)  # hard context cap (e.g. 32800)
    _dbg = os.getenv("SM_DEBUG_BUDGET")

    def _write_event(kind: str, **fields):
        rec = {
            "time": datetime.utcnow().isoformat() + "Z",
            "query_id": qid,
            "step": step,
            "kind": kind,
            "budget_used": budget_used,
            "budget_limit": args.max_total_tokens,
        }
        rec.update(fields)
        event_file.parent.mkdir(parents=True, exist_ok=True)
        with event_file.open("a", encoding="utf-8") as f:
            f.write(json.dumps(rec, ensure_ascii=False, default=str) + "\n")

    def _preview(value, limit: int = 1000) -> str:
        text = value if isinstance(value, str) else json.dumps(value, ensure_ascii=False, default=str)
        return text[:limit]

    def _spend_new_tokens(tok: int) -> bool:
        """Charge newly-created trajectory content without crossing the run budget."""
        nonlocal budget_used
        tok = max(0, int(tok or 0))
        if budget_used + tok > args.max_total_tokens:
            return False
        budget_used += tok
        return True

    def _overhead_now(tools):
        if not show_dashboard:
            return _tool_overhead_tokens(tools)
        dash = manager.get_dashboard()
        dmsg = {"role": "user",
                "content": f"<context_workspace_status>\n{dash}\n</context_workspace_status>"}
        try:
            dtok = count_msg_tokens([dmsg], _TKT_ENC)
        except Exception:
            dtok = max(1, len(json.dumps([dmsg], ensure_ascii=False)) // 3)
        return _tool_overhead_tokens(tools) + dtok

    async def _replay_pending():
        """Replay the model's previously-rejected external calls after an archive,
        without spending an LLM call. Synthesizes one assistant turn carrying the
        same calls (new ids) and runs each through the same admission gate. Any call
        that still does not fit is re-stashed and the model archives again next turn.
        """
        nonlocal must_archive_before_external, n_rejected, status
        actions = list(pending_external_retry)
        pending_external_retry.clear()
        synth = [(name, a, f"retry_{uuid.uuid4().hex[:8]}") for (name, a) in actions]
        assistant_record = {
            "role": "assistant", "content": "",
            "tool_calls": [
                {"id": tcid, "type": "function",
                 "function": {"name": name, "arguments": json.dumps(a, ensure_ascii=False)}}
                for (name, a, tcid) in synth
            ],
        }
        messages.append(assistant_record)
        manager.register_message(assistant_record, len(messages) - 1)
        _write_event("auto_retry_replay", n_actions=len(synth),
                     actions=[name for (name, a, tcid) in synth])
        overhead_tok = _overhead_now(all_tools)
        for name, tool_args, tcid in synth:
            tool_counts[name] = tool_counts.get(name, 0) + 1
            _write_event("tool_call", tool_name=name, arguments=tool_args, source="auto_retry")
            # A prior replayed call in this batch overflowed → re-stash the rest.
            if must_archive_before_external:
                pending_external_retry.append((name, tool_args))
                notice = _format_protocol_violation(name, tool_args)
                tmsg = {"role": "tool", "tool_call_id": tcid, "name": name, "content": notice}
                messages.append(tmsg); manager.register_message(tmsg, len(messages) - 1)
                control_notice_steps[tcid] = step
                _write_event("tool_result", tool_name=name, arguments=tool_args, source="auto_retry",
                             disposition="protocol_violation_not_executed", charged_tokens=0,
                             budget_exempt_reason="dynamic_protocol_notice")
                continue
            if name in context_fns:
                try:
                    output = context_fns[name](tool_args)
                    raw_output = output
                except Exception as exc:
                    output = f"Error: {name} failed: {exc}"
                    raw_output = output
            else:
                try:
                    res = await _call_mcp_tool_with_retries(
                        mcp_client,
                        name,
                        tool_args,
                        timeout_seconds=args.tool_timeout_seconds,
                        retries=args.tool_retries,
                    )
                    raw_output = _extract_mcp_text(res); output = raw_output
                except Exception as exc:
                    if _is_mcp_transport_error(exc):
                        raise
                    output = f"Error: tool {name} failed: {exc}"; raw_output = output
            tool_content = output if isinstance(output, str) else json.dumps(output, ensure_ascii=False)
            try:
                _parsed = json.loads(tool_content)
                if isinstance(_parsed, list):
                    tool_content = json.dumps({"results": _parsed}, ensure_ascii=False)
            except Exception:
                pass
            tmsg = {"role": "tool", "tool_call_id": tcid, "name": name, "content": tool_content}
            tok_est = _msg_tokens(tmsg)
            single_result_limit = int(hard_available * args.max_result_context_fraction)
            if tok_est > single_result_limit:
                notice = _format_result_too_large(
                    name, tool_args, tok_est, hard_available, args.max_result_context_fraction
                )
                rmsg = {"role": "tool", "tool_call_id": tcid, "name": name, "content": notice}
                messages.append(rmsg); manager.register_message(rmsg, len(messages) - 1)
                control_notice_steps[tcid] = step
                _write_event("tool_result", tool_name=name, arguments=tool_args, source="auto_retry",
                             disposition="result_too_large_rejected",
                             raw_result_tokens=tok_est, charged_tokens=0,
                             budget_exempt_reason="dynamic_oversize_notice",
                             output_preview=_preview(notice),
                             raw_output_preview=_preview(raw_output if raw_output is not None else output))
                continue
            if budget_used + tok_est > args.max_total_tokens:
                pending_external_retry.append((name, tool_args))
                status = "budget_exhausted"
                _write_event("trajectory_budget_exhausted", source="auto_retry_postcheck",
                             tool_name=name, arguments=tool_args, raw_result_tokens=tok_est)
                return
            conv_now = manager.conv_tokens(messages, _TKT_ENC)
            hard_remaining = hard_available - overhead_tok - conv_now
            if tok_est > hard_remaining:
                n_rejected += 1
                must_archive_before_external = True
                pending_external_retry.append((name, tool_args))  # retry after next archive
                reject_content = _format_context_reject(
                    name, tool_args, tok_est, conv_now, overhead_tok, hard_available)
                rmsg = {"role": "tool", "tool_call_id": tcid, "name": name, "content": reject_content}
                messages.append(rmsg); manager.register_message(rmsg, len(messages) - 1)
                control_notice_steps[tcid] = step
                _write_event("tool_result", tool_name=name, arguments=tool_args, source="auto_retry",
                             disposition="context_limit_rejected", raw_result_tokens=tok_est,
                             charged_tokens=0, budget_exempt_reason="dynamic_rejection_notice")
            else:
                if not _spend_new_tokens(tok_est):
                    pending_external_retry.append((name, tool_args))
                    status = "budget_exhausted"
                    _write_event("trajectory_budget_exhausted", source="auto_retry_spend",
                                 tool_name=name, arguments=tool_args, raw_result_tokens=tok_est)
                    return
                result_entries.append(
                    {"type": "tool_call", "tool_name": name, "arguments": tool_args, "output": output})
                messages.append(tmsg)
                block_id = manager.register_message(tmsg, len(messages) - 1)
                try:
                    manager.set_tool_source(block_id, name, tool_args, tcid)
                except Exception:
                    pass
                if name not in context_fns:
                    try:
                        manager.set_tool_source(block_id, name, tool_args, tcid)
                    except Exception:
                        pass
                if name == "search":
                    search_admitted_counts[tool_args.get("query", "")] = \
                        search_admitted_counts.get(tool_args.get("query", ""), 0) + 1
                _write_event("tool_result", tool_name=name, arguments=tool_args, source="auto_retry",
                             disposition="admitted_external", charged_tokens=tok_est,
                             docids=re.findall(r'"docid"\s*:\s*"([^"]+)"', tool_content)[:10])

    step = 0
    # Primary bound is the new-content budget; --max-steps is only a safety net so a
    # model that keeps triggering rejections without archiving cannot spin forever.
    # Set --max-steps <= 0 to disable the safety net.
    while budget_used < args.max_total_tokens and (args.max_steps <= 0 or step < args.max_steps):
        step += 1
        # Resync in-memory cache with on-disk state written by last step's tools.
        manager._state_cache = None

        # When a result was rejected by the hard limit, only offer archive/delete
        # this turn so the model is forced to free space instead of spinning on
        # rejected search/get_document calls.
        turn_tools = recovery_tools if must_archive_before_external else all_tools
        if not turn_tools:
            turn_tools = all_tools
        must_archive_at_turn_start = must_archive_before_external

        # Auto-retry: space was just freed (must_archive cleared) and the model has
        # rejected external calls still pending → replay them here without an LLM
        # call, then let the next real turn reason over the results.
        if auto_retry and pending_external_retry and not must_archive_before_external:
            await _replay_pending()
            if status != "incomplete":
                break
            continue

        # Expire stale control notices (reject/protocol/budget/invalid) so they
        # stop occupying context once the agent has had a few turns to react.
        _expire_control_notices(messages, control_notice_steps, step, control_notice_ttl)

        # ── Per-turn overhead (tool schemas + dashboard) for the admission gate ──
        # NO preflight auto-offload: the agent itself must free space via
        # context_workspace_archive; oversized tool results are rejected until it
        # does (faithful strict_long_context behaviour).
        if show_dashboard:
            dashboard_text = manager.get_dashboard()
            _dash_msg = {"role": "user",
                         "content": f"<context_workspace_status>\n{dashboard_text}\n</context_workspace_status>"}
            try:
                _dash_tok = count_msg_tokens([_dash_msg], _TKT_ENC)
            except Exception:
                _dash_tok = max(1, len(json.dumps([_dash_msg], ensure_ascii=False)) // 3)
        else:
            _dash_tok = 0
        overhead_tok = _tool_overhead_tokens(turn_tools) + _dash_tok
        _asm = None
        if _dbg:
            try:
                _asm = manager.conv_tokens(messages, _TKT_ENC)
                _narch = len([b for b in manager.get_state().get("blocks", {}).values()
                              if b.get("status") in ("compressed", "archived")])
                print(f"[DBG {qid} step{step}] assembled={_asm} overhead={overhead_tok} "
                      f"used={_asm + overhead_tok}/{hard_available} archived_blocks={_narch}", flush=True)
            except Exception as _e:
                print(f"[DBG {qid} step{step}] debug failed: {_e}", flush=True)
        else:
            try:
                _asm = manager.conv_tokens(messages, _TKT_ENC)
            except Exception:
                _asm = None
        _write_event("turn_start", assembled_tokens=_asm, overhead_tokens=overhead_tok,
                     hard_context_limit=hard_available)

        # ── Assemble visible context + append the dashboard ──
        try:
            api_messages = manager.assemble(messages)
        except Exception as exc:
            if args.verbose:
                print(f"[{qid}] assemble failed, using raw messages: {exc}")
            api_messages = list(messages)
        if show_dashboard:
            dashboard_text = manager.get_dashboard()
            api_messages = api_messages + [
                {
                    "role": "user",
                    "content": f"<context_workspace_status>\n{dashboard_text}\n</context_workspace_status>",
                }
            ]
        if _is_gemini(args.model) and args.gemini_fold_tool_history:
            api_messages = _gemini_safe_messages(api_messages)

        # ── Call the backbone (OpenAI-compatible endpoint, e.g. Venus) ──
        turn_max_tokens = max(1, min(args.max_tokens, args.max_total_tokens - budget_used))
        try:
            resp = await _chat_completion_with_retries(
                oai,
                model=args.model,
                messages=api_messages,
                tools=turn_tools,
                tool_choice="auto",
                max_tokens=turn_max_tokens,
                temperature=args.temperature,
                extra_headers={"X-TraceId": uuid.uuid4().hex},
            )
        except Exception as exc:
            print(f"[Error] query={qid} step={step} model call failed: {exc}")
            status = "model_error"
            break

        u = getattr(resp, "usage", None)
        if u:
            usage_in += getattr(u, "prompt_tokens", 0) or 0
            usage_out += getattr(u, "completion_tokens", 0) or 0
            usage_total += getattr(u, "total_tokens", 0) or 0
        completion_tok = (getattr(u, "completion_tokens", 0) or 0) if u else 0

        choice = resp.choices[0]
        msg = choice.message
        assistant_text = msg.content or ""
        tool_calls = msg.tool_calls or []
        if not tool_calls:
            dsml_calls, cleaned_text = _parse_dsml_tool_calls(assistant_text)
            if dsml_calls:
                tool_calls = dsml_calls
                assistant_text = cleaned_text

        if not tool_calls and _is_gemini(args.model) and _looks_like_text_tool_call(assistant_text):
            # Gemini drifted to writing a tool call as text. Correct without
            # polluting history with the textual call, and retry.
            corrective = {"role": "user", "content": _DRIFT_CORRECTION}
            gen_tok = max(completion_tok, _msg_tokens({"role": "assistant", "content": assistant_text}))
            if not _spend_new_tokens(gen_tok):
                status = "budget_exhausted"
                _write_event("trajectory_budget_exhausted", source="assistant_drift_correction",
                             attempted_tokens=gen_tok)
                break
            messages.append(corrective)
            manager.register_message(corrective, len(messages) - 1)
            _write_event("assistant_drift_correction", charged_tokens=gen_tok,
                         corrective_tokens=_msg_tokens(corrective),
                         corrective_charged_tokens=0,
                         budget_exempt_reason="dynamic_harness_notice",
                         assistant_preview=_preview(assistant_text))
            continue

        if assistant_text and getattr(msg, "reasoning_content", None):
            result_entries.append(
                {"type": "reasoning", "tool_name": None, "arguments": None,
                 "output": [msg.reasoning_content]}
            )

        # Record assistant message into history (OpenAI format) + register block.
        assistant_record = {"role": "assistant", "content": assistant_text}
        if tool_calls:
            assistant_record["tool_calls"] = [
                {
                    "id": tc.id,
                    "type": "function",
                    "function": {"name": tc.function.name, "arguments": tc.function.arguments},
                }
                for tc in tool_calls
            ]
        assistant_tok = max(completion_tok, _msg_tokens(assistant_record))
        if not _spend_new_tokens(assistant_tok):
            status = "budget_exhausted"
            _write_event("trajectory_budget_exhausted", source="assistant",
                         attempted_tokens=assistant_tok,
                         assistant_preview=_preview(assistant_text))
            if args.verbose or _dbg:
                print(
                    f"[{qid}] step{step} BUDGET_EXHAUSTED assistant: "
                    f"used {budget_used} + gen {assistant_tok} > {args.max_total_tokens}",
                    flush=True,
                )
            break
        messages.append(assistant_record)
        manager.register_message(assistant_record, len(messages) - 1)
        _write_event(
            "assistant",
            charged_tokens=assistant_tok,
            completion_tokens=completion_tok,
            tool_calls=[
                {
                    "name": tc.function.name,
                    "arguments": tc.function.arguments,
                }
                for tc in tool_calls
            ],
            assistant_preview=_preview(assistant_text),
        )

        if not tool_calls:
            if assistant_text.strip():
                # Non-empty final answer → done.
                final_text = assistant_text
                status = "completed"
                break
            # Empty response (no tool calls, no text). This is a model hiccup, not
            # budget exhaustion: don't kill the episode while budget remains. Nudge
            # and retry up to max_empty_retries consecutive times.
            empty_retry_count += 1
            _write_event("empty_output_retry", attempt=empty_retry_count,
                         max_retries=max_empty_retries, budget_used=budget_used)
            if empty_retry_count <= max_empty_retries and budget_used < args.max_total_tokens:
                nudge = {
                    "role": "user",
                    "content": (
                        "Your previous response was empty. Either continue using the "
                        "available tools to gather more evidence, or, if you already "
                        "have enough information, reply with your final answer now."
                    ),
                }
                messages.append(nudge)
                manager.register_message(nudge, len(messages) - 1)
                continue
            # Retries exhausted → give up honestly (labelled distinctly from a real
            # budget_exhausted so the two failure modes are not conflated).
            final_text = assistant_text
            status = "empty_output"
            _write_event("empty_final_output", budget_used=budget_used,
                         max_total_tokens=args.max_total_tokens,
                         empty_retries=empty_retry_count)
            break

        # A turn with tool calls means the model is making progress: reset the
        # consecutive-empty counter so isolated hiccups don't accumulate.
        empty_retry_count = 0

        # ── Execute each tool call ──
        for tc in tool_calls:
            name = tc.function.name
            try:
                tool_args = json.loads(tc.function.arguments or "{}")
            except Exception:
                tool_args = {}
            tool_counts[name] = tool_counts.get(name, 0) + 1
            _write_event("tool_call", tool_name=name, arguments=tool_args)

            if name not in context_fns and must_archive_before_external:
                n_protocol_violations += 1
                # Same-turn cascade: an earlier call in THIS batch overflowed and set
                # must_archive, so this call was never executed. It is a real request
                # the model made — stash it to replay after the archive. (A turn that
                # *started* in must_archive is gated to archive-only, so this only
                # captures cascades, not defiance.)
                if auto_retry and not must_archive_at_turn_start:
                    pending_external_retry.append((name, tool_args))
                output = _format_protocol_violation(name, tool_args)
                tool_msg = {
                    "role": "tool",
                    "tool_call_id": tc.id,
                    "name": name,
                    "content": output,
                }
                tok_est = _msg_tokens(tool_msg)
                result_entries.append(
                    {"type": "tool_call", "tool_name": name, "arguments": tool_args, "output": output}
                )
                messages.append(tool_msg)
                manager.register_message(tool_msg, len(messages) - 1)
                control_notice_steps[tc.id] = step
                _write_event("tool_result", tool_name=name, arguments=tool_args,
                             disposition="protocol_violation_not_executed",
                             notice_tokens=tok_est, charged_tokens=0,
                             budget_exempt_reason="dynamic_protocol_notice",
                             output_preview=_preview(output))
                continue

            if name == "get_document" and _is_workspace_id(tool_args.get("docid", "")):
                n_protocol_violations += 1
                output = _format_invalid_workspace_id_tool_arg(name, tool_args)
                tool_msg = {
                    "role": "tool",
                    "tool_call_id": tc.id,
                    "name": name,
                    "content": output,
                }
                tok_est = _msg_tokens(tool_msg)
                result_entries.append(
                    {"type": "tool_call", "tool_name": name, "arguments": tool_args, "output": output}
                )
                messages.append(tool_msg)
                manager.register_message(tool_msg, len(messages) - 1)
                control_notice_steps[tc.id] = step
                _write_event("tool_result", tool_name=name, arguments=tool_args,
                             disposition="invalid_workspace_id_not_executed",
                             notice_tokens=tok_est, charged_tokens=0,
                             budget_exempt_reason="dynamic_invalid_tool_argument_notice",
                             output_preview=_preview(output))
                continue

            raw_output = None
            if name in context_fns:
                try:
                    output = context_fns[name](tool_args)
                except Exception as exc:
                    output = f"Error: {name} failed: {exc}"
            else:
                try:
                    res = await _call_mcp_tool_with_retries(
                        mcp_client,
                        name,
                        tool_args,
                        timeout_seconds=args.tool_timeout_seconds,
                        retries=args.tool_retries,
                    )
                    raw_output = _extract_mcp_text(res)
                    output = raw_output
                except Exception as exc:
                    if _is_mcp_transport_error(exc):
                        raise
                    output = f"Error: tool {name} failed: {exc}"
                    raw_output = output

            # Some OpenAI->Gemini proxies map a tool message into a Gemini
            # function_response whose `response` must be an object (struct), not a
            # list. A search result is a JSON array, so wrap bare arrays in an
            # object. result_entries keeps the original output for docid parsing.
            tool_content = output if isinstance(output, str) else json.dumps(output, ensure_ascii=False)
            try:
                _parsed = json.loads(tool_content)
                if isinstance(_parsed, list):
                    tool_content = json.dumps({"results": _parsed}, ensure_ascii=False)
            except Exception:
                pass
            tool_msg = {
                "role": "tool",
                "tool_call_id": tc.id,
                "name": name,
                "content": tool_content,
            }
            tok_est = _msg_tokens(tool_msg)

            if name in context_bypass:
                # LOCAbench bypasses only archive: the agent must always be able
                # to free context. The archive receipt is still registered so the
                # workspace/dashboard/pruning behavior matches the original path.
                result_entries.append(
                    {"type": "tool_call", "tool_name": name, "arguments": tool_args, "output": output}
                )
                manager._state_cache = None
                messages.append(tool_msg)
                manager.register_message(tool_msg, len(messages) - 1)
                manager.update_dashboard_cache()
                manager._state_cache = None
                if not str(output).startswith("Error:"):
                    must_archive_before_external = False
                _write_event("tool_result", tool_name=name, arguments=tool_args,
                             disposition="admitted_context_tool_bypass", result_tokens=tok_est,
                             charged_tokens=0,
                             budget_exempt_reason="context_workspace_archive_bypass",
                             output_preview=_preview(output))
                continue

            # Single-result gate: if one external result consumes most of the context
            # window, archiving cannot produce a useful VISTA state. Reject it once
            # and force the model to use snippets, a narrower query, or another source.
            single_result_limit = int(hard_available * args.max_result_context_fraction)
            if name not in context_fns and tok_est > single_result_limit:
                notice = _format_result_too_large(
                    name, tool_args, tok_est, hard_available, args.max_result_context_fraction
                )
                big_msg = {"role": "tool", "tool_call_id": tc.id, "name": name, "content": notice}
                result_entries.append(
                    {"type": "tool_call", "tool_name": name, "arguments": tool_args, "output": notice}
                )
                messages.append(big_msg)
                manager.register_message(big_msg, len(messages) - 1)
                control_notice_steps[tc.id] = step
                _write_event("tool_result", tool_name=name, arguments=tool_args,
                             disposition="result_too_large_rejected",
                             raw_result_tokens=tok_est, charged_tokens=0,
                             budget_exempt_reason="dynamic_oversize_notice",
                             output_preview=_preview(notice),
                             raw_output_preview=_preview(raw_output if raw_output is not None else output))
                continue

            # Strict admission gate for non-bypass results (search / get_document
            # and recover/delete): reject if admitting would push the assembled
            # context past the hard limit. The agent must archive to free space,
            # then retry the call.
            if budget_used + tok_est > args.max_total_tokens:
                budget_left = max(0, args.max_total_tokens - budget_used)
                budget_msg = {
                    "role": "tool",
                    "tool_call_id": tc.id,
                    "name": name,
                    "content": (
                        "[TRAJECTORY_BUDGET_EXHAUSTED]\n"
                        f"The result of {name} (~{tok_est:,} tok) was NOT added because it would "
                        f"exceed the per-query new-content budget.\n"
                        f"Budget used: ~{budget_used:,} tok | budget left: ~{budget_left:,} tok | "
                        f"budget limit: ~{args.max_total_tokens:,} tok."
                    ),
                }
                summary_tok = _msg_tokens(budget_msg)
                result_entries.append(
                    {
                        "type": "tool_call",
                        "tool_name": name,
                        "arguments": tool_args,
                        "output": budget_msg["content"],
                    }
                )
                messages.append(budget_msg)
                manager.register_message(budget_msg, len(messages) - 1)
                control_notice_steps[tc.id] = step
                status = "budget_exhausted"
                _write_event("trajectory_budget_exhausted", source="tool_result",
                             tool_name=name, arguments=tool_args, raw_result_tokens=tok_est,
                             notice_tokens=summary_tok,
                             notice_admitted=True,
                             notice_charged_tokens=0,
                             budget_exempt_reason="dynamic_budget_notice",
                             output_preview=_preview(budget_msg["content"]),
                             raw_output_preview=_preview(raw_output if raw_output is not None else output))
                if args.verbose or _dbg:
                    print(
                        f"[{qid}] step{step} BUDGET_EXHAUSTED {name}: "
                        f"used {budget_used} + est {tok_est} > {args.max_total_tokens}",
                        flush=True,
                    )
                break

            conv_now = manager.conv_tokens(messages, _TKT_ENC)
            hard_remaining = hard_available - overhead_tok - conv_now
            if tok_est > hard_remaining:
                n_rejected += 1
                reject_content = _format_context_reject(
                    name, tool_args, tok_est, conv_now, overhead_tok, hard_available
                )
                reject_msg = {
                    "role": "tool",
                    "tool_call_id": tc.id,
                    "name": name,
                    "content": reject_content,
                }
                summary_tok = _msg_tokens(reject_msg)
                result_entries.append(
                    {
                        "type": "tool_call",
                        "tool_name": name,
                        "arguments": tool_args,
                        "output": reject_msg["content"],
                    }
                )
                messages.append(reject_msg)
                manager.register_message(reject_msg, len(messages) - 1)
                control_notice_steps[tc.id] = step
                must_archive_before_external = True
                # This external call could not fit; stash it to replay once the
                # model frees space with an archive (no extra LLM call needed).
                if auto_retry:
                    pending_external_retry.append((name, tool_args))
                _write_event("tool_result", tool_name=name, arguments=tool_args,
                             disposition="context_limit_rejected",
                             raw_result_tokens=tok_est, notice_tokens=summary_tok,
                             charged_tokens=0,
                             budget_exempt_reason="dynamic_rejection_notice",
                             output_preview=_preview(reject_content),
                             raw_output_preview=_preview(raw_output if raw_output is not None else output))
                if args.verbose or _dbg:
                    print(f"[{qid}] step{step} REJECT {name}: est {tok_est} > remaining {hard_remaining}", flush=True)
            else:
                if not _spend_new_tokens(tok_est):
                    status = "budget_exhausted"
                    _write_event("trajectory_budget_exhausted", source="external_tool_result_postcheck",
                                 tool_name=name, arguments=tool_args, raw_result_tokens=tok_est,
                                 output_preview=_preview(output),
                                 raw_output_preview=_preview(raw_output if raw_output is not None else output))
                    break
                result_entries.append(
                    {"type": "tool_call", "tool_name": name, "arguments": tool_args, "output": output}
                )
                messages.append(tool_msg)
                block_id = manager.register_message(tool_msg, len(messages) - 1)
                try:
                    if name not in context_fns:
                        manager.set_tool_source(block_id, name, tool_args, tc.id)
                except Exception:
                    pass
                if name == "search":
                    query = tool_args.get("query", "")
                    search_admitted_counts[query] = search_admitted_counts.get(query, 0) + 1
                _write_event("tool_result", tool_name=name, arguments=tool_args,
                             disposition=("admitted_context_tool" if name in context_fns else "admitted_external"),
                             charged_tokens=tok_est,
                             docids=re.findall(r'"docid"\s*:\s*"([^"]+)"', tool_content)[:10],
                             output_preview=_preview(output),
                             raw_output_preview=_preview(raw_output if raw_output is not None else output))

        # Auto-archive fallback for a defied gate. This turn STARTED in
        # must_archive (only archive/delete were offered), yet the model ignored
        # it and emitted external calls instead — all protocol-violated, freeing
        # nothing. The backbone API rejects a forced tool_choice (verified: 502),
        # so we cannot hard-compel an archive; instead the harness frees space
        # itself by archiving the OLDEST visible block, and warns loudly so we can
        # see how often (and on which queries) the model defies the gate.
        if (status == "incomplete" and must_archive_at_turn_start
                and must_archive_before_external):
            blocks = (manager.get_state() or {}).get("blocks", {})
            visible = sorted(
                (b for b in blocks.values() if b.get("status") == "visible"),
                key=lambda b: int(re.sub(r"\D", "", str(b.get("id", "B0"))) or 0),
            )
            if visible:
                oldest = visible[0]["id"]
                warn = (f"VISTA WARNING [{qid} step{step}]: model stayed in "
                        f"must_archive but archived nothing (emitted external calls "
                        f"instead); harness auto-archiving oldest block {oldest}.")
                print(warn, flush=True)
                try:
                    out = context_fns["context_workspace_archive"]({"block_id": oldest})
                    manager._state_cache = None
                    manager.update_dashboard_cache()
                    manager._state_cache = None
                    if not str(out).startswith("Error:"):
                        must_archive_before_external = False
                        n_auto_archived += 1
                    _write_event("auto_archive_oldest", source="must_archive_defied",
                                 block_id=oldest, step=step, warning=warn,
                                 disposition="harness_forced_archive")
                except Exception as exc:
                    _write_event("auto_archive_failed", block_id=oldest,
                                 step=step, error=str(exc)[:200])
            else:
                _write_event("auto_archive_nothing_visible", step=step)

        if status != "incomplete":
            break

    if status == "incomplete":
        status = "budget_exhausted"

    # Budget exhausted without a committed answer: take one final no-tools turn to
    # force the model to answer from the evidence it already gathered. Stays
    # labelled budget_exhausted; the answer is still scored by the evaluator.
    if status == "budget_exhausted" and not final_text.strip():
        try:
            close_msgs = manager.assemble(messages)
        except Exception:
            close_msgs = list(messages)
        if show_dashboard:
            close_msgs = close_msgs + [
                {"role": "user",
                 "content": f"<context_workspace_status>\n{manager.get_dashboard()}\n</context_workspace_status>"}
            ]
        if _is_gemini(args.model) and args.gemini_fold_tool_history:
            close_msgs = _gemini_safe_messages(close_msgs)
        final_text = await _forced_final_answer(
            oai, args.model, close_msgs, args.max_tokens, args.temperature)
        _write_event("budget_close_final_answer", had_answer=bool(final_text.strip()),
                     budget_used=budget_used)

    # The evaluator reads the final answer from the last result entry, which must
    # be an output_text. Always append it.
    result_entries.append(
        {"type": "output_text", "tool_name": None, "arguments": None, "output": final_text}
    )

    record = {
        "metadata": {
            "model": args.model,
            "method": ("vista_strict_lc_better_dashboard" if ablate == "none"
                       else f"vista_ablate_{ablate}"),
            "vista_ablate": ablate,
            "archive_placeholder_style": args.archive_placeholder_style,
            "max_output_tokens": args.max_tokens,
            "max_context_size": args.max_context_size,
            "max_total_tokens": args.max_total_tokens,
            "budget_used": budget_used,
            "steps": step,
            "rejected_results": n_rejected,
            "protocol_violations": n_protocol_violations,
            "auto_archived_oldest": n_auto_archived,
            "output_dir": str(out_dir),
            "event_file": str(event_file),
        },
        "query_id": qid,
        "tool_call_counts": tool_counts,
        "usage": {
            "input_tokens": usage_in,
            "output_tokens": usage_out,
            "total_tokens": usage_total,
        },
        "status": status,
        "retrieved_docids": extract_retrieved_docids_from_result(result_entries),
        "result": result_entries,
    }

    out_dir.mkdir(parents=True, exist_ok=True)
    ts = datetime.utcnow().strftime("%Y%m%dT%H%M%S%fZ")
    fname = out_dir / f"run_{ts}.json"
    with open(fname, "w", encoding="utf-8") as f:
        json.dump(record, f, indent=2, default=str)
    print(f"[{qid or 'single'}] status={status} steps={step} new_tok={budget_used:,} billed={usage_total:,} rejected={n_rejected} tools={tool_counts} -> {fname.name}")


def _load_queries(tsv_path: Path):
    queries = []
    with tsv_path.open(newline="", encoding="utf-8") as f:
        for row in csv.reader(f, delimiter="\t"):
            if len(row) != 2:
                continue
            queries.append((row[0].strip(), row[1].strip()))
    return queries


def _write_timeout_record(out_dir: Path, qid: str, mode: str, args, timeout_seconds: int):
    out_dir.mkdir(parents=True, exist_ok=True)
    method = f"react_{args.react_cm}" if mode == "react" else (
        "vista_strict_lc_better_dashboard" if args.vista_ablate == "none"
        else f"vista_ablate_{args.vista_ablate}"
    )
    record = {
        "metadata": {
            "model": args.model,
            "method": method,
            "max_output_tokens": args.max_tokens,
            "max_context_size": args.max_context_size,
            "max_total_tokens": args.max_total_tokens,
            "timeout_seconds": timeout_seconds,
            "budget_used": 0,
            "steps": 0,
            "output_dir": str(out_dir),
        },
        "query_id": qid,
        "tool_call_counts": {},
        "usage": {"input_tokens": 0, "output_tokens": 0, "total_tokens": 0},
        "status": "timeout",
        "retrieved_docids": [],
        "result": [{"type": "output_text", "tool_name": None, "arguments": None, "output": ""}],
    }
    ts = datetime.utcnow().strftime("%Y%m%dT%H%M%S%fZ")
    fname = out_dir / f"run_{ts}.json"
    with open(fname, "w", encoding="utf-8") as f:
        json.dump(record, f, indent=2, default=str)
    print(f"[{qid or 'single'}] status=timeout after {timeout_seconds}s -> {fname.name}", flush=True)


def _write_error_record(out_dir: Path, qid: str, mode: str, args, status: str, message: str):
    out_dir.mkdir(parents=True, exist_ok=True)
    record = {
        "query_id": qid,
        "metadata": {
            "mode": mode,
            "max_context_size": args.max_context_size,
            "max_total_tokens": args.max_total_tokens,
            "steps": 0,
            "output_dir": str(out_dir),
        },
        "tool_call_counts": {},
        "usage": {"input_tokens": 0, "output_tokens": 0, "total_tokens": 0},
        "status": status,
        "retrieved_docids": [],
        "result": [{"type": "output_text", "tool_name": None, "arguments": None, "output": message}],
    }
    ts = datetime.utcnow().strftime("%Y%m%dT%H%M%S%fZ")
    fname = out_dir / f"run_{ts}.json"
    with open(fname, "w", encoding="utf-8") as f:
        json.dump(record, f, indent=2, default=str)
    print(f"[{qid or 'single'}] status={status}: {message} -> {fname.name}", flush=True)


def _already_done(out_dir: Path):
    terminal_statuses = {
        "completed",
        "budget_exhausted",
        "empty_output",
        "timeout",
        "context_overflow",
        "model_error",
    }
    done = set()
    if out_dir.exists():
        for p in out_dir.glob("run_*.json"):
            try:
                with p.open() as f:
                    record = json.load(f)
                qid = record.get("query_id")
                if qid and record.get("status") in terminal_statuses:
                    done.add(str(qid))
            except Exception:
                continue
    return done


async def main():
    parser = argparse.ArgumentParser(
        description="VISTA (strict_lc_better_dashboard) agent for BrowseComp-Plus."
    )
    parser.add_argument("--query", default="topics-qrels/queries.tsv",
                        help="A .tsv dataset (qid<TAB>query) or a single query string.")
    parser.add_argument("--model", default="gemini-3-flash", help="Backbone model name (default: %(default)s).")
    parser.add_argument("--mcp-url", required=True,
                        help="URL of the BrowseComp-Plus searcher MCP server, e.g. http://127.0.0.1:8080/mcp")
    parser.add_argument("--output-dir", default="runs/bm25/vista_gemini_3_flash")
    parser.add_argument("--query-template", choices=["QUERY_TEMPLATE", "QUERY_TEMPLATE_NO_GET_DOCUMENT", "GAIA_TEMPLATE"],
                        default="QUERY_TEMPLATE", help="Default keeps get_document available for recovery.")
    parser.add_argument("--max-context-size", type=int, default=32800,
                        help="HARD context limit in tokens (default: %(default)s). vista: tool results are "
                             "rejected once assembled context + overhead would exceed this, forcing the agent "
                             "to archive; no preflight auto-offload. react: drives truncation.")
    parser.add_argument("--max-result-context-fraction", type=float, default=0.80,
                        help="Reject any single non-context tool result whose token count exceeds this "
                             "fraction of the hard context window. This prevents one huge result from "
                             "causing archive/retry loops. Default: %(default)s.")
    parser.add_argument("--max-total-tokens", type=int, default=196608,
                        help="New-content token budget for the whole query (default: %(default)s = 6x the "
                             "32k window). Counts only assistant-generated content/tool calls and newly "
                             "admitted non-bypass tool results (search/get_document and recover/delete). "
                             "Dashboard/status messages, archive receipts, rejected raw results, and "
                             "rejection/protocol notices are NOT charged. Primary stopping bound.")
    parser.add_argument("--max-tokens", type=int, default=10000, help="Max output tokens per turn.")
    parser.add_argument("--max-steps", type=int, default=150,
                        help="Safety net only: the loop is primarily bounded by --max-total-tokens. Caps "
                             "runaway archive/retry loops that never make progress. Set <=0 for no step cap.")
    parser.add_argument("--timeout-seconds", type=int, default=0,
                        help="Optional per-query wall-clock timeout. When >0, a timed-out query writes a "
                             "status=timeout run record with an empty answer so batch scoring can continue.")
    parser.add_argument("--tool-timeout-seconds", type=int, default=180,
                        help="Per-attempt wall-clock timeout for one MCP tool call. This prevents a single "
                             "hung SSE search/get_document call from blocking the whole query.")
    parser.add_argument("--tool-retries", type=int, default=2,
                        help="Retry count after a failed or timed-out MCP tool call. The final failure is "
                             "returned to the agent as a tool error instead of crashing the batch.")
    parser.add_argument("--query-retries", type=int, default=1,
                        help="Retry a whole query with a fresh MCP client after a fatal MCP transport error.")
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--num-threads", type=int, default=5, help="Concurrent queries.")
    parser.add_argument("--enable-delete", action="store_true", help="Expose context_workspace_delete.")
    parser.add_argument("--archive-placeholder-style", choices=["metadata", "locabench"], default="metadata",
                        help="metadata: archived blocks keep generic provenance metadata. locabench: use the "
                             "original LOCAbench placeholder format '[ARCHIVED:Bx Ln] replacement'.")
    parser.add_argument("--base-url", default=None,
                        help="OpenAI-compatible base URL (default: $OPENAI_BASE_URL or $LOCA_OPENAI_BASE_URL).")
    parser.add_argument("--gemini-fold-tool-history", action="store_true",
                        help="Fallback for broken Gemini-compatible gateways: fold historical tool calls/results "
                             "into text. Default is LOCAbench-style structured assistant.tool_calls + tool messages.")
    parser.add_argument("--mode", choices=["vista", "react"], default="vista",
                        help="vista = full method; react = fair append-only baseline with budget truncation.")
    parser.add_argument("--vista-ablate", choices=["none", "no_dashboard", "no_recovery", "auto_archive"],
                        default="none",
                        help="Ablate a VISTA component (for --mode vista). none = full method. no_dashboard = "
                             "hide the per-turn <context_workspace_status> dashboard (archive+recover stay). "
                             "no_recovery = remove context_workspace_recover so archiving is one-way (lossy). "
                             "auto_archive = remove the agent's archive tool; the harness auto-archives the "
                             "oldest block under pressure (no agent choice, no dashboard), recover stays.")
    parser.add_argument("--react-cm",
                        choices=["truncate", "clear", "mask", "skeleton", "summary",
                                 "active_compress", "fold"],
                        default="truncate",
                        help="Context management for --mode react. truncate = drop oldest whole rounds "
                             "(default ReAct). clear = Tool-result Clearing. mask = Stale-observation Masking. "
                             "skeleton = Skeleton Compression. summary = SLIM periodic summarization. "
                             "active_compress = Active Context Compression (agent calls compress_context). "
                             "fold = Context-Folding (agent calls branch/return_to_main sub-agents).")
    parser.add_argument("--verbose", action="store_true")
    args = parser.parse_args()

    base_url = args.base_url or os.getenv("OPENAI_BASE_URL") or os.getenv("LOCA_OPENAI_BASE_URL")
    api_key = os.getenv("OPENAI_API_KEY") or os.getenv("LOCA_OPENAI_API_KEY")
    if not base_url or not api_key:
        raise SystemExit(
            "Set an OpenAI-compatible endpoint via OPENAI_BASE_URL/OPENAI_API_KEY "
            "(or LOCA_OPENAI_BASE_URL/LOCA_OPENAI_API_KEY)."
        )
    oai = AsyncOpenAI(base_url=base_url, api_key=api_key)

    out_dir = Path(args.output_dir).expanduser().resolve()

    # Probe the MCP server once to get a stable tool schema, then close this
    # session. Each query gets its own MCP client below, so one broken SSE stream
    # cannot poison other workers.
    async with Client(SSETransport(url=args.mcp_url)) as probe_client:
        mcp_tools = await probe_client.list_tools()
    searcher_openai_tools = _mcp_tools_to_openai(mcp_tools)
    searcher_names = [t["function"]["name"] for t in searcher_openai_tools]
    if args.verbose:
        print("Searcher tools:", searcher_names)
    # Match the prompt to the tools actually served: if get_document is not
    # available, use the search-only template so the agent is not told to use
    # a tool it does not have.
    if "get_document" not in searcher_names and args.query_template == "QUERY_TEMPLATE":
        args.query_template = "QUERY_TEMPLATE_NO_GET_DOCUMENT"
        print("[vista] get_document not served; using QUERY_TEMPLATE_NO_GET_DOCUMENT")

    qstr = args.query.strip()
    is_tsv = qstr.lower().endswith(".tsv") and Path(qstr).is_file()

    async def _dispatch(qid, qt):
        async with Client(SSETransport(url=args.mcp_url)) as query_mcp_client:
            if args.mode == "react":
                await _run_one_query_react(
                    oai=oai, mcp_client=query_mcp_client, searcher_openai_tools=searcher_openai_tools,
                    args=args, qid=qid, qtext=qt, out_dir=out_dir,
                )
            else:
                await _run_one_query(
                    oai=oai, mcp_client=query_mcp_client, searcher_openai_tools=searcher_openai_tools,
                    args=args, qid=qid, qtext=qt, out_dir=out_dir,
                )

    if not is_tsv:
        await _dispatch("single", args.query)
        return

    queries = _load_queries(Path(qstr))
    done = _already_done(out_dir)
    remaining = [(qid, qt) for qid, qt in queries if qid not in done]
    print(f"Processing {len(remaining)} queries (skipping {len(done)} done) with "
          f"{args.num_threads} worker(s).")

    sem = asyncio.Semaphore(max(1, args.num_threads))

    async def _worker(qid, qt):
        async with sem:
            attempts = max(1, int(args.query_retries) + 1)
            for attempt in range(attempts):
                try:
                    if args.timeout_seconds > 0:
                        await asyncio.wait_for(_dispatch(qid, qt), timeout=args.timeout_seconds)
                    else:
                        await _dispatch(qid, qt)
                    return
                except asyncio.TimeoutError:
                    _write_timeout_record(out_dir, qid, args.mode, args, args.timeout_seconds)
                    return
                except Exception as exc:
                    if _is_mcp_transport_error(exc) and attempt < attempts - 1:
                        print(
                            f"[query retry] query={qid} MCP transport failed "
                            f"({type(exc).__name__}: {exc}); rebuilding MCP client "
                            f"attempt {attempt + 2}/{attempts}",
                            flush=True,
                        )
                        await asyncio.sleep(2.0)
                        continue
                    status = "mcp_transport_error" if _is_mcp_transport_error(exc) else "query_error"
                    _write_error_record(out_dir, qid, args.mode, args, status, f"{type(exc).__name__}: {exc}")
                    return

    tasks = [_worker(qid, qt) for qid, qt in remaining]
    with tqdm(total=len(tasks), desc="Queries", unit="q") as pbar:
        for fut in asyncio.as_completed(tasks):
            await fut
            pbar.update(1)


if __name__ == "__main__":
    asyncio.run(main())

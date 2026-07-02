# Copyright 2025 AxonRL Team. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Flexible parallel inference runner with configurable environments and tools."""

import contextlib
import io
import json
import os
import re
import signal
import sys
import time
import importlib
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
from typing import Optional, List, Dict, Any

import fire
import requests
from dotenv import load_dotenv
import random
from requests.exceptions import RequestException, HTTPError

# Suppress npm update notices
os.environ["npm_config_update_notifier"] = "false"
os.environ["NO_UPDATE_NOTIFIER"] = "true"
os.environ["NPM_CONFIG_UPDATE_NOTIFIER"] = "false"
os.environ["npm_config_loglevel"] = "error"
os.environ["NPM_CONFIG_LOGLEVEL"] = "error"

# Suppress FastMCP banner
os.environ["FASTMCP_SHOW_CLI_BANNER"] = "false"

# Suppress MCP server verbose output by default (can be overridden)
os.environ.setdefault("LOCA_QUIET", "1")

# Suppress MCP/FastMCP logging output
import logging
# Set root logger to WARNING to suppress INFO messages
logging.basicConfig(level=logging.WARNING, force=True)
logging.getLogger().setLevel(logging.WARNING)
# Suppress specific noisy loggers
for logger_name in ["mcp", "fastmcp", "mcp.server", "mcp.client", "httpx", "httpcore", "asyncio"]:
    logging.getLogger(logger_name).setLevel(logging.WARNING)

# Import all potential tools and wrappers
from gem.tools.mcp_tool import MCPTool
from gem.tools.mcp_server.programmatic_tool_calling.helper import ProgrammaticToolCallingTool
from gem.tools.tool_env_wrapper import ToolEnvWrapperClaimDone, ToolEnvWrapperOpenAI
from gem.tools.mcp_server.config_loader import build_server_config
from gem.tools.mcp_server.canvas.helper import get_canvas_stdio_config
from gem.tools.mcp_server.claim_done.helper import get_claim_done_stdio_config
from gem.tools.mcp_server.filesystem.helper import get_filesystem_stdio_config
from gem.tools.mcp_server.memory.helper import get_memory_stdio_config
from gem.tools.mcp_server.memory_tool.helper import get_memory_tool_stdio_config
from gem.tools.mcp_server.python_execute.helper import get_python_execute_stdio_config
from gem.tools.mcp_server.programmatic_tool_calling.helper import get_programmatic_tool_calling_stdio_config
from gem.tools.mcp_server.emails.helper import get_email_stdio_config
from gem.tools.mcp_server.excel.helper import get_excel_stdio_config
from gem.tools.mcp_server.terminal.helper import get_terminal_stdio_config
from gem.tools.mcp_server.google_cloud.helper import get_google_cloud_stdio_config
from gem.tools.mcp_server.google_sheet.helper import get_google_sheet_stdio_config
from gem.tools.mcp_server.pdf_tools.helper import get_pdf_tools_stdio_config
from gem.tools.mcp_server.calendar_server.helper import get_calendar_stdio_config
from gem.tools.mcp_server.woocommerce.helper import get_woocommerce_stdio_config
from gem.tools.mcp_server.snowflake.helper import get_snowflake_stdio_config
from inference.common.output_io import (
    build_task_workspace,
    write_all_trajectories_file,
    write_eval_file,
    write_json_file,
    write_results_file,
    write_trajectory_file,
)
from inference.common.trajectory_schema import (
    attach_conversation,
    attach_events,
    attach_metrics,
    attach_provider_payload,
    make_base_envelope,
)

load_dotenv()


_TOTAL_ROWS_RE = re.compile(r"\bTotal rows:\s*([\d,]+)", re.I)
_SHOWING_ROWS_RE = re.compile(r"\bResults\s*\(showing\s+([\d,]+)\s+rows\)", re.I)


def _append_warning_log(path: Optional[Path], message: str) -> None:
    """Best-effort per-task warning log for console-only safety signals."""
    if not path:
        return
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        ts = time.strftime("%Y-%m-%d %H:%M:%S")
        with path.open("a", encoding="utf-8") as f:
            f.write(f"[{ts}] {message.rstrip()}\n")
    except Exception:
        pass


def _partial_rows_info(content: str) -> str:
    """Extract partial row metadata already present in a tool transcript."""
    text = content or ""
    total_match = _TOTAL_ROWS_RE.search(text)
    showing_match = _SHOWING_ROWS_RE.search(text)
    if not total_match or not showing_match:
        return ""
    try:
        total = int(total_match.group(1).replace(",", ""))
        showing = int(showing_match.group(1).replace(",", ""))
    except ValueError:
        return ""
    if total > showing:
        return f"Shown rows: {showing:,} / Total rows: {total:,}"
    return ""


def _short_json(value: Any, max_chars: int = 320) -> str:
    """Compact bounded JSON/string rendering for tool-call metadata."""
    try:
        if isinstance(value, str):
            text = value
        else:
            text = json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    except Exception:
        text = str(value)
    text = " ".join(text.split())
    if len(text) > max_chars:
        return text[: max(0, max_chars - 3)] + "..."
    return text


def _result_shape_meta(raw_content: Any, max_chars: int = 360) -> str:
    """Return short, generic result-shape metadata with no task inference."""
    raw_text = raw_content if isinstance(raw_content, str) else json.dumps(raw_content, ensure_ascii=False)
    parsed = None
    if isinstance(raw_content, (list, dict)):
        parsed = raw_content
    else:
        text = raw_text.strip()
        if text:
            try:
                parsed = json.loads(text)
            except Exception:
                parsed = None

    parts: List[str] = []
    if isinstance(parsed, list):
        parts.append(f"Result shape: list, returned_items={len(parsed):,}")
    elif isinstance(parsed, dict):
        parts.append(f"Result shape: dict, keys={len(parsed):,}")
        list_fields = [
            f"{k}={len(v):,}"
            for k, v in parsed.items()
            if isinstance(k, str) and isinstance(v, list)
        ][:4]
        if list_fields:
            parts.append("List fields: " + ", ".join(list_fields))
    else:
        lines = raw_text.splitlines()
        parts.append(f"Result shape: text, chars={len(raw_text):,}, lines={len(lines):,}")
        if lines and "," in lines[0]:
            cols = len(lines[0].split(","))
            parts.append(f"CSV-like header columns={cols:,}")

    meta = "\n".join(parts)
    if len(meta) > max_chars:
        meta = meta[: max(0, max_chars - 3)] + "..."
    return meta


def _tool_result_metadata(tool_args: Any, raw_content: Any, max_chars: int = 760) -> str:
    """Short metadata preserved when a tool result is represented by a placeholder."""
    lines = []
    args = _short_json(tool_args, 320)
    if args:
        lines.append(f"Args: {args}")
    shape = _result_shape_meta(raw_content, 360)
    if shape:
        lines.append(shape)
    text = "\n".join(lines)
    if len(text) > max_chars:
        text = text[: max(0, max_chars - 3)] + "..."
    return text


@contextlib.contextmanager
def suppress_stdout():
    """Context manager to suppress stdout output from preprocessing scripts."""
    old_stdout = sys.stdout
    sys.stdout = io.StringIO()
    try:
        yield
    finally:
        sys.stdout = old_stdout


@contextlib.contextmanager
def suppress_all_output():
    """Suppress all stdout/stderr at both Python and OS file-descriptor levels.

    This catches output that the Python-level sys.stdout replacement misses:
    - Loggers whose StreamHandlers captured the original stream object
    - C extensions writing directly to fd 1/2
    - Subprocess output inherited from the parent
    """
    # Save Python-level streams
    old_stdout = sys.stdout
    old_stderr = sys.stderr

    # Save OS-level file descriptors
    fd_ok = True
    try:
        stdout_fd = old_stdout.fileno()
        stderr_fd = old_stderr.fileno()
        saved_stdout_fd = os.dup(stdout_fd)
        saved_stderr_fd = os.dup(stderr_fd)
    except (io.UnsupportedOperation, AttributeError, OSError):
        fd_ok = False

    # Redirect OS-level fds to /dev/null
    if fd_ok:
        devnull = os.open(os.devnull, os.O_WRONLY)
        os.dup2(devnull, stdout_fd)
        os.dup2(devnull, stderr_fd)
        os.close(devnull)

    # Replace Python-level streams
    sys.stdout = io.StringIO()
    sys.stderr = io.StringIO()

    # Suppress logging at INFO level and below
    prev_disable = logging.root.manager.disable
    logging.disable(logging.INFO)

    try:
        yield
    finally:
        # Restore logging
        logging.disable(prev_disable)

        # Restore Python-level streams
        sys.stdout = old_stdout
        sys.stderr = old_stderr

        # Restore OS-level file descriptors
        if fd_ok:
            os.dup2(saved_stdout_fd, stdout_fd)
            os.dup2(saved_stderr_fd, stderr_fd)
            os.close(saved_stdout_fd)
            os.close(saved_stderr_fd)


def dynamic_import_class(class_path: str):
    """Dynamically import a class from a module path.
    
    Args:
        class_path: Full path to class, e.g., 'gem.envs.canvas_list_test_s2l.canvas_list_test_s2l.CanvasListTestS2LEnv'
    
    Returns:
        The imported class
    """
    module_path, class_name = class_path.rsplit('.', 1)
    module = importlib.import_module(module_path)
    return getattr(module, class_name)


def setup_mcp_servers(
    mcp_configs: Dict[str, Any],
    task_workspace: Path,
    agent_workspace: Path,
) -> Dict[str, Any]:
    """Setup MCP servers based on configuration.
    
    Args:
        mcp_configs: Dictionary of MCP server configurations
        task_workspace: Path to task workspace
        agent_workspace: Path to agent workspace
    
    Returns:
        Dictionary with mcpServers configuration
    """
    config = {"mcpServers": {}}
    
    for server_name, server_config in mcp_configs.items():
        if not server_config.get("enabled", True):
            continue
            
        server_type = server_config.get("type")
        params = server_config.get("params", {})
        if server_type == "context_workspace":
            # Keep SMS state outside agent_workspace. Several benchmark
            # environments rebuild agent_workspace during setup/evaluation; the
            # context workspace is runner-owned memory and must survive that.
            params["workspace_dir"] = str(task_workspace / "context_workspace")
            # Payloads also get an agent-readable copy so normal file tools
            # and Python can inspect them like ordinary files.
            params["payload_dir"] = str(agent_workspace / ".sms_payloads")
        
        # Replace path placeholders
        for key, value in params.items():
            if isinstance(value, str):
                value = value.replace("{task_workspace}", str(task_workspace))
                value = value.replace("{agent_workspace}", str(agent_workspace))
                params[key] = value

        # Add workspace paths to params for placeholder replacement in YAML loader
        params["task_workspace"] = str(task_workspace)
        params["agent_workspace"] = str(agent_workspace)

        # Try YAML-based config first, fall back to legacy helpers
        try:
            server_cfg = build_server_config(
                server_type=server_type,
                params=params,
                server_name=server_name
            )
        except FileNotFoundError:
            # Fallback to legacy helpers during migration
            if server_type == "canvas":
                server_cfg = get_canvas_stdio_config(**params)
            elif server_type == "email":
                server_cfg = get_email_stdio_config(**params)
            elif server_type == "excel":
                server_cfg = get_excel_stdio_config(**params)
            elif server_type == "python_execute":
                server_cfg = get_python_execute_stdio_config(**params)
            elif server_type == "programmatic_tool_calling" or server_type == "programmatic-tool-calling":
                server_cfg = get_programmatic_tool_calling_stdio_config(**params)
            elif server_type == "claim_done":
                server_cfg = get_claim_done_stdio_config(**params)
            elif server_type == "memory":
                server_cfg = get_memory_stdio_config(**params)
            elif server_type == "memory_tool" or server_type == "memory-tool":
                server_cfg = get_memory_tool_stdio_config(**params)
            elif server_type == "filesystem":
                server_cfg = get_filesystem_stdio_config(**params)
            elif server_type == "terminal":
                server_cfg = get_terminal_stdio_config(**params)
            elif server_type == "google_cloud":
                server_cfg = get_google_cloud_stdio_config(**params)
            elif server_type == "google_sheet" or server_type == "google-sheet":
                server_cfg = get_google_sheet_stdio_config(**params)
            elif server_type == "pdf_tools" or server_type == "pdf-tools":
                server_cfg = get_pdf_tools_stdio_config(**params)
            elif server_type == "calendar":
                server_cfg = get_calendar_stdio_config(**params)
            elif server_type == "woocommerce":
                server_cfg = get_woocommerce_stdio_config(**params)
            elif server_type == "snowflake":
                server_cfg = get_snowflake_stdio_config(**params)
            else:
                raise ValueError(f"Unknown MCP server type: {server_type}")
        
        config["mcpServers"].update(server_cfg)
    
    return config


def _remove_schema_keys(obj):
    """Recursively remove '$schema' keys from tool parameter schemas.

    Gemini (via Venus) rejects JSON schemas that contain '$schema' fields,
    returning INVALID_ARGUMENT errors. This strips them before sending.
    """
    if isinstance(obj, dict):
        return {k: _remove_schema_keys(v) for k, v in obj.items() if k != "$schema"}
    if isinstance(obj, list):
        return [_remove_schema_keys(item) for item in obj]
    return obj


def _fix_nested_arrays(obj):
    """Recursively fix nested array schemas for Gemini compatibility.

    Gemini requires that every `type: array` field has an `items` sub-schema,
    including arrays nested inside other arrays' `items`.  When the original
    schema declares `{"type": "array", "items": {"type": "array"}}` (a 2D
    array without an inner items), Gemini returns:
      INVALID_ARGUMENT: …items.items: missing field
    We patch by adding a default `items: {"type": "string"}` to any
    `type: array` that is missing its `items` field.
    """
    if isinstance(obj, dict):
        fixed = {k: _fix_nested_arrays(v) for k, v in obj.items()}
        if fixed.get("type") == "array" and "items" not in fixed:
            fixed["items"] = {"type": "string"}
        return fixed
    if isinstance(obj, list):
        return [_fix_nested_arrays(item) for item in obj]
    return obj


def _limit_tools_for_api(tools: Optional[List], max_tools: Optional[int] = None) -> Optional[List]:
    """Keep advertised tools within OpenAI-compatible API limits.

    Some LOCA configs expose more tools than Venus/OpenAI accepts (128).  Keep
    the self-management and completion tools first, then preserve original order
    for the remaining tools.  The full local tool registry is unchanged; this
    only limits what is advertised in a single API request.
    """
    if not tools:
        return tools
    if max_tools is None:
        raw_limit = os.getenv("LOCA_MAX_API_TOOLS", "128").strip()
        max_tools = int(raw_limit) if raw_limit else 128
    if max_tools <= 0 or len(tools) <= max_tools:
        return tools

    priority_names = {
        "context_workspace_archive",
        "context_workspace_delete",
        "context_workspace_update_state_board",
        "claim_done",
    }

    selected = []
    selected_ids = set()

    def _add(tool_obj):
        if len(selected) >= max_tools:
            return
        ident = id(tool_obj)
        if ident in selected_ids:
            return
        selected.append(tool_obj)
        selected_ids.add(ident)

    for t in tools:
        name = (t.get("function") or {}).get("name")
        if name in priority_names or (isinstance(name, str) and name.startswith("context_workspace_")):
            _add(t)
    for t in tools:
        _add(t)

    omitted = len(tools) - len(selected)
    print(
        f"[TOOL-LIMIT] Advertised tools capped at {len(selected)}/{len(tools)} "
        f"(omitted {omitted}) to satisfy API max_tools={max_tools}.",
        file=sys.stderr,
    )
    return selected


def _estimate_prompt_tokens(messages, tools, model_name: str, sms_mode: bool = False):
    """Estimate prompt tokens with the active strategy's accounting policy.

    SMS uses the same selected-field estimator as the dashboard. Other
    strategies use the same selected-field estimator as the API trim path.
    """
    try:
        import tiktoken
        try:
            tokenizer = tiktoken.encoding_for_model(model_name)
        except Exception:
            tokenizer = tiktoken.get_encoding("cl100k_base")

        if sms_mode:
            try:
                from gem.tools.mcp_server.context_workspace.workspace_manager import count_msg_tokens
            except Exception:
                count_msg_tokens = None

            def _count_msgs(msgs):
                if count_msg_tokens is not None:
                    return count_msg_tokens(msgs, tokenizer)
                total = 0
                for m in msgs:
                    c = m.get("content") or ""
                    if not isinstance(c, str):
                        c = json.dumps(c, ensure_ascii=False)
                    total += len(tokenizer.encode(c, disallowed_special=()))
                    for tc in m.get("tool_calls") or []:
                        fn = tc.get("function") or {}
                        text = (fn.get("name") or "") + (fn.get("arguments") or "")
                        total += len(tokenizer.encode(text, disallowed_special=()))
                    if m.get("name"):
                        total += len(tokenizer.encode(m["name"], disallowed_special=()))
                return total

            messages_tokens = _count_msgs(messages)
            tools_tokens = 0
            for t in (tools[0] if tools else []):
                fn = t.get("function") or {}
                tool_text = (
                    (fn.get("name") or "")
                    + (fn.get("description") or "")
                    + json.dumps(fn.get("parameters") or {}, ensure_ascii=False)
                )
                tools_tokens += _count_msgs([{"content": tool_text}])
            return messages_tokens, tools_tokens, messages_tokens + tools_tokens

        def _count_msgs(msgs):
            total = 0
            for m in msgs:
                c = m.get("content") or ""
                if not isinstance(c, str):
                    c = json.dumps(c, ensure_ascii=False)
                total += len(tokenizer.encode(c, disallowed_special=()))
                for tc in m.get("tool_calls") or []:
                    fn = tc.get("function") or {}
                    text = (fn.get("name") or "") + (fn.get("arguments") or "")
                    total += len(tokenizer.encode(text, disallowed_special=()))
                if m.get("name"):
                    total += len(tokenizer.encode(m["name"], disallowed_special=()))
            return total

        messages_tokens = _count_msgs(messages)
        tools_tokens = 0
        for t in (tools[0] if tools else []):
            fn = t.get("function") or {}
            tool_text = (
                (fn.get("name") or "")
                + (fn.get("description") or "")
                + json.dumps(fn.get("parameters") or {}, ensure_ascii=False)
            )
            tools_tokens += _count_msgs([{"content": tool_text}])
        return messages_tokens, tools_tokens, messages_tokens + tools_tokens
    except Exception:
        return 0, 0, 0


def make_aihubmix_api_request(
    messages: List[Dict],
    model_name: str,
    aihubmix_api_keys: str,
    aihubmix_api_url: str = "https://aihubmix.com/v1/chat/completions",
    tools: Optional[List] = None,
    tool_choice: Optional[str] = None,
    max_retries: int = 200,
    temperature: float = 1.0,
    top_p: float = 1.0,
    max_tokens: int = 4096,
    max_context_size: Optional[int] = None,
    context_awareness: bool = False,
    protect_archived_placeholders: bool = False,
    reasoning_effort: Optional[str] = None,
    reasoning_max_tokens: Optional[int] = None,
    reasoning_enabled: bool = True,
    reasoning_exclude: bool = False,
    trace_id: Optional[str] = None,
    warning_log_path: Optional[Path] = None,
):
    """Make AIHubMix API request with retry logic.

    Args:
        messages: The messages to send to the API
        model_name: Name of the model to use
        aihubmix_api_keys: API key(s) for AIHubMix (comma-separated)
        aihubmix_api_url: The AIHubMix API endpoint URL
        tools: Optional list of tools to include in the request
        tool_choice: Optional tool choice parameter
        max_retries: Maximum number of retry attempts
        temperature: Sampling temperature
        top_p: Top-p sampling parameter
        max_tokens: Maximum number of tokens to generate
        max_context_size: Maximum context size in tokens (if set, will trim messages to fit)
        context_awareness: If True, will also remove token usage user messages when trimming

    Returns:
        Processed response object with type and data
    """
    verbose = False
    # Determine API keys to use
    api_keys = []
    if isinstance(aihubmix_api_keys, list):
        api_keys = aihubmix_api_keys
    elif isinstance(aihubmix_api_keys, str) and ',' in aihubmix_api_keys:
        api_keys = aihubmix_api_keys.split(',')
    else:
        api_keys = [aihubmix_api_keys]
    
    # Randomly select an API key for this request
    current_api_key = random.choice(api_keys)
    sms_mode = protect_archived_placeholders
    tools = _limit_tools_for_api(tools)
    
    
    # Prepare headers for API request
    headers = {
        "Content-Type": "application/json",
        "Authorization": "Bearer " + str(current_api_key)
    }
    if sms_mode:
        # SMS only: sticky routing keeps dashboard-managed turns on a stable backend.
        import uuid as _uuid
        _trace_id = trace_id or _uuid.uuid4().hex  # 32-char hex
        _span_id = _uuid.uuid4().hex[:16]           # 16-char hex
        headers["Venus-Sticky-Routing"] = "trace"
        headers["traceparent"] = f"00-{_trace_id}-{_span_id}-01"
    elif "claude" in model_name.lower():
        headers.setdefault("Venus-Sticky-Routing", os.getenv("VENUS_STICKY_ROUTING", "token"))

    if verbose:
        print(f"Headers: {headers}")
    
    # Track whether messages were trimmed
    trimmed_messages = None
    trim_info = None  # Store trim information
    original_message_count = len(messages)
    
    # Prepare request data
    # gpt-5 and later require max_completion_tokens instead of max_tokens
    _max_tokens_key = "max_completion_tokens" if "gpt-5" in model_name.lower() else "max_tokens"
    json_data = {
        "model": model_name,
        "messages": messages,
        "temperature": temperature,
        _max_tokens_key: max_tokens
    }
    if "claude" not in model_name.lower():
        json_data["top_p"] = top_p

    # Add reasoning control parameters for OpenAI models that support reasoning
    # Convert empty strings to None for proper handling
    if reasoning_effort == "":
        reasoning_effort = None
    if reasoning_max_tokens == "":
        reasoning_max_tokens = None

    if reasoning_effort is not None:
        # OpenAI-compatible proxies used by Gemini/Venus accept this top-level
        # field. Keeping it absent by default preserves original benchmark runs.
        json_data["reasoning_effort"] = reasoning_effort

    if reasoning_max_tokens is not None:
        reasoning_config = {}

        # Keep the structured field for providers that expose a reasoning object.
        reasoning_config["max_tokens"] = reasoning_max_tokens

        # Set enabled flag (default: inferred from effort or max_tokens)
        reasoning_config["enabled"] = reasoning_enabled

        # Set exclude flag (default: false)
        reasoning_config["exclude"] = reasoning_exclude

        json_data["reasoning"] = reasoning_config
    
    # ── Timing accumulators (returned in response for caller to aggregate) ────
    _t_tok_est_s = 0.0   # time spent on token estimation + trim logic
    _t_http_s    = 0.0   # cumulative time spent waiting on Gemini HTTP

    # Estimate tokens before making the API call.
    # Uses count_msg_tokens from workspace_manager — the canonical token estimator
    # shared by bulk-output gate, dashboard, and trim so all three always agree.
    _t0_tok = time.perf_counter()
    try:
        import tiktoken
        try:
            tokenizer = tiktoken.encoding_for_model(model_name)
        except:
            tokenizer = tiktoken.get_encoding("cl100k_base")

        if sms_mode:
            try:
                from gem.tools.mcp_server.context_workspace.workspace_manager import count_msg_tokens as _count_msg_tokens
            except ImportError:
                _count_msg_tokens = None

            def _content_tokens(msgs):
                """SMS estimator shared by dashboard, bulk gate, and trim."""
                if _count_msg_tokens is not None:
                    return _count_msg_tokens(msgs, tokenizer)
                total = 0
                for m in msgs:
                    c = m.get("content") or ""
                    if not isinstance(c, str):
                        c = json.dumps(c, ensure_ascii=False)
                    total += len(tokenizer.encode(c, disallowed_special=()))
                    for tc in m.get("tool_calls") or []:
                        fn = tc.get("function") or {}
                        total += len(tokenizer.encode(
                            (fn.get("name") or "") + (fn.get("arguments") or ""),
                            disallowed_special=(),
                        ))
                    if m.get("name"):
                        total += len(tokenizer.encode(m["name"], disallowed_special=()))
                return total
        else:
            def _content_tokens(msgs):
                """LOCA-bench estimator for non-SMS baselines."""
                total = 0
                for m in msgs:
                    c = m.get("content") or ""
                    if not isinstance(c, str):
                        c = json.dumps(c, ensure_ascii=False)
                    total += len(tokenizer.encode(c, disallowed_special=()))
                    for tc in m.get("tool_calls") or []:
                        fn = tc.get("function") or {}
                        total += len(tokenizer.encode(
                            (fn.get("name") or "") + (fn.get("arguments") or ""),
                            disallowed_special=(),
                        ))
                    if m.get("name"):
                        total += len(tokenizer.encode(m["name"], disallowed_special=()))
                return total

        messages_tokens = _content_tokens(messages)
        
        tools_tokens = 0
        has_agent_context_management = False
        if tools:
            if sms_mode:
                for t in tools:
                    fn = t.get("function") or {}
                    t_name = fn.get("name") or ""
                    if t_name in {"context_workspace_archive", "context_workspace_delete"}:
                        has_agent_context_management = True
                    t_desc = fn.get("description") or ""
                    t_para = json.dumps(fn.get("parameters") or {}, ensure_ascii=False)
                    tools_tokens += _content_tokens([{"content": t_name + t_desc + t_para}])
            else:
                for t in tools:
                    fn = t.get("function") or {}
                    t_name = fn.get("name") or ""
                    t_desc = fn.get("description") or ""
                    t_para = json.dumps(fn.get("parameters") or {}, ensure_ascii=False)
                    tools_tokens += _content_tokens([{"content": t_name + t_desc + t_para}])
        
        total_estimated_tokens = messages_tokens + tools_tokens
        if verbose:
            print(f"📊 Estimated tokens - Messages: {messages_tokens:,}, Tools: {tools_tokens:,}, Total: {total_estimated_tokens:,}")

        # Non-SMS baselines keep original LOCA-bench behavior: reserve max_tokens
        # from the context window. SMS uses full input accounting so dashboard,
        # bulk gate, and trim agree with the payload it manages.
        available_context = (
            max_context_size
            if sms_mode or max_context_size is None
            else max_context_size - max_tokens
        )
        if max_context_size is not None and total_estimated_tokens > available_context:
            if sms_mode:
                error_msg = (
                    "[SM-NO-TRIM] Context exceeds the request window, and automatic trim is disabled "
                    "in self-managed mode. "
                    f"Messages: {messages_tokens:,} tok, Tools: {tools_tokens:,} tok, "
                    f"Total: {total_estimated_tokens:,} tok, Limit: {available_context:,} tok. "
                    "The agent should archive/delete context before retrying."
                )
                print(error_msg)
                _append_warning_log(warning_log_path, error_msg)
                return {
                    'type': 'error',
                    'data': [error_msg],
                    'call_messages': {
                        'role': 'assistant',
                        'content': error_msg
                    },
                    'timing': {'tok_est_s': time.perf_counter() - _t0_tok, 'http_s': _t_http_s},
                }
            if protect_archived_placeholders:
                # In SM mode the bulk-output gate should have prevented this.
                # If trim fires here, keep it payload-only: the raw conversation
                # and workspace message indexes must not be rewritten from the
                # assembled view.
                _warning = (
                    "[SM-TRIM] ⚠️  Payload-only trim fired in self-managed mode — "
                    "bulk gate underestimated. "
                    f"Messages: {messages_tokens:,} tok, Tools: {tools_tokens:,} tok, "
                    f"Total: {total_estimated_tokens:,} tok, Limit: {available_context:,} tok."
                )
                print(_warning)
                _append_warning_log(warning_log_path, _warning)
            elif verbose:
                print(f"⚠️  Total tokens ({total_estimated_tokens:,}) exceeds available context ({available_context:,}). Trimming messages...")
            
            original_message_count = len(messages)
            
            # Strategy: Remove assistant and tool messages from the beginning until we fit
            # Keep removing messages one by one from the start
            current_messages = messages.copy()
            removed_count = 0
            
            # Compute initial token count for the loop
            current_total = messages_tokens + tools_tokens

            while len(current_messages) > 0:
                # If we fit within the limit, we're done
                if current_total <= available_context:
                    messages = current_messages
                    break
                
                # Otherwise, try to remove the first assistant or tool message
                # If assistant has tool_calls, also remove all corresponding tool results
                # If context_awareness is enabled, also remove paired token usage user messages
                removed_any = False
                for i in range(len(current_messages)):
                    msg_role = current_messages[i].get('role')

                    # SM mode: skip archived placeholders — they are the agent's
                    # only record of compressed history. Deleting them causes
                    # complete amnesia which is worse than a slightly full context.
                    if protect_archived_placeholders:
                        content = current_messages[i].get('content', '')
                        if isinstance(content, str) and content.startswith('[ARCHIVED:'):
                            continue

                    # Remove assistant messages
                    if msg_role == 'assistant':
                        msg = current_messages.pop(i)
                        removed_count += 1
                        removed_any = True

                        # If this assistant message has tool_calls, remove all corresponding tool results
                        if 'tool_calls' in msg and msg['tool_calls']:
                            tool_call_ids = {tc['id'] for tc in msg['tool_calls']}

                            # Find and remove all tool messages with matching tool_call_id
                            j = 0
                            while j < len(current_messages):
                                if current_messages[j].get('role') == 'tool':
                                    if current_messages[j].get('tool_call_id') in tool_call_ids:
                                        current_messages.pop(j)
                                        removed_count += 1
                                        # Don't increment j since we just removed an element
                                        continue
                                j += 1

                            # If context_awareness is enabled, remove the token usage message
                            # that comes after all the tool results
                            if context_awareness:
                                # Look for the next user message with token usage starting from position i
                                j = i
                                while j < len(current_messages):
                                    if current_messages[j].get('role') == 'user':
                                        content = current_messages[j].get('content', '')
                                        if isinstance(content, str) and '<system_warning>Token usage:' in content:
                                            current_messages.pop(j)
                                            removed_count += 1
                                            break
                                    j += 1

                            # Also remove memory warning messages if present after tool results
                            j = i
                            while j < len(current_messages):
                                if current_messages[j].get('role') == 'user':
                                    content = current_messages[j].get('content', '')
                                    if isinstance(content, str) and '**You are nearing the context window limit.**' in content:
                                        current_messages.pop(j)
                                        removed_count += 1
                                        break
                                j += 1
                        break

                    # Remove standalone tool messages
                    elif msg_role == 'tool':
                        current_messages.pop(i)
                        removed_count += 1
                        removed_any = True

                        # If context_awareness is enabled, check if next message is token usage
                        if context_awareness and i < len(current_messages):
                            next_msg = current_messages[i]
                            if next_msg.get('role') == 'user':
                                next_content = next_msg.get('content', '')
                                if isinstance(next_content, str) and '<system_warning>Token usage:' in next_content:
                                    current_messages.pop(i)
                                    removed_count += 1

                        # Also remove memory warning messages if present
                        # Check if next message is memory warning
                        if i < len(current_messages):
                            next_msg = current_messages[i]
                            if next_msg.get('role') == 'user':
                                next_content = next_msg.get('content', '')
                                if isinstance(next_content, str) and '**You are nearing the context window limit.**' in next_content:
                                    current_messages.pop(i)
                                    removed_count += 1
                        break

                # If no assistant or tool message found to remove, we can't trim further
                if not removed_any:
                    # Keep what we have and break
                    messages = current_messages
                    break

                # Recompute token count after removals (only when we actually removed something)
                current_total = _content_tokens(current_messages) + tools_tokens

            # Recalculate final token count using the active estimator.
            final_messages_tokens = _content_tokens(messages)
            final_total_tokens = final_messages_tokens + tools_tokens
            
            if verbose:
                print(f"✂️  Trimmed messages: {original_message_count} -> {len(messages)} messages (removed {removed_count} assistant/tool messages)")
                print(f"📊 After trimming - Messages: {final_messages_tokens:,}, Tools: {tools_tokens:,}, Total: {final_total_tokens:,}")
                print(f"📊 Context limit: {max_context_size:,} tokens; max output request: {max_tokens:,} tokens")

            # Check if trimming removed all assistant/tool messages (only user messages left)
            has_non_user_messages = any(msg.get('role') in ['assistant', 'tool'] for msg in messages)

            context_lost = (not has_non_user_messages and removed_count > 0)
            if context_lost:
                prefix = "[SM-TRIM]" if protect_archived_placeholders else "[TRIM]"
                if protect_archived_placeholders:
                    _warning = (
                        f"{prefix} ⚠️  Trim removed all assistant/tool messages for this API call "
                        f"({removed_count} removed). Continuing with user/system context only. "
                        f"Current total: {final_total_tokens:,} tokens (limit: {available_context:,})."
                    )
                    print(_warning)
                    _append_warning_log(warning_log_path, _warning)
                else:
                    error_msg = (
                        f"ERROR: Context trimming removed all {removed_count} assistant/tool messages. "
                        f"Only user messages remain ({len(messages)} messages, {final_messages_tokens:,} tokens). "
                        f"The conversation context has been lost. "
                        f"Current total: {final_total_tokens:,} tokens (max available: {available_context:,}). "
                        f"Please increase max_context_size or reduce conversation length."
                    )
                    if verbose:
                        print(f"🚨 {error_msg}")
                    return {
                        'type': 'error',
                        'data': [error_msg],
                        'call_messages': {
                            'role': 'assistant',
                            'content': error_msg
                        },
                        'timing': {'tok_est_s': _t_tok_est_s, 'http_s': _t_http_s},
                    }
            
            # Check if we still exceed the limit after trimming
            if final_total_tokens > available_context:
                error_msg = (
                    f"ERROR: Cannot fit messages within available context ({available_context:,} = {max_context_size:,} tokens). "
                    f"After removing {removed_count} assistant/tool messages, "
                    f"still have {final_total_tokens:,} tokens (Messages: {final_messages_tokens:,}, Tools: {tools_tokens:,}). "
                    f"Cannot trim further without losing user messages. "
                    f"Please increase max_context_size."
                )
                if verbose:
                    print(f"🚨 {error_msg}")
                return {
                    'type': 'error',
                    'data': [error_msg],
                    'call_messages': {
                        'role': 'assistant',
                        'content': error_msg
                    }
                }

            # Update json_data with trimmed messages for this request.
            json_data["messages"] = messages

            # Create trim info to return to caller
            import copy
            trim_info = {
                'original_message_count': original_message_count,
                'trimmed_message_count': len(messages),
                'removed_count': removed_count,
                'original_total_tokens': total_estimated_tokens,
                'trimmed_total_tokens': final_total_tokens,
                'messages_tokens': final_messages_tokens,
                'tools_tokens': tools_tokens,
                'max_context_size': max_context_size,
                'max_tokens': max_tokens,
                'available_context': available_context,
                'payload_only': protect_archived_placeholders,
                'context_lost': context_lost,
                'messages_after_trim_sample': copy.deepcopy(messages)  # Sample of messages after trimming
            }
            if protect_archived_placeholders:
                # The caller owns raw messages. In SM mode `messages` here is the
                # assembled dashboard/context payload, not the raw trajectory.
                trimmed_messages = None
            else:
                trimmed_messages = messages  # Save trimmed messages to return to caller

    except Exception as e:
        if verbose:
            print(f"⚠️  Token estimation failed: {e}")
    _t_tok_est_s = time.perf_counter() - _t0_tok

    # Provider routing removed: Venus handles routing internally and does not accept
    # the AIHubMix-specific "provider" field. Sending it causes 400 errors.

    if verbose:
        print(f"JSON data: {json_data}")

    # Add tools if provided
    if tools:
        tools_for_request = _fix_nested_arrays(_remove_schema_keys(tools))
        if "claude" in model_name.lower() and tools_for_request:
            tools_for_request[-1].setdefault("function", {})["cache_control"] = {"type": "ephemeral"}
        json_data["tools"] = tools_for_request
    if tool_choice:
        json_data["tool_choice"] = tool_choice

    # Track retry attempts
    times = 0
    last_error_detail = None
    consecutive_network_failures = 0  # Timeout/ConnectionError in a row
    MAX_CONSECUTIVE_NETWORK_FAILURES = 3  # Give up early if upstream looks dead

    while times < max_retries:
        try:
            # Make API request
            if verbose:
                print(f"Making API request to: {aihubmix_api_url}")
            _t0_http = time.perf_counter()
            # Split timeout: 10s to establish connection, 300s for the server
            # to start streaming bytes back. Keeps a single hung request from
            # eating a worker for 10 minutes.
            response = requests.post(
                aihubmix_api_url,
                headers=headers,
                json=json_data,
                timeout=(10, 300)
            )
            _t_http_s += time.perf_counter() - _t0_http
            # Got bytes back from the server: reset the network-failure streak.
            consecutive_network_failures = 0
            if verbose:
                print(f"Response status: {response.status_code}")
            
            if response.status_code == 200:
                try:
                    res = json.loads(response.text)
                    
                    # Extract token usage information if available
                    if "usage" in res:
                        usage = res.get("usage", {})
                        prompt_tokens = usage.get("prompt_tokens", 0)
                        completion_tokens = usage.get("completion_tokens", 0)
                        total_tokens = usage.get("total_tokens", 0)
                        if verbose:
                            print(f"Token usage: prompt_tokens={prompt_tokens}, completion_tokens={completion_tokens}, total_tokens={total_tokens}")

                    # Process response
                    result = []
                    is_tool = False
                    should_retry = False

                    for choice in res['choices']:
                        finish_reason = choice.get('finish_reason', '')
                        if verbose:
                            print(f"Finish reason: {finish_reason}")

                        if finish_reason == 'error':
                            if verbose:
                                print(f"Received error finish reason. Retrying request...")
                            should_retry = True
                            break
                        elif finish_reason == 'length':
                            if verbose:
                                print(f"WARNING: Response truncated due to max_tokens limit!")
                                print(f"Current max_tokens: {max_tokens}")
                                print(f"Retrying request to get complete response...")
                            should_retry = True
                            break
                        elif finish_reason is None:
                            if verbose:
                                print(f"Received None finish reason. Retrying request...")
                            should_retry = True
                            break
                        else:
                            # Check for tool_calls in the message first
                            # Some providers (e.g., Google Gemini) return finish_reason="stop" even with tool_calls
                            message = choice.get('message', {})
                            has_tool_calls = 'tool_calls' in message and message['tool_calls']

                            if has_tool_calls:
                                # Handle tool calls regardless of finish_reason
                                result.extend(message.get('tool_calls', []))
                                is_tool = True
                                if verbose:
                                    print(f"Detected tool_calls in message with finish_reason={finish_reason}")
                            else:
                                # Normal content response
                                content = message.get('content', '')

                                # Check if content is empty and no tool_calls
                                if not content or not content.strip():
                                    if verbose:
                                        print(f"Received empty content without tool_calls. Retrying request...")
                                    should_retry = True
                                    break
                                
                                result.append(content)
                    
                    # If we should retry, continue to the next iteration
                    if should_retry:
                        times += 1
                        # Switch to a random API key for the retry
                        current_api_key = random.choice(api_keys)
                        headers["Authorization"] = "Bearer " + str(current_api_key)
                        if verbose:
                            print(f"Switched to random API key for error retry")

                        # Simple backoff with some randomness
                        sleep_time = 1 + random.random()
                        if verbose:
                            print(f"Retrying in {sleep_time:.2f} seconds...")
                        time.sleep(sleep_time)
                        continue
                    
                    # Return the content based on type
                    if is_tool:
                        return {
                            'type': 'tool',
                            'data': result,
                            'call_messages': res['choices'][0]['message'],
                            'raw_response': res,
                            'trimmed_messages': trimmed_messages,
                            'trim_info': trim_info,
                            'timing': {'tok_est_s': _t_tok_est_s, 'http_s': _t_http_s},
                        }
                    else:
                        return {
                            'type': 'normal',
                            'data': result,
                            'call_messages': res['choices'][0]['message'],
                            'raw_response': res,
                            'trimmed_messages': trimmed_messages,
                            'trim_info': trim_info,
                            'timing': {'tok_est_s': _t_tok_est_s, 'http_s': _t_http_s},
                        }
                    
                except (KeyError, json.JSONDecodeError) as e:
                    last_error_detail = f"Error parsing API response: {e}. Response text: {response.text[:1000]}"
                    if verbose:
                        print(f"Error parsing API response: {e}")
                        print(f"Response text: {response.text}")

            # Handle rate limiting
            if response.status_code == 429:
                last_error_detail = f"HTTP 429 rate limited. Response: {response.text[:1000]}"
                import re
                # Extract retry time if available
                pattern_milliseconds = re.compile(r'(?<=Please retry after )\d+(?= milliseconds)')
                milliseconds = pattern_milliseconds.findall(str(response.text))
                if milliseconds:
                    wait_time = int(milliseconds[0])/1000
                else:
                    wait_time = 1 + random.random()

                if verbose:
                    print(f"Rate limited. Retrying after {wait_time} seconds.")
                time.sleep(wait_time)
                times += 1

                # Select a different random API key for the next attempt
                current_api_key = random.choice(api_keys)
                headers["Authorization"] = "Bearer " + str(current_api_key)
                if verbose:
                    print(f"Switching to a random API key")
                continue

            # Handle authentication errors - use a different random API key
            if response.status_code == 401:
                last_error_detail = f"HTTP 401 authentication error. Response: {response.text[:1000]}"
                if verbose:
                    print("Authentication error. Trying a different API key.")
                # Remove the failed key from the list if we have multiple keys
                if len(api_keys) > 1:
                    api_keys = [key for key in api_keys if key != current_api_key]

                # Select a new random key
                current_api_key = random.choice(api_keys)
                headers["Authorization"] = "Bearer " + str(current_api_key)
                if verbose:
                    print(f"Switched to random API key")
                times += 1
                continue

            # Handle 400 errors
            if response.status_code == 400:
                last_error_detail = f"HTTP 400. Response: {response.text[:1000]}"
                # Print the actual error response
                if verbose:
                    print(f"API request failed with status 400: {response.text}")

                # Try to parse error message to determine if it's a parameter error
                try:
                    error_data = response.json()
                    error_msg = error_data.get("error", {}).get("message", "")
                    error_code = error_data.get("error", {}).get("code", "")

                    # Check if it's a context-length error (Gemini actually exceeded)
                    _ctx_keywords = (
                        "context", "token", "length", "exceed", "too long",
                        "maximum", "limit", "window", "overflow",
                    )
                    if any(kw in error_msg.lower() for kw in _ctx_keywords):
                        _mtok = locals().get("messages_tokens", "?")
                        _ttok = locals().get("tools_tokens", "?")
                        _tot  = locals().get("total_estimated_tokens", "?")
                        print(
                            f"[GEMINI-CTX] 🚨 Gemini 真的爆了 — context length error from API. "
                            f"messages_tokens={_mtok}, tools_tokens={_ttok}, total={_tot}. "
                            f"Error: {error_msg}"
                        )

                    # Check if it's a parameter format error (non-retriable)
                    if "InvalidParameter" in error_msg or "invalid_parameter_error" in error_code:
                        if verbose:
                            print(f"Parameter format error detected. This is non-retriable.")
                        return {
                            "type": "error", 
                            "data": [f"Error: Invalid parameter format. Response: {response.text}"],
                            "call_messages": {"role": "assistant", "content": f"Error: Invalid parameter format. Response: {response.text}"}
                        }
                except:
                    pass
                
                # For other 400 errors (e.g., high risk content), reduce max retries
                #reduced_max_retries = max(1, max_retries // 10)

                reduced_max_retries = max_retries
                if times >= reduced_max_retries:
                    if verbose:
                        print(f"Reached reduced retry limit ({reduced_max_retries}) for 400 error. Giving up.")
                    return {
                        "type": "error",
                        "data": [f"Error: Request failed with 400 status. Response: {response.text}"],
                        "call_messages": {"role": "assistant", "content": f"Error: Request failed with 400 status. Response: {response.text}"}
                    }
                if verbose:
                    print(f"400 error detected. Using reduced retry limit: {reduced_max_retries}")
            # For other errors
            else:
                last_error_detail = f"HTTP {response.status_code}. Response: {response.text[:1000]}"
                if verbose:
                    print(f"API request failed with status {response.status_code}: {response.text}")
                    print(f"Response: {response}")

            # Switch to a random API key after several failures
            if times % 3 == 2:  # Every 3rd attempt
                current_api_key = random.choice(api_keys)
                headers["Authorization"] = "Bearer " + str(current_api_key)
                if verbose:
                    print(f"Switched to random API key")

        except requests.exceptions.Timeout:
            last_error_detail = f"Request timed out (connect=10s, read=300s) on attempt {times+1}/{max_retries}."
            consecutive_network_failures += 1
            if verbose:
                print(f"Request timed out. Retrying {times+1}/{max_retries}...")
            # Try a different random API key on timeout
            current_api_key = random.choice(api_keys)
            headers["Authorization"] = "Bearer " + str(current_api_key)
            if verbose:
                print(f"Switched to random API key after timeout")
        except requests.exceptions.ConnectionError:
            last_error_detail = f"Connection error on attempt {times+1}/{max_retries}."
            consecutive_network_failures += 1
            if verbose:
                print(f"Connection error. Retrying {times+1}/{max_retries}...")
            # Try a different random API key on connection error
            current_api_key = random.choice(api_keys)
            headers["Authorization"] = "Bearer " + str(current_api_key)
            if verbose:
                print(f"Switched to random API key after connection error")
        except Exception as e:
            last_error_detail = f"Request error on attempt {times+1}/{max_retries}: {e}"
            if verbose:
                print(f"Request error: {e}")

        # Bail out early when the network layer keeps failing. Burning the full
        # 200 retries on a dead upstream just hangs the worker.
        if consecutive_network_failures >= MAX_CONSECUTIVE_NETWORK_FAILURES:
            print(
                f"[NET-GIVEUP] {consecutive_network_failures} consecutive timeouts/"
                f"connection-errors; aborting retry loop early. Last: {last_error_detail}",
                file=sys.stderr,
            )
            break

        # Simple backoff with some randomness
        sleep_time = 1 + random.random()
        if verbose:
            print(f"Retrying in {sleep_time:.2f} seconds...")
        time.sleep(sleep_time)
        times += 1
    
    error_msg = "Error: Failed to get response after multiple retries."
    if last_error_detail:
        error_msg = f"{error_msg} Last error: {last_error_detail}"
    return {
        "type": "error", 
        "data": [error_msg],
        "call_messages": {"role": "assistant", "content": error_msg},
        "timing": {'tok_est_s': _t_tok_est_s, 'http_s': _t_http_s},
    }


def perform_thinking_reset(messages: List[Dict], keep_thinking: int = 1) -> tuple:
    """
    Remove reasoning content from assistant messages to reduce token usage.

    This function clears reasoning/thinking content from assistant messages by:
    - Setting 'reasoning_content' or 'reasoning' to empty string ""
    - Removing 'reasoning_details' field entirely (if present)

    Args:
        messages: List of message dictionaries
        keep_thinking: Number of most recent assistant messages to keep reasoning_content for (default: 1)

    Returns:
        Tuple of (new_messages, reset_info)
    """
    # Find all assistant message indices
    assistant_indices = []

    for i, msg in enumerate(messages):
        if msg.get('role') == 'assistant':
            assistant_indices.append(i)

    if len(assistant_indices) == 0:
        return messages, None

    # Determine which assistant messages to clear reasoning_content from
    # Keep the last 'keep_thinking' assistant messages
    if keep_thinking > 0 and len(assistant_indices) > keep_thinking:
        # Clear all except the last 'keep_thinking' assistant messages
        indices_to_clear = assistant_indices[:-keep_thinking]
    elif keep_thinking == 0:
        # Clear all assistant messages
        indices_to_clear = assistant_indices
    else:
        # keep_thinking >= total assistants, don't clear any
        return messages, None

    if len(indices_to_clear) == 0:
        return messages, None

    # Create new messages list with cleared reasoning_content
    new_messages = []
    cleared_count = 0
    total_reasoning_content_length = 0

    for i, msg in enumerate(messages):
        if i in indices_to_clear:
            # Check if this assistant message has reasoning fields
            # Support both 'reasoning' and 'reasoning_content' field names
            has_reasoning_content = 'reasoning_content' in msg and msg['reasoning_content']
            has_reasoning = 'reasoning' in msg and msg['reasoning']
            has_reasoning_details = 'reasoning_details' in msg and msg['reasoning_details']

            if has_reasoning_content or has_reasoning or has_reasoning_details:
                # Track the length of removed content
                if has_reasoning_content:
                    total_reasoning_content_length += len(str(msg['reasoning_content']))
                if has_reasoning:
                    total_reasoning_content_length += len(str(msg['reasoning']))
                if has_reasoning_details:
                    total_reasoning_content_length += len(str(msg['reasoning_details']))

                # Create a copy of the message and clear reasoning fields
                new_msg = msg.copy()

                # Clear reasoning_content (set to empty string)
                if 'reasoning_content' in new_msg:
                    new_msg['reasoning_content'] = ""

                # Clear reasoning (set to empty string)
                if 'reasoning' in new_msg:
                    new_msg['reasoning'] = ""

                # Remove reasoning_details entirely
                if 'reasoning_details' in new_msg:
                    del new_msg['reasoning_details']

                new_messages.append(new_msg)
                cleared_count += 1
            else:
                # No reasoning content to clear
                new_messages.append(msg)
        else:
            new_messages.append(msg)

    # Create reset info
    reset_info = {
        'num_cleared': cleared_count,
        'total_assistants': len(assistant_indices),
        'cleared_indices': sorted(indices_to_clear),
        'keep_thinking': keep_thinking,
        'total_reasoning_content_length': total_reasoning_content_length
    }

    return new_messages, reset_info


def perform_context_reset(messages: List[Dict], reset_ratio: float, keep_last_tool_call: bool = True) -> tuple:
    """
    Remove reset_ratio of tool calls and corresponding tool results from messages.

    Args:
        messages: List of message dictionaries
        reset_ratio: Ratio of tool calls to remove (0.0 to 1.0)
        keep_last_tool_call: If True, always keep the most recent assistant tool_calls (default: True)

    Returns:
        Tuple of (new_messages, reset_info)
    """
    # Find all assistant messages with tool_calls and their corresponding tool results
    tool_call_pairs = []
    
    i = 0
    while i < len(messages):
        msg = messages[i]
        if msg.get('role') == 'assistant' and 'tool_calls' in msg and msg['tool_calls']:
            # This is an assistant message with tool calls
            tool_call_ids = [tc['id'] for tc in msg['tool_calls']]
            
            # Look for tool results after this message
            j = i + 1
            tool_results_indices = []
            while j < len(messages):
                if messages[j].get('role') == 'tool':
                    if messages[j].get('tool_call_id') in tool_call_ids:
                        tool_results_indices.append(j)
                j += 1
            
            tool_call_pairs.append({
                'assistant_idx': i,
                'tool_result_indices': tool_results_indices,
                'num_tool_calls': len(msg['tool_calls'])
            })
        i += 1
    
    # If keep_last_tool_call is True and we have tool call pairs,
    # exclude the last pair from removal consideration
    if keep_last_tool_call and len(tool_call_pairs) > 0:
        # Only consider the pairs except the last one for removal
        pairs_available_for_removal = tool_call_pairs[:-1]
        num_to_remove = int(len(pairs_available_for_removal) * reset_ratio)
    else:
        pairs_available_for_removal = tool_call_pairs
        num_to_remove = int(len(tool_call_pairs) * reset_ratio)
    
    if num_to_remove == 0:
        return messages, None
    
    # Remove from the earliest tool calls (FIFO), excluding the last one if keep_last_tool_call is True
    pairs_to_remove = pairs_available_for_removal[:num_to_remove]
    
    # Collect indices to remove and modify
    tool_indices_to_remove = set()
    assistant_indices_to_modify = set()
    
    # Collect tool_call_ids that will be removed (for filtering reasoning_details)
    tool_call_ids_to_remove = set()
    
    for pair in pairs_to_remove:
        assistant_indices_to_modify.add(pair['assistant_idx'])
        tool_indices_to_remove.update(pair['tool_result_indices'])
        # Collect the tool_call_ids from the assistant message
        assistant_msg = messages[pair['assistant_idx']]
        if 'tool_calls' in assistant_msg:
            for tc in assistant_msg['tool_calls']:
                tool_call_ids_to_remove.add(tc['id'])
    
    # Create new messages list
    new_messages = []
    reset_info = {
        'num_pairs_removed': num_to_remove,
        'total_pairs': len(tool_call_pairs),
        'removed_assistant_indices': sorted(list(assistant_indices_to_modify)),
        'removed_tool_indices': sorted(list(tool_indices_to_remove)),
        'reset_ratio': reset_ratio,
        'kept_last_tool_call': keep_last_tool_call and len(tool_call_pairs) > 0
    }
    
    for i, msg in enumerate(messages):
        # Skip tool result messages that should be removed
        if i in tool_indices_to_remove:
            continue
        
        # If this is an assistant message that should have tool_calls removed
        if i in assistant_indices_to_modify:
            # Remove tool_calls but keep the rest (like content)
            new_msg = {k: v for k, v in msg.items() if k != 'tool_calls'}
            # Also remove corresponding reasoning_details entries if present
            if 'reasoning_details' in new_msg and new_msg['reasoning_details']:
                new_msg['reasoning_details'] = [
                    rd for rd in new_msg['reasoning_details']
                    if rd.get('id') not in tool_call_ids_to_remove
                ]
                # If reasoning_details is now empty, remove it entirely
                if not new_msg['reasoning_details']:
                    del new_msg['reasoning_details']
            new_messages.append(new_msg)
        else:
            new_messages.append(msg)
    
    return new_messages, reset_info


def run_single_task(
    task_id: int,
    config_id: int,
    run_id: int,
    base_task_dir: str,
    output_dir: str,
    env_class: str,
    env_params: Dict[str, Any],
    mcp_configs: Dict[str, Any],
    api_key: str,
    base_url: str,
    model: str,
    max_tool_uses: int,
    max_tokens: int,
    timeout: int,
    max_retries: int = 5,
    initial_retry_delay: float = 2.0,
    reset_size: Optional[int] = None,
    reset_ratio: float = 0.5,
    context_reset: bool = False,
    context_summary: bool = False,
    context_awareness: bool = False,
    max_context_size: Optional[int] = None,
    memory_warning_threshold: float = 0.8,
    thinking_reset: bool = False,
    keep_thinking: int = 1,
    reasoning_effort: Optional[str] = None,
    reasoning_max_tokens: Optional[int] = None,
    reasoning_enabled: bool = True,
    reasoning_exclude: bool = False,
    config_name: str = "",
):
    """Run a single task with configurable environment and tools.

    Args:
        task_id: Global unique identifier for this task instance
        config_id: Configuration group ID
        run_id: Run number within this configuration
        base_task_dir: Base directory for task data
        output_dir: Directory to save results
        env_class: Full path to environment class
        env_params: Parameters for environment initialization
        mcp_configs: MCP server configurations
        api_key: API key for the model
        base_url: Base URL for the API
        model: Model name to use
        max_tool_uses: Maximum number of tool uses
        max_tokens: Maximum tokens for generation
        timeout: Request timeout in seconds
        max_retries: Maximum API retry attempts
        initial_retry_delay: Initial delay for retry in seconds
        reset_size: Token threshold for context management (None to disable all management)
        reset_ratio: Ratio of tool calls to remove during reset (0.0 to 1.0)
        context_reset: If True, remove old tool calls when exceeding token limit
        context_summary: If True, generate summary when exceeding token limit
        context_awareness: If True, inform the model about token budget and usage at each step
        max_context_size: Maximum context size in tokens (if set, will trim messages to fit)
        memory_warning_threshold: Threshold ratio (0.0-1.0) for memory warning when memory_tool is enabled.
                                  Warning is issued when total_tokens >= reset_size * threshold and < reset_size.
        thinking_reset: If True, clear reasoning_content from assistant messages when exceeding token limit
        keep_thinking: Number of most recent assistant messages to keep reasoning_content for (default: 1)

        Note: context_reset and context_summary are mutually exclusive.
              If both are False, no context management is performed even if reset_size is set.

    Returns:
        Dictionary with task results
    """
    verbose = False
    task_label = f"{config_name}-State{run_id}" if config_name else f"Config{config_id}-Run{run_id}"
    if verbose:
        print(f"[Task {task_id} | {task_label}] Starting...")
        print(f"[Task {task_id} | {task_label}] Environment: {env_class}")
        print(f"[Task {task_id} | {task_label}] Params: {env_params}")

    # Create isolated directories for this task
    task_workspace = build_task_workspace(
        base_task_dir=base_task_dir,
        config_name=config_name,
        config_id=config_id,
        run_id=run_id,
    )
    task_workspace.mkdir(parents=True, exist_ok=True)
    
    local_db_dir = task_workspace / "local_db"
    agent_workspace = task_workspace / "agent_workspace"
    memory_dir = agent_workspace / "memory"
    
    # Ensure directories exist
    local_db_dir.mkdir(parents=True, exist_ok=True)
    agent_workspace.mkdir(parents=True, exist_ok=True)
    memory_dir.mkdir(parents=True, exist_ok=True)
    
    episode = []
    full_messages_history = []  # Store complete message history including reset messages
    reset_events = []  # Store information about reset events
    summary_events = []  # Store information about summary events

    # Strip Gemini-specific opaque fields before writing to trajectory.
    # These fields (reasoning_content, thought_signature, extra_content) are
    # not meaningful for analysis and can inflate trajectory files 10-50x.
    _TRAJ_STRIP = ("reasoning_content", "reasoning", "reasoning_details",
                   "thought_signature", "extra_content")
    def _strip_for_trajectory(msgs):
        result = []
        for m in msgs:
            m2 = {k: v for k, v in m.items() if k not in _TRAJ_STRIP}
            if m2.get("tool_calls"):
                m2["tool_calls"] = [
                    {k: v for k, v in tc.items() if k not in _TRAJ_STRIP}
                    for tc in m2["tool_calls"]
                ]
            result.append(m2)
        return result
    trim_events = []  # Store information about trim/truncation events
    thinking_reset_events = []  # Store information about thinking reset events
    usage_tracking = []  # Store per-step API usage
    initial_user_message = None  # Store the initial user message for summary mode
    memory_warning_issued = False  # Track if memory warning has been issued
    tool = None  # Initialize tool to None for cleanup in finally block

    try:
        # Dynamically import and instantiate environment class
        EnvClass = dynamic_import_class(env_class)
        
        # Prepare environment parameters with path replacements
        prepared_env_params = {}
        for key, value in env_params.items():
            if isinstance(value, str):
                value = value.replace("{task_workspace}", str(task_workspace))
                value = value.replace("{agent_workspace}", str(agent_workspace))
            prepared_env_params[key] = value
        
        # Add task_dir if not specified
        if "task_dir" not in prepared_env_params:
            prepared_env_params["task_dir"] = str(task_workspace)
        
        # Add random seed if not specified
        if "seed" not in prepared_env_params:
            prepared_env_params["seed"] = random.randint(0, 1000000)

        # Always suppress preprocessing output in runner mode.
        with suppress_all_output():
            env = EnvClass(**prepared_env_params)
        if verbose:
            print(f"[Task {task_id} | {task_label}] Environment created successfully")

        # Setup MCP servers
        mcp_config = setup_mcp_servers(mcp_configs, task_workspace, agent_workspace)

        # Create tool - use ProgrammaticToolCallingTool if programmatic_tool_calling is enabled
        has_programmatic = any(
            config.get("type") in ["programmatic_tool_calling", "programmatic-tool-calling"]
            and config.get("enabled", True)
            for config in mcp_configs.values()
        )
        has_context_workspace = any(
            config.get("type") == "context_workspace" and config.get("enabled", True)
            for config in mcp_configs.values()
        )
        disable_sms_dashboard = os.getenv("SM_DISABLE_DASHBOARD", "").lower() in (
            "1",
            "true",
            "yes",
            "on",
        )
        disable_sms_archive = os.getenv("SM_DISABLE_ARCHIVE", "").lower() in (
            "1",
            "true",
            "yes",
            "on",
        )
        disable_agent_archive = os.getenv("SM_DISABLE_AGENT_ARCHIVE", "").lower() in (
            "1",
            "true",
            "yes",
            "on",
        )
        enable_sms_delete = os.getenv("SM_ENABLE_DELETE", "").lower() in (
            "1",
            "true",
            "yes",
            "on",
        )
        enable_sms_state_board = os.getenv("SM_ENABLE_STATE_BOARD", "").lower() in (
            "1",
            "true",
            "yes",
            "on",
        )
        special_sms_archive_prompt = os.getenv("SM_SPECIAL_ARCHIVE_PROMPT", "").lower() in (
            "1",
            "true",
            "yes",
            "on",
        )
        better_sms_dashboard = os.getenv("SM_BETTER_DASHBOARD", "").lower() in (
            "1",
            "true",
            "yes",
            "on",
        )
        archive_placeholder_only = os.getenv("SM_ARCHIVE_PLACEHOLDER_ONLY", "").lower() in (
            "1",
            "true",
            "yes",
            "on",
        )
        strict_long_context = os.getenv("SM_STRICT_LONG_CONTEXT", "").lower() in (
            "1",
            "true",
            "yes",
            "on",
        )
        fixed_archive_policy = disable_agent_archive and not disable_sms_archive
        has_agent_context_management = ((not disable_sms_archive) and (not disable_agent_archive)) or enable_sms_delete
        strict_long_context = (
            strict_long_context
            and has_context_workspace
            and (has_agent_context_management or fixed_archive_policy)
        )

        # Only fix schema for OpenAI-compatible models (gpt-* or explicit openai prefix)
        fix_schema = "openai" in model.lower() or "gpt" in model.lower()

        if has_programmatic:
            tool = ProgrammaticToolCallingTool(mcp_config, validate_on_init=False, execution_timeout=120.0, fix_schema_for_openai=fix_schema)
        else:
            tool = MCPTool(mcp_config, validate_on_init=False, execution_timeout=120.0, fix_schema_for_openai=fix_schema)

        env = ToolEnvWrapperOpenAI(
            env,
            tools=[tool],
            max_tool_uses=max_tool_uses,
            sms_mode=has_context_workspace,
        )

        # Always suppress preprocessing output in runner mode.
        with suppress_all_output():
            obs, info, user_prompt, tools = env.reset()

        if has_context_workspace and tools:
            hidden_context_tools = set()
            if disable_sms_archive or disable_agent_archive:
                hidden_context_tools.add("context_workspace_archive")
            if not enable_sms_delete:
                hidden_context_tools.add("context_workspace_delete")
            if hidden_context_tools:
                tools = [
                    [
                        t for t in tool_group
                        if t.get("function", {}).get("name") not in hidden_context_tools
                    ]
                    for tool_group in tools
                ]
            if not enable_sms_state_board:
                tools = [
                    [
                        t for t in tool_group
                        if t.get("function", {}).get("name") != "context_workspace_update_state_board"
                    ]
                    for tool_group in tools
                ]
            if special_sms_archive_prompt or archive_placeholder_only:
                for tool_group in tools:
                    for t in tool_group:
                        fn = t.get("function", {})
                        if fn.get("name") != "context_workspace_archive":
                            continue
                        params = fn.get("parameters", {}) or {}
                        props = params.get("properties", {}) if isinstance(params, dict) else {}
                        replacement_prop = props.get("replacement") if isinstance(props, dict) else None
                        if archive_placeholder_only:
                            fn["description"] = (
                                "Archive block IDs into bare placeholders only. The replacement argument is ignored; "
                                "the visible context will contain only [ARCHIVED:...] markers with no semantic hint."
                            )
                            if isinstance(replacement_prop, dict):
                                replacement_prop["description"] = (
                                    "Ignored in this ablation; archive output is placeholder-only."
                                )
                        elif special_sms_archive_prompt:
                            fn["description"] = (
                                "Archive block IDs and write a compact retrieval guide. Do not copy full data. "
                                "Use replacement as a manual entry describing what information types the archived "
                                "block contains and what loading the original block would help recover."
                            )
                            if isinstance(replacement_prop, dict):
                                replacement_prop["description"] = (
                                    "A retrieval guide/manual entry for the archived block: describe the information "
                                    "types present and when loading the original block would be useful; do not copy "
                                    "complete raw data."
                                )

        # Save tools information for later storage
        tools_info = tools[0] if tools else None

        if verbose:
            print(f"[Task {task_id} | {task_label}] Environment initialized")
            print(f"[Task {task_id} | {task_label}] Initial observation length: {len(obs)}")

        # Check if memory_tool is included in mcp_configs
        has_memory_tool = any(
            config.get("type") in ["memory_tool", "memory-tool"] and config.get("enabled", True)
            for config in mcp_configs.values()
        )

        has_claim_done = any(
            config.get("type") == "claim_done" and config.get("enabled", True)
            for config in mcp_configs.values()
        )

        # Build the user prompt with optional enhancements
        enhanced_user_prompt = user_prompt

        # Add memory protocol if memory_tool is included
        if has_memory_tool:
            memory_protocol = (
                "\n\n"
                "IMPORTANT: ALWAYS VIEW YOUR MEMORY DIRECTORY BEFORE DOING ANYTHING ELSE.\n"
                "MEMORY PROTOCOL:\n"
                "1. Use the `view` command of your `memory_tool` to check for earlier progress.\n"
                "2. ... (work on the task) ...\n"
                "     - As you make progress, record status / progress / thoughts etc in your memory.\n"
                "ASSUME INTERRUPTION: Your context window might be reset at any moment, so you risk losing any progress that is not recorded in your memory directory."
            )
            enhanced_user_prompt += memory_protocol
            if verbose:
                print(f"[Task {task_id} | {task_label}] Memory tool detected: Added MEMORY PROTOCOL to user prompt")

        # Add self-managed context protocol if context_workspace is included
        if has_context_workspace:
            context_action = (
                "Delete completed context that has no remaining task value; deletion is permanent. "
                if disable_sms_archive and enable_sms_delete
                else "Archive and delete tools are unavailable to the agent in this run. Fixed archive policy externalizes old bulky tool transcripts only when the context would exceed the limit. "
                if disable_agent_archive
                else "Archive completed context that is no longer needed verbatim. "
            )
            if better_sms_dashboard:
                context_workspace_protocol = (
                    "\n\n"
                    "CONTEXT MANAGEMENT PROTOCOL:\n"
                    "A <context_workspace_status> dashboard is shown every turn as a compact map of context blocks. "
                    "Use context tools only when clearly needed. "
                    "Do not archive, delete, or offload content solely because it is old, large, or listed in context metadata; "
                    "leave content visible when the context budget is sufficient. "
                    "Large payloads may be represented by placeholders; inspect originals only when needed. "
                    "Use ordinary file/terminal/python tools, source metadata, and any in-context payload placeholders "
                    "to inspect external evidence when details are needed. "
                    "For structured data or calculations, use the source file, source tool, or query directly. "
                    "Do not copy table, CSV, or JSON rows from the conversation into code."
                )
            else:
                context_workspace_protocol = (
                    "\n\n"
                    "CONTEXT MANAGEMENT PROTOCOL:\n"
                    "A <context_workspace_status> dashboard is shown every turn as a compact map of context blocks. "
                    "Use context tools only when they help keep the conversation within the window. "
                    f"{context_action}"
                    "For structured data or calculations, use the source file, source tool, or query directly. "
                    "Do not copy table, CSV, or JSON rows from the conversation into code."
                )
            if enable_sms_state_board:
                context_workspace_protocol += (
                    " Use context_workspace_update_state_board(...) to keep the dashboard's State Board current "
                    "with compact work status, progress, decisions, and next actions."
                )
            if special_sms_archive_prompt:
                context_workspace_protocol += (
                    " When archiving, write the replacement as a retrieval guide or manual entry: do not copy full "
                    "data; describe what kinds of information the archived block contains and what loading the "
                    "original block would help recover, so a future reader can choose the right block to inspect."
                )
            enhanced_user_prompt += context_workspace_protocol
            if verbose:
                print(f"[Task {task_id} | {task_label}] Context workspace detected: Added CONTEXT MANAGEMENT PROTOCOL to user prompt")
        if has_context_workspace and strict_long_context and verbose:
            print(f"[Task {task_id} | {task_label}] Strict long-context mode enabled")

        # Add context awareness if enabled
        if context_awareness and max_context_size is not None:
            # Determine the context size to display based on whether context_reset or context_summary is enabled
            # When context_reset or context_summary is enabled with reset_size, use reset_size instead of max_context_size
            #display_context_size = reset_size if (reset_size is not None and (context_reset or context_summary)) else max_context_size
            display_context_size = max_context_size

            context_notice = (
                "\n\n"
                f"You need to complete the task within the following context window size:\n"
                f"<budget:token_budget>{display_context_size}</budget:token_budget>\n\n"
                f"Your context window will be automatically compacted as it approaches its limit, "
                f"allowing you to continue working indefinitely from where you left off. "
                f"Therefore, do not stop tasks early due to token budget concerns."
            )
            enhanced_user_prompt += context_notice
            if verbose:
                print(f"[Task {task_id} | {task_label}] Context awareness enabled: Added token budget ({display_context_size}) and context management notice to user prompt")

        # Generate a per-task trace_id for Venus sticky routing (trace mode).
        # All API calls within this task share the same trace_id → same backend → cache hits.
        import uuid as _uuid
        _task_trace_id = _uuid.uuid4().hex  # 32-char hex, stable for this task

        # For Claude models on Venus, use content block format with cache_control on the initial message.
        # The initial user message (system prompt + task) is large and fixed throughout the task,
        # so caching it saves significant tokens on every subsequent step.
        if "claude" in model.lower():
            initial_content = [{"type": "text", "text": enhanced_user_prompt, "cache_control": {"type": "ephemeral"}}]
            initial_user_message = {"role": "user", "content": initial_content}
        else:
            initial_user_message = {"role": "user", "content": enhanced_user_prompt}
        messages = [initial_user_message]
        full_messages_history.append(initial_user_message.copy())  # Add initial user prompt to full history

        # ── Context Workspace: initialize WorkspaceManager ─────────────────────
        _cw_manager = None
        _cw_workspace_dir = None
        _tkt_enc = None
        if has_context_workspace:
            try:
                import pathlib as _pl
                from gem.tools.mcp_server.context_workspace.workspace_manager import WorkspaceManager
                for _srv_cfg in mcp_configs.values():
                    if _srv_cfg.get("type") == "context_workspace":
                        _cw_workspace_dir = _srv_cfg.get("params", {}).get("workspace_dir", "")
                        if not _cw_workspace_dir:
                            _cw_workspace_dir = str(task_workspace / "context_workspace")
                        _cw_workspace_dir = _cw_workspace_dir.replace("{task_workspace}", str(task_workspace))
                        _cw_workspace_dir = _cw_workspace_dir.replace("{agent_workspace}", str(agent_workspace))
                        break
                if _cw_workspace_dir:
                    _cw_payload_dir = str(agent_workspace / ".sms_payloads")
                    for _srv_cfg in mcp_configs.values():
                        if _srv_cfg.get("type") == "context_workspace":
                            _cw_payload_dir = _srv_cfg.get("params", {}).get("payload_dir", _cw_payload_dir)
                            _cw_payload_dir = _cw_payload_dir.replace("{task_workspace}", str(task_workspace))
                            _cw_payload_dir = _cw_payload_dir.replace("{agent_workspace}", str(agent_workspace))
                            break
                    _cw_manager = WorkspaceManager(
                        _pl.Path(_cw_workspace_dir),
                        token_budget=max_context_size or 200_000,
                        public_payload_dir=_pl.Path(_cw_payload_dir),
                    )
                    # Register the initial task message as B1
                    _cw_manager.register_message(initial_user_message, 0)
                    # Measure fixed overhead (ALL tools definitions) once using
                    # the same per-tool estimate as trim/gate.
                    try:
                        import tiktoken as _tkt
                        _tkt_enc = _tkt.get_encoding("cl100k_base")
                        _overhead = 0
                        for _t in (tools[0] if tools else []):
                            _fn = _t.get("function") or {}
                            _s = (
                                (_fn.get("name") or "")
                                + (_fn.get("description") or "")
                                + json.dumps(_fn.get("parameters") or {}, ensure_ascii=False)
                            )
                            _overhead += len(_tkt_enc.encode(_s, disallowed_special=()))
                        _cw_manager.set_overhead(_overhead)
                    except Exception:
                        pass  # overhead stays 0, calibration still works without it
                    if verbose:
                        print(f"[Task {task_id} | {task_label}] Context workspace manager initialized at {_cw_workspace_dir}")
            except Exception as _e:
                if verbose:
                    print(f"[Task {task_id} | {task_label}] Warning: WorkspaceManager init failed: {_e}")
                _cw_manager = None

        # Prepare output path - save trajectory in task workspace
        save_file = task_workspace / "trajectory.json"
        warning_log_path = task_workspace / "logs" / "warnings.log"

        # Run interaction loop
        done = False
        step_count = 0

        # Per-task timing accumulators (seconds)
        _perf_api_s   = 0.0  # total time inside make_aihubmix_api_request
        _perf_http_s  = 0.0  # Gemini HTTP wait only (subset of api)
        _perf_tok_s   = 0.0  # token estimation only (subset of api)
        _perf_tool_s  = 0.0  # env.step_openai (tool execution)
        _perf_cw_s    = 0.0  # CW assembly + bulk-output gate processing
        _perf_total_s = 0.0  # total loop wall time
        _omitted_result_fingerprints = set()
        try:
            _task_timeout_s = int(os.getenv("LOCA_TASK_TIMEOUT_SECONDS", "1800"))
        except ValueError:
            _task_timeout_s = 1800
        _task_timeout_s = max(0, _task_timeout_s)
        _task_started_at = time.perf_counter()

        while not done:
            if _task_timeout_s and (time.perf_counter() - _task_started_at) >= _task_timeout_s:
                _elapsed = time.perf_counter() - _task_started_at
                raise TimeoutError(
                    f"Task wall-clock timeout after {_elapsed:.1f}s "
                    f"(limit: {_task_timeout_s}s / {_task_timeout_s / 60:.1f} min)"
                )

            _t_step_start = time.perf_counter()
            step_count += 1
            if verbose:
                print(f"[Task {task_id} | {task_label}] Step {step_count}")

            # ── Sync in-memory cache with disk state (once per step) ─────────
            # CW server tools (archive/delete/note) write directly to state.json
            # without updating _state_cache. Clear the cache once here so the
            # next _load() reads the fresh on-disk state (including any block
            # status changes from the previous step's CW tool calls).
            if _cw_manager is not None:
                _cw_manager._state_cache = None

            # ── Context Workspace: assemble visible messages + inject dashboard ─
            _t_cw_start = time.perf_counter()
            if _cw_manager is not None:
                try:
                    _management_api_messages = None
                    # SMS preflight: before building the API payload, replace
                    # old large raw tool-result blocks with stable offload
                    # placeholders until the full assembled request fits the
                    # configured preflight ratio. This is a final safety guard:
                    # with the default ratio 0.98 and reserve 0, it really
                    # triggers near 98% of the total input budget.
                    try:
                        _state = _cw_manager.get_state()
                        _fixed_overhead = int(_state.get("overhead_tokens", 0) or 0)
                        _dashboard_overhead = 0
                        _dashboard_preview = _cw_manager.get_dashboard()
                        _dashboard_msg = {
                            "role": "user",
                            "content": f"<context_workspace_status>\n{_dashboard_preview}\n</context_workspace_status>",
                        }
                        try:
                            from gem.tools.mcp_server.context_workspace.workspace_manager import count_msg_tokens as _count_msg_tokens
                            _dashboard_overhead = _count_msg_tokens([_dashboard_msg], _tkt_enc)
                        except Exception:
                            _dashboard_overhead = max(1, len(json.dumps([_dashboard_msg], ensure_ascii=False)) // 3)
                        try:
                            _preflight_ratio = float(os.getenv("SM_PREFLIGHT_TARGET_RATIO", "0.98"))
                        except ValueError:
                            _preflight_ratio = 0.98
                        _preflight_ratio = max(0.50, min(1.00, _preflight_ratio))
                        try:
                            _turn_reserve = int(os.getenv("SM_PREFLIGHT_TURN_RESERVE_TOKENS", "0"))
                        except ValueError:
                            _turn_reserve = 0
                        _target_conv = (
                            int((max_context_size or 200_000) * _preflight_ratio)
                            - _fixed_overhead
                            - _dashboard_overhead
                            - _turn_reserve
                        )
                        # In strict-LC ablations, only disable automatic
                        # preflight offload when archive itself is disabled.
                        # With archive available, preflight offload is just the
                        # final automatic archive guard.
                        _disable_preflight_offload = strict_long_context and disable_sms_archive
                        if _disable_preflight_offload:
                            _current_conv = _cw_manager.conv_tokens(messages, _tkt_enc)
                            if _current_conv > _target_conv:
                                _warning = (
                                    f"[SM-STRICT-LC] ⚠️  Preflight rejected automatic offload for {task_label} "
                                    f"step {step_count}: context ~{_current_conv:,} tok, "
                                    f"target ~{max(1, _target_conv):,} tok."
                                )
                                print(_warning)
                                _append_warning_log(warning_log_path, _warning)
                                _pressure_msg = {
                                    "role": "user",
                                    "content": (
                                        "[CONTEXT_LIMIT_REJECTED]\n"
                                        "The conversation is too large for the next model request.\n"
                                        f"Current context: ~{_current_conv:,} tok\n"
                                        f"Target before continuing: ~{max(1, _target_conv):,} tok\n"
                                        "Free context with the available context-management tools, then retry the blocked action."
                                    ),
                                }
                                _management_api_messages = []
                                for _msg in messages:
                                    if _msg.get("role") in ("system", "developer"):
                                        _management_api_messages.append(_msg)
                                for _msg in messages:
                                    if _msg.get("role") == "user":
                                        _management_api_messages.append(_msg)
                                        break
                                if not disable_sms_dashboard:
                                    _management_api_messages.append(_dashboard_msg)
                                _management_api_messages.append(_pressure_msg)
                        else:
                            if fixed_archive_policy:
                                try:
                                    _target_ratio = float(os.getenv("SM_FIXED_ARCHIVE_TARGET_RATIO", "0.98"))
                                except ValueError:
                                    _target_ratio = 0.98
                                _target_ratio = max(0.50, min(1.00, _target_ratio))
                                _target_conv = (
                                    int((max_context_size or 200_000) * _target_ratio)
                                    - _fixed_overhead
                                    - _dashboard_overhead
                                    - _turn_reserve
                                )
                            _offloaded = _cw_manager.preflight_offload_raw_tool_results(
                                messages,
                                target_conv_tokens=max(1, _target_conv),
                                tkt_enc=_tkt_enc,
                                policy="oldest" if fixed_archive_policy else "largest",
                            )
                            if _offloaded:
                                if fixed_archive_policy:
                                    _warning = (
                                        f"[SM-FIXED-ARCHIVE] ⚠️  Fixed archive policy externalized {_offloaded} block(s) "
                                        f"for {task_label} step {step_count}."
                                    )
                                else:
                                    _warning = (
                                        f"[SM-SAFETY] ⚠️  Preflight offload fired for {task_label} "
                                        f"step {step_count} ({_offloaded} block(s))."
                                    )
                                print(_warning)
                                _append_warning_log(warning_log_path, _warning)
                    except Exception as _e:
                        if verbose:
                            print(f"[Task {task_id} | {task_label}] Warning: SMS preflight offload failed: {_e}")
                        _append_warning_log(
                            warning_log_path,
                            f"[Task {task_id} | {task_label}] Warning: SMS preflight offload failed at step {step_count}: {_e}",
                        )

                    api_messages = _cw_manager.assemble(messages)
                    n_compressed = sum(
                        1 for b in _cw_manager.get_state().get("blocks", {}).values()
                        if b.get("status") == "compressed"
                    )

                    if _management_api_messages is not None:
                        api_messages = _management_api_messages
                    else:
                        # Append dashboard as an independent user message at the end of
                        # api_messages so the agent sees the current context state right
                        # before it responds. Using a standalone message keeps each prior
                        # message at its original size.
                        if api_messages and not disable_sms_dashboard:
                            dashboard_text = _cw_manager.get_dashboard()
                            dashboard_msg = {
                                "role": "user",
                                "content": f"<context_workspace_status>\n{dashboard_text}\n</context_workspace_status>",
                            }
                            api_messages = api_messages + [dashboard_msg]

                    if verbose:
                        print(f"[Task {task_id} | {task_label}] Assembled {len(api_messages)} msgs (of {len(messages)} raw, {n_compressed} compressed)")
                except Exception as _e:
                    if verbose:
                        print(f"[Task {task_id} | {task_label}] Warning: assembly failed, using raw messages: {_e}")
                    api_messages = messages
            else:
                api_messages = messages

            _perf_cw_s += time.perf_counter() - _t_cw_start  # assembly portion only

            # ── Save full assembled context for each step (outside CW timing) ──
            try:
                _ctx_log_dir = task_workspace / "logs"
                _ctx_log_dir.mkdir(parents=True, exist_ok=True)
                _ctx_log_path = _ctx_log_dir / f"context_step_{step_count:04d}.json"
                with open(_ctx_log_path, "w") as _f:
                    import json as _json
                    _json.dump({
                        "step": step_count,
                        "total_raw_messages": len(messages),
                        "assembled_messages": len(api_messages),
                        "messages": api_messages,
                    }, _f, indent=2, ensure_ascii=False)
            except Exception:
                pass

            # Make API request (use assembled visible messages if workspace is active)
            _t_api_start = time.perf_counter()
            response = make_aihubmix_api_request(
                messages=api_messages,
                model_name=model,
                aihubmix_api_keys=api_key,
                aihubmix_api_url=f"{base_url}/chat/completions",
                tools=tools[0] if tools else None,
                max_retries=max_retries,
                temperature=1.0,
                top_p=1.0,
                max_tokens=max_tokens,
                max_context_size=max_context_size,
                context_awareness=context_awareness,
                protect_archived_placeholders=(_cw_manager is not None),
                reasoning_effort=reasoning_effort,
                reasoning_max_tokens=reasoning_max_tokens,
                reasoning_enabled=reasoning_enabled,
                reasoning_exclude=reasoning_exclude,
                trace_id=_task_trace_id,
                warning_log_path=warning_log_path,
            )

            _perf_api_s += time.perf_counter() - _t_api_start
            _resp_timing = response.get('timing', {})
            _perf_http_s += _resp_timing.get('http_s', 0.0)
            _perf_tok_s  += _resp_timing.get('tok_est_s', 0.0)

            if response.get('type') == 'error':
                error_text = "; ".join(str(x) for x in response.get('data', [])) or "Unknown API error"
                raise RuntimeError(f"Model API request failed at step {step_count}: {error_text}")

            # Track API usage per step
            raw_resp = response.get('raw_response', {})
            if raw_resp and 'usage' in raw_resp:
                usage = raw_resp['usage']
                # Venus returns cache info under prompt_tokens_details
                prompt_tokens_details = usage.get('prompt_tokens_details', {})
                cache_read_tokens = prompt_tokens_details.get('cache_read_tokens', 0)
                cache_creation_tokens = prompt_tokens_details.get('cache_creation_tokens', 0)
                usage_tracking.append({
                    'step': step_count,
                    'prompt_tokens': usage.get('prompt_tokens', 0),
                    'completion_tokens': usage.get('completion_tokens', 0),
                    'total_tokens': usage.get('total_tokens', 0),
                    'prompt_cache_hit_tokens': cache_read_tokens,
                    'prompt_cache_miss_tokens': usage.get('prompt_cache_miss_tokens', 0),
                    'cache_read_tokens': cache_read_tokens,
                    'cache_creation_tokens': cache_creation_tokens,
                })

            # Update raw messages only when the API helper returns a real raw
            # trajectory trim. SM trim is payload-only, because the request was
            # built from assembled context_workspace messages.
            if 'trimmed_messages' in response and response['trimmed_messages'] is not None:
                original_count = len(messages)
                original_messages = messages  # Save original messages for comparison
                messages = response['trimmed_messages']
                if verbose:
                    print(f"[Task {task_id} | {task_label}] Messages updated after trimming: {original_count} -> {len(messages)}")

                # Check if memory warning was trimmed away
                if memory_warning_issued:
                    # Check if memory warning existed in original but not in trimmed
                    had_memory_warning = any(
                        msg.get('role') == 'user' and
                        isinstance(msg.get('content', ''), str) and
                        '**You are nearing the context window limit.**' in msg.get('content', '')
                        for msg in original_messages
                    )
                    has_memory_warning = any(
                        msg.get('role') == 'user' and
                        isinstance(msg.get('content', ''), str) and
                        '**You are nearing the context window limit.**' in msg.get('content', '')
                        for msg in messages
                    )
                    if had_memory_warning and not has_memory_warning:
                        memory_warning_issued = False
                        if verbose:
                            print(f"[Task {task_id} | {task_label}] Memory warning was trimmed away, resetting memory_warning_issued flag")

            # Record trim event even when SM used payload-only trimming and raw
            # messages were intentionally left untouched.
            if 'trim_info' in response and response['trim_info'] is not None:
                import copy
                trim_event = {
                    'step': step_count,
                    'trim_info': copy.deepcopy(response['trim_info']),
                    'context': 'main_api_call'  # Distinguish from summary trim
                }
                trim_events.append(trim_event)
                if verbose:
                    mode = "payload-only " if response['trim_info'].get('payload_only') else ""
                    print(f"[Task {task_id} | {task_label}] {mode}Trim event recorded: removed {response['trim_info']['removed_count']} messages")

            # Check if response has call_messages (defensive check)
            if 'call_messages' not in response:
                print(f"ERROR: Response missing 'call_messages' key. Response: {response}")
                # Create a default error message
                call_messages = {
                    "role": "assistant", 
                    "content": f"Error: Invalid response format - {response.get('type', 'unknown')}: {response.get('data', ['Unknown error'])}"
                }
            else:
                call_messages = response['call_messages']
            
            # Ensure all tool_calls have arguments field (fix for consistency)
            if 'tool_calls' in call_messages and call_messages['tool_calls']:
                for tool_call in call_messages['tool_calls']:
                    if 'function' in tool_call:
                        if 'arguments' not in tool_call['function']:
                            # If no arguments key, add empty dict
                            tool_call['function']['arguments'] = "{}"
                        elif tool_call['function']['arguments'] == "":
                            # If arguments is empty string, convert to empty dict JSON
                            tool_call['function']['arguments'] = "{}"
                        # elif isinstance(tool_call['function']['arguments'], str):
                        #     # If arguments is a JSON string, parse it
                        #     try:
                        #         tool_call['function']['arguments'] = json.loads(tool_call['function']['arguments'])
                        #     except json.JSONDecodeError:
                        #         # If parsing fails, use empty dict
                        #         tool_call['function']['arguments'] = "{}"

            # Add assistant's message to conversation
            messages.append(call_messages)
            # Register with workspace manager (enables selective archiving)
            if _cw_manager is not None:
                try:
                    _cw_manager.register_message(call_messages, len(messages) - 1)
                except Exception:
                    pass

            # Save a copy of messages to full history before potential reset
            full_messages_history.append(call_messages.copy())

            if verbose:
                print("response", response)

            _t_tool_start = time.perf_counter()
            with suppress_all_output():
                next_obs, reward, terminated, truncated, info = env.step_openai(response, verbose=False)
            _perf_tool_s += time.perf_counter() - _t_tool_start

            if verbose:
                print("next_obs", next_obs)
                print("reward", reward)
                print("terminated", terminated)
                print("truncated", truncated)
                print("info", info)

            # Update state
            done = terminated or truncated

            if not done:
                try:
                    tool_results = json.loads(next_obs)

                    if _cw_manager is not None:
                        _t_cw_gate_start = time.perf_counter()
                        # ── Bulk-output gate: keep oversized raw results out of context ─
                        # Instead of asking the agent to pull a huge payload, store
                        # it out of band and return a short observation. The intended
                        # recovery path is a narrower source-tool call.
                        try:
                            import tiktoken as _tkt_mod
                            _tkt_enc = _tkt_mod.get_encoding("cl100k_base")
                        except Exception:
                            _tkt_enc = None

                        def _tok_estimate(text):
                            if _tkt_enc is not None:
                                return len(_tkt_enc.encode(str(text), disallowed_special=()))
                            return max(1, len(str(text)) // 3)

                        try:
                            from gem.tools.mcp_server.context_workspace.workspace_manager import count_msg_tokens as _count_msg_tokens
                        except ImportError:
                            _count_msg_tokens = None

                        def _msg_tok_estimate(msgs):
                            if _count_msg_tokens is not None:
                                return _count_msg_tokens(msgs, _tkt_enc)
                            total = 0
                            for _m in msgs:
                                _c = _m.get("content") or ""
                                if not isinstance(_c, str):
                                    _c = json.dumps(_c, ensure_ascii=False)
                                total += _tok_estimate(_c)
                                for _tc in _m.get("tool_calls") or []:
                                    _fn = _tc.get("function") or {}
                                    total += _tok_estimate(
                                        (_fn.get("name") or "") + (_fn.get("arguments") or "")
                                    )
                                if _m.get("name"):
                                    total += _tok_estimate(_m["name"])
                            return total

                        def _get_tool_name(tr, call_msgs):
                            """Find the tool name for a tool result via its tool_call_id."""
                            tc_id = tr.get("tool_call_id", "")
                            for tc in call_msgs.get("tool_calls", []) if isinstance(call_msgs, dict) else []:
                                if tc.get("id") == tc_id:
                                    return tc.get("function", {}).get("name", "tool")
                            return "tool"

                        def _get_tool_source(tr, call_msgs):
                            """Return (tool_name, parsed_args, tool_call_id) for a tool result."""
                            tc_id = tr.get("tool_call_id", "")
                            for tc in call_msgs.get("tool_calls", []) if isinstance(call_msgs, dict) else []:
                                if tc.get("id") != tc_id:
                                    continue
                                fn = tc.get("function", {}) or {}
                                raw_args = fn.get("arguments") or ""
                                try:
                                    args_obj = json.loads(raw_args) if isinstance(raw_args, str) else raw_args
                                except Exception:
                                    args_obj = raw_args
                                return fn.get("name", "tool"), args_obj, tc_id
                            return _get_tool_name(tr, call_msgs), "", tc_id

                        def _get_tool_fingerprint(tr, call_msgs):
                            """Stable fingerprint of the source tool call that produced this result."""
                            import hashlib as _hashlib
                            tc_id = tr.get("tool_call_id", "")
                            for tc in call_msgs.get("tool_calls", []) if isinstance(call_msgs, dict) else []:
                                if tc.get("id") == tc_id:
                                    fn = tc.get("function", {}) or {}
                                    raw_args = fn.get("arguments") or ""
                                    try:
                                        args_obj = json.loads(raw_args) if isinstance(raw_args, str) else raw_args
                                        args_norm = json.dumps(args_obj, ensure_ascii=False, sort_keys=True)
                                    except Exception:
                                        args_norm = str(raw_args)
                                    key = (fn.get("name", "tool") or "tool") + "\n" + args_norm
                                    return _hashlib.sha256(key.encode("utf-8")).hexdigest()
                            fallback = (tr.get("tool_call_id", "") or "") + "\n" + str(tr.get("name", "tool"))
                            return _hashlib.sha256(fallback.encode("utf-8")).hexdigest()

                        try:
                            # Compute remaining context space for the bulk-output gate.
                            #
                            # This follows Claude Code's shape: large raw tool
                            # results are handled as tool-output hygiene, while
                            # the context percentage is only a warning/compaction
                            # signal. A soft budget must not turn into an
                            # admission gate for small results; otherwise the
                            # agent can get stuck near 80% with even tiny tool
                            # results omitted.
                            #
                            # Trim still owns the true hard limit at 100%, but
                            # the admission gate must leave room for the next
                            # assistant turn and token-estimation drift. Claude
                            # Code treats near-capacity as pressure before the
                            # hard limit; doing the same here prevents SMS from
                            # flying exactly at 128k and then needing trim one
                            # turn later.
                            # Tools overhead uses the SAME per-tool calculation
                            # as trim so gate and trim agree.
                            try:
                                _gate_ratio = float(os.getenv("SM_GATE_HARD_RATIO", "0.98"))
                            except ValueError:
                                _gate_ratio = 0.98
                            _gate_ratio = max(0.50, min(1.00, _gate_ratio))
                            _hard_available = int((max_context_size or 200_000) * _gate_ratio)
                            _output_reserve = 0  # trim is at 100%, no output reserve needed
                            _tools_tok = 0
                            for _t in (tools[0] if tools else []):
                                _fn = _t.get("function") or {}
                                _tools_tok += _tok_estimate(
                                    (_fn.get("name") or "")
                                    + (_fn.get("description") or "")
                                    + json.dumps(_fn.get("parameters") or {}, ensure_ascii=False)
                                )
                            # Reserve dashboard cost even in no-dashboard ablations
                            # so the only ablated signal is visibility, not budget.
                            _dashboard_tok = 0
                            _dashboard_text = _cw_manager.get_dashboard()
                            _dashboard_tok = _msg_tok_estimate([{
                                "role": "user",
                                "content": f"<context_workspace_status>\n{_dashboard_text}\n</context_workspace_status>",
                            }])
                            _overhead_tok = _tools_tok + _dashboard_tok

                            # Compute baseline token count ONCE before processing this
                            # batch of tool results. Then track incremental additions so
                            # remaining is always current without re-encoding the whole
                            # history on every iteration.
                            #
                            # IMPORTANT: use _cw_manager.conv_tokens() — this assembles
                            # messages first (replacing compressed blocks with replacement
                            # placeholders) before counting, so the estimate matches
                            # exactly what the API will receive. Using raw messages would
                            # count full content for compressed blocks and overestimate
                            # by 6x+, causing all tool results to be incorrectly flagged
                            # as too large for the remaining context.
                            _base_conv_tok = _cw_manager.conv_tokens(messages, _tkt_enc)
                            _added_tok = 0  # tokens added to messages in this loop

                            try:
                                _large_result_threshold = int(
                                    os.getenv("SM_LARGE_RESULT_BLOCK_TOKENS", "12000")
                                )
                            except ValueError:
                                _large_result_threshold = 12000
                            try:
                                # Claude Code also enforces an aggregate budget
                                # for one turn's batch of parallel tool results
                                # (200k chars ≈ 50k tokens). Without this, many
                                # medium-sized results can collectively fill the
                                # window even though no single result is large.
                                _batch_budget_tokens = int(
                                    os.getenv("SM_TOOL_RESULTS_PER_TURN_TOKENS", "50000")
                                )
                            except ValueError:
                                _batch_budget_tokens = 50000
                            try:
                                _preview_chars = int(
                                    os.getenv("SM_LARGE_RESULT_PREVIEW_CHARS", "2000")
                                )
                            except ValueError:
                                _preview_chars = 2000
                            _preview_chars = max(0, min(8000, _preview_chars))

                            _CW_BYPASS = {
                                "context_workspace_archive",
                            }
                            _tool_result_meta = {}
                            _batch_candidates = []
                            for _i, _tr in enumerate(tool_results):
                                _tool_name = _get_tool_name(_tr, call_messages)
                                _tok_est = _msg_tok_estimate([_tr])
                                _tool_result_meta[_i] = (_tool_name, _tok_est)
                                if _tool_name not in _CW_BYPASS:
                                    _batch_candidates.append((_tok_est, _i))

                            _batch_total = sum(tok for tok, _ in _batch_candidates)
                            _batch_hard_remaining = (
                                _hard_available
                                - _output_reserve
                                - _overhead_tok
                                - _base_conv_tok
                            )
                            if fixed_archive_policy and _batch_total > _batch_hard_remaining:
                                _target_for_batch = (
                                    _hard_available
                                    - _output_reserve
                                    - _overhead_tok
                                    - _batch_total
                                )
                                _fixed_offloaded = _cw_manager.preflight_offload_raw_tool_results(
                                    messages,
                                    target_conv_tokens=max(1, _target_for_batch),
                                    tkt_enc=_tkt_enc,
                                    policy="oldest",
                                )
                                if _fixed_offloaded:
                                    _warning = (
                                        f"[SM-FIXED-ARCHIVE] ⚠️  Fixed archive policy externalized {_fixed_offloaded} block(s) "
                                        f"before admitting tool results for {task_label} step {step_count}."
                                    )
                                    print(_warning)
                                    _append_warning_log(warning_log_path, _warning)
                                    _base_conv_tok = _cw_manager.conv_tokens(messages, _tkt_enc)
                                    _added_tok = 0
                                    _batch_hard_remaining = (
                                        _hard_available
                                        - _output_reserve
                                        - _overhead_tok
                                        - _base_conv_tok
                                    )
                            _strict_reject_turn = (
                                strict_long_context
                                and _batch_total > _batch_hard_remaining
                            )

                            if _strict_reject_turn:
                                _attempt_lines = []
                                for _tok_est, _i in _batch_candidates[:30]:
                                    _name, _ = _tool_result_meta.get(_i, ("tool", _tok_est))
                                    _attempt_lines.append(f"- {_name}: ~{_tok_est:,} tok")
                                if len(_batch_candidates) > 30:
                                    _attempt_lines.append(
                                        f"- ... {len(_batch_candidates) - 30} more tool result(s)"
                                    )
                                _attempt_summary = "\n".join(_attempt_lines) or "- none"
                                _warning = (
                                    f"[SM-STRICT-LC] ⚠️  Rejected tool result turn for {task_label} "
                                    f"step {step_count}: attempted ~{_batch_total:,} tok, "
                                    f"remaining ~{_batch_hard_remaining:,} tok."
                                )
                                print(_warning)
                                _append_warning_log(warning_log_path, _warning)

                                for _tr_idx, tr in enumerate(tool_results):
                                    tool_name, tok_est = _tool_result_meta.get(
                                        _tr_idx, (_get_tool_name(tr, call_messages), 0)
                                    )

                                    if tool_name in _CW_BYPASS:
                                        if tool_name.startswith("context_workspace_"):
                                            _cw_manager._state_cache = None
                                        messages.append(tr)
                                        _cw_manager.register_message(
                                            tr, len(messages) - 1, blocked=False
                                        )
                                        if tool_name.startswith("context_workspace_"):
                                            _cw_manager.update_dashboard_cache()
                                        continue

                                    reject_msg = {
                                        "role": tr.get("role", "tool"),
                                        "tool_call_id": tr.get("tool_call_id", ""),
                                        "content": (
                                            "[CONTEXT_LIMIT_REJECTED]\n"
                                            "This strict long-context run did not add this "
                                            "tool result to the conversation because the full "
                                            "tool-result turn would exceed the context limit.\n"
                                            f"Current assembled context: ~{_base_conv_tok:,} tok\n"
                                            f"Fixed overhead: ~{_overhead_tok:,} tok\n"
                                            f"Hard admission limit: ~{_hard_available:,} tok\n"
                                            f"Remaining for this tool turn: ~{_batch_hard_remaining:,} tok\n"
                                            f"Attempted total: ~{_batch_total:,} tok\n"
                                            "Attempted tool results:\n"
                                            f"{_attempt_summary}\n"
                                            "Clean more context, then try again with the same "
                                            "or a narrower tool call."
                                        ),
                                    }
                                    messages.append(reject_msg)
                                    _cw_manager.register_message(
                                        reject_msg, len(messages) - 1, blocked=False
                                    )

                                _cw_manager.update_dashboard_cache()
                                tool_results = []

                            _batch_omit_indices = set()
                            if (
                                not strict_long_context
                                and _batch_total > _batch_budget_tokens
                            ):
                                _remaining_batch_total = _batch_total
                                for _tok_est, _i in sorted(_batch_candidates, reverse=True):
                                    if _remaining_batch_total <= _batch_budget_tokens:
                                        break
                                    _batch_omit_indices.add(_i)
                                    _remaining_batch_total -= _tok_est

                            for _tr_idx, tr in enumerate(tool_results):
                                tool_name, tok_est = _tool_result_meta.get(
                                    _tr_idx, (_get_tool_name(tr, call_messages), 0)
                                )

                                # Context-workspace control tools bypass the bulk gate.
                                if tool_name in _CW_BYPASS:
                                    # Context-workspace tools mutate workspace_state.json
                                    # in their own MCP server process. Drop the runner's
                                    # cached state before registering their tool result,
                                    # otherwise register_message() can save stale state
                                    # over the server's archive/note/defer changes.
                                    if tool_name.startswith("context_workspace_"):
                                        _cw_manager._state_cache = None
                                    messages.append(tr)
                                    _cw_manager.register_message(tr, len(messages) - 1, blocked=False)
                                    if tool_name.startswith("context_workspace_"):
                                        _cw_manager.update_dashboard_cache()
                                    # After a CW tool (especially archive), re-compute the
                                    # assembled baseline — archiving shrinks the assembled
                                    # context (compressed blocks → short placeholders), so
                                    # incremental tracking would overestimate without a refresh.
                                    _base_conv_tok = _cw_manager.conv_tokens(messages, _tkt_enc)
                                    _added_tok = 0
                                    continue

                                # Check if adding this result would overflow the
                                # hard context limit. The 80% budget shown in the
                                # dashboard is advisory; it should not suppress
                                # small results needed to keep working.
                                # Use baseline + incremental delta so remaining stays
                                # accurate as we add bypass messages in this loop.
                                _conv_tok = _base_conv_tok + _added_tok
                                _hard_remaining = _hard_available - _output_reserve - _overhead_tok - _conv_tok
                                _would_overflow = tok_est > _hard_remaining
                                if fixed_archive_policy and _would_overflow:
                                    _target_for_result = (
                                        _hard_available
                                        - _output_reserve
                                        - _overhead_tok
                                        - _added_tok
                                        - tok_est
                                    )
                                    _fixed_offloaded = _cw_manager.preflight_offload_raw_tool_results(
                                        messages,
                                        target_conv_tokens=max(1, _target_for_result),
                                        tkt_enc=_tkt_enc,
                                        policy="oldest",
                                    )
                                    if _fixed_offloaded:
                                        _warning = (
                                            f"[SM-FIXED-ARCHIVE] ⚠️  Fixed archive policy externalized {_fixed_offloaded} block(s) "
                                            f"before admitting {tool_name} for {task_label} step {step_count}."
                                        )
                                        print(_warning)
                                        _append_warning_log(warning_log_path, _warning)
                                        _base_conv_tok = _cw_manager.conv_tokens(messages, _tkt_enc)
                                        _conv_tok = _base_conv_tok + _added_tok
                                        _hard_remaining = _hard_available - _output_reserve - _overhead_tok - _conv_tok
                                        _would_overflow = tok_est > _hard_remaining
                                _large_result = (
                                    not strict_long_context
                                    and tok_est >= _large_result_threshold
                                )
                                _batch_over_budget = (
                                    not strict_long_context
                                    and _tr_idx in _batch_omit_indices
                                )

                                if not _would_overflow and not _large_result and not _batch_over_budget:
                                    # Fits fine — put directly into context.
                                    messages.append(tr)
                                    _cw_manager.register_message(tr, len(messages) - 1, blocked=False)
                                    _added_tok += tok_est  # track so next tr sees accurate remaining
                                    continue

                                if strict_long_context and _would_overflow:
                                    _warning = (
                                        f"[SM-STRICT-LC] ⚠️  Rejected tool result for {task_label} "
                                        f"step {step_count} ({tool_name}): attempted ~{tok_est:,} tok, "
                                        f"remaining ~{_hard_remaining:,} tok."
                                    )
                                    print(_warning)
                                    _append_warning_log(warning_log_path, _warning)
                                    reject_msg = {
                                        "role": tr.get("role", "tool"),
                                        "tool_call_id": tr.get("tool_call_id", ""),
                                        "content": (
                                            "[CONTEXT_LIMIT_REJECTED]\n"
                                            "This strict long-context run did not add this "
                                            "tool result to the conversation because it would "
                                            "exceed the context limit.\n"
                                            f"Current assembled context: ~{_conv_tok:,} tok\n"
                                            f"Fixed overhead: ~{_overhead_tok:,} tok\n"
                                            f"Hard admission limit: ~{_hard_available:,} tok\n"
                                            f"Remaining for this result: ~{_hard_remaining:,} tok\n"
                                            f"Attempted result: ~{tok_est:,} tok\n"
                                            "Clean more context, then try again with the same "
                                            "or a narrower tool call."
                                        ),
                                    }
                                    messages.append(reject_msg)
                                    _cw_manager.register_message(
                                        reject_msg, len(messages) - 1, blocked=False
                                    )
                                    _added_tok += _msg_tok_estimate([reject_msg])
                                    continue

                                _reasons = []
                                if _would_overflow:
                                    _reasons.append(f"would exceed hard remaining {_hard_remaining:,} tok")
                                if _large_result:
                                    _reasons.append(f"single result >= {_large_result_threshold:,} tok")
                                if _batch_over_budget:
                                    _reasons.append(f"turn tool results exceed {_batch_budget_tokens:,} tok")
                                if _would_overflow:
                                    _warning = (
                                        f"[SM-SAFETY] ⚠️  Bulk-output gate fired for {task_label} "
                                        f"step {step_count} ({tool_name})."
                                    )
                                    print(_warning)
                                    _append_warning_log(warning_log_path, _warning)

                                # Keep the observation factual. Tool docs already
                                # describe how to request narrower views when needed.
                                fp = _get_tool_fingerprint(tr, call_messages)
                                duplicate_omission = fp in _omitted_result_fingerprints
                                if not duplicate_omission:
                                    _omitted_result_fingerprints.add(fp)
                                    blocked_idx = len(messages)
                                    block_id = _cw_manager.register_message(tr, blocked_idx, blocked=True)
                                    _src_tool, _src_args, _src_tc_id = _get_tool_source(tr, call_messages)
                                    _cw_manager.set_tool_source(block_id, _src_tool, _src_args, _src_tc_id)
                                else:
                                    block_id = None
                                _payload_path = ""
                                _source_meta_path = ""
                                _partial_info = _partial_rows_info(str(tr.get("content", "")))
                                if block_id is not None:
                                    try:
                                        _block_state = (
                                            _cw_manager.get_state()
                                            .get("blocks", {})
                                            .get(block_id, {})
                                        )
                                        _payload_path = (
                                            _block_state
                                            .get("public_payload_path", "")
                                        )
                                        _source_meta_path = _block_state.get("public_source_metadata_path", "")
                                        _partial_info = _block_state.get("payload_partial_info", "") or _partial_info
                                    except Exception:
                                        _payload_path = ""
                                        _source_meta_path = ""

                                _result_label = "duplicate tool transcript" if duplicate_omission else "tool transcript"
                                _note_line = f"Transcript note: {_partial_info}\n" if _partial_info else ""
                                _raw_content = str(tr.get("content", ""))
                                _preview = ""
                                if _preview_chars > 0 and _raw_content:
                                    _preview = _raw_content[:_preview_chars]
                                    _last_nl = _preview.rfind("\n")
                                    if _last_nl > _preview_chars // 2:
                                        _preview = _preview[:_last_nl]
                                    _preview = _preview.rstrip()
                                _preview_line = (
                                    f"\nPreview (first {len(_preview):,} chars):\n{_preview}\n"
                                    if _preview else ""
                                )
                                _src_tool, _src_args, _src_tc_id = _get_tool_source(tr, call_messages)
                                _meta = _tool_result_metadata(_src_args, _raw_content)
                                _meta_line = f"{_meta}\n" if _meta else ""
                                notify_content = (
                                    f"[{_result_label.upper()} OUTSIDE CONTEXT]\n"
                                    f"Block: {block_id or 'duplicate'}\n"
                                    f"Transcript file: {_payload_path or '(see dashboard)'}\n"
                                    f"{_note_line}"
                                    f"Tool: {tool_name}\n"
                                    f"{_meta_line}"
                                    f"Approx size: ~{tok_est:,} tokens\n"
                                    f"{_preview_line}"
                                    "This file stores only the transcript returned by this tool call, "
                                    "not a guarantee of complete source data. Use the source tool again "
                                    "if the transcript indicates omitted, shown, paged, or truncated data."
                                )
                                notify_msg = {
                                    "role": tr.get("role", "tool"),
                                    "tool_call_id": tr.get("tool_call_id", ""),
                                    "content": notify_content,
                                }
                                messages.append(notify_msg)
                                notify_idx = len(messages) - 1
                                if block_id is not None:
                                    _cw_manager.set_notify_msg_idx(block_id, notify_idx)
                                _cw_manager.register_message(notify_msg, notify_idx, blocked=False)
                                # notify msg itself costs tokens — track so next iteration
                                # has an accurate remaining estimate
                                _added_tok += _msg_tok_estimate([notify_msg])

                            _cw_manager.update_dashboard_cache()
                        except Exception as _e:
                            # Fallback: old behavior if bulk-output gate fails.
                            # Always print — this means the gate threw an exception and
                            # tool results bypassed the bulk-output gate entirely.
                            import traceback as _tb
                            _warning = (
                                f"[GATE-FALLBACK] ⚠️  Bulk-output gate threw exception for task {task_label} "
                                f"step {step_count} — falling back to direct extend (gate bypassed!)\n"
                                f"  Exception: {_e}\n"
                                f"  {_tb.format_exc().splitlines()[-1]}"
                            )
                            print(_warning)
                            _append_warning_log(warning_log_path, _warning)
                            messages.extend(tool_results)
                            try:
                                n = len(messages)
                                for _offset, _tr in enumerate(tool_results):
                                    _cw_manager.register_message(_tr, n - len(tool_results) + _offset)
                                _cw_manager.update_dashboard_cache()
                            except Exception:
                                pass
                    else:
                        # No workspace manager: put tool results directly into context
                        messages.extend(tool_results)

                    if _cw_manager is not None:
                        _perf_cw_s += time.perf_counter() - _t_cw_gate_start

                    # Also add to full history (always store real results for trajectory)
                    full_messages_history.extend(tool_results)

                    # Add token usage information if context_awareness is enabled
                    if context_awareness and max_context_size is not None:
                        # Calculate current token usage
                        try:
                            _, _, current_tokens = _estimate_prompt_tokens(
                                messages,
                                tools,
                                model,
                                sms_mode=(_cw_manager is not None),
                            )

                            # Determine the context size to display based on whether context_reset or context_summary is enabled
                            # When context_reset or context_summary is enabled with reset_size, use reset_size instead of max_context_size
                            #display_context_size = reset_size if (reset_size is not None and (context_reset or context_summary)) else max_context_size
                            display_context_size = max_context_size
                            remaining_tokens = display_context_size - current_tokens

                            # Add token usage warning message
                            token_usage_message = {
                                "role": "user",
                                "content": f"Current token usage:\n<system_warning>Token usage: {current_tokens}/{display_context_size}; {remaining_tokens} remaining</system_warning>"
                            }
                            messages.append(token_usage_message)
                            full_messages_history.append(token_usage_message)

                            if verbose:
                                print(f"[Task {task_id} | {task_label}] Context awareness: Token usage {current_tokens}/{display_context_size} ({remaining_tokens} remaining)")
                        except Exception as e:
                            print(f"[Task {task_id} | {task_label}] Warning: Failed to calculate tokens for context awareness: {e}", file=sys.stderr)

                except (json.JSONDecodeError, TypeError) as e:
                    print(f"[Task {task_id} | {task_label}] Warning: Failed to parse next_obs as JSON, skipping. Error: {e}", file=sys.stderr)
                    if verbose:
                        print(f"[Task {task_id} | {task_label}] next_obs content: {next_obs[:200]}...")
            obs = next_obs
            
            # Check if context RESET is needed (must be done AFTER tool results)
            if context_reset and reset_size is not None and 'raw_response' in response:
                # Calculate total_tokens using tiktoken (same method as in call_openai_with_tools)
                try:
                    _, tools_tokens, total_tokens = _estimate_prompt_tokens(
                        messages,
                        tools,
                        model,
                        sms_mode=(_cw_manager is not None),
                    )
                except Exception as e:
                    print(f"[Task {task_id} | {task_label}] Warning: Failed to calculate tokens using tiktoken: {e}", file=sys.stderr)
                    # Fallback to usage from API response
                    usage = response.get('raw_response', {}).get('usage', {})
                    total_tokens = usage.get('total_tokens', 0)

                # Check if memory warning should be issued (when memory_tool is enabled)
                memory_warning_threshold_tokens = reset_size * memory_warning_threshold
                if has_memory_tool and not memory_warning_issued and total_tokens >= memory_warning_threshold_tokens and total_tokens < reset_size:
                    if verbose:
                        print(f"[Task {task_id} | {task_label}] Memory warning threshold reached ({total_tokens} >= {memory_warning_threshold_tokens:.0f}). Inserting memory warning message...")

                    # Calculate remaining tokens
                    remaining_tokens = reset_size - total_tokens if reset_size else max_context_size - total_tokens

                    # Insert memory warning message
                    memory_warning_message = {
                        "role": "user",
                        "content": (
                            "<system_warning>\n\n"
                            "**You are nearing the context window limit.**\n\n"
                            "Your context will be automatically compacted soon.\n\n"
                            "Please save any important information from tool results into memory files before it is removed from the context. "
                            f"Token usage: {total_tokens}/{reset_size if reset_size else max_context_size}; {remaining_tokens} remaining"
                            "</system_warning>"
                        )
                    }
                    messages.append(memory_warning_message)
                    memory_warning_issued = True  # Mark warning as issued to prevent duplicates
                    if verbose:
                        print(f"[Task {task_id} | {task_label}] Memory warning message inserted into conversation")

                if total_tokens > reset_size:
                    # Use context reset approach
                    if verbose:
                        print(f"[Task {task_id} | {task_label}] Token usage ({total_tokens}) exceeds reset_size ({reset_size}). Performing context reset...")
                    
                    # Perform context reset
                    messages_before_reset = messages.copy()
                    tokens_before_reset = total_tokens  # Record tokens before reset
                    new_messages, reset_info = perform_context_reset(messages, reset_ratio)
                    
                    if reset_info is not None:
                        messages = new_messages
                        
                        # Calculate tokens after reset
                        try:
                            _, _, tokens_after_reset = _estimate_prompt_tokens(
                                messages,
                                tools,
                                model,
                                sms_mode=(_cw_manager is not None),
                            )
                        except Exception as e:
                            print(f"[Task {task_id} | {task_label}] Warning: Failed to calculate tokens after reset: {e}", file=sys.stderr)
                            tokens_after_reset = None

                        # Save the complete messages after reset for inspection
                        # Deep copy to preserve the exact state at this moment
                        import copy
                        messages_after_reset_sample = copy.deepcopy(messages)

                        # Record reset event
                        reset_event = {
                            'step': step_count,
                            'total_tokens': total_tokens,
                            'tokens_before_reset': tokens_before_reset,
                            'tokens_after_reset': tokens_after_reset,
                            'reset_size': reset_size,
                            'reset_info': reset_info,
                            'messages_before_count': len(messages_before_reset),
                            'messages_after_count': len(messages),
                            'messages_after_reset_sample': messages_after_reset_sample
                        }
                        reset_events.append(reset_event)

                        if verbose:
                            print(f"[Task {task_id} | {task_label}] Context reset completed:")
                            print(f"  - Removed {reset_info['num_pairs_removed']}/{reset_info['total_pairs']} tool call pairs")
                            if reset_info.get('kept_last_tool_call'):
                                print(f"  - Kept the most recent tool call pair")
                            print(f"  - Messages count: {len(messages_before_reset)} -> {len(messages)}")
                            print(f"  - Tokens: {tokens_before_reset} -> {tokens_after_reset}")

                        # Reset memory warning flag after context reset
                        memory_warning_issued = False

            # Check if thinking RESET is needed (must be done AFTER tool results and potentially after context_reset)
            if thinking_reset and reset_size is not None and 'raw_response' in response:
                # Calculate total_tokens using tiktoken (same method as in call_openai_with_tools)
                try:
                    _, tools_tokens, total_tokens = _estimate_prompt_tokens(
                        messages,
                        tools,
                        model,
                        sms_mode=(_cw_manager is not None),
                    )
                except Exception as e:
                    print(f"[Task {task_id} | {task_label}] Warning: Failed to calculate tokens using tiktoken: {e}", file=sys.stderr)
                    # Fallback to usage from API response
                    usage = response.get('raw_response', {}).get('usage', {})
                    total_tokens = usage.get('total_tokens', 0)

                # Check if memory warning should be issued (when memory_tool is enabled) for thinking_reset
                thinking_memory_warning_threshold_tokens = reset_size * memory_warning_threshold
                if has_memory_tool and not memory_warning_issued and total_tokens >= thinking_memory_warning_threshold_tokens and total_tokens < reset_size:
                    if verbose:
                        print(f"[Task {task_id} | {task_label}] Thinking memory warning threshold reached ({total_tokens} >= {thinking_memory_warning_threshold_tokens:.0f}). Inserting memory warning message...")

                    # Calculate remaining tokens
                    remaining_tokens = reset_size - total_tokens

                    # Insert memory warning message
                    thinking_memory_warning_message = {
                        "role": "user",
                        "content": (
                            "<system_warning>\n\n"
                            "**You are nearing the context window limit.**\n\n"
                            "Your context will be automatically compacted soon.\n\n"
                            "Please save any important reasoning information from your thinking process into memory files before it is removed from the context. "
                            f"Token usage: {total_tokens}/{reset_size}; {remaining_tokens} remaining"
                            "</system_warning>"
                        )
                    }
                    messages.append(thinking_memory_warning_message)
                    memory_warning_issued = True  # Mark warning as issued to prevent duplicates
                    if verbose:
                        print(f"[Task {task_id} | {task_label}] Thinking memory warning message inserted into conversation")

                if total_tokens > reset_size:
                    # Use thinking reset approach
                    if verbose:
                        print(f"[Task {task_id} | {task_label}] Token usage ({total_tokens}) exceeds reset_size ({reset_size}). Performing thinking reset...")

                    # Perform thinking reset
                    messages_before_thinking_reset = messages.copy()
                    tokens_before_thinking_reset = total_tokens  # Record tokens before reset
                    new_messages, thinking_reset_info = perform_thinking_reset(messages, keep_thinking)

                    if thinking_reset_info is not None:
                        messages = new_messages

                        # Calculate tokens after thinking reset
                        try:
                            _, _, tokens_after_thinking_reset = _estimate_prompt_tokens(
                                messages,
                                tools,
                                model,
                                sms_mode=(_cw_manager is not None),
                            )
                        except Exception as e:
                            print(f"[Task {task_id} | {task_label}] Warning: Failed to calculate tokens after thinking reset: {e}", file=sys.stderr)
                            tokens_after_thinking_reset = None

                        # Save the complete messages after thinking reset for inspection
                        # Deep copy to preserve the exact state at this moment
                        import copy
                        messages_after_thinking_reset_sample = copy.deepcopy(messages)

                        # Record thinking reset event
                        thinking_reset_event = {
                            'step': step_count,
                            'total_tokens': total_tokens,
                            'tokens_before_reset': tokens_before_thinking_reset,
                            'tokens_after_reset': tokens_after_thinking_reset,
                            'reset_size': reset_size,
                            'thinking_reset_info': thinking_reset_info,
                            'messages_before_count': len(messages_before_thinking_reset),
                            'messages_after_count': len(messages),
                            'messages_after_thinking_reset_sample': messages_after_thinking_reset_sample
                        }
                        thinking_reset_events.append(thinking_reset_event)

                        if verbose:
                            print(f"[Task {task_id} | {task_label}] Thinking reset completed:")
                            print(f"  - Cleared reasoning_content from {thinking_reset_info['num_cleared']}/{thinking_reset_info['total_assistants']} assistant messages")
                            print(f"  - Kept reasoning_content for last {keep_thinking} assistant message(s)")
                            print(f"  - Total reasoning_content length removed: {thinking_reset_info['total_reasoning_content_length']}")
                            print(f"  - Tokens: {tokens_before_thinking_reset} -> {tokens_after_thinking_reset}")

                        # Reset memory warning flag after thinking reset
                        memory_warning_issued = False

            # Check if context SUMMARY is needed (must be done AFTER tool results)
            if context_summary and reset_size is not None and 'raw_response' in response:
                # Calculate total_tokens using tiktoken (same method as in call_openai_with_tools)
                try:
                    _, tools_tokens, total_tokens = _estimate_prompt_tokens(
                        messages,
                        tools,
                        model,
                        sms_mode=(_cw_manager is not None),
                    )
                except Exception as e:
                    print(f"[Task {task_id} | {task_label}] Warning: Failed to calculate tokens using tiktoken: {e}", file=sys.stderr)
                    # Fallback to usage from API response
                    usage = response.get('raw_response', {}).get('usage', {})
                    total_tokens = usage.get('total_tokens', 0)

                # Check if memory warning should be issued (when memory_tool is enabled)
                memory_warning_threshold_tokens = reset_size * memory_warning_threshold
                if has_memory_tool and not memory_warning_issued and total_tokens >= memory_warning_threshold_tokens and total_tokens < reset_size:
                    if verbose:
                        print(f"[Task {task_id} | {task_label}] Memory warning threshold reached ({total_tokens} >= {memory_warning_threshold_tokens:.0f}). Inserting memory warning message...")

                    # Calculate remaining tokens
                    remaining_tokens = reset_size - total_tokens if reset_size else max_context_size - total_tokens

                    # Insert memory warning message
                    memory_warning_message = {
                        "role": "user",
                        "content": (
                            "<system_warning>\n\n"
                            "**You are nearing the context window limit.**\n\n"
                            "Your context will be automatically compacted soon.\n\n"
                            "Please save any important information from tool results into memory files before it is removed from the context. "
                            f"Token usage: {total_tokens}/{reset_size if reset_size else max_context_size}; {remaining_tokens} remaining"
                            "</system_warning>"
                        )
                    }
                    messages.append(memory_warning_message)
                    memory_warning_issued = True  # Mark warning as issued to prevent duplicates
                    if verbose:
                        print(f"[Task {task_id} | {task_label}] Memory warning message inserted into conversation")

                if total_tokens > reset_size:
                    # Use context summary approach
                    if verbose:
                        print(f"[Task {task_id} | {task_label}] Token usage ({total_tokens}) exceeds reset_size ({reset_size}). Generating context summary...")

                    # Add summary request message
                    summary_request_message = {
                        "role": "user",
                        "content": "You are approaching the context window's length limit. To continue the task, you must produce an operational summary of the overflowing conversation trajectory. This summary will be transferred into a fresh context window and will serve—together with the user's original task description—as your only available reference. The full conversation history will no longer be accessible, so ensure the summary captures all essential information needed to proceed effectively."
                    }
                    messages.append(summary_request_message)

                    # Call API to get summary
                    if verbose:
                        print(f"[Task {task_id} | {task_label}] Requesting summary from model...")
                    summary_response = make_aihubmix_api_request(
                        messages=messages,
                        model_name=model,
                        aihubmix_api_keys=api_key,
                        aihubmix_api_url=f"{base_url}/chat/completions",
                        tools=None,  # Don't allow tool calls for summary
                        max_retries=max_retries,
                        temperature=0.7,
                        top_p=1.0,
                        max_tokens=max_tokens,
                        max_context_size=max_context_size,
                        context_awareness=context_awareness,
                        reasoning_effort=reasoning_effort,
                        reasoning_max_tokens=reasoning_max_tokens,
                        reasoning_enabled=reasoning_enabled,
                        reasoning_exclude=reasoning_exclude,
                    )
                    
                    # Update messages if they were trimmed (before generating summary)
                    if 'trimmed_messages' in summary_response and summary_response['trimmed_messages'] is not None:
                        messages_before_trim = len(messages)
                        messages = summary_response['trimmed_messages']
                        if verbose:
                            print(f"[Task {task_id} | {task_label}] Messages trimmed before summary: {messages_before_trim} -> {len(messages)}")

                        # Record trim event for summary call
                        if 'trim_info' in summary_response and summary_response['trim_info'] is not None:
                            import copy
                            trim_event = {
                                'step': step_count,
                                'trim_info': copy.deepcopy(summary_response['trim_info']),
                                'context': 'summary_api_call'  # Distinguish from main trim
                            }
                            trim_events.append(trim_event)
                            if verbose:
                                print(f"[Task {task_id} | {task_label}] Trim event recorded (summary): removed {summary_response['trim_info']['removed_count']} messages")
                    
                    # Get summary message
                    if 'call_messages' in summary_response:
                        summary_message = summary_response['call_messages']
                        
                        # Extract summary content and create a new user message
                        if 'content' in summary_message:
                            summary_content = "You previously worked on this task in an earlier context window. This is a new context window, and the text provided here is a summary of the portion you completed before.\n\n" + summary_message['content']
                            
                            # Create a new user message with the summary
                            summary_user_message = {
                                "role": "user",
                                "content": summary_content
                            }
                            
                            # Reset messages to initial + summary as user message
                            messages_before_summary = messages.copy()
                            messages = [initial_user_message, summary_user_message]
                        
                            # Record summary event
                            import copy
                            summary_event = {
                                'step': step_count,
                                'total_tokens': total_tokens,
                                'reset_size': reset_size,
                                'messages_before_count': len(messages_before_summary),
                                'messages_after_count': len(messages),
                                'summary_request': summary_request_message,
                                'summary_response_original': copy.deepcopy(summary_message),  # Original assistant response
                                'summary_user_message': copy.deepcopy(summary_user_message),  # Converted to user message
                                'messages_before_summary': copy.deepcopy(messages_before_summary),
                            }
                            summary_events.append(summary_event)

                            if verbose:
                                print(f"[Task {task_id} | {task_label}] Context summary completed:")
                                print(f"  - Messages count: {len(messages_before_summary)} -> {len(messages)}")
                                print(f"  - Summary converted to user message and context reset to initial + summary")

                            # Reset memory warning flag after context summary
                            memory_warning_issued = False
                        else:
                            print(f"[Task {task_id} | {task_label}] ERROR: Summary response has no content", file=sys.stderr)
                    else:
                        print(f"[Task {task_id} | {task_label}] ERROR: Failed to get summary response", file=sys.stderr)
            
            # Record episode data (without messages to save space)
            episode.append({
                "observation": obs,
                "action": response,
                "reward": reward,
                "info": info,
            })

            # Save current progress after each step (simplified format)
            episode_data = {
                "messages": _strip_for_trajectory(messages),
                "events": {
                    "reset": reset_events or [],
                    "summary": summary_events or [],
                    "trim": trim_events or [],
                    "thinking_reset": thinking_reset_events or [],
                },
                "metrics": {
                    "accuracy": reward,
                    "total_steps": step_count,
                    "completed": done,
                },
            }

            envelope = make_base_envelope(
                backend="openai",
                task={
                    "task_id": task_id,
                    "config_id": config_id,
                    "run_id": run_id,
                    "config_name": config_name,
                    "env_class": env_class,
                    "env_params": env_params,
                },
            )
            attach_conversation(
                envelope,
                messages=messages,
                full_messages_history=_strip_for_trajectory(full_messages_history),
            )
            attach_events(
                envelope,
                reset=reset_events or [],
                summary=summary_events or [],
                trim=trim_events or [],
                thinking_reset=thinking_reset_events or [],
            )
            attach_metrics(
                envelope,
                accuracy=reward,
                total_steps=step_count,
                completed=done,
            )
            attach_provider_payload(
                envelope,
                model=model,
                usage_tracking=usage_tracking,
            )
            write_trajectory_file(
                save_file,
                envelope=envelope,
                legacy_payload=episode_data,
                indent=2,
            )

            # Save stats.json with API usage tracking (progress)
            if usage_tracking:
                stats_data = {"usage_tracking": usage_tracking}
                stats_file = save_file.parent / "token_stats.json"
                write_json_file(stats_file, stats_data, indent=2)

            if verbose:
                print(f"[Task {task_id} | {task_label}] Progress saved to: {save_file}")

            _perf_total_s += time.perf_counter() - _t_step_start

        # ── Timing summary (always printed so it's visible in logs) ────────────
        if _perf_total_s > 0:
            _t_api_internal_s = max(0.0, _perf_api_s - _perf_http_s - _perf_tok_s)
            _t_other_s = max(0.0, _perf_total_s - _perf_api_s - _perf_tool_s - _perf_cw_s)
            def _pct(t): return f"{100*t/_perf_total_s:.1f}%"
            print(
                f"[PERF {task_label}] Total={_perf_total_s:.1f}s over {step_count} steps | "
                f"Gemini HTTP={_perf_http_s:.1f}s ({_pct(_perf_http_s)}) | "
                f"API internal={_t_api_internal_s:.1f}s ({_pct(_t_api_internal_s)}) | "
                f"Token est={_perf_tok_s:.1f}s ({_pct(_perf_tok_s)}) | "
                f"Tool exec={_perf_tool_s:.1f}s ({_pct(_perf_tool_s)}) | "
                f"CW overhead={_perf_cw_s:.1f}s ({_pct(_perf_cw_s)}) | "
                f"Other={_t_other_s:.1f}s ({_pct(_t_other_s)})"
            )

        # Update final episode data (simplified format)
        episode_data = {
            "messages": _strip_for_trajectory(messages),
            "events": {
                "reset": reset_events or [],
                "summary": summary_events or [],
                "trim": trim_events or [],
                "thinking_reset": thinking_reset_events or [],
            },
            "metrics": {
                "accuracy": reward,
                "total_steps": step_count,
                "completed": True,
            },
        }

        envelope = make_base_envelope(
            backend="openai",
            task={
                "task_id": task_id,
                "config_id": config_id,
                "run_id": run_id,
                "config_name": config_name,
                "env_class": env_class,
                "env_params": env_params,
            },
        )
        attach_conversation(
            envelope,
            messages=messages,
            full_messages_history=full_messages_history,
        )
        attach_events(
            envelope,
            reset=reset_events or [],
            summary=summary_events or [],
            trim=trim_events or [],
            thinking_reset=thinking_reset_events or [],
        )
        attach_metrics(
            envelope,
            accuracy=reward,
            total_steps=step_count,
            completed=True,
        )
        attach_provider_payload(
            envelope,
            model=model,
            usage_tracking=usage_tracking,
        )
        write_trajectory_file(
            save_file,
            envelope=envelope,
            legacy_payload=episode_data,
            indent=2,
        )

        # Save token_stats.json with API usage tracking
        if usage_tracking:
            stats_data = {"usage_tracking": usage_tracking}
            stats_file = save_file.parent / "token_stats.json"
            write_json_file(stats_file, stats_data, indent=2)

        # Save eval.json alongside trajectory.json
        feedback = info.get("env_observation", "") if info else ""
        write_eval_file(
            task_workspace=save_file.parent,
            status="success",
            accuracy=reward,
            steps=step_count,
            feedback=feedback,
        )

        if verbose:
            print(f"[Task {task_id} | {task_label}] Completed successfully!")
            print(f"[Task {task_id} | {task_label}] Episode saved to: {save_file}")
            print(f"[Task {task_id} | {task_label}] Total steps: {step_count}")
            print(f"[Task {task_id} | {task_label}] Final reward (accuracy): {reward}")
            if reset_events:
                print(f"[Task {task_id} | {task_label}] Total context resets: {len(reset_events)}")
            if summary_events:
                print(f"[Task {task_id} | {task_label}] Total context summaries: {len(summary_events)}")
            if trim_events:
                total_trimmed = sum(event['trim_info']['removed_count'] for event in trim_events)
                print(f"[Task {task_id} | {task_label}] Total trim events: {len(trim_events)} (removed {total_trimmed} messages total)")
            if thinking_reset_events:
                total_cleared = sum(event['thinking_reset_info']['num_cleared'] for event in thinking_reset_events)
                total_length = sum(event['thinking_reset_info']['total_reasoning_content_length'] for event in thinking_reset_events)
                print(f"[Task {task_id} | {task_label}] Total thinking resets: {len(thinking_reset_events)} (cleared {total_cleared} assistant messages, {total_length} characters total)")

        # Compute token aggregates from usage_tracking (MAX/SUM logic same as ana_all_configs.py)
        api_prompt_tokens = 0
        api_completion_tokens = 0
        api_total_tokens = 0
        for ut in usage_tracking:
            step_total = ut.get('total_tokens', 0)
            if step_total > api_total_tokens:
                api_total_tokens = step_total
                api_prompt_tokens = ut.get('prompt_tokens', 0)
            api_completion_tokens += ut.get('completion_tokens', 0)

        # Tokens removed by trim events
        trimmed_tokens = sum(
            e.get('trim_info', {}).get('original_total_tokens', 0) - e.get('trim_info', {}).get('trimmed_total_tokens', 0)
            for e in (trim_events or [])
        )
        # Tokens removed by context reset events
        reset_tokens = sum(
            e.get('tokens_before_reset', 0) - e.get('tokens_after_reset', 0)
            for e in (reset_events or [])
            if e.get('tokens_before_reset', 0) and e.get('tokens_after_reset', 0)
        )
        # Tokens removed by thinking reset events
        thinking_reset_tokens = sum(
            e.get('tokens_before_reset', 0) - e.get('tokens_after_reset', 0)
            for e in (thinking_reset_events or [])
            if e.get('tokens_before_reset', 0) and e.get('tokens_after_reset', 0)
        )
        # Tokens removed by summary events (estimate based on message ratio)
        summary_tokens = sum(
            e.get('total_tokens', 0) - int(e.get('total_tokens', 0) * (e.get('messages_after_count', 1) / max(e.get('messages_before_count', 1), 1)))
            for e in (summary_events or [])
            if e.get('total_tokens', 0)
        )

        return {
            "task_id": task_id,
            "config_id": config_id,
            "run_id": run_id,
            "config_name": config_name,
            "status": "success",
            "steps": step_count,
            "final_reward": reward,
            "accuracy": reward,  # Final reward as accuracy
            "save_file": str(save_file),
            "env_class": env_class,
            "env_params": env_params,
            "tool_calls": info.get("tool_use_counter", 0) if info else 0,
            "api_prompt_tokens": api_prompt_tokens,
            "api_completion_tokens": api_completion_tokens,
            "api_total_tokens": api_total_tokens,
            "trimmed_tokens": trimmed_tokens,
            "reset_tokens": reset_tokens,
            "thinking_reset_tokens": thinking_reset_tokens,
            "summary_tokens": summary_tokens,
        }
        
    except Exception as e:
        if isinstance(e, TimeoutError):
            timeout_reason = str(e).splitlines()[0]
            timeout_feedback = f"{timeout_reason} | timeout finalization: pending"
            timeout_reward = 0.0
            timeout_info = {}
            try:
                _warning = (
                    f"[Task {task_id} | {task_label}] Timeout reached; "
                    "evaluating current workspace state."
                )
                print(_warning, file=sys.stderr)
                _append_warning_log(task_workspace / "logs" / "warnings.log", _warning)
                _timeout_obs, timeout_reward, _terminated, _truncated, timeout_info = env.step(
                    "<timeout_finalization />"
                )
                verdict = "passed" if timeout_reward > 0 else "failed"
                timeout_feedback = (
                    f"{timeout_reason} | timeout finalization: {verdict}, "
                    f"accuracy={timeout_reward}"
                )
            except Exception as eval_error:
                timeout_feedback = (
                    f"{timeout_reason} | timeout finalization failed: {eval_error}"
                )
                timeout_info = {"error": str(eval_error), "evaluation": "error"}

            timeout_passed = timeout_reward > 0
            timeout_status = "timeout"
            timeout_steps = step_count if step_count else len(episode)

            task_workspace.mkdir(parents=True, exist_ok=True)
            write_eval_file(
                task_workspace=task_workspace,
                status=timeout_status,
                accuracy=timeout_reward,
                steps=timeout_steps,
                feedback=timeout_feedback,
            )

            timeout_save_file = task_workspace / "trajectory.json"
            envelope = make_base_envelope(
                backend="openai",
                task={
                    "task_id": task_id,
                    "config_id": config_id,
                    "run_id": run_id,
                    "config_name": config_name,
                    "env_class": env_class,
                    "env_params": env_params,
                },
            )
            attach_conversation(
                envelope,
                messages=_strip_for_trajectory(messages) if "messages" in locals() else [],
                full_messages_history=_strip_for_trajectory(full_messages_history),
            )
            attach_events(
                envelope,
                reset=reset_events or [],
                summary=summary_events or [],
                trim=trim_events or [],
                thinking_reset=thinking_reset_events or [],
            )
            attach_metrics(
                envelope,
                accuracy=timeout_reward,
                total_steps=timeout_steps,
                completed=timeout_passed,
            )
            attach_provider_payload(
                envelope,
                model=model,
                usage_tracking=usage_tracking,
                error=None,
            )
            write_trajectory_file(
                timeout_save_file,
                envelope=envelope,
                legacy_payload={
                    "status": timeout_status,
                    "error": None,
                    "timeout_finalization": True,
                    "timeout_passed": timeout_passed,
                    "total_steps": timeout_steps,
                    "feedback": timeout_feedback,
                    "metrics": {
                        "accuracy": timeout_reward,
                        "total_steps": timeout_steps,
                        "completed": timeout_passed,
                    },
                },
                indent=2,
            )

            if usage_tracking:
                write_json_file(
                    timeout_save_file.parent / "token_stats.json",
                    {"usage_tracking": usage_tracking},
                    indent=2,
                )

            api_prompt_tokens = 0
            api_completion_tokens = 0
            api_total_tokens = 0
            for ut in usage_tracking:
                step_total = ut.get("total_tokens", 0)
                if step_total > api_total_tokens:
                    api_total_tokens = step_total
                    api_prompt_tokens = ut.get("prompt_tokens", 0)
                api_completion_tokens += ut.get("completion_tokens", 0)

            return {
                "task_id": task_id,
                "config_id": config_id,
                "run_id": run_id,
                "config_name": config_name,
                "status": timeout_status,
                "error": None,
                "steps": timeout_steps,
                "final_reward": timeout_reward,
                "accuracy": timeout_reward,
                "feedback": timeout_feedback,
                "save_file": str(timeout_save_file),
                "env_class": env_class,
                "env_params": env_params,
                "tool_calls": timeout_info.get("tool_use_counter", 0) if timeout_info else 0,
                "api_prompt_tokens": api_prompt_tokens,
                "api_completion_tokens": api_completion_tokens,
                "api_total_tokens": api_total_tokens,
                "timeout_finalization": True,
            }

        # Always print errors to stderr
        _error_warning = f"[Task {task_id} | {task_label}] Error: {e}"
        print(_error_warning, file=sys.stderr)
        _append_warning_log(task_workspace / "logs" / "warnings.log", _error_warning)
        import traceback
        traceback.print_exc()

        # Always write eval.json so results aggregation never misses a task
        task_workspace.mkdir(parents=True, exist_ok=True)
        write_eval_file(
            task_workspace=task_workspace,
            status="error",
            accuracy=0.0,
            steps=len(episode),
            feedback=str(e),
        )

        # Save partial episode on error
        if episode:
            error_save_file = task_workspace / "trajectory.json"
            error_save_file.parent.mkdir(parents=True, exist_ok=True)

            # Create error episode data
            episode_data = {
                "error": str(e),
                "total_steps": len(episode),
            }

            envelope = make_base_envelope(
                backend="openai",
                task={
                    "task_id": task_id,
                    "config_id": config_id,
                    "run_id": run_id,
                    "config_name": config_name,
                    "env_class": env_class,
                    "env_params": env_params,
                },
            )
            attach_conversation(
                envelope,
                full_messages_history=_strip_for_trajectory(full_messages_history),
            )
            attach_metrics(
                envelope,
                accuracy=0.0,
                total_steps=len(episode),
                completed=False,
            )
            attach_provider_payload(
                envelope,
                model=model,
                error=str(e),
            )
            write_trajectory_file(
                error_save_file,
                envelope=envelope,
                legacy_payload=episode_data,
                indent=4,
            )

            # Save eval.json for error case
            write_eval_file(
                task_workspace=error_save_file.parent,
                status="error",
                accuracy=0.0,
                steps=len(episode),
                feedback=str(e),
            )

            if verbose:
                print(f"[Task {task_id} | {task_label}] Partial episode saved to: {error_save_file}")

        return {
            "task_id": task_id,
            "config_id": config_id,
            "run_id": run_id,
            "config_name": config_name,
            "status": "error",
            "error": str(e),
            "steps": len(episode),
            "env_class": env_class,
            "env_params": env_params,
        }

    finally:
        # Always clean up the tool to prevent ghost MCP server processes
        if tool is not None:
            try:
                tool.close()
                if verbose:
                    print(f"[Task {task_id} | {task_label}] Tool closed successfully")
            except Exception as cleanup_error:
                print(f"[Task {task_id} | {task_label}] Warning: Error closing tool: {cleanup_error}", file=sys.stderr)


def normalize_config_for_grouping(config: Dict[str, Any]) -> tuple:
    """Create a normalized representation of a config for grouping purposes.
    
    Configs are considered the same if they differ only in the 'seed' parameter.
    
    Args:
        config: Configuration dictionary
    
    Returns:
        Tuple representing the normalized config (hashable)
    """
    # Create a copy of env_params without seed
    env_params = config.get("env_params", {}).copy()
    seed = env_params.pop("seed", None)
    
    # Create a normalized representation
    normalized = {
        "env_class": config.get("env_class"),
        "env_params": tuple(sorted(env_params.items())),
        "mcp_servers": json.dumps(config.get("mcp_servers", {}), sort_keys=True)
    }
    
    return (normalized["env_class"], normalized["env_params"], normalized["mcp_servers"])


def group_configs_by_similarity(configs: List[Dict[str, Any]]) -> Dict[int, List[int]]:
    """Group configurations that differ only by seed.
    
    Args:
        configs: List of configuration dictionaries
    
    Returns:
        Dictionary mapping group_id to list of original config indices
    """
    groups = {}
    config_to_group = {}
    
    for idx, config in enumerate(configs):
        normalized = normalize_config_for_grouping(config)
        
        if normalized not in config_to_group:
            group_id = len(config_to_group)
            config_to_group[normalized] = group_id
            groups[group_id] = []
        
        group_id = config_to_group[normalized]
        groups[group_id].append(idx)
    
    return groups


def check_episode_needs_resume(episode_file: Path) -> bool:
    """Check if an episode file indicates a failed run that needs to be resumed.
    
    An episode needs resume if:
    1. The file exists and can be parsed
    2. The last message in final_messages is an error message
    
    Args:
        episode_file: Path to the episode JSON file
        
    Returns:
        True if the episode needs to be resumed, False otherwise
    """
    try:
        with open(episode_file, 'r') as f:
            episode_data = json.load(f)
        
        final_messages = episode_data.get('final_messages', [])
        if not final_messages:
            return True  # No messages means incomplete
        
        last_message = final_messages[-1]
        content = last_message.get('content', '')
        
        # Check for error patterns
        error_patterns = [
            "Error: Failed to get response after multiple retries.",
            "Error: Invalid parameter format.",
            "Error: Request failed with 400 status.",
            "ERROR: Context trimming removed all",
            "ERROR: Cannot fit messages within available context",
        ]
        
        for pattern in error_patterns:
            if pattern in content:
                return True
        
        # Also check if completed flag is False or missing
        if not episode_data.get('completed', False):
            return True
            
        return False
        
    except (json.JSONDecodeError, FileNotFoundError, KeyError) as e:
        print(f"Warning: Could not parse episode file {episode_file}: {e}")
        return True  # If we can't read it, assume it needs resume


def scan_resume_directory(resume_dir: str, delete_failed: bool = True) -> Dict[int, List[int]]:
    """Scan a resume directory to find which configs/runs need to be re-run.

    Supports both old-style (config_N/run_N) and new-style (TaskName/stateN) layouts.
    For new-style, reads task_mapping.json to map task names back to group IDs.

    Args:
        resume_dir: Path to the existing output directory
        delete_failed: If True, delete the failed episode files that will be resumed

    Returns:
        Dictionary mapping config_id to list of run_ids that need to be resumed
    """
    resume_path = Path(resume_dir)
    if not resume_path.exists():
        print(f"Resume directory does not exist: {resume_dir}")
        return {}

    # Task files are stored under tasks/ subdirectory
    tasks_path = resume_path / "tasks"
    if not tasks_path.exists():
        # Fall back to scanning resume_path directly for backward compatibility
        tasks_path = resume_path

    configs_to_resume = {}
    files_to_delete = []  # Track files to delete

    # Try to load task_mapping.json for new-style directories
    task_mapping = {}
    task_mapping_file = tasks_path / "task_mapping.json"
    if task_mapping_file.exists():
        try:
            with open(task_mapping_file, 'r') as f:
                task_mapping = json.load(f)
        except (json.JSONDecodeError, IOError):
            pass

    # Scan for task/config directories
    for config_dir in tasks_path.iterdir():
        if not config_dir.is_dir():
            continue

        # Determine config_id based on directory naming style
        if config_dir.name.startswith('config_'):
            # Old-style: config_N
            try:
                config_id = int(config_dir.name.split('_')[1])
            except (IndexError, ValueError):
                continue
        elif config_dir.name in task_mapping:
            # New-style: TaskName mapped via task_mapping.json
            config_id = task_mapping[config_dir.name]
        else:
            continue

        # Check for new-style state directories (stateN/trajectory.json)
        state_dirs = [d for d in config_dir.iterdir() if d.is_dir() and d.name.startswith('state')]
        if state_dirs:
            for state_dir in state_dirs:
                try:
                    run_id = int(state_dir.name.replace('state', ''))
                except ValueError:
                    continue
                traj_file = state_dir / "trajectory.json"
                eval_file = state_dir / "eval.json"
                if traj_file.exists():
                    # Resume should fill interrupted/system-failed runs only.
                    # A completed task with accuracy=0 is still a valid result and
                    # must not be re-run, otherwise resume changes the experiment.
                    needs_resume = False
                    status = None
                    try:
                        if eval_file.exists():
                            with open(eval_file, 'r') as f:
                                eval_data = json.load(f)
                            status = eval_data.get("status")
                        else:
                            with open(traj_file, 'r') as f:
                                traj_data = json.load(f)
                            status = traj_data.get("status")
                            if status is None:
                                metrics = traj_data.get("metrics", {})
                                if metrics.get("completed") is True:
                                    status = "success"
                                elif metrics.get("completed") is False:
                                    status = "error"
                    except (json.JSONDecodeError, IOError):
                        needs_resume = True

                    if status in {"error", "exception", "timeout"}:
                        needs_resume = True
                    elif status == "success":
                        needs_resume = False
                    elif status is None:
                        needs_resume = True

                    if needs_resume:
                        if config_id not in configs_to_resume:
                            configs_to_resume[config_id] = []
                        configs_to_resume[config_id].append(run_id)
                        files_to_delete.append(traj_file)
                        if eval_file.exists():
                            files_to_delete.append(eval_file)
                        print(f"  {config_dir.name} state{run_id}: needs resume")
                    else:
                        print(f"  {config_dir.name} state{run_id}: completed successfully")
                else:
                    # No trajectory file means this run was never started
                    if config_id not in configs_to_resume:
                        configs_to_resume[config_id] = []
                    configs_to_resume[config_id].append(run_id)
                    print(f"  {config_dir.name} state{run_id}: no trajectory found, will re-run")
            continue

        # Old-style: check for episode files in config_N directories
        episode_files = list(config_dir.glob('config*_run*-episode-*.json'))

        if not episode_files:
            # No episode files means this config was never started
            print(f"  Config {config_id}: no episode files found, will run all runs")
            if config_id not in configs_to_resume:
                configs_to_resume[config_id] = [-1]  # -1 indicates all runs need to be done
            continue

        # Group episode files by run_id
        runs_found = {}
        for episode_file in episode_files:
            filename = episode_file.name
            try:
                run_part = filename.split('_run')[1].split('-')[0]
                run_id = int(run_part)
                if run_id not in runs_found or episode_file.stat().st_mtime > runs_found[run_id].stat().st_mtime:
                    runs_found[run_id] = episode_file
            except (IndexError, ValueError):
                continue

        # Check each run's latest episode file
        for run_id, episode_file in runs_found.items():
            if check_episode_needs_resume(episode_file):
                if config_id not in configs_to_resume:
                    configs_to_resume[config_id] = []
                configs_to_resume[config_id].append(run_id)
                files_to_delete.append(episode_file)
                print(f"  Config {config_id} Run {run_id}: needs resume (file: {episode_file.name})")
            else:
                print(f"  Config {config_id} Run {run_id}: completed successfully")

    # Delete failed episode files if requested
    if delete_failed and files_to_delete:
        print(f"\nDeleting {len(files_to_delete)} failed episode files...")
        for file_path in files_to_delete:
            try:
                file_path.unlink()
                print(f"  Deleted: {file_path.name}")
            except Exception as e:
                print(f"  Warning: Failed to delete {file_path.name}: {e}")

    return configs_to_resume


def run_config_combinations(
    config_file: str,
    runs_per_config: int = 1,
    base_task_dir: str = "",
    output_dir: str = "",
    api_key: str = "",
    base_url: str="",
    model: str = "gpt-5-nano",
    max_tool_uses: int = 500,
    max_tokens: int = 32768,
    timeout: int = 600,
    max_workers: Optional[int] = None,
    max_retries: int = 50,
    initial_retry_delay: float = 2.0,
    reset_size: Optional[int] = None,
    reset_ratio: float = 0.5,
    context_reset: bool = False,
    context_summary: bool = False,
    context_awareness: bool = False,
    group_by_seed: bool = True,
    max_context_size: Optional[int] = None,
    memory_warning_threshold: float = 0.8,
    thinking_reset: bool = False,
    keep_thinking: int = 1,
    reasoning_effort: Optional[str] = None,
    reasoning_max_tokens: Optional[int] = None,
    reasoning_enabled: bool = True,
    reasoning_exclude: bool = False,
    resume_dir: Optional[str] = None,
):
    """Run multiple configurations in parallel with flexible environment and tool setup.

    Args:
        config_file: Path to JSON configuration file
        runs_per_config: Number of runs per configuration
        base_task_dir: Base directory for task workspaces
        output_dir: Directory to save episode results
        api_key: API key (if None, will use from env)
        base_url: API base URL
        model: Model name
        max_tool_uses: Maximum tool uses per episode
        max_tokens: Maximum tokens per generation
        timeout: API request timeout
        max_workers: Maximum parallel workers
        max_retries: Maximum API retry attempts
        initial_retry_delay: Initial delay between retries in seconds
        reset_size: Token threshold for context management (None to disable all)
        reset_ratio: Ratio of tool calls to remove during reset (0.0 to 1.0)
        context_reset: If True, remove old tool calls when exceeding token limit
        context_summary: If True, generate summary when exceeding token limit
        context_awareness: If True, inform the model about token budget and usage at each step
        group_by_seed: If True, group configs that differ only by seed as same config
        max_context_size: Maximum context size in tokens (if set, will trim messages to fit)
        memory_warning_threshold: Threshold ratio (0.0-1.0) for memory warning when memory_tool is enabled
        thinking_reset: If True, clear reasoning_content from assistant messages when exceeding token limit
        keep_thinking: Number of most recent assistant messages to keep reasoning_content for (default: 1)
        reasoning_effort: The reasoning effort level for OpenAI models that support reasoning.
                         Supported values: "none", "minimal", "low", "medium", "high", "xhigh".
                         If set, takes precedence over reasoning_max_tokens.
        reasoning_max_tokens: Specific token limit for reasoning (Anthropic-style). Used if reasoning_effort is not set.
        reasoning_enabled: Whether to enable reasoning (default: True). Automatically inferred from effort or max_tokens.
        reasoning_exclude: Set to True to exclude reasoning tokens from response (default: False).
        resume_dir: Path to existing output directory to resume from. If provided, only failed runs will be re-executed.

        Note: context_reset and context_summary are mutually exclusive.
              If both are False, no context management is performed.
    """
    verbose = False
    # Runner mode is always quiet.
    os.environ["LOCA_QUIET"] = "1"
    logging.getLogger().setLevel(logging.WARNING)

    # Check for resume mode
    configs_to_resume = None
    if resume_dir:
        if verbose:
            print(f"\n{'=' * 80}")
            print("RESUME MODE ENABLED")
            print(f"{'=' * 80}")
            print(f"Scanning resume directory: {resume_dir}")
        configs_to_resume = scan_resume_directory(resume_dir)

        if not configs_to_resume:
            if verbose:
                print("\nNo configs need to be resumed. All runs completed successfully!")
                print(f"{'=' * 80}\n")
            return

        if verbose:
            total_to_resume = 0
            for runs in configs_to_resume.values():
                if -1 in runs:
                    total_to_resume += 1  # Will be updated based on actual runs later
                else:
                    total_to_resume += len(runs)
            print(f"\nFound runs to resume across {len(configs_to_resume)} configs:")
            for config_id, run_ids in sorted(configs_to_resume.items()):
                if -1 in run_ids:
                    print(f"  Config {config_id}: all runs (no episode files found)")
                else:
                    print(f"  Config {config_id}: runs {run_ids}")
            print(f"{'=' * 80}\n")

        # Use the resume directory as output directory
        output_dir = resume_dir
        # Note: base_task_dir remains unchanged (new task workspace)
        # This means each resume run starts with a fresh task workspace
        if verbose:
            print(f"Resume mode: Results will be saved to: {output_dir}")
            print(f"Resume mode: Using new task workspace: {base_task_dir}")
    
    # Load configurations
    with open(config_file, "r") as f:
        config_data = json.load(f)

    configs = config_data.get("configurations", [])
    if verbose:
        print(f"Loaded {len(configs)} configurations from {config_file}")

    # Group configurations if group_by_seed is enabled
    if group_by_seed:
        config_groups = group_configs_by_similarity(configs)
        if verbose:
            print(f"\nGrouping enabled: Found {len(config_groups)} unique configuration groups")
            for group_id, config_indices in config_groups.items():
                if len(config_indices) > 1:
                    print(f"  Group {group_id}: {len(config_indices)} configs with different seeds (indices: {config_indices})")
    else:
        # No grouping - each config is its own group
        config_groups = {i: [i] for i in range(len(configs))}
        if verbose:
            print(f"Grouping disabled: Treating each config separately")
    
    # # Get API key from environment if not provided
    # if api_key is None:
    #     api_key = os.environ.get("OPENAI_API_KEY")
    #     if not api_key:
    #         raise ValueError("API key not provided and OPENAI_API_KEY not set in environment")
    
    # Create base directories
    Path(base_task_dir).mkdir(parents=True, exist_ok=True)
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    
    # Calculate total tasks based on grouping and resume mode
    if configs_to_resume is not None:
        # Resume mode: count only the tasks that need to be resumed
        # -1 in runs list means all runs need to be done
        total_tasks = 0
        for config_id, runs in configs_to_resume.items():
            if -1 in runs:
                # All runs need to be done for this config
                if group_by_seed and config_id in config_groups:
                    total_tasks += max(len(config_groups[config_id]), runs_per_config)
                else:
                    total_tasks += runs_per_config
            else:
                total_tasks += len(runs)
    elif group_by_seed:
        # For each group: max(configs in group, runs_per_config)
        total_tasks = sum(max(len(indices), runs_per_config) for indices in config_groups.values())
    else:
        total_tasks = len(configs) * runs_per_config
    
    # Set default max_workers
    if max_workers is None:
        max_workers = min(total_tasks, os.cpu_count() or 4) if total_tasks > 0 else 1
    
    if verbose:
        print("=" * 80)
        print("FLEXIBLE PARALLEL INFERENCE")
        print("=" * 80)
        print(f"Total configurations: {len(configs)}")
        print(f"Unique config groups: {len(config_groups)}")
        print(f"Runs per configuration: {runs_per_config}")
        print(f"Total tasks: {total_tasks}")
        print(f"Max workers: {max_workers}")
        print(f"Model: {model}")
        print(f"Base task directory: {base_task_dir}")
        print(f"Output directory: {output_dir}")
        if max_context_size is not None:
            print(f"Max context size: {max_context_size:,} tokens (messages will be trimmed if exceeded)")
        if context_awareness:
            print(f"Context awareness enabled:")
            print(f"  - Model will be informed about token budget: {max_context_size:,} tokens" if max_context_size else "  - Warning: max_context_size not set")
            print(f"  - Token usage will be reported after each tool call")
        if reset_size is not None:
            if context_summary:
                print(f"Context summary enabled:")
                print(f"  - Reset size: {reset_size} tokens")
                print(f"  - Will generate summary when exceeding token limit")
            elif context_reset:
                print(f"Context reset enabled:")
                print(f"  - Reset size: {reset_size} tokens")
                print(f"  - Reset ratio: {reset_ratio}")
            else:
                print(f"Context monitoring enabled (no management):")
                print(f"  - Reset size: {reset_size} tokens")
                print(f"  - Will only log when exceeding token limit")

        if group_by_seed:
            print("\nConfiguration Groups:")
            for group_id in sorted(config_groups.keys()):
                config_indices = config_groups[group_id]
                print(f"  Group {group_id}: {len(config_indices)} configs")
                for idx in config_indices:
                    config = configs[idx]
                    seed = config.get('env_params', {}).get('seed', 'N/A')
                    print(f"    - Config index {idx}: seed={seed}")
                # Show details for first config in group
                config = configs[config_indices[0]]
                print(f"    Environment: {config.get('env_class', 'N/A')}")
                print(f"    Params (excluding seed): {{{', '.join(f'{k}: {v}' for k, v in config.get('env_params', {}).items() if k != 'seed')}}}")
                print(f"    MCP Servers: {list(config.get('mcp_servers', {}).keys())}")
        else:
            print("\nConfigurations:")
            for i, config in enumerate(configs):
                print(f"  Config {i}:")
                print(f"    Environment: {config.get('env_class', 'N/A')}")
                print(f"    Params: {config.get('env_params', {})}")
                print(f"    MCP Servers: {list(config.get('mcp_servers', {}).keys())}")
        print("=" * 80)

        # Display reasoning configuration if any parameters are set
        reasoning_configured = reasoning_effort is not None or reasoning_max_tokens is not None
        if reasoning_configured:
            print("\nReasoning Configuration:")
            if reasoning_effort is not None:
                print(f"  - Reasoning effort: {reasoning_effort}")
            if reasoning_max_tokens is not None:
                print(f"  - Reasoning max tokens: {reasoning_max_tokens}")
            print(f"  - Reasoning enabled: {reasoning_enabled}")
            print(f"  - Reasoning exclude: {reasoning_exclude}")
        print("\n" + "=" * 80)

    # Prepare task arguments based on grouping
    task_args = []
    task_id = 0
    skipped_count = 0

    # Build group_id -> config_name mapping for results aggregation
    group_id_to_name = {}

    if group_by_seed:
        # Build group_id -> config_name mapping first
        for group_id, config_indices in sorted(config_groups.items()):
            template_config = configs[config_indices[0]]
            config_name = template_config.get("name", "")
            if not config_name:
                env_cls = template_config.get("env_class", "")
                config_name = env_cls.rsplit(".", 1)[-1] if "." in env_cls else f"config_{group_id}"
            group_id_to_name[group_id] = config_name

        # Determine the max number of seeds across all groups
        max_seeds = max(len(config_indices) for config_indices in config_groups.values())

        # Iterate seed-round-first: run all 15 tasks with seed_idx=0, then all 15 with seed_idx=1, etc.
        # This way the user can see preliminary results after each round of 15 tasks.
        for seed_idx in range(max_seeds):
            for group_id, config_indices in sorted(config_groups.items()):
                if seed_idx >= len(config_indices):
                    continue

                config = configs[config_indices[seed_idx]]
                config_name = group_id_to_name[group_id]

                # Check if we should skip this run (resume mode)
                if configs_to_resume is not None:
                    if group_id not in configs_to_resume:
                        skipped_count += 1
                        continue
                    elif -1 not in configs_to_resume[group_id] and seed_idx not in configs_to_resume[group_id]:
                        skipped_count += 1
                        continue

                # Check if config provides specific reasoning settings
                config_reasoning_effort = config.get('reasoning_effort', reasoning_effort)
                config_reasoning_max_tokens = config.get('reasoning_max_tokens', reasoning_max_tokens)
                config_reasoning_enabled = config.get('reasoning_enabled', reasoning_enabled)
                config_reasoning_exclude = config.get('reasoning_exclude', reasoning_exclude)

                # Convert empty strings to None
                if config_reasoning_effort == "":
                    config_reasoning_effort = None
                if config_reasoning_max_tokens == "":
                    config_reasoning_max_tokens = None

                task_args.append((
                    task_id,
                    group_id,
                    seed_idx,
                    base_task_dir,
                    output_dir,
                    config["env_class"],
                    config["env_params"],
                    config["mcp_servers"],
                    api_key,
                    base_url,
                    model,
                    max_tool_uses,
                    max_tokens,
                    timeout,
                    max_retries,
                    initial_retry_delay,
                    reset_size,
                    reset_ratio,
                    context_reset,
                    context_summary,
                    context_awareness,
                    max_context_size,
                    memory_warning_threshold,
                    thinking_reset,
                    keep_thinking,
                    config_reasoning_effort,
                    config_reasoning_max_tokens,
                    config_reasoning_enabled,
                    config_reasoning_exclude,
                    config_name,
                ))
                task_id += 1
    else:
        # No grouping - original behavior
        for config_id, config in enumerate(configs):
            # Derive config_name for non-grouped mode
            cfg_name = config.get("name", "")
            if not cfg_name:
                env_cls = config.get("env_class", "")
                cfg_name = env_cls.rsplit(".", 1)[-1] if "." in env_cls else f"config_{config_id}"
            group_id_to_name[config_id] = cfg_name

            for run_id in range(runs_per_config):
                # Check if we should skip this run (resume mode)
                if configs_to_resume is not None:
                    # -1 in the list means all runs need to be done
                    if config_id not in configs_to_resume:
                        # This config doesn't need to be resumed at all, skip
                        skipped_count += 1
                        continue
                    elif -1 not in configs_to_resume[config_id] and run_id not in configs_to_resume[config_id]:
                        # This specific run doesn't need to be resumed, skip it
                        skipped_count += 1
                        continue

                # Check if config provides specific reasoning settings
                config_reasoning_effort = config.get('reasoning_effort', reasoning_effort)
                config_reasoning_max_tokens = config.get('reasoning_max_tokens', reasoning_max_tokens)
                config_reasoning_enabled = config.get('reasoning_enabled', reasoning_enabled)
                config_reasoning_exclude = config.get('reasoning_exclude', reasoning_exclude)

                # Convert empty strings to None
                if config_reasoning_effort == "":
                    config_reasoning_effort = None
                if config_reasoning_max_tokens == "":
                    config_reasoning_max_tokens = None

                task_args.append((
                    task_id,
                    config_id,
                    run_id,
                    base_task_dir,
                    output_dir,
                    config["env_class"],
                    config["env_params"],
                    config["mcp_servers"],
                    api_key,
                    base_url,
                    model,
                    max_tool_uses,
                    max_tokens,
                    timeout,
                    max_retries,
                    initial_retry_delay,
                    reset_size,
                    reset_ratio,
                    context_reset,
                    context_summary,
                    context_awareness,
                    max_context_size,
                    memory_warning_threshold,
                    thinking_reset,
                    keep_thinking,
                    config_reasoning_effort,
                    config_reasoning_max_tokens,
                    config_reasoning_enabled,
                    config_reasoning_exclude,
                    cfg_name,
                ))
                task_id += 1

    # Save task_mapping.json for resume support
    task_mapping = {name: gid for gid, name in group_id_to_name.items()}
    task_mapping_file = Path(base_task_dir) / "task_mapping.json"
    task_mapping_file.parent.mkdir(parents=True, exist_ok=True)
    with open(task_mapping_file, "w") as f:
        json.dump(task_mapping, f, indent=2)

    # Print resume mode summary if applicable
    if configs_to_resume is not None and verbose:
        print(f"Resume mode: {skipped_count} runs skipped (already completed), {len(task_args)} runs to execute")

    # Check if there are any tasks to run
    if not task_args:
        if verbose:
            print("\nNo tasks to run. All runs completed successfully!")
        return

    # Shuffle removed: seed-round-first order already gives representative intermediate results

    # Run tasks in parallel
    start_time = time.time()
    results = []
    executor = None

    # Signal handler for graceful shutdown
    def signal_handler(signum, frame):
        import multiprocessing
        print("\n\nReceived interrupt signal. Shutting down...", file=sys.stderr)

        # First, shutdown the executor to prevent new tasks
        if executor is not None:
            executor.shutdown(wait=False, cancel_futures=True)

        # Kill all child processes (worker processes and their MCP server children)
        try:
            # Get all active child processes from multiprocessing
            children = multiprocessing.active_children()
            for child in children:
                try:
                    # Terminate the child process
                    child.terminate()
                except Exception:
                    pass

            # Also try to kill any remaining processes in the process group of each worker
            # This catches MCP server subprocesses that may have been spawned
            if executor is not None and hasattr(executor, '_processes'):
                for pid in list(executor._processes.keys()):
                    try:
                        # Kill the entire process group of each worker
                        os.killpg(os.getpgid(pid), signal.SIGTERM)
                    except (ProcessLookupError, PermissionError, OSError):
                        pass
                    try:
                        # Also send SIGKILL to ensure termination
                        os.kill(pid, signal.SIGKILL)
                    except (ProcessLookupError, PermissionError, OSError):
                        pass

            # Wait briefly for processes to terminate
            for child in children:
                try:
                    child.join(timeout=0.5)
                except Exception:
                    pass

        except Exception as e:
            print(f"Warning: Error during process cleanup: {e}", file=sys.stderr)

        sys.exit(130)  # 128 + SIGINT(2)

    # Set up signal handlers
    original_sigint = signal.signal(signal.SIGINT, signal_handler)
    original_sigterm = signal.signal(signal.SIGTERM, signal_handler)

    # Progress tracking for Rich display
    completed_count = 0
    success_count = 0
    timeout_count = 0
    error_count = 0
    total_accuracy = 0.0
    accuracy_count = 0

    try:
        with ProcessPoolExecutor(max_workers=max_workers, max_tasks_per_child=1) as executor:
            # Submit all tasks
            futures = {
                executor.submit(run_single_task, *args): (args[0], args[1], args[2])
                for args in task_args
            }

            from rich.progress import Progress, SpinnerColumn, TextColumn, BarColumn, TaskProgressColumn, TimeElapsedColumn
            from rich.console import Console

            console = Console()

            with Progress(
                SpinnerColumn(),
                TextColumn("[progress.description]{task.description}"),
                BarColumn(),
                TaskProgressColumn(),
                TimeElapsedColumn(),
                console=console,
                transient=False,
            ) as progress:
                task_progress = progress.add_task(
                    f"[cyan]Running {len(task_args)} tasks ({max_workers} workers)",
                    total=len(task_args)
                )

                for future in as_completed(futures):
                    task_id, config_id, run_id = futures[future]
                    try:
                        result = future.result()
                        results.append(result)
                        completed_count += 1
                        result_status = result["status"]
                        if result_status == 'success':
                            success_count += 1
                            acc = result.get('accuracy', result.get('final_reward', 0))
                            if acc is not None:
                                total_accuracy += acc
                                accuracy_count += 1
                        elif result_status == 'timeout':
                            timeout_count += 1
                            acc = result.get('accuracy', result.get('final_reward', 0))
                            if acc is not None:
                                total_accuracy += acc
                                accuracy_count += 1
                        else:
                            error_count += 1

                        # Print per-task completion line
                        task_name = result.get("config_name", f"config_{config_id}")
                        state = f"state{run_id}"
                        r_acc = result.get('accuracy', 0)
                        r_steps = result.get('steps', '?')
                        r_tokens = result.get('api_total_tokens', 0)
                        r_trimmed = result.get('trimmed_tokens', 0)
                        r_tokens_incl = r_tokens + r_trimmed
                        tok_str = f"{r_tokens:,} tok" if r_trimmed == 0 else f"{r_tokens:,} tok ({r_tokens_incl:,} incl. trimmed)"
                        if result_status == 'success' and r_acc > 0:
                            progress.console.print(f"  [green]\u2713[/green] {task_name} {state} \u2014 passed (acc: {r_acc}, {r_steps} steps, {tok_str})")
                        elif result_status == 'timeout':
                            progress.console.print(f"  [yellow]⚠[/yellow] {task_name} {state} \u2014 timeout-finalized (acc: {r_acc}, {r_steps} steps, {tok_str})")
                        else:
                            progress.console.print(f"  [red]\u2717[/red] {task_name} {state} \u2014 failed (acc: {r_acc}, {r_steps} steps, {tok_str})")
                    except Exception as e:
                        results.append({
                            "task_id": task_id,
                            "config_id": config_id,
                            "run_id": run_id,
                            "status": "exception",
                            "error": str(e),
                        })
                        completed_count += 1
                        error_count += 1
                        progress.console.print(f"  [red]\u2717[/red] config_{config_id} state{run_id} \u2014 exception: {e}")

                    # Update progress bar description with stats
                    avg_acc = total_accuracy / accuracy_count if accuracy_count > 0 else 0
                    progress.update(
                        task_progress,
                        advance=1,
                        description=f"[cyan]Tasks: {completed_count}/{len(task_args)} | Success: {success_count} | Timeout: {timeout_count} | Errors: {error_count} | Avg Acc: {avg_acc:.2%}"
                    )

    finally:
        # Restore original signal handlers
        signal.signal(signal.SIGINT, original_sigint)
        signal.signal(signal.SIGTERM, original_sigterm)
    
    elapsed_time = time.time() - start_time
    
    # Analyze results by configuration
    config_stats = {}
    for result in results:
        config_id = result.get("config_id", 0)
        if config_id not in config_stats:
            config_stats[config_id] = {
                "total": 0,
                "success": 0,
                "timeout": 0,
                "error": 0,
                "accuracies": [],
                "steps": [],
                "tool_calls": [],
                "api_total_tokens": [],
                "trimmed_tokens": [],
                "reset_tokens": [],
                "thinking_reset_tokens": [],
                "summary_tokens": [],
            }

        config_stats[config_id]["total"] += 1
        if result["status"] in {"success", "timeout"}:
            if result["status"] == "success":
                config_stats[config_id]["success"] += 1
            else:
                config_stats[config_id]["timeout"] += 1
            accuracy = result.get("accuracy", result["final_reward"])
            config_stats[config_id]["accuracies"].append(accuracy)
            config_stats[config_id]["steps"].append(result["steps"])
            config_stats[config_id]["tool_calls"].append(result.get("tool_calls", 0))
            config_stats[config_id]["api_total_tokens"].append(result.get("api_total_tokens", 0))
            config_stats[config_id]["trimmed_tokens"].append(result.get("trimmed_tokens", 0))
            config_stats[config_id]["reset_tokens"].append(result.get("reset_tokens", 0))
            config_stats[config_id]["thinking_reset_tokens"].append(result.get("thinking_reset_tokens", 0))
            config_stats[config_id]["summary_tokens"].append(result.get("summary_tokens", 0))
        else:
            config_stats[config_id]["error"] += 1
    
    total_success = sum(1 for r in results if r["status"] == "success")
    total_timeout = sum(1 for r in results if r["status"] == "timeout")
    total_error = sum(1 for r in results if r["status"] in ["error", "exception"])

    # Calculate overall averages
    all_accuracies = []
    all_steps = []
    all_tool_calls = []
    all_api_total_tokens = []
    all_trimmed_tokens = []
    all_reset_tokens = []
    all_thinking_reset_tokens = []
    all_summary_tokens = []
    for stats in config_stats.values():
        all_accuracies.extend(stats['accuracies'])
        all_steps.extend(stats['steps'])
        all_tool_calls.extend(stats['tool_calls'])
        all_api_total_tokens.extend(stats['api_total_tokens'])
        all_trimmed_tokens.extend(stats['trimmed_tokens'])
        all_reset_tokens.extend(stats['reset_tokens'])
        all_thinking_reset_tokens.extend(stats['thinking_reset_tokens'])
        all_summary_tokens.extend(stats['summary_tokens'])
    avg_accuracy = sum(all_accuracies) / len(all_accuracies) if all_accuracies else None
    avg_steps = sum(all_steps) / len(all_steps) if all_steps else None
    avg_tool_calls = sum(all_tool_calls) / len(all_tool_calls) if all_tool_calls else None
    total_api_tokens_all = sum(all_api_total_tokens)
    avg_api_tokens = sum(all_api_total_tokens) / len(all_api_total_tokens) if all_api_total_tokens else None
    # Compute per-run "inclusive" token sums, then average
    all_tokens_incl_trimmed = [a + t for a, t in zip(all_api_total_tokens, all_trimmed_tokens)]
    all_tokens_incl_reset = [a + t + r for a, t, r in zip(all_api_total_tokens, all_trimmed_tokens, all_reset_tokens)]
    all_tokens_incl_all = [a + t + r + th + s for a, t, r, th, s in zip(all_api_total_tokens, all_trimmed_tokens, all_reset_tokens, all_thinking_reset_tokens, all_summary_tokens)]
    avg_tokens_incl_trimmed = sum(all_tokens_incl_trimmed) / len(all_tokens_incl_trimmed) if all_tokens_incl_trimmed else None
    avg_tokens_incl_reset = sum(all_tokens_incl_reset) / len(all_tokens_incl_reset) if all_tokens_incl_reset else None
    avg_tokens_incl_all = sum(all_tokens_incl_all) / len(all_tokens_incl_all) if all_tokens_incl_all else None

    # Save results.json (use task names as keys when available)
    results_file = Path(output_dir) / "results.json"
    per_config_data = {}
    for k, v in config_stats.items():
        key = group_id_to_name.get(k, str(k))
        n = len(v["api_total_tokens"]) or 1
        cfg_incl_trimmed = [a + t for a, t in zip(v["api_total_tokens"], v["trimmed_tokens"])]
        cfg_incl_reset = [a + t + r for a, t, r in zip(v["api_total_tokens"], v["trimmed_tokens"], v["reset_tokens"])]
        cfg_incl_all = [a + t + r + th + s for a, t, r, th, s in zip(v["api_total_tokens"], v["trimmed_tokens"], v["reset_tokens"], v["thinking_reset_tokens"], v["summary_tokens"])]
        per_config_data[key] = {
            "success": v["success"],
            "timeout": v["timeout"],
            "error": v["error"],
            "avg_accuracy": round(sum(v["accuracies"]) / len(v["accuracies"]), 4) if v["accuracies"] else None,
            "avg_steps": round(sum(v["steps"]) / len(v["steps"]), 2) if v["steps"] else None,
            "avg_tool_calls": round(sum(v["tool_calls"]) / len(v["tool_calls"]), 2) if v["tool_calls"] else None,
            "total_api_tokens": sum(v["api_total_tokens"]),
            "avg_api_tokens": round(sum(v["api_total_tokens"]) / n, 0) if v["api_total_tokens"] else None,
            "avg_api_tokens_incl_trimmed": round(sum(cfg_incl_trimmed) / n, 0) if cfg_incl_trimmed else None,
            "avg_api_tokens_incl_reset": round(sum(cfg_incl_reset) / n, 0) if cfg_incl_reset else None,
            "avg_api_tokens_incl_all": round(sum(cfg_incl_all) / n, 0) if cfg_incl_all else None,
        }

    results_data = {
        "metadata": {
            "model": model,
            "timestamp": int(time.time()),
            "elapsed_seconds": round(elapsed_time, 2),
            "total_tasks": len(task_args),
        },
        "summary": {
            "total_success": total_success,
            "total_timeout": total_timeout,
            "total_error": total_error,
            "avg_accuracy": round(avg_accuracy, 4) if avg_accuracy is not None else None,
            "avg_steps": round(avg_steps, 2) if avg_steps is not None else None,
            "avg_tool_calls": round(avg_tool_calls, 2) if avg_tool_calls is not None else None,
            "total_api_tokens": total_api_tokens_all,
            "avg_api_tokens": round(avg_api_tokens, 0) if avg_api_tokens is not None else None,
            "avg_api_tokens_incl_trimmed": round(avg_tokens_incl_trimmed, 0) if avg_tokens_incl_trimmed is not None else None,
            "avg_api_tokens_incl_reset": round(avg_tokens_incl_reset, 0) if avg_tokens_incl_reset is not None else None,
            "avg_api_tokens_incl_all": round(avg_tokens_incl_all, 0) if avg_tokens_incl_all is not None else None,
        },
        "per_config": per_config_data,
    }
    write_results_file(
        path=results_file,
        metadata=results_data["metadata"],
        summary=results_data["summary"],
        per_config=results_data["per_config"],
        indent=2,
    )

    # all_trajectories.json skipped — individual trajectory.json per task is sufficient
    # and the aggregated file can be several GB due to Gemini reasoning fields.
    all_traj_file = None

    # Print summary table.
    from rich.console import Console as _SummaryConsole
    _sc = _SummaryConsole()
    _line = "=" * 50
    _sc.print(f"\n[bold]{_line}[/bold]")
    _sc.print(f"[bold]  LOCA Benchmark Summary[/bold]")
    _sc.print(f"[bold]{_line}[/bold]")
    _sc.print(f"  Total Tasks:              {len(task_args)}")
    _sc.print(f"  Success / Timeout / Error:{total_success} / {total_timeout} / {total_error}")
    _sc.print(f"  Average Accuracy:         {avg_accuracy:.4f}" if avg_accuracy is not None else "  Average Accuracy:         N/A")
    _sc.print(f"  Average Steps:            {avg_steps:.2f}" if avg_steps is not None else "  Average Steps:            N/A")
    _sc.print(f"  Average Tool Calls:       {avg_tool_calls:.1f}" if avg_tool_calls is not None else "  Average Tool Calls:       N/A")
    _sc.print(f"  Total API Tokens:         {total_api_tokens_all}")
    _sc.print(f"  Average API Tokens:       {int(avg_api_tokens)}" if avg_api_tokens is not None else "  Average API Tokens:       N/A")
    _sc.print(f"  Avg API Tokens (+Trim):   {int(avg_tokens_incl_trimmed)}" if avg_tokens_incl_trimmed is not None else "  Avg API Tokens (+Trim):   N/A")
    _sc.print(f"  Avg API Tokens (+Reset):  {int(avg_tokens_incl_reset)}" if avg_tokens_incl_reset is not None else "  Avg API Tokens (+Reset):  N/A")
    _sc.print(f"  Avg API Tokens (+All):    {int(avg_tokens_incl_all)}" if avg_tokens_incl_all is not None else "  Avg API Tokens (+All):    N/A")
    _sc.print(f"  Elapsed Time:             {int(elapsed_time)}s")
    _sc.print(f"[bold]{_line}[/bold]")
    _sc.print(f"  Results: {results_file}")
    _sc.print(f"[bold]{_line}[/bold]\n")


def main():
    """Main entry point.
    
    Example usage:
        python -m gem.inference.run_multi_openai_v2 --config_file example_flexible_config.json --runs_per_config 3
    """
    fire.Fire(run_config_combinations)


if __name__ == "__main__":
    main()

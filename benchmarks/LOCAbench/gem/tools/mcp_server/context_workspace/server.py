#!/usr/bin/env python3
"""
Context Workspace MCP Server

Tools for agent-controlled selective context compression.

Design: archive = replace in-place with a compact index, not remove.
  - visible   : full content in context
  - compressed: index placeholder in context (original stored as external payload)
  - deleted   : content intentionally removed; not retrievable
  - Batch archive works at the lowest listed compression level and skips higher levels

Tools:
  context_workspace_archive    - replace one or more blocks with compact indexes
"""

import json
import logging
import os
import re
import sys
import time
from pathlib import Path
from typing import Any, Dict

os.environ["FASTMCP_SHOW_CLI_BANNER"] = "false"
os.environ["FASTMCP_LOG_LEVEL"] = "ERROR"

if os.environ.get("LOCA_QUIET", "").lower() in ("1", "true", "yes"):
    logging.basicConfig(level=logging.ERROR, force=True)
    for _n in ["mcp", "fastmcp", "mcp.server", "mcp.client", "uvicorn"]:
        logging.getLogger(_n).setLevel(logging.ERROR)

gem_root = Path(__file__).parent.parent.parent.parent.parent
if str(gem_root) not in sys.path:
    sys.path.insert(0, str(gem_root))

from fastmcp import FastMCP

app = FastMCP("Context Workspace Server")

DEFAULT_WORKSPACE_DIR = "./context_workspace"
PUBLIC_PAYLOAD_DIRECT_MAX_CHARS = 50_000
PUBLIC_PAYLOAD_CHUNK_MAX_CHARS = 50_000
PUBLIC_PAYLOAD_CHUNK_MAX_LINES = 2_000


def get_workspace_dir() -> Path:
    d = Path(os.environ.get("CONTEXT_WORKSPACE_DIR", DEFAULT_WORKSPACE_DIR)).resolve()
    d.mkdir(parents=True, exist_ok=True)
    return d


def state_path() -> Path:
    return get_workspace_dir() / "workspace_state.json"


def payload_dir() -> Path:
    d = get_workspace_dir() / "payloads"
    d.mkdir(parents=True, exist_ok=True)
    return d


def public_payload_dir() -> Path | None:
    raw = os.environ.get("CONTEXT_WORKSPACE_PAYLOAD_DIR", "").strip()
    if not raw:
        return None
    d = Path(raw).resolve()
    d.mkdir(parents=True, exist_ok=True)
    return d


def _env_flag(name: str) -> bool:
    return os.environ.get(name, "").lower() in ("1", "true", "yes", "on")


def _payload_path_for(block_id: str) -> Path:
    return payload_dir() / f"{block_id}.txt"


def _public_payload_content(
    block_id: str,
    content: str,
    payload_kind: str = "",
    payload_maybe_complete: bool = True,
    source_metadata_path: str = "",
) -> str:
    return content or ""


def _write_public_payload_manifest(
    block_id: str,
    content: str,
    source_metadata_path: str = "",
) -> str:
    public_dir = public_payload_dir()
    if public_dir is None:
        return content or ""
    parts_dir = public_dir / f"{block_id}.parts"
    parts_dir.mkdir(parents=True, exist_ok=True)
    for stale in parts_dir.glob("part_*.txt"):
        try:
            stale.unlink()
        except Exception:
            pass

    parts = []
    lines = content.splitlines(keepends=True) or [content]
    idx = 0
    part_no = 1
    char_pos = 0
    while idx < len(lines):
        chunk_lines = []
        chunk_chars = 0
        start_line = idx + 1
        start_char = char_pos
        while idx < len(lines):
            line = lines[idx]
            if (
                chunk_lines
                and (
                    len(chunk_lines) >= PUBLIC_PAYLOAD_CHUNK_MAX_LINES
                    or chunk_chars + len(line) > PUBLIC_PAYLOAD_CHUNK_MAX_CHARS
                )
            ):
                break
            if not chunk_lines and len(line) > PUBLIC_PAYLOAD_CHUNK_MAX_CHARS:
                line = line[:PUBLIC_PAYLOAD_CHUNK_MAX_CHARS]
                lines[idx] = lines[idx][PUBLIC_PAYLOAD_CHUNK_MAX_CHARS:]
                if not lines[idx]:
                    idx += 1
                chunk_lines.append(line)
                chunk_chars += len(line)
                char_pos += len(line)
                break
            chunk_lines.append(line)
            chunk_chars += len(line)
            char_pos += len(line)
            idx += 1

        part_path = parts_dir / f"part_{part_no:04d}.txt"
        part_path.write_text("".join(chunk_lines), encoding="utf-8")
        parts.append({
            "file": str(part_path),
            "lines": f"{start_line}-{start_line + max(0, len(chunk_lines) - 1)}",
            "chars": f"{start_char}-{char_pos}",
        })
        part_no += 1

    parts_index = "\n".join(
        f"- {Path(p['file']).name}: lines {p['lines']}, chars {p['chars']}"
        for p in parts[:80]
    )
    if len(parts) > 80:
        parts_index += f"\n- ... {len(parts) - 80} more parts in {parts_dir}"
    source_line = f"Source metadata: {source_metadata_path}\n" if source_metadata_path else ""
    return (
        f"[SMS PAYLOAD MANIFEST: {block_id}]\n"
        "This complete payload is large, so the raw content is stored in bounded chunk files.\n"
        "Do not read all chunks at once. Search the parts directory or read only the needed part file.\n"
        f"Parts directory: {parts_dir}\n"
        f"{source_line}"
        f"Total parts: {len(parts)}\n"
        f"Chunk limits: <= {PUBLIC_PAYLOAD_CHUNK_MAX_LINES} lines and <= {PUBLIC_PAYLOAD_CHUNK_MAX_CHARS} chars per part.\n"
        "Parts:\n"
        f"{parts_index}\n"
    )


def _write_source_metadata(block: Dict[str, Any]) -> None:
    if block.get("type") != "tool_result" or not block.get("source_tool_name"):
        return
    metadata = {
        "block_id": block.get("id"),
        "payload_kind": block.get("payload_kind") or "tool transcript",
        "payload_maybe_complete": bool(block.get("payload_maybe_complete", True)),
        "source_kind": "tool_call",
        "source_tool_name": block.get("source_tool_name"),
        "source_tool_arguments": block.get("source_tool_arguments", ""),
        "source_tool_call_id": block.get("source_tool_call_id", ""),
        "guidance": (
            "This payload stores only the transcript returned by this tool call, not a guarantee of "
            "complete source data. Use the source tool again if the transcript indicates omitted, "
            "shown, paged, or truncated data."
        ),
    }
    path = payload_dir() / f"{block['id']}.source.json"
    path.write_text(json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8")
    block["source_metadata_path"] = str(path.relative_to(get_workspace_dir()))
    public_dir = public_payload_dir()
    if public_dir is not None and not _env_flag("SM_DISABLE_RECOVER"):
        public_path = public_dir / f"{block['id']}.source.json"
        public_path.write_text(json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8")
        block["public_source_metadata_path"] = str(public_path)
        if block.get("public_payload_path"):
            try:
                raw = _read_payload(block)
            except Exception:
                raw = block.get("content", "")
            Path(block["public_payload_path"]).write_text(
                _public_payload_content(
                    block["id"],
                    raw,
                    block.get("payload_kind", ""),
                    bool(block.get("payload_maybe_complete", True)),
                    str(public_path),
                ),
                encoding="utf-8",
            )
    else:
        block.pop("public_source_metadata_path", None)


def _write_payload(block: Dict[str, Any]) -> None:
    content = block.get("content", "")
    if not content:
        return
    path = _payload_path_for(block["id"])
    path.write_text(content, encoding="utf-8")
    block["payload_path"] = str(path.relative_to(get_workspace_dir()))
    kind, maybe_complete = _payload_kind(block.get("type", ""), content)
    block["payload_kind"] = kind
    block["payload_maybe_complete"] = maybe_complete
    block["payload_partial_info"] = _partial_rows_info(content)
    public_dir = public_payload_dir()
    if public_dir is not None and not _env_flag("SM_DISABLE_RECOVER"):
        public_path = public_dir / f"{block['id']}.txt"
        public_path.write_text(
            _public_payload_content(
                block["id"],
                content,
                kind,
                maybe_complete,
                block.get("public_source_metadata_path", ""),
            ),
            encoding="utf-8",
        )
        block["public_payload_path"] = str(public_path)
    else:
        block.pop("public_payload_path", None)
    block["content_len"] = len(content)
    _write_source_metadata(block)
    block["content"] = ""


def _read_payload(block: Dict[str, Any]) -> str:
    content = block.get("content", "")
    if content:
        return content
    rel = block.get("payload_path")
    if rel:
        try:
            return (get_workspace_dir() / rel).read_text(encoding="utf-8")
        except Exception:
            pass
    public = block.get("public_payload_path")
    if public:
        try:
            return Path(public).read_text(encoding="utf-8")
        except Exception:
            return ""
    return ""


# ── Shared state I/O ──────────────────────────────────────────────────────────

def _load() -> Dict[str, Any]:
    p = state_path()
    if p.exists():
        try:
            with open(p) as f:
                return json.load(f)
        except Exception:
            pass
    return {
        "blocks": {},
        "next_id": 1,
        "current_episode_id": None,
        "archive_groups": {},
        "next_archive_group_id": 1,
        "temporary_observations": {},
    }


def _save(state: Dict[str, Any]) -> None:
    p = state_path()
    p.parent.mkdir(parents=True, exist_ok=True)
    with open(p, "w") as f:
        json.dump(state, f, indent=2)


def _invalidate_dashboard(state: Dict[str, Any]) -> None:
    state["dashboard_cache"] = None


def _record_temporary_observation(state: Dict[str, Any], block_id: str, kind: str) -> None:
    block = state.get("blocks", {}).get(block_id, {})
    obs = state.setdefault("temporary_observations", {}).setdefault(block_id, {
        "block_id": block_id,
        "summary": block.get("summary", ""),
        "status": block.get("status", ""),
    })
    obs["summary"] = block.get("summary", obs.get("summary", ""))
    obs["status"] = block.get("status", obs.get("status", ""))
    obs["last_seen_at"] = time.time()


def _mark_promoted(state: Dict[str, Any], block_ids: list[str], promoted_by: str) -> None:
    for bid in block_ids:
        obs = state.setdefault("temporary_observations", {}).setdefault(bid, {
            "block_id": bid,
            "summary": state.get("blocks", {}).get(bid, {}).get("summary", ""),
        })
        obs["promoted_at"] = time.time()
        obs["promoted_by"] = promoted_by


def _estimate_tokens(text: str) -> int:
    try:
        import tiktoken as _tkt
        enc = _tkt.get_encoding("cl100k_base")
        return len(enc.encode(str(text), disallowed_special=()))
    except Exception:
        return max(1, len(str(text)) // 3)


def _block_tokens(block: Dict[str, Any]) -> int:
    if block.get("status") == "compressed":
        return _estimate_tokens(block.get("summary", ""))
    return block.get("tiktoken_tokens") or _estimate_tokens(block.get("content", ""))


_INCOMPLETE_TRANSCRIPT_RE = re.compile(
    r"showing\s+\d+\s+rows|first\s+\d+\s+rows|displaying\s+\d+|"
    r"total_pages|page_size|truncated|TRUNCATED|next page|has_more|"
    r"more results|omitted",
    re.I,
)

_TOTAL_ROWS_RE = re.compile(r"\bTotal rows:\s*([\d,]+)", re.I)
_SHOWING_ROWS_RE = re.compile(r"\bResults\s*\(showing\s+([\d,]+)\s+rows\)", re.I)


def _partial_rows_info(content: str) -> str:
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
        return f"shown rows: {showing:,} / total rows: {total:,}"
    return ""


def _payload_kind(block_type: str, content: str) -> tuple[str, bool]:
    if block_type == "tool_result":
        return "tool transcript", not bool(_INCOMPLETE_TRANSCRIPT_RE.search(content or ""))
    return "conversation transcript", True


def _current_context_tokens(state: Dict[str, Any]) -> int:
    blocks = state.get("blocks", {})
    in_context = [
        b for b in blocks.values()
        if b.get("status") in ("visible", "pinned", "compressed")
    ]
    return sum(_block_tokens(b) for b in in_context) + int(state.get("overhead_tokens", 0) or 0)


def _parse_block_ids(block_ids: str) -> list[str]:
    ids = []
    text = str(block_ids)

    def _add_id(bid: str) -> None:
        if bid and bid not in ids:
            ids.append(bid)

    # Accept common range forms the model naturally writes:
    #   B10-B20, B10 - B20, B10:B20, B10 to B20
    # Only explicit B-number ranges are expanded; other text remains subject
    # to normal target parsing so invalid syntax is surfaced as missing.
    token_pattern = re.compile(
        r"\bB(\d+)\s*(?:-|:|\bto\b)\s*B(\d+)\b|\b[Bb]\d+\b|\bG\d+\b|[^\s,]+",
        re.IGNORECASE,
    )
    for match in token_pattern.finditer(text):
        if match.group(1) and match.group(2):
            start = int(match.group(1))
            end = int(match.group(2))
            step = 1 if start <= end else -1
            for n in range(start, end + step, step):
                _add_id(f"B{n}")
            continue

        token = match.group(0).strip()
        if re.fullmatch(r"[bB]\d+", token):
            _add_id(f"B{int(token[1:])}")
        else:
            _add_id(token)
    return ids


def _live_group_block_ids(state: Dict[str, Any], group_id: str) -> list[str]:
    """Return current compressed members for an archive group.

    A block can be re-archived into a newer group, so the group's original
    block_ids are not necessarily current. Use archive_group_id to avoid stale
    group handles operating on blocks that now belong elsewhere.
    """
    group = state.get("archive_groups", {}).get(group_id)
    if not group:
        return []
    blocks = state.get("blocks", {})
    ids = []
    for bid in group.get("block_ids", []):
        block = blocks.get(bid)
        if (
            block
            and block.get("status") == "compressed"
            and block.get("archive_group_id") == group_id
        ):
            ids.append(bid)
    return ids


def _parse_block_or_group_targets(
    state: Dict[str, Any],
    target_ids: str,
) -> tuple[list[str], list[str], list[str], list[str]]:
    """Parse block/group targets.

    Returns (expanded_block_ids, group_ids, missing_targets, stale_groups).
    """
    ids: list[str] = []
    groups: list[str] = []
    missing: list[str] = []
    stale: list[str] = []
    archive_groups = state.get("archive_groups", {})
    blocks = state.get("blocks", {})

    for target in _parse_block_ids(target_ids):
        if target in archive_groups:
            groups.append(target)
            live_ids = _live_group_block_ids(state, target)
            if not live_ids:
                stale.append(target)
            for bid in live_ids:
                if bid not in ids:
                    ids.append(bid)
        elif target in blocks:
            if target not in ids:
                ids.append(target)
        else:
            missing.append(target)

    return ids, groups, missing, stale


def _archive_level(block: Dict[str, Any]) -> int:
    """Visible/raw blocks are L0; compressed blocks are their current level."""
    if block.get("status") == "compressed":
        return int(block.get("compression_level", 1) or 1)
    return 0


def _next_archive_group_id(state: Dict[str, Any]) -> str:
    n = int(state.get("next_archive_group_id", 1) or 1)
    state["next_archive_group_id"] = n + 1
    return f"G{n}"


def _archive_one(
    state: Dict[str, Any],
    block_id: str,
    replacement: str = "",
    group_id: str | None = None,
) -> int:
    block = state["blocks"][block_id]

    current_level = block.get("compression_level", 0)
    new_level = current_level + 1

    # Update replacement. Re-archiving a compressed block moves it to the next
    # index level. Do not append old summaries: mixing levels inside one
    # replacement makes provenance ambiguous.
    # Stored under the legacy key "summary" for state-file compatibility.
    if _env_flag("SM_ARCHIVE_PLACEHOLDER_ONLY"):
        block["summary"] = ""
    elif replacement:
        block["summary"] = replacement
    elif not block.get("summary"):
        # Older state files may have empty content for visible blocks; new state
        # files store full content so compressed blocks remain retrievable.
        content = block.get("content", "")
        block["summary"] = (content[:120] + ("…" if len(content) > 120 else "")) if content else f"Block {block_id}"

    block["status"] = "compressed"
    block["compression_level"] = new_level
    _write_payload(block)
    if group_id:
        block["archive_group_id"] = group_id

    return new_level


def _archive_paths_note(state: Dict[str, Any], block_ids: list[str]) -> str:
    if _env_flag("SM_DISABLE_RECOVER"):
        return ""
    payload_blocks = [
        state["blocks"][bid]
        for bid in block_ids
        if state["blocks"].get(bid, {}).get("public_payload_path")
    ]
    paths = [b.get("public_payload_path") for b in payload_blocks if b.get("public_payload_path")]
    if not paths:
        return ""
    if len(paths) == 1:
        return f"\nPayload file: {paths[0]}"
    return f"\nPayload files: {len(paths)} files, e.g. {paths[0]}"


def _delete_one(block_id: str, block: Dict[str, Any], reason: str, deleted: list[str]) -> None:
    block["status"] = "deleted"
    block["content"] = ""
    block["payload_path"] = ""
    block["public_payload_path"] = ""
    block["summary"] = ""
    block["tiktoken_tokens"] = 0
    block["delete_reason"] = reason.strip()
    block["deleted_at"] = time.time()
    deleted.append(block_id)


_SPACE_ONLY_DELETE_TERMS = (
    "context budget",
    "context is tight",
    "free space",
    "save space",
    "make room",
    "too large",
    "token budget",
    "reduce tokens",
)


# ── Tools ─────────────────────────────────────────────────────────────────────

@app.tool()
def context_workspace_archive(block_id: str, replacement: str = "") -> str:
    """Replace one or more blocks with compact indexes.

    The original content is stored externally as a payload file. Operations
    are block-level: only listed block IDs are archived. Multiple IDs and ranges
    are accepted. Mixed-level batches archive the lowest-level listed blocks and
    skip higher levels.

    Args:
        block_id: Block IDs, ranges, or group IDs, e.g. "B3", "B3,B4",
                  "B10-B20", "B10 to B20", or "G2".
        replacement: Short index text for the archived block(s).
    """
    state = _load()
    ids, groups, missing, stale_groups = _parse_block_or_group_targets(state, block_id)
    if not ids:
        return "Error: archive requires at least one block ID."

    if missing:
        return f"Error: target(s) not found: {', '.join(missing)}. Check the dashboard for valid block/group IDs."
    if stale_groups:
        return (
            f"Error: archive group(s) have no current member blocks: {', '.join(stale_groups)}. "
            "Use the current archive group shown in the dashboard or list block IDs explicitly."
        )

    protected = [bid for bid in ids if state["blocks"][bid]["status"] == "pinned"]
    if protected:
        return f"Error: cannot archive initial task message block(s): {', '.join(protected)}."

    deleted = [bid for bid in ids if state["blocks"][bid]["status"] == "deleted"]
    if deleted:
        return f"Error: deleted block(s) cannot be archived: {', '.join(deleted)}."

    skipped_higher = []
    if len(ids) > 1:
        levels = {bid: _archive_level(state["blocks"][bid]) for bid in ids}
        min_level = min(levels.values())
        skipped_higher = [bid for bid in ids if levels[bid] != min_level]
        ids = [bid for bid in ids if levels[bid] == min_level]

    requested_targets = _parse_block_ids(block_id)
    rearchive_existing_group = (
        len(requested_targets) == 1
        and len(groups) == 1
        and len(ids) > 1
    )
    group_id = None
    if len(ids) > 1:
        group_id = groups[0] if rearchive_existing_group else _next_archive_group_id(state)
        input_level = _archive_level(state["blocks"][ids[0]])
        state.setdefault("archive_groups", {})[group_id] = {
            "id": group_id,
            "block_ids": ids,
            "replacement": "" if _env_flag("SM_ARCHIVE_PLACEHOLDER_ONLY") else replacement,
            "input_level": input_level,
            "output_level": input_level + 1,
            "created_at": time.time(),
        }

    levels = {}
    for bid in ids:
        levels[bid] = _archive_one(state, bid, replacement, group_id)
    _save(state)

    label = ids[0] if len(ids) == 1 else f"{group_id} ({', '.join(ids)})"
    level_label = f"L{levels[ids[0]]}" if len(ids) == 1 else "group"
    skipped_note = (
        f"\nSkipped higher-level blocks: {', '.join(skipped_higher)}. "
        "Archive them separately if they still need a higher-level index."
        if skipped_higher else ""
    )
    return (
        f"[ARCHIVED:{label} {level_label}] "
        f"{'(placeholder only)' if _env_flag('SM_ARCHIVE_PLACEHOLDER_ONLY') else (replacement[:120] if replacement else '(existing/auto index)')}"
        f"{_archive_paths_note(state, ids)}"
        f"{skipped_note}"
    )


@app.tool()
def context_workspace_delete(block_id: str, reason: str) -> str:
    """Permanently remove one or more blocks.

    Deleted content cannot be recovered. Operations are block-level: only listed
    block IDs are deleted. Multiple IDs and ranges are accepted.

    Args:
        block_id: Block IDs, ranges, or group IDs.
        reason: Short reason why the content has no future task value.
    """
    if not reason or not reason.strip():
        return "Error: delete requires a reason explaining why the content has no future task value."

    state = _load()
    ids, groups, missing, stale_groups = _parse_block_or_group_targets(state, block_id)
    if not ids:
        return "Error: delete requires at least one block ID."

    if missing:
        return f"Error: target(s) not found: {', '.join(missing)}. Check the dashboard for valid block/group IDs."
    if stale_groups:
        return (
            f"Error: archive group(s) have no current member blocks: {', '.join(stale_groups)}. "
            "Use the current archive group shown in the dashboard or list block IDs explicitly."
        )

    protected = [bid for bid in ids if state["blocks"][bid]["status"] == "pinned"]
    if protected:
        return f"Error: cannot delete initial task message block(s): {', '.join(protected)}."

    reason_lower = reason.lower()
    space_only_warning = any(term in reason_lower for term in _SPACE_ONLY_DELETE_TERMS)

    deleted = []
    for bid in ids:
        block = state["blocks"][bid]
        if block["status"] == "deleted":
            continue
        _delete_one(bid, block, reason, deleted)

    _save(state)
    label = block_id if groups else (block_id if len(ids) == 1 else ", ".join(ids))
    warning = (
        "\nWarning: delete is for content with no remaining task value, not just for freeing context. "
        "If any deleted content may matter later, keep it visible or archive it instead."
        if space_only_warning else ""
    )
    return f"[DELETED:{label}] Content permanently removed: {', '.join(deleted)}.{warning}"


def context_workspace_checkpoint(content: str, title: str = "", source_blocks: str = "") -> str:
    """Append a checkpoint block to the conversation.

    Use this only when a compact task state would help future steps. Prefer
    source files and source tools for structured data and calculations.

    Args:
        content: Checkpoint content to append.
        title: Optional short title for the checkpoint.
        source_blocks: Optional source block IDs.
    """
    if not content or not content.strip():
        return "Error: checkpoint content cannot be empty."

    state = _load()
    title = title.strip() or "Checkpoint"
    source_blocks = source_blocks.strip()
    source_ids = [bid for bid in _parse_block_ids(source_blocks) if re.fullmatch(r"B\d+", bid)]
    if source_ids:
        state.setdefault("pending_checkpoint_sources", [])
        for bid in source_ids:
            if bid not in state["pending_checkpoint_sources"]:
                state["pending_checkpoint_sources"].append(bid)
        _invalidate_dashboard(state)
        _save(state)

    lines = [
        "[CHECKPOINT]",
        f"Title: {title}",
    ]
    if source_blocks:
        lines.append(f"Source blocks: {source_blocks}")
    lines.extend(["", content.strip()])
    return "\n".join(lines)


@app.tool()
def context_workspace_update_state_board(content: str) -> str:
    """Update the dashboard State Board with compact current work status.

    The State Board is an external status note displayed at the top of every
    dashboard. Keep it short and operational: current progress, decisions,
    unresolved gaps, and next actions.

    Args:
        content: Compact current work status to show in the dashboard.
    """
    if not _env_flag("SM_ENABLE_STATE_BOARD"):
        return "Error: state board is disabled for this run."
    state = _load()
    state["state_board"] = (content or "").strip()
    state["state_board_updated_at"] = time.time()
    _invalidate_dashboard(state)
    _save(state)
    return "[STATE_BOARD_UPDATED] Dashboard State Board updated."


@app.tool()
def context_workspace_recover(block_id: str) -> str:
    """Read back the exact original content of archived block(s).

    Archived blocks appear in the dashboard as [ARCHIVED:Bx Ln] placeholders.
    Use this to restore the verbatim payload of a block or archive group when the
    current step needs the exact original evidence, for example a specific
    document, table, or search result that was archived earlier. This is the
    recover side of the archive-then-recover loop for environments that do not
    expose file or python tools to read payload files directly.

    Args:
        block_id: Block IDs, ranges, or group IDs, e.g. "B3", "B3,B4",
                  "B10-B20", "B10 to B20", or "G2".
    """
    if _env_flag("SM_DISABLE_RECOVER"):
        return "Error: recover is disabled for this run."

    state = _load()
    ids, groups, missing, stale_groups = _parse_block_or_group_targets(state, block_id)
    if not ids:
        return "Error: recover requires at least one block ID."
    if missing:
        return f"Error: target(s) not found: {', '.join(missing)}. Check the dashboard for valid block/group IDs."
    if stale_groups:
        return (
            f"Error: archive group(s) have no current member blocks: {', '.join(stale_groups)}. "
            "Use the current archive group shown in the dashboard or list block IDs explicitly."
        )

    parts = []
    for bid in ids:
        block = state["blocks"].get(bid, {})
        if block.get("status") == "deleted":
            parts.append(f"[RECOVER:{bid}] Error: block was deleted and cannot be recovered.")
            continue
        content = _read_payload(block) or block.get("content", "")
        if not content:
            parts.append(f"[RECOVER:{bid}] (no stored payload; the block may still be visible in context).")
            continue
        parts.append(f"[RECOVERED:{bid}]\n{content}")
    return "\n\n".join(parts)


# ── Entry point ────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--workspace-dir", default=DEFAULT_WORKSPACE_DIR)
    parser.add_argument("--payload-dir", default="")
    args = parser.parse_args()

    os.environ["CONTEXT_WORKSPACE_DIR"] = str(Path(args.workspace_dir).resolve())
    if args.payload_dir:
        os.environ["CONTEXT_WORKSPACE_PAYLOAD_DIR"] = str(Path(args.payload_dir).resolve())
    app.run(transport="stdio", show_banner=False)

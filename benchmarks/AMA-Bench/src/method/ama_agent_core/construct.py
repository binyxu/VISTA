"""
Memory Construction Module for AMA-Agent

This module handles the construction of state memory from trajectory data.
It processes trajectory text into different turns and embeds them for retrieval.
"""
import json
from concurrent.futures import ThreadPoolExecutor
from typing import Dict, List, Any, Optional, Callable, Tuple
from .utils import extract_state_memory_from_response
from .prompt import COMPRESS_PROMPT_TEMPLATE, CAUSAL_PROMPT_TEMPLATE


def construct_state_memory(
    trajectory_text: str,
    task: str = "",
    call_llm_func: Optional[Callable] = None,
    chunk_size: int = 2048,
    session_size: int = 16384,
    max_context_length: Optional[int] = None,
    embed_engine: Optional[Callable] = None,
    causal: bool = False
) -> Dict[str, Any]:
    """
    Construct state memory from trajectory text.

    Process:
    1. Parse trajectory text into different turns
    2. Compress trajectory into state memory using LLM
    3. Optionally embed each turn for retrieval
    4. Optionally extract causal graph (if causal=True)

    Args:
        trajectory_text: String-formatted trajectory text with turns
        task: Task description
        call_llm_func: Function for LLM interaction
        chunk_size: Maximum size for each chunk (default: 8192)
        max_context_length: Model context window used to enforce a safe
            upper bound for chunk size (0.8 * max_context_length)
        embed_engine: Optional embedding function for turn-level embeddings
        causal: If True, also extract causal relationships to build a causal
                graph alongside the state memory (default: False)

    Returns:
        Dictionary containing:
            - state_mem: Compressed state memory string
            - causal_graph: Extracted causal relationships (if causal=True, else None)
            - text_mem: Original trajectory data
            - embed_mem: Turn-level embeddings (if embed_engine provided, else None)
            - trajectory: Parsed trajectory list
    """
    # Parse trajectory text into turns
    trajectory = _parse_trajectory_text(trajectory_text)

    # Build text memory
    trajectory_data = {
        'trajectory': trajectory,
        'task': task,
        'episode_id': 'episode'
    }
    text_mem = {
        'task': task,
        'trajectory_text': trajectory_text,
        'trajectory_data': trajectory_data,
        'episode_id': 'episode',
        'num_turns': len(trajectory)
    }


    # Build state memory (and optionally causal graph)
    if causal:
        state_mem, causal_graph = _process_trajectory_causal(
            trajectory_text=trajectory_text,
            task=task,
            session_size=session_size,
            max_context_length=max_context_length,
            call_llm_func=call_llm_func
        )
    else:
        state_mem = _process_trajectory(
            trajectory_text=trajectory_text,
            task=task,
            session_size=session_size,
            max_context_length=max_context_length,
            call_llm_func=call_llm_func
        )
        causal_graph = None

    # Build turn-level embeddings only if embed_engine is provided
    embed_mem = _build_turn_embeddings(
        trajectory=trajectory,
        embed_engine=embed_engine,
        min_chunk_size=chunk_size,
    )

    return {
        'state_mem': state_mem,
        'causal_graph': causal_graph,
        'text_mem': text_mem,
        'embed_mem': embed_mem,
        'trajectory': trajectory
    }


def _parse_trajectory_text(trajectory_text: str) -> List[Dict[str, Any]]:
    """
    Parse trajectory text into list of turn dictionaries.

    Expected format:
        Turn 0:
          Action: ...
          Observation: ...
        Turn 1:
          Action: ...
          Observation: ...

    Multi-line action/observation values are supported: lines that follow an
    Action/Observation header and do not match any other recognized header are
    appended to the current field.

    Args:
        trajectory_text: Formatted trajectory text

    Returns:
        List of turn dictionaries with keys: turn_idx, action, observation
    """
    trajectory = []
    lines = trajectory_text.strip().split('\n')

    current_turn: Dict[str, Any] = {}
    current_field: Optional[str] = None

    for line in lines:
        stripped = line.strip()

        # Match "Turn X:" or "Step X:" header
        if (stripped.startswith('Turn ') or stripped.startswith('Step ')) and ':' in stripped:
            if current_turn:
                trajectory.append(current_turn)
            try:
                turn_num = int(stripped.split(':')[0].split()[-1])
                current_turn = {'turn_idx': turn_num}
            except (ValueError, IndexError):
                current_turn = {}
            current_field = None

        elif stripped.startswith('Action:'):
            current_turn['action'] = stripped[7:].strip()
            current_field = 'action'

        elif stripped.startswith('Observation:'):
            current_turn['observation'] = stripped[12:].strip()
            current_field = 'observation'

        elif current_field and stripped and current_turn:
            # Continuation of a multi-line action or observation
            current_turn[current_field] = current_turn.get(current_field, '') + '\n' + stripped

    if current_turn:
        trajectory.append(current_turn)

    return trajectory


def _extract_turn_blocks(text: str) -> List[Tuple[int, str]]:
    """
    Split trajectory text into complete turn blocks.

    Returns:
        List of (turn_idx, turn_block_text) preserving original order.
    """
    lines = text.splitlines()
    turn_blocks: List[Tuple[int, str]] = []
    current_turn_idx: Optional[int] = None
    current_lines: List[str] = []

    for line in lines:
        stripped = line.strip()
        is_turn_header = (stripped.startswith('Turn ') or stripped.startswith('Step ')) and ':' in stripped

        if is_turn_header:
            if current_turn_idx is not None and current_lines:
                turn_blocks.append((current_turn_idx, "\n".join(current_lines).strip()))

            try:
                current_turn_idx = int(stripped.split(':')[0].split()[-1])
            except (ValueError, IndexError):
                # If parsing fails, continue current numbering if possible.
                current_turn_idx = (turn_blocks[-1][0] + 1) if turn_blocks else 0
            current_lines = [line]
            continue

        if current_turn_idx is not None:
            current_lines.append(line)

    if current_turn_idx is not None and current_lines:
        turn_blocks.append((current_turn_idx, "\n".join(current_lines).strip()))

    return turn_blocks


def _truncate_text_strict(text: str, max_length: int) -> str:
    """
    Strictly truncate text to at most max_length characters.

    Uses head-tail truncation with a marker while guaranteeing the final length
    does not exceed max_length.
    """
    max_length = max(1, int(max_length))
    if len(text) <= max_length:
        return text

    marker = "\n...[truncated]...\n"
    if max_length <= len(marker) + 8:
        return text[:max_length]

    content_budget = max_length - len(marker)
    head_len = max(1, int(content_budget * 0.7))
    tail_len = max(1, content_budget - head_len)
    truncated = text[:head_len] + marker + text[-tail_len:]
    return truncated[:max_length]


def _build_trajectory_turn_chunks(
    text: str,
    chunk_size: int,
    hard_cap_size: Optional[int] = None,
) -> List[Dict[str, Any]]:
    """
    Build trajectory chunks on turn boundaries using session-size threshold.

    Each chunk accumulates complete turns until it minimally reaches/exceeds
    `chunk_size`, then starts a new chunk. If `hard_cap_size` is set, chunk
    growth is also bounded by that cap. Oversized single turns are truncated
    to fit the cap.
    """
    chunk_size = max(1, chunk_size)
    if hard_cap_size is not None:
        hard_cap_size = max(1, hard_cap_size)
    turn_blocks = _extract_turn_blocks(text)
    if not turn_blocks:
        # Fallback: preserve legacy behavior if turn headers are absent.
        raw_text = text.strip()
        if not raw_text:
            return []
        fallback_size = hard_cap_size if hard_cap_size is not None else chunk_size
        return [
            {
                'text': raw_text[i:i + fallback_size],
                'start_turn': 'unknown',
                'end_turn': 'unknown',
            }
            for i in range(0, len(raw_text), fallback_size)
        ]

    chunks: List[Dict[str, Any]] = []

    current_parts: List[str] = []
    current_size = 0
    chunk_start_turn: Optional[int] = None
    chunk_end_turn: Optional[int] = None

    for turn_idx, block_text in turn_blocks:
        if not block_text:
            continue

        if hard_cap_size is not None and len(block_text) > hard_cap_size:
            block_text = _truncate_text_strict(block_text, hard_cap_size)

        addition_size = len(block_text) + (1 if current_parts else 0)

        # Respect hard cap when possible without splitting turns.
        if (
            hard_cap_size is not None
            and current_parts
            and (current_size + addition_size) > hard_cap_size
        ):
            chunks.append({
                'text': "\n".join(current_parts),
                'start_turn': chunk_start_turn,
                'end_turn': chunk_end_turn,
            })
            current_parts = []
            current_size = 0
            chunk_start_turn = None
            chunk_end_turn = None
            addition_size = len(block_text)

        if chunk_start_turn is None:
            chunk_start_turn = turn_idx
        chunk_end_turn = turn_idx
        current_parts.append(block_text)
        current_size += addition_size

        # Minimal turn span that reaches/exceeds chunk_size.
        if current_size >= chunk_size:
            chunks.append({
                'text': "\n".join(current_parts),
                'start_turn': chunk_start_turn,
                'end_turn': chunk_end_turn,
            })
            current_parts = []
            current_size = 0
            chunk_start_turn = None
            chunk_end_turn = None

    if current_parts and chunk_start_turn is not None and chunk_end_turn is not None:
        chunks.append({
            'text': "\n".join(current_parts),
            'start_turn': chunk_start_turn,
            'end_turn': chunk_end_turn,
        })

    return chunks


def _build_trajectory_chunks(text: str, chunk_size: int) -> List[str]:
    """
    Backward-compatible helper returning only chunk texts.
    """
    return [chunk['text'] for chunk in _build_trajectory_turn_chunks(text, chunk_size)]


def _process_trajectory(
    trajectory_text: str,
    task: str,
    session_size: int,
    max_context_length: Optional[int],
    call_llm_func: Optional[Callable]
) -> Optional[str]:
    """
    Process trajectory text to build state memory.

    Fits in one session → single LLM call.
    Exceeds session_size → sessions processed concurrently (Phase 1),
    partial state memories concatenated then capped at session_size * 4 (Phase 2).
    """
    if not call_llm_func:
        return None

    total_chars = len(trajectory_text)
    safe_cap = int(max_context_length * 0.7) if max_context_length else None
    effective_session_size = min(session_size, safe_cap) if safe_cap else session_size
    effective_session_size = max(1, effective_session_size)

    compress_template = COMPRESS_PROMPT_TEMPLATE

    # Single-session path
    if total_chars <= effective_session_size:
        compress_prompt = compress_template.format(
            task=task,
            trajectory_text=trajectory_text,
            previous_state_text=""
        )
        _, llm_response = call_llm_func(compress_prompt)
        if llm_response:
            return extract_state_memory_from_response(llm_response)
        return None

    # Multi-session path
    chunks = _build_trajectory_turn_chunks(
        trajectory_text,
        effective_session_size,
        hard_cap_size=safe_cap
    )

    # Phase 1: process all sessions concurrently (each independently)
    def _compress_chunk(chunk: Dict[str, Any]) -> Dict[str, Any]:
        prompt = compress_template.format(
            task=task,
            trajectory_text=chunk['text'],
            previous_state_text=""
        )
        _, response = call_llm_func(prompt)
        return {
            'start_turn': chunk.get('start_turn'),
            'end_turn': chunk.get('end_turn'),
            'state': extract_state_memory_from_response(response) if response else None
        }

    with ThreadPoolExecutor() as executor:
        chunk_states: List[Dict[str, Any]] = list(executor.map(_compress_chunk, chunks))

    # Phase 2: merge partial state memories, capped to avoid downstream context overflows.
    valid_states = [s for s in chunk_states if s.get('state')]
    if not valid_states:
        return None

    merged_parts: List[str] = []
    for state_info in valid_states:
        start_turn = state_info.get('start_turn')
        end_turn = state_info.get('end_turn')
        state_text = state_info['state']
        range_line = f"summary_turn_range: turn {start_turn} to turn {end_turn}"
        merged_parts.append(f"{range_line}\n{state_text}")

    merged = "\n\n".join(merged_parts)
    return merged



def _build_turn_embeddings(
    trajectory: List[Dict[str, Any]],
    embed_engine: Optional[Callable],
    min_chunk_size: int = 2048,
) -> Optional[Dict[str, Any]]:
    """
    Build turn-level embeddings for retrieval.

    Turns are grouped into chunks of at least _MIN_EMBED_CHUNK_SIZE characters
    before embedding.  A turn whose text already exceeds the minimum is embedded
    alone.  Grouped turns share the resulting embedding vector so the retrieval
    interface (parallel flat lists) stays unchanged.

    Args:
        trajectory: List of turns
        embed_engine: Synchronous embedding function, or None to skip embedding

    Returns:
        Dictionary containing embeddings/turn_texts/turn_indices,
        or None if embed_engine is not provided
    """
    if embed_engine is None:
        return None

    # Build per-turn texts
    turns_data: List[tuple] = []
    for turn in trajectory:
        turn_idx = turn.get('turn_idx', 0)
        action = turn.get('action', '')
        observation = turn.get('observation', '')
        turn_text = f"Turn {turn_idx}: Action={action}, Observation={observation}"
        turns_data.append((turn_idx, turn_text))

    # Group turns into chunks with minimum size _MIN_EMBED_CHUNK_SIZE
    # Each chunk: (chunk_text_to_embed, [turn_indices_in_chunk])
    chunks: List[tuple] = []
    buf_text = ""
    buf_indices: List[int] = []
    '''
    for turn_idx, turn_text in turns_data:
        if len(turn_text) >= min_chunk_size:
            # Flush accumulated buffer first
            if buf_indices:
                chunks.append((buf_text, buf_indices))
                buf_text = ""
                buf_indices = []
            # Large turn is its own chunk
            chunks.append((turn_text, [turn_idx]))
        else:
            buf_text = (buf_text + "\n" + turn_text).lstrip("\n") if buf_text else turn_text
            buf_indices.append(turn_idx)
            if len(buf_text) >= min_chunk_size:
                chunks.append((buf_text, buf_indices))
                buf_text = ""
                buf_indices = []
    '''
    for turn_idx, turn_text in turns_data:
        # Large turn is its own chunk
        chunks.append((turn_text, [turn_idx]))
        
   
    # Embed one text per chunk in parallel
    chunk_texts = [c[0] for c in chunks]
    with ThreadPoolExecutor() as executor:
        chunk_embeddings = list(executor.map(embed_engine, chunk_texts))

    # Expand back to per-turn flat lists (grouped turns share the same embedding)
    turn_texts_out: List[str] = []
    turn_indices_out: List[int] = []
    embeddings_out: List[Any] = []

    tidx_to_text = {tidx: txt for tidx, txt in turns_data}
    for (_, turn_idxs), emb in zip(chunks, chunk_embeddings):
        for tidx in turn_idxs:
            turn_texts_out.append(tidx_to_text[tidx])
            turn_indices_out.append(tidx)
            embeddings_out.append(emb)

    return {
        'embeddings': embeddings_out,
        'turn_texts': turn_texts_out,
        'turn_indices': turn_indices_out
    }

def _extract_causal_graph_from_response(llm_response: str) -> Optional[List[Dict[str, Any]]]:
    """
    Extract the causal graph JSON array from LLM response after **CAUSAL_GRAPH** marker.

    Args:
        llm_response: Raw LLM response text

    Returns:
        Parsed list of causal relationship dicts, or None if extraction fails
    """
    if not llm_response:
        return None

    marker = "**CAUSAL_GRAPH**"
    pos = llm_response.find(marker)
    if pos == -1:
        pos = llm_response.upper().find(marker)
    if pos == -1:
        return None

    after_marker = llm_response[pos + len(marker):].strip()

    # Find the JSON array
    import re
    json_match = re.search(r'(\[.*?\])', after_marker, re.DOTALL)
    if not json_match:
        return None

    try:
        return json.loads(json_match.group(1))
    except json.JSONDecodeError:
        return None


def _process_trajectory_causal(
    trajectory_text: str,
    task: str,
    session_size: int,
    max_context_length: Optional[int],
    call_llm_func: Optional[Callable]
) -> tuple:
    """
    Process trajectory to extract both state memory and causal graph.

    Args:
        trajectory: Parsed trajectory list
        trajectory_text: Formatted trajectory text
        task: Task description
        chunk_size: Maximum size for each chunk
        call_llm_func: LLM function

    Returns:
        Tuple of (state_mem, causal_graph) where causal_graph is a list of
        causal relationship dicts
    """
    if not call_llm_func:
        return None, None

    total_chars = len(trajectory_text)
    safe_cap = int(max_context_length * 0.8) if max_context_length else None
    effective_session_size = min(session_size, safe_cap) if safe_cap else session_size
    effective_session_size = max(1, effective_session_size)
    accumulated_state = ""
    all_causal_edges: List[Dict[str, Any]] = []

    chunks = (
        [{'text': trajectory_text}]
        if total_chars <= effective_session_size
        else _build_trajectory_turn_chunks(
            trajectory_text,
            effective_session_size,
            hard_cap_size=safe_cap
        )
    )

    for chunk in chunks:
        chunk_text = chunk['text']
        previous_state_text = f"Previous State Memory:\n{accumulated_state}" if accumulated_state else ""

        causal_prompt = CAUSAL_PROMPT_TEMPLATE.format(
            task=task,
            trajectory_text=chunk_text,
            previous_state_text=previous_state_text
        )

        _, llm_response = call_llm_func(causal_prompt)

        if llm_response:
            # Extract state memory
            chunk_state = extract_state_memory_from_response(llm_response)
            if chunk_state:
                accumulated_state = chunk_state

            # Extract causal graph
            causal_edges = _extract_causal_graph_from_response(llm_response)
            if causal_edges:
                all_causal_edges.extend(causal_edges)

    state_mem = accumulated_state if accumulated_state else None
    causal_graph = all_causal_edges if all_causal_edges else None
    return state_mem, causal_graph

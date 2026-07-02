"""
Memory Retrieval Module for AMA-Agent

Multi-stage retrieval pipeline (Fig. 8 (B)):
  Stage 1 — Async similarity-based top-k node retrieval via embedding engine.
  Stage 2 — LLM sufficiency judgment (CHUNK_SUFFICIENCY_JUDGMENT_PROMPT_TEMPLATE):
             SUFFICIENT  → return immediately.
             NEED_GRAPH  → parse the spec, retrieve adjacent/range/individual turns.
             NEED_CODE   → generate and execute a Python search script.
  Stage 3 — Synthesize all gathered evidence into a final context string.
"""
import json
import re
from typing import Any, Callable, Dict, List, Optional
import time

from .prompt import CHUNK_SUFFICIENCY_JUDGMENT_PROMPT_TEMPLATE
from .utils import (
    _similarity_retrieve,
    _extract_chunks,
    _format_chunks,
    _retrieve_graph_turns,
    _run_keyword_search,
    truncate_trajectory_text,
)


# Sentinel used to pass a direct answer (produced by the sufficiency judgment)
# back to memory_interface so it can short-circuit the answering LLM call.
DIRECT_ANSWER_PREFIX = "<<<AMA_DIRECT_ANSWER>>>"
DIRECT_ANSWER_SUFFIX = "<<<END_AMA_DIRECT_ANSWER>>>"


def _extract_step_numbers(question: str, max_turn: Optional[int] = None) -> List[int]:
    """Return turn indices the question explicitly references.

    Supported phrasings (case-insensitive):
      * single   : "step 5", "turn 5"
      * range    : "between step 5 and step 10", "from step 5 to step 10",
                   "steps 5 to 10", "steps 5-10", "steps 5–10", "turns 5..10"
      * prefix   : "by step 26", "up to step 26", "until step 26", "through step 26"
                   (only when max_turn is given; otherwise we don't know the upper bound)
      * list     : "steps 3, 7, 12"
    """
    nums: set = set()

    # Range forms: "between step X and step Y", "from step X to step Y", "steps X to Y"
    for m in re.finditer(
        r'\b(?:between|from)\s+(?:step|turn)s?\s+(\d+)\s+(?:and|to|through|until|\-|–)\s+(?:step|turn)?\s*(\d+)\b',
        question, re.IGNORECASE,
    ):
        lo, hi = int(m.group(1)), int(m.group(2))
        if lo > hi:
            lo, hi = hi, lo
        nums.update(range(lo, hi + 1))

    # "steps X to Y" / "turns X to Y"
    for m in re.finditer(
        r'\b(?:step|turn)s?\s+(\d+)\s+(?:to|through|until|\-|–)\s+(\d+)\b',
        question, re.IGNORECASE,
    ):
        lo, hi = int(m.group(1)), int(m.group(2))
        if lo > hi:
            lo, hi = hi, lo
        nums.update(range(lo, hi + 1))

    # "steps X-Y" / "steps X–Y" without word "to"
    for m in re.finditer(
        r'\b(?:step|turn)s?\s+(\d+)\s*[–\-]\s*(\d+)\b',
        question, re.IGNORECASE,
    ):
        lo, hi = int(m.group(1)), int(m.group(2))
        if lo > hi:
            lo, hi = hi, lo
        nums.update(range(lo, hi + 1))

    # List form: "steps 3, 7, 12" / "turns 5, 8, 15" — capture the numbers
    # that follow a "step(s)/turn(s)" introducer (comma- or "and"-separated).
    for m in re.finditer(
        r'\b(?:step|turn)s?\s+((?:\d+\s*(?:,|and|\s)\s*)+\d+)\b',
        question, re.IGNORECASE,
    ):
        for n in re.findall(r'\d+', m.group(1)):
            nums.add(int(n))

    # Prefix forms: "by/until/up to/through step N" → 0..N inclusive
    if max_turn is not None:
        for m in re.finditer(
            r'\b(?:by|until|up\s+to|through|before)\s+(?:step|turn)\s+(\d+)\b',
            question, re.IGNORECASE,
        ):
            hi = min(int(m.group(1)), max_turn)
            nums.update(range(0, hi + 1))

    # Single references: "step N" / "turn N"
    for m in re.finditer(r'\b(?:step|turn)\s+(\d+)\b', question, re.IGNORECASE):
        nums.add(int(m.group(1)))

    return sorted(nums)


def _extract_inline_answer(sufficiency_response: str) -> Optional[str]:
    """Extract the inline ANSWER from a SUFFICIENT sufficiency-judgment response."""
    if not sufficiency_response:
        return None
    m = re.search(r'ANSWER\s*:\s*(.+?)\s*$', sufficiency_response, re.DOTALL | re.IGNORECASE)
    if not m:
        return None
    answer = m.group(1).strip()
    # Strip any trailing NEED_GRAPH/NEED_CODE noise if the model concatenated formats.
    answer = re.split(r'\b(?:NEED_GRAPH|NEED_CODE)\s*:', answer, maxsplit=1)[0].strip()
    return answer or None


# ═══════════════════════════════════════════════════════════════════════════════
# Public entry point
# ═══════════════════════════════════════════════════════════════════════════════

def memory_retrieve(
    memory: Dict[str, Any],
    question: str,
    call_llm_func: Callable,
    top_k: int = 5,
    embed_engine: Optional[Any] = None,
    max_context_length: int = 114688,
) -> str:
    """
    Retrieve relevant context from memory for answering a question.

    Pipeline:
      1. Async similarity retrieval → top_k seed chunks stored in ``mem``.
      2. CHUNK_SUFFICIENCY_JUDGMENT_PROMPT_TEMPLATE → SUFFICIENT / NEED_GRAPH / NEED_CODE.
         - SUFFICIENT : return immediately.
         - NEED_CODE  : generate & execute a Python search script via
                        CODE_GENERATION_PROMPT_TEMPLATE.
         - NEED_GRAPH : parse the adjacency / range / index spec from the
                        response and extend ``mem`` with the requested turns.

    Args:
        memory:         Memory dict with keys state_mem, causal_graph, text_mem, embed_mem.
        question:       Question to answer.
        call_llm_func:  Synchronous LLM call: (prompt: str) -> (_, response: str).
        top_k:          Number of top chunks for similarity retrieval (default: 5).
        embed_engine:   EmbeddingEngine whose base_url/port identifies the server;
                        falls back to BM25 when None.

    Returns:
        Context string ready to prepend to the answering prompt.
    """
    state_mem    = memory.get('state_mem', '')
    text_mem     = memory.get('text_mem', {})
    embed_mem    = memory.get('embed_mem')

    state_mem_str   = str(state_mem) if state_mem else ""
    task            = text_mem.get('task', '')
    trajectory_data = text_mem.get('trajectory_data', {})
    trajectory      = trajectory_data.get('trajectory', [])

    # ── Stage 1: Similarity retrieval (top_k) ────────────────────────────────────────
    # embed_engine is a synchronous callable — call it directly, no event loop.
    seed_indices = _similarity_retrieve(question, trajectory, embed_engine, embed_mem, top_k)

    # Internal accumulated memory — extended as additional turns are retrieved.
    mem: List[Dict[str, Any]] = _extract_chunks(trajectory, seed_indices)
    # Track seen turns from Stage 1 onwards to prevent duplicates across all stages.
    existing_turns: set = {c["turn"] for c in mem}
    # Cumulative evidence starts from similarity retrieval and keeps growing.
    extra_evidence_chunks: List[Dict[str, Any]] = list(mem)

    # ── Stage 1b: Pin turns explicitly referenced by step/turn number ────────
    # Supports single ("step 5"), range ("between step 5 and step 10"),
    # prefix ("by step 26"), and list ("steps 3, 7, 12") forms.
    _max_turn = max((t.get('turn_idx', -1) for t in trajectory), default=-1)
    step_nums = _extract_step_numbers(question, max_turn=_max_turn if _max_turn >= 0 else None)
    pinned_chunks: List[Dict[str, Any]] = []
    if step_nums:
        pinned_chunks = _extract_chunks(trajectory, step_nums)
        pinned_new = [c for c in pinned_chunks if c["turn"] not in existing_turns]
        mem.extend(pinned_new)
        extra_evidence_chunks.extend(pinned_new)
        existing_turns.update(c["turn"] for c in pinned_new)

    # ── Stage 2: Sufficiency judgment loop (up to 3 iterations) ──────────────
    # Each NEED_GRAPH response extends ``mem`` with the requested turns and the
    # judgment is re-run with the enriched context.  NEED_CODE or exhausting
    # all iterations breaks out of the loop.
    _MAX_ITERS = 3
    _TIMEOUT = 60.0  # seconds
    extra_evidence: str = ""
    sufficiency_response: str = ""
    _need_code = False
    _deadline = time.monotonic() + _TIMEOUT

    for _iter in range(_MAX_ITERS):
        if time.monotonic() >= _deadline:
            break
        _chunks_str = _format_chunks(mem)
        # Guard against oversized sufficiency prompts: cap the chunks portion.
        _overhead = len(question) + 4096  # template boilerplate + answer budget
        _chunks_budget = max(4096, max_context_length - _overhead)
        if len(_chunks_str) > _chunks_budget:
            _chunks_str = truncate_trajectory_text(_chunks_str, _chunks_budget)
        sufficiency_prompt = CHUNK_SUFFICIENCY_JUDGMENT_PROMPT_TEMPLATE.format(
            query=question,
            retrieved_chunks=_chunks_str,
        )
        _, sufficiency_response = call_llm_func(sufficiency_prompt)
        resp_upper = (sufficiency_response or "").upper()

        # SUFFICIENT — can answer immediately. Reuse the inline ANSWER to skip
        # the second LLM call in memory_interface (problem 4 fix).
        if "SUFFICIENT" in resp_upper and "NEED_" not in resp_upper:
            context = _synthesize(
                state_mem_str=state_mem_str,
                task=task,
                chunks=mem,
                extra_evidence_chunks=extra_evidence_chunks,
                extra_evidence="",
                pinned_chunks=pinned_chunks,
                max_context_length=max_context_length,
            )
            inline_answer = _extract_inline_answer(sufficiency_response or "")
            if inline_answer:
                return (
                    f"{DIRECT_ANSWER_PREFIX}{inline_answer}{DIRECT_ANSWER_SUFFIX}\n"
                    f"{context}"
                )
            return context

        if "NEED_CODE" in resp_upper:
            _need_code = True
            break

        # NEED_GRAPH — retrieve the requested turns and loop.
        new_chunks = _retrieve_graph_turns(trajectory, sufficiency_response or "")
        new_unique = [c for c in new_chunks if c["turn"] not in existing_turns]
        if not new_unique:
            break  # Nothing new to add; stop looping.
        mem.extend(new_unique)
        extra_evidence_chunks.extend(new_unique)
        existing_turns.update(c["turn"] for c in new_unique)

    if _need_code:
        # ── Stage 3a: Code-generation search with retry (up to 3 attempts) ───
        # Each failed attempt feeds its error back to the next code generation.
        _prev_error = ""
        _code_deadline = time.monotonic() + _TIMEOUT
        for _ in range(_MAX_ITERS):
            if time.monotonic() >= _code_deadline:
                break
            _remaining = max(1.0, _code_deadline - time.monotonic())
            extra_evidence = _run_keyword_search(
                trajectory_data=trajectory_data,
                question=question,
                task=task,
                call_llm_func=call_llm_func,
                previous_error=_prev_error,
                timeout=_remaining,
            )
            if not (extra_evidence.startswith("error:") or "Traceback" in extra_evidence):
                break
            _prev_error = extra_evidence  # feed error to next iteration

    # ── Stage 4: Synthesize all evidence ─────────────────────────────────────
    return _synthesize(
        state_mem_str=state_mem_str,
        task=task,
        chunks=mem,
        extra_evidence_chunks=extra_evidence_chunks,
        extra_evidence=extra_evidence,
        pinned_chunks=pinned_chunks,
        max_context_length=max_context_length,
    )


# ═══════════════════════════════════════════════════════════════════════════════
# Context / output builders
# ═══════════════════════════════════════════════════════════════════════════════


def _synthesize(
    state_mem_str: str,
    task: str,
    chunks: List[Dict[str, Any]],
    extra_evidence_chunks: Optional[List[Dict[str, Any]]] = None,
    extra_evidence: str = "",
    pinned_chunks: Optional[List[Dict[str, Any]]] = None,
    max_context_length: int = 114688,
) -> str:
    """Stage 4: assemble all gathered evidence into the final context string.

    State-memory budget is allocated dynamically based on remaining space after
    evidence is laid out, with a soft floor (10%) and ceiling (50%) so neither
    side starves.
    """
    all_chunks = list(chunks)
    if extra_evidence_chunks:
        all_chunks.extend(extra_evidence_chunks)

    # Deduplicate by turn, then sort ascending.
    seen: Dict[int, Dict[str, Any]] = {}
    for c in all_chunks:
        turn = c.get("turn")
        if not isinstance(turn, int):
            continue
        if turn not in seen:
            seen[turn] = c
            continue
        # Keep the richer chunk when duplicate turns appear.
        old = seen[turn]
        old_len = len(str(old.get("action", ""))) + len(str(old.get("observation", "")))
        new_len = len(str(c.get("action", ""))) + len(str(c.get("observation", "")))
        if new_len > old_len:
            seen[turn] = c
    sorted_chunks = sorted(seen.values(), key=lambda x: x["turn"])

    # Build evidence body first so we can size state_mem against what's left.
    evidence_body = _format_chunks(sorted_chunks)
    code_search_section = (
        f"\n\n# Code Search Result\n{extra_evidence}" if extra_evidence else ""
    )

    # Render pinned (step-N) chunks with FULL observation (no truncation) and
    # put them in a separate, prominently-labelled section. The compressed
    # state memory often disagrees with the raw turn — for point-lookup
    # questions ("at step N, what action…?") the raw turn is authoritative.
    pinned_section = ""
    if pinned_chunks:
        # Deduplicate by turn idx and sort ascending.
        pinned_seen: Dict[int, Dict[str, Any]] = {}
        for c in pinned_chunks:
            t = c.get("turn")
            if isinstance(t, int) and t not in pinned_seen:
                pinned_seen[t] = c
        pinned_sorted = sorted(pinned_seen.values(), key=lambda x: x["turn"])
        if pinned_sorted:
            # No obs truncation for pinned turns (cap individually only if huge).
            pinned_body = _format_chunks(pinned_sorted, max_obs_chars=20000)
            pinned_section = (
                "# Pinned Turns (referenced explicitly in the question — TRUST THIS OVER STATE MEMORY)\n"
                + pinned_body
            )

    # Dynamic state_mem budget: floor 10%, ceiling 50%, take what evidence
    # doesn't consume. Reserve overhead for task line + section headers.
    _overhead = len(task) + 1024
    _floor = max(2048, int(max_context_length * 0.10))
    _ceiling = max(_floor, int(max_context_length * 0.50))
    _remaining = (
        max_context_length
        - len(evidence_body)
        - len(code_search_section)
        - len(pinned_section)
        - _overhead
    )
    _state_mem_budget = max(_floor, min(_ceiling, _remaining))
    if len(state_mem_str) > _state_mem_budget:
        state_mem_str = truncate_trajectory_text(state_mem_str, _state_mem_budget)

    evidence_str = (
        "The retrieved evidence from memory is as follows:\n\n"
        + evidence_body
        + code_search_section
    )
    parts = [f"# Task\n{task}"]
    if pinned_section:
        parts.append(pinned_section)
    parts.extend([
        f"# The State memory summary\n{state_mem_str}",
        f"# The Retrieved Evidence: \n{evidence_str}",
    ])

    context = "\n\n".join(parts)
    return truncate_trajectory_text(context, max_context_length)


def _extract_chunks_from_extra_evidence(extra_evidence: str) -> List[Dict[str, Any]]:
    """Parse turn chunks from keyword-search output."""
    if not extra_evidence:
        return []

    text = extra_evidence.strip()
    chunks: List[Dict[str, Any]] = []

    # 1) Prefer JSON output from keyword-search script (most common path).
    try:
        payload = json.loads(text)
    except (TypeError, json.JSONDecodeError):
        payload = None

    if payload is not None:
        def _walk(node: Any) -> None:
            if isinstance(node, dict):
                turn_raw = node.get("turn", node.get("turn_idx"))
                try:
                    turn = int(turn_raw)
                except (TypeError, ValueError):
                    turn = None
                if turn is not None:
                    chunks.append({
                        "turn": turn,
                        "action": str(node.get("action", "")),
                        "observation": str(node.get("observation", "")),
                    })
                for v in node.values():
                    _walk(v)
            elif isinstance(node, list):
                for item in node:
                    _walk(item)

        _walk(payload)
        if chunks:
            return chunks

    # 2) Fallback: parse already-formatted plain text.
    pattern = re.compile(
        r"Turn\s+(-?\d+):\s*\n\s*Action:\s*(.*?)\n\s*Observation:\s*(.*?)(?=\n\s*Turn\s+-?\d+:|\Z)",
        flags=re.IGNORECASE | re.DOTALL,
    )
    for match in pattern.finditer(text):
        chunks.append({
            "turn": int(match.group(1)),
            "action": match.group(2).strip(),
            "observation": match.group(3).strip()[:500],
        })

    return chunks

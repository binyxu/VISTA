#!/usr/bin/env python3
"""Proprioceptive-blindness diagnostic probe (reviewer Q6).

Claim under test: while acting inside an accumulated tool-agent transcript an
LLM cannot perceive its own context STATE (total size, per-block size, which
block is largest) because that state is runtime metadata, not text in the
prompt. VISTA's dashboard supplies exactly those fields.

Anchor: the moment the agent issued its FIRST archive. That is the decision
point: "when archiving was triggered, how badly would a model misjudge its
state if the dashboard were not there?" We snapshot the raw transcript as it
stood just before that archive call.

Leakage control (verified): the runtime dashboard ledger is NOT persisted in
trajectory.messages, so the stored transcript is already dashboard-free except
for three injected artifacts that we strip:
  1. the `CONTEXT MANAGEMENT PROTOCOL:` trailer on the first task message,
  2. `[CONTEXT_LIMIT_REJECTED]` notices (they print exact token counts),
  3. `[ARCHIVED:...]` placeholders.
After stripping, a scan over all archive trajectories shows zero residual
token/budget/usage leaks, and there is no per-block usage annotation.

Conditions:
  noboard  cleaned transcript only.
  board    same transcript with the canonical ledger prepended (the fields the
           dashboard renders), an upper-bound control isolating perception from
           skill.

Each quantity is asked in its OWN call so the measurements stay independent
(asking everything at once lets the model make its answers mutually consistent).
"""
import argparse
import json
import os
import random
import re
import time
import urllib.request
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import tiktoken

ROOT = Path(__file__).resolve().parent
REPO = ROOT.parent
DEFAULT_RUN_DIR = (
    REPO
    / "external/backup/outputs/"
    "inf_self_managed_final_128k_set_config_strict_lc_better_dashboard."
    "generated_gemini-3-flash_RS200000_RR0.5_MC128000_MW0.5_20260609_215555."
    "compact_debug_reconstructable"
)

API_URL = os.environ.get(
    "LOCA_OPENAI_BASE_URL", "https://api.example.com/v1"
).rstrip("/") + "/chat/completions"
API_KEY = os.environ.get("LOCA_OPENAI_API_KEY", "")

BUDGET = 128000
ENC = tiktoken.get_encoding("cl100k_base")


def ntok(text: str) -> int:
    return len(ENC.encode(text, disallowed_special=())) if text else 0


# --------------------------------------------------------------------------- #
# Data loading: anchor at first archive, strip injected dashboard artifacts   #
# --------------------------------------------------------------------------- #
@dataclass
class Block:
    bid: str
    role: str
    content: str
    tokens: int


@dataclass
class Snapshot:
    task: str
    blocks: list = field(default_factory=list)
    total_tokens: int = 0
    archive_idx: int = -1

    @property
    def n(self):
        return len(self.blocks)


def _tc_name(tc):
    return ((tc.get("function") or {}).get("name")) or tc.get("name")


def _is_meta(c: str) -> bool:
    return isinstance(c, str) and (
        "[CONTEXT_LIMIT_REJECTED]" in c or c.lstrip().startswith("[ARCHIVED:")
    )


def _content_of(m: dict) -> str:
    c = m.get("content")
    if not isinstance(c, str):
        c = json.dumps(c, ensure_ascii=False) if c is not None else ""
    if m.get("tool_calls"):
        c = (c + "\n" + json.dumps(m["tool_calls"], ensure_ascii=False)).strip()
    if "CONTEXT MANAGEMENT PROTOCOL:" in c:
        c = c.split("CONTEXT MANAGEMENT PROTOCOL:")[0].strip()
    return c


def discover_archive_snapshots(run_dir: Path, cap_tokens: int = 0) -> list:
    """Anchor each trajectory at its first archive. If cap_tokens > 0, truncate
    the pre-archive context by dropping the trailing blocks once the cumulative
    size would exceed the cap (guards against exceeding any backend window)."""
    import gzip

    snaps = []
    for gz in sorted((run_dir / "files").rglob("trajectory.json.gz")):
        try:
            d = json.load(gzip.open(gz))
        except Exception:
            continue
        msgs = d.get("messages")
        if not isinstance(msgs, list):
            continue
        a = None
        for i, m in enumerate(msgs):
            if any("archive" in str(_tc_name(tc)).lower()
                   for tc in (m.get("tool_calls") or [])):
                a = i
                break
        if a is None:
            continue
        blocks, total = [], 0
        for m in msgs[:a]:
            c = _content_of(m)
            if not c or _is_meta(c):
                continue
            t = ntok(c)
            if cap_tokens and total + t > cap_tokens and blocks:
                break
            blocks.append(Block(f"B{len(blocks)+1}", m.get("role", "?"), c, t))
            total += t
        if len(blocks) < 8:
            continue
        task = gz.parent.parent.name + "/" + gz.parent.name
        snaps.append(Snapshot(task, blocks, total, a))
    return snaps


# --------------------------------------------------------------------------- #
# Ground truth                                                                #
# --------------------------------------------------------------------------- #
def largest_block(snap: Snapshot) -> str:
    return max(snap.blocks, key=lambda b: b.tokens).bid


def turns_ago(snap: Snapshot, bid: str) -> int:
    idx = next(i for i, b in enumerate(snap.blocks) if b.bid == bid)
    return sum(1 for b in snap.blocks[idx + 1:] if b.role == "assistant")


def pick_probe_blocks(snap: Snapshot, k: int, seed: int) -> list:
    rng = random.Random(seed)
    cand = [b.bid for b in snap.blocks[:-1]]
    rng.shuffle(cand)
    return sorted(cand[:k], key=lambda x: int(x[1:]))


def pick_pairs(snap: Snapshot, k: int, seed: int) -> list:
    """Sample k distinct block pairs for the 'which is larger' probe."""
    rng = random.Random(seed + 7)
    ids = [b.bid for b in snap.blocks]
    pairs, seen = [], set()
    tries = 0
    while len(pairs) < k and tries < 200:
        tries += 1
        a, b = rng.sample(ids, 2)
        key = tuple(sorted((a, b), key=lambda x: int(x[1:])))
        if key in seen:
            continue
        seen.add(key)
        pairs.append(key)
    return pairs


def _tok(snap, bid):
    return next(x for x in snap.blocks if x.bid == bid).tokens


# --------------------------------------------------------------------------- #
# Prompt construction                                                         #
# --------------------------------------------------------------------------- #
def render_transcript(snap: Snapshot) -> str:
    out = []
    for b in snap.blocks:
        out.append(f"===== BLOCK {b.bid} | role={b.role} =====")
        out.append(b.content)
    return "\n".join(out)


def render_plain(snap: Snapshot) -> str:
    """Transcript WITHOUT block-id headers: role-delimited only. Used by the
    quote-based recency probe so the model is not handed an ordinal index."""
    out = []
    for b in snap.blocks:
        out.append(f"[{b.role}]")
        out.append(b.content)
    return "\n".join(out)


def sample_quotes(snap: Snapshot, k: int, seed: int) -> list:
    """Pick k passages that each occur exactly once in the plain transcript,
    returned as (label, snippet, bid). Avoids the final block (age 0 trivial)."""
    full = render_plain(snap)
    rng = random.Random(seed + 13)
    cand = [b for b in snap.blocks[:-1] if len(b.content) >= 90]
    rng.shuffle(cand)
    out = []
    for b in cand:
        c = " ".join(b.content.split())
        if len(c) < 90:
            continue
        start = len(c) // 3
        snip = c[start:start + 140].strip()
        if len(snip) < 60:
            snip = c[:140].strip()
        # uniqueness in the plain transcript (whitespace-normalized)
        full_norm = " ".join(full.split())
        if full_norm.count(snip) != 1:
            continue
        out.append((f"P{len(out)+1}", snip, b.bid))
        if len(out) >= k:
            break
    return out


def render_dashboard(snap: Snapshot) -> str:
    lines = [
        f"<budget:token_budget>{BUDGET}</budget:token_budget>  "
        f"used={snap.total_tokens}  free={BUDGET - snap.total_tokens}",
        f"{'ID':<6}{'~Tok':>8}{'Age':>6}  {'Type':<10}{'Status'}",
    ]
    for b in snap.blocks:
        lines.append(
            f"{b.bid:<6}{b.tokens:>8}{turns_ago(snap, b.bid):>5}r  "
            f"{b.role:<10}visible"
        )
    return "\n".join(lines)


QUESTION_TEXT = {
    "total": (
        'Estimate the total number of tokens the entire transcript above '
        'currently occupies. Answer with ONLY this JSON object, no prose:\n'
        '{"total_tokens": <int>}'
    ),
    "largest": (
        'Which single BLOCK above occupies the most tokens? Answer with ONLY '
        'this JSON object, no prose:\n{"largest_block": "B<number>"}'
    ),
    "block": (
        'Estimate the token size of each of these blocks: {blocks}. Answer with '
        'ONLY this JSON object, no prose:\n{{"block_tokens": {{{kv}}}}}'
    ),
    "pairwise": (
        'For each listed pair of blocks, say which block occupies MORE tokens. '
        'Pairs: {pairs}. Answer with ONLY this JSON object, no prose:\n'
        '{{"pairs": {{{kv}}}}}'
    ),
    "turns_ago": (
        'For each of these blocks, state how many model/assistant turns ago it '
        'appeared, counting the most recent model turn as 0: {blocks}. Answer '
        'with ONLY this JSON object, no prose:\n{{"turns_ago": {{{kv}}}}}'
    ),
    "recency": (
        'For each passage below, state how many model/assistant turns ago it '
        'appeared in the conversation above, counting the most recent model '
        'turn as 0.\n{passages}\nAnswer with ONLY this JSON object, no prose:\n'
        '{{"turns_ago": {{{kv}}}}}'
    ),
}

PREAMBLE = (
    "You are an agent operating inside the conversation/transcript shown above. "
    "It is your entire current context window.\n\n"
)


def build_messages(snap, condition, question, probe_blocks=None, pairs=None,
                   quotes=None):
    # recency: noboard uses the id-free plain transcript (no ordinal index
    # handed to the model); board uses the id transcript + dashboard (full info)
    if question == "recency":
        transcript = (render_transcript(snap) if condition == "board"
                      else render_plain(snap))
    else:
        transcript = render_transcript(snap)
    if question == "recency":
        passages = "\n".join(f'{lbl}: "{snip}"' for lbl, snip, _ in quotes)
        kv = ", ".join(f'"{lbl}": <int>' for lbl, _, _ in quotes)
        q = QUESTION_TEXT["recency"].format(passages=passages, kv=kv)
    elif question == "block":
        kv = ", ".join(f'"{b}": <int>' for b in probe_blocks)
        q = QUESTION_TEXT["block"].format(
            blocks=", ".join(probe_blocks), kv=kv
        )
    elif question == "turns_ago":
        kv = ", ".join(f'"{b}": <int>' for b in probe_blocks)
        q = QUESTION_TEXT["turns_ago"].format(
            blocks=", ".join(probe_blocks), kv=kv
        )
    elif question == "pairwise":
        pair_str = ", ".join(f"({a}, {b})" for a, b in pairs)
        kv = ", ".join(f'"{a}|{b}": "B<id>"' for a, b in pairs)
        q = QUESTION_TEXT["pairwise"].format(pairs=pair_str, kv=kv)
    else:
        q = QUESTION_TEXT[question]
    body = transcript
    if condition == "board":
        body = (
            "CONTEXT STATE DASHBOARD (authoritative, machine-generated):\n"
            + render_dashboard(snap)
            + "\n\n"
            + transcript
        )
    # The large prefix (transcript, plus dashboard for board) is identical
    # across the three questions of one snapshot+condition, so we mark it for
    # prompt caching; only the trailing question differs and stays uncached.
    return [{"role": "user", "content": [
        {"type": "text", "text": body, "cache_control": {"type": "ephemeral"}},
        {"type": "text", "text": "\n\n" + PREAMBLE + q},
    ]}]


# --------------------------------------------------------------------------- #
# API                                                                         #
# --------------------------------------------------------------------------- #
def call_model(model, messages, max_retries=5):
    body = json.dumps(
        {"model": model, "messages": messages, "temperature": 0,
         "max_tokens": 4096}
    ).encode()
    for attempt in range(max_retries):
        try:
            req = urllib.request.Request(
                API_URL, data=body,
                headers={"Content-Type": "application/json",
                         "Authorization": "Bearer " + API_KEY,
                         # token-sticky routing keeps the prompt cache warm
                         "Venus-Sticky-Routing": "token"},
            )
            d = json.load(urllib.request.urlopen(req, timeout=240))
            ch = (d.get("choices") or [{}])[0]
            content = (ch.get("message") or {}).get("content") or ""
            return content, (d.get("usage") or {})
        except Exception as e:
            wait = 2 ** attempt
            print(f"    [retry {attempt+1}/{max_retries}] {repr(e)[:80]} wait{wait}s")
            time.sleep(wait)
    return None, {}


def parse_json(text):
    if not text:
        return None
    m = re.search(r"\{.*\}", text, re.S)
    if not m:
        return None
    for cand in (m.group(0), re.sub(r",\s*([}\]])", r"\1", m.group(0))):
        try:
            return json.loads(cand)
        except Exception:
            continue
    return None


def _to_int(x):
    if isinstance(x, (int, float)):
        return int(x)
    if isinstance(x, str):
        m = re.search(r"-?\d[\d,]*", x)
        if m:
            return int(m.group(0).replace(",", ""))
    return None


def rel_err(pred, gt):
    if pred is None or gt is None or gt == 0:
        return None
    return abs(pred - gt) / abs(gt)


# --------------------------------------------------------------------------- #
# Main                                                                         #
# --------------------------------------------------------------------------- #
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run-dir", default=str(DEFAULT_RUN_DIR))
    ap.add_argument("--models", nargs="+", default=["gemini-3-flash-preview"])
    ap.add_argument("--conditions", nargs="+", default=["noboard", "board"])
    ap.add_argument("--questions", nargs="+",
                    default=["total", "block", "pairwise"])
    ap.add_argument("--n-traj", type=int, default=5)
    ap.add_argument("--blocks-per-traj", type=int, default=4)
    ap.add_argument("--pairs-per-traj", type=int, default=6)
    ap.add_argument("--cap-tokens", type=int, default=100000)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default=str(ROOT / "results/probe_results.jsonl"))
    args = ap.parse_args()

    print(f"API_URL = {API_URL}")
    snaps = discover_archive_snapshots(Path(args.run_dir), args.cap_tokens)
    rng = random.Random(args.seed)
    rng.shuffle(snaps)
    snaps = snaps[: args.n_traj]
    print(f"Loaded {len(snaps)} archive-anchored snapshots (cap={args.cap_tokens}):")
    for s in snaps:
        print(f"  {s.task:46s} blocks={s.n:3d} total_tok={s.total_tokens:7d} "
              f"largest={largest_block(s)} archive@msg{s.archive_idx}")

    def _aux(q, si, snap):
        pb = (pick_probe_blocks(snap, args.blocks_per_traj, args.seed + si)
              if q in ("block", "turns_ago") else None)
        pairs = (pick_pairs(snap, args.pairs_per_traj, args.seed + si)
                 if q == "pairwise" else None)
        quotes = (sample_quotes(snap, args.blocks_per_traj, args.seed + si)
                  if q == "recency" else None)
        return pb, pairs, quotes

    out_path = Path(args.out)
    rows = []
    cache_tok = 0
    with out_path.open("w") as fh:
        for model in args.models:
            for cond in args.conditions:
                # snapshot outer, question inner: the 3 questions reuse the same
                # cached transcript prefix back-to-back (warm cache)
                for si, snap in enumerate(snaps):
                    for q in args.questions:
                        pb, pairs, quotes = _aux(q, si, snap)
                        msgs = build_messages(snap, cond, q, pb, pairs, quotes)
                        raw, usage = call_model(model, msgs)
                        ans = parse_json(raw) or {}
                        sc = score(snap, q, ans, pb, pairs, quotes)
                        cr = ((usage.get("prompt_tokens_details") or {}).get("cached_tokens", 0)
                              or usage.get("prompt_cache_hit_tokens", 0) or 0)
                        cache_tok += cr
                        rec = {"model": model, "condition": cond, "question": q,
                               "task": snap.task, "answer": ans,
                               "gt": gt_of(snap, q, pb, pairs, quotes), "score": sc,
                               "probe_blocks": pb,
                               "pairs": [list(p) for p in pairs] if pairs else None,
                               "quotes": [[l, b] for l, _, b in quotes] if quotes else None,
                               "usage": usage, "parsed": bool(ans)}
                        fh.write(json.dumps(rec) + "\n")
                        fh.flush()
                        rows.append(rec)
                        print(f"  {model[:14]:14s}|{cond:7s}|{snap.task[:24]:24s}|"
                              f"{q:8s} {_fmt_score(q, sc)} cache={cr} parsed={bool(ans)}")
    print(f"\nWrote {len(rows)} records to {out_path}. total cached_tokens={cache_tok:,}")


def gt_of(snap, q, pb, pairs=None, quotes=None):
    if q == "total":
        return {"total_tokens": snap.total_tokens}
    if q == "largest":
        return {"largest_block": largest_block(snap), "n_blocks": snap.n}
    if q == "block":
        return {"block_tokens": {b: _tok(snap, b) for b in pb}}
    if q == "turns_ago":
        return {"turns_ago": {b: turns_ago(snap, b) for b in pb}}
    if q == "recency":
        return {"turns_ago": {lbl: turns_ago(snap, bid) for lbl, _, bid in quotes}}
    if q == "pairwise":
        return {"pairs": {f"{a}|{b}": (a if _tok(snap, a) >= _tok(snap, b) else b)
                          for a, b in pairs},
                "ratios": {f"{a}|{b}": round(max(_tok(snap, a), _tok(snap, b))
                                             / max(1, min(_tok(snap, a), _tok(snap, b))), 2)
                           for a, b in pairs}}
    return {}


def score(snap, q, ans, pb, pairs=None, quotes=None):
    if q == "recency":
        ta = ans.get("turns_ago") or {}
        out = []
        for lbl, _, bid in quotes:
            pv = _to_int(ta.get(lbl))
            if pv is not None:
                out.append(abs(pv - turns_ago(snap, bid)))
        return {"turns_abs_errs": out}
    if q == "total":
        return {"total_rel_err": rel_err(_to_int(ans.get("total_tokens")),
                                         snap.total_tokens)}
    if q == "largest":
        pred = ans.get("largest_block")
        mm = re.search(r"B\d+", str(pred)) if pred else None
        pred = mm.group(0) if mm else None
        return {"largest_correct": int(pred == largest_block(snap)),
                "largest_chance": 1.0 / snap.n, "pred": pred}
    if q == "block":
        bt = ans.get("block_tokens") or {}
        errs = [rel_err(_to_int(bt.get(b)), _tok(snap, b)) for b in pb]
        return {"block_rel_errs": [e for e in errs if e is not None]}
    if q == "turns_ago":
        ta = ans.get("turns_ago") or {}
        out = []
        for b in pb:
            pv = _to_int(ta.get(b))
            if pv is not None:
                out.append(abs(pv - turns_ago(snap, b)))
        return {"turns_abs_errs": out}
    if q == "pairwise":
        pr = ans.get("pairs") or {}
        hits, hard_hits, hard_n = [], [], 0
        for a, b in pairs:
            gt = a if _tok(snap, a) >= _tok(snap, b) else b
            pred = pr.get(f"{a}|{b}")
            mm = re.search(r"B\d+", str(pred)) if pred else None
            pred = mm.group(0) if mm else None
            ok = int(pred == gt)
            hits.append(ok)
            ratio = max(_tok(snap, a), _tok(snap, b)) / max(1, min(_tok(snap, a), _tok(snap, b)))
            if ratio < 2.0:
                hard_n += 1
                hard_hits.append(ok)
        return {"pairwise_hits": hits, "pairwise_hard_hits": hard_hits}
    return {}


def _fmt_score(q, sc):
    if q == "total":
        e = sc["total_rel_err"]
        return f"total_rel_err={'  -  ' if e is None else f'{e:5.2f}'}"
    if q == "largest":
        return (f"largest={'OK' if sc['largest_correct'] else 'x'} "
                f"(chance={sc['largest_chance']:.2f}) pred={sc.get('pred')}")
    if q == "block":
        es = sc["block_rel_errs"]
        med = sorted(es)[len(es)//2] if es else None
        return f"block_med_err={'  -  ' if med is None else f'{med:5.2f}'}"
    if q in ("turns_ago", "recency"):
        es = sc["turns_abs_errs"]
        med = sorted(es)[len(es)//2] if es else None
        return f"turns_med_abserr={'  -  ' if med is None else f'{med:4.1f}'} (n={len(es)})"
    if q == "pairwise":
        h = sc["pairwise_hits"]
        hh = sc["pairwise_hard_hits"]
        acc = sum(h) / len(h) if h else None
        hacc = sum(hh) / len(hh) if hh else None
        return (f"pairwise_acc={'-' if acc is None else f'{acc:.2f}'} "
                f"(hard<2x: {'-' if hacc is None else f'{hacc:.2f}'} n={len(hh)})")
    return ""


if __name__ == "__main__":
    main()

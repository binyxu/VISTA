/**
 * SimpleSummarizer — type-aware extractive summarizer.
 *
 * Each block type gets a dedicated heuristic:
 *   bash_output    → last meaningful line + error/status detection
 *   file_read      → filename prefix + first substantive lines
 *   assistant_message → first sentence/paragraph (the agent's answer)
 *   search_result  → first non-empty line
 *   tool_call      → "Call <source>" placeholder (index.js overrides with inferToolMetaFromArgs)
 *   episode/others → keyword-scored extractive fallback
 */

const KEYWORDS = [
  "error", "exception", "failed", "function", "class", "def ", "return",
  "log", "request", "response", "config", "route", "handler", "test",
  "todo", "fix", "current", "final", "decision", "root cause",
];

const ERROR_RE = /error|exception|traceback|fail(ed|ure)?|cannot|invalid|undefined|null pointer/i;
const STATUS_RE = /^\s*(exit|return|status|code|result)[:\s]+[-\d]/i;

export class SimpleSummarizer {
  /**
   * @param {import("./workspace.js").ContextBlock} block
   * @param {string} [taskGoal]
   * @param {number} [maxWords]
   * @returns {string}
   */
  summarizeBlock(block, taskGoal = "", maxWords = 45) {
    switch (block.type) {
      case "bash_output":    return this._summarizeBashOutput(block.content, maxWords);
      case "file_read":      return this._summarizeFileRead(block.source, block.content, maxWords);
      case "file_write":     return `Wrote ${block.source || "file"}`;
      case "assistant_message": return this._summarizeAssistantMessage(block.content, maxWords);
      case "search_result":  return this._firstLine(block.content, maxWords);
      case "tool_call":      return `Call ${block.source}`;
      default:               return this._extractive(block.content, taskGoal, maxWords);
    }
  }

  /**
   * Episode summary: prefer the last assistant_message among the children
   * (the agent's final answer for that episode). Falls back to concatenating
   * child summaries only when no assistant reply exists yet.
   *
   * @param {import("./workspace.js").ContextBlock[]} blocks  child blocks
   * @param {string} taskGoal  episode's user query
   * @param {number} [maxWords]
   * @returns {string}
   */
  summarizeEpisode(blocks, taskGoal = "", maxWords = 90) {
    // Prefer the last assistant answer — it IS the episode summary.
    const assistantMsgs = blocks.filter((b) => b.type === "assistant_message");
    if (assistantMsgs.length > 0) {
      const last = assistantMsgs[assistantMsgs.length - 1];
      return this._summarizeAssistantMessage(last.content, maxWords);
    }

    // Fallback: stitch together whatever child summaries exist.
    const parts = [];
    if (taskGoal) parts.push(`Goal: ${taskGoal}`);
    for (const block of blocks) {
      if (block.summary) parts.push(`${block.id}: ${block.summary}`);
    }
    const text = parts.length > 0
      ? parts.join(" ")
      : blocks.slice(-3).map((b) => b.content).join(" ");
    return this._truncate(text, maxWords);
  }

  // ---------------------------------------------------------------------------
  // Private helpers
  // ---------------------------------------------------------------------------

  /** bash_output: error lines first, then last meaningful line, then first line. */
  _summarizeBashOutput(content, maxWords) {
    const lines = content.split("\n").map((l) => l.trim()).filter(Boolean);
    if (lines.length === 0) return "";

    // Pick up explicit error/failure lines.
    const errorLine = lines.find((l) => ERROR_RE.test(l));
    if (errorLine) return this._truncate(errorLine, maxWords);

    // Pick up explicit status/exit-code lines.
    const statusLine = lines.find((l) => STATUS_RE.test(l));
    if (statusLine) return this._truncate(statusLine, maxWords);

    // Last non-trivial line (often the result or final status).
    const last = [...lines].reverse().find((l) => l.length > 3);
    if (last) return this._truncate(last, maxWords);

    return this._truncate(lines[0], maxWords);
  }

  /** file_read: filename + first substantive lines (skip blanks and shebangs). */
  _summarizeFileRead(source, content, maxWords) {
    const filename = source ? source.split(/[/\\]/).pop() : "";
    const lines = content.split("\n")
      .map((l) => l.trim())
      .filter((l) => l && !l.startsWith("#!") && l !== "---");
    const snippet = lines.slice(0, 3).join(" ");
    const prefix = filename ? `${filename}: ` : "";
    return this._truncate(prefix + snippet, maxWords);
  }

  /** assistant_message: first sentence or first paragraph — the agent's answer. */
  _summarizeAssistantMessage(content, maxWords) {
    // Strip fenced code blocks so a period inside code doesn't become the boundary.
    const stripped = content.replace(/```[\s\S]*?```/g, "[code]").replace(/`[^`]+`/g, "[code]");
    const text = stripped.replace(/\s+/g, " ").trim();
    // If the response has an intro sentence followed by code, take just the intro.
    const codeIdx = text.indexOf("[code]");
    if (codeIdx > 0) {
      return this._truncate(text.slice(0, codeIdx).replace(/[:\s]+$/, "."), maxWords);
    }
    // Otherwise: first sentence ending in . ! ?
    const sentenceEnd = text.search(/[.!?]\s/);
    const candidate = sentenceEnd > 0 ? text.slice(0, sentenceEnd + 1) : text;
    return this._truncate(candidate, maxWords);
  }

  /** First non-empty line, truncated. */
  _firstLine(content, maxWords) {
    const line = content.split("\n").map((l) => l.trim()).find(Boolean) ?? "";
    return this._truncate(line, maxWords);
  }

  /** Original keyword-scored extractive summarizer (fallback). */
  _extractive(content, taskGoal, maxWords) {
    const text = content.replace(/\s+/g, " ").trim();
    if (!text) return "";

    const candidates = content.split(/(?<=[.!?])\s+|\n+/).filter(Boolean);
    const goalWords = taskGoal
      ? taskGoal.toLowerCase().split(/\s+/).filter((w) => w.length > 4)
      : [];

    const scored = candidates
      .map((line) => {
        const clean = line.replace(/\s+/g, " ").trim();
        if (!clean) return null;
        const lower = clean.toLowerCase();
        let score = KEYWORDS.reduce((s, k) => s + (lower.includes(k) ? 1 : 0), 0);
        score += goalWords.reduce((s, w) => s + (lower.includes(w) ? 1 : 0), 0);
        score -= Math.max(0, Math.floor((clean.split(/\s+/).length - 60) / 20));
        return [score, clean];
      })
      .filter(Boolean)
      .sort((a, b) => b[0] - a[0]);

    const chosen = scored.length > 0 && scored[0][0] > 0 ? scored[0][1] : text;
    return this._truncate(chosen, maxWords);
  }

  /** Truncate to maxWords. */
  _truncate(text, maxWords) {
    const words = text.replace(/\s+/g, " ").trim().split(" ");
    return words.slice(0, maxWords).join(" ") + (words.length > maxWords ? "..." : "");
  }
}

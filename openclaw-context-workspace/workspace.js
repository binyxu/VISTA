/**
 * ContextWorkspace — JS port of prototype/context_workspace.py
 *
 * Manages a structured, addressable context for long-horizon agents:
 * - Visible Workspace: blocks currently shown to the agent.
 * - Archive: full-fidelity hidden blocks, still searchable/recoverable.
 * - Dashboard: compact view of context composition and token usage.
 * - Actions: ABSTRACT, HIDE, ASK_ARCHIVE, SHOW_BLOCK, RESTORE, ABSTRACT_AND_HIDE.
 */

import { readFileSync, writeFileSync } from "fs";
import { SimpleSummarizer } from "./summarizer.js";
import { getEmbedFn, cosine } from "./embedder.js";

// ---------------------------------------------------------------------------
// Types (JSDoc only — no TypeScript runtime dependency)
// ---------------------------------------------------------------------------

/**
 * @typedef {"visible"|"hidden"|"pinned"|"dropped"} BlockStatus
 * @typedef {"session"|"episode"|"react_step"|"user_message"|"assistant_message"|"tool_call"|"tool_result"|"file_read"|"file_write"|"bash_output"|"search_result"|"summary"|"scratchpad"} BlockType
 */

/**
 * @typedef {Object} ContextBlock
 * @property {string} id
 * @property {BlockType} type
 * @property {string} content
 * @property {string} source
 * @property {BlockStatus} status
 * @property {string|null} parentId
 * @property {string} summary
 * @property {number} tokens
 * @property {number} createdAt
 * @property {number} updatedAt
 * @property {Record<string, unknown>} metadata
 */

// ---------------------------------------------------------------------------
// Constants
// ---------------------------------------------------------------------------

export const ActionType = Object.freeze({
  ABSTRACT: "ABSTRACT",
  HIDE: "HIDE",
  ASK_ARCHIVE: "ASK_ARCHIVE",
  SHOW_BLOCK: "SHOW_BLOCK",
  RESTORE: "RESTORE",
  RECALL: "RECALL",
  ABSTRACT_AND_HIDE: "ABSTRACT_AND_HIDE",
});

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

export class ContextActionError extends Error {}

/**
 * @param {string} text
 * @returns {number}
 */
export function estimateTokens(text) {
  if (!text) return 0;
  return Math.max(1, Math.ceil(text.split(/\s+/).length * 1.3));
}

/**
 * @param {string} text
 * @param {number} words
 * @returns {string}
 */
function short(text, words = 20) {
  const toks = text.replace(/\s+/g, " ").trim().split(" ");
  return toks.slice(0, words).join(" ") + (toks.length > words ? "..." : "");
}

/**
 * @param {string} text
 * @returns {Set<string>}
 */
function terms(text) {
  return new Set(
    (text.match(/[A-Za-z_][A-Za-z0-9_./-]{2,}/g) || []).map((t) =>
      t.toLowerCase()
    )
  );
}

// ---------------------------------------------------------------------------
// ContextWorkspace
// ---------------------------------------------------------------------------

export class ContextWorkspace {
  /**
   * @param {object} opts
   * @param {number} [opts.tokenBudget]
   * @param {number} [opts.dashboardTopK]
   * @param {SimpleSummarizer} [opts.summarizer]
   */
  constructor({ tokenBudget = 32_000, dashboardTopK = 6, summarizer } = {}) {
    this.tokenBudget = tokenBudget;
    this.dashboardTopK = dashboardTopK;
    this.summarizer = summarizer ?? new SimpleSummarizer();

    /** @type {Map<string, ContextBlock>} */
    this.blocks = new Map();
    /** @type {Map<string, string[]>} */
    this.children = new Map();
    this._nextId = 1;
    /** @type {string|null} */
    this.currentEpisodeId = null;

  }

  // -------------------------------------------------------------------------
  // Block creation
  // -------------------------------------------------------------------------

  _newId(prefix = "B") {
    return `${prefix}${this._nextId++}`;
  }

  /**
   * @param {object} opts
   * @param {BlockType} opts.type
   * @param {string} opts.content
   * @param {string} [opts.source]
   * @param {string|null} [opts.parentId]
   * @param {BlockStatus} [opts.status]
   * @param {string} [opts.summary]
   * @param {Record<string,unknown>} [opts.metadata]
   * @param {string} [opts.blockId]
   * @returns {string} block id
   */
  addBlock({ type, content, source = "", parentId = null, status = "visible", summary = "", metadata = {}, blockId }) {
    const id = blockId ?? this._newId("B");
    if (this.blocks.has(id)) {
      throw new ContextActionError(`Duplicate block id: ${id}`);
    }
    /** @type {ContextBlock} */
    const block = {
      id,
      type,
      content,
      source,
      status,
      parentId,
      summary,
      tokens: estimateTokens(content),
      createdAt: Date.now(),
      updatedAt: Date.now(),
      metadata,
    };
    this.blocks.set(id, block);
    if (parentId) {
      if (!this.children.has(parentId)) this.children.set(parentId, []);
      this.children.get(parentId).push(id);
    }
    return id;
  }

  /**
   * @param {string} userQuery
   * @param {string} [source]
   * @returns {string} episode id
   */
  startEpisode(userQuery, source = "user") {
    const episodeId = this.addBlock({
      type: "episode",
      content: userQuery,
      source,
      status: "visible",
      summary: this.summarizer._truncate(userQuery, 20),
    });
    this.currentEpisodeId = episodeId;
    return episodeId;
  }

  /**
   * @param {string} [content]
   * @returns {string}
   */
  addReactStep(content = "") {
    return this.addBlock({
      type: "react_step",
      content: content || "ReAct step",
      source: "agent",
      parentId: this.currentEpisodeId,
    });
  }

  /**
   * @param {string} toolName
   * @param {unknown} args
   * @param {string|null} [parentId]
   * @returns {string}
   */
  addToolCall(toolName, args, parentId = null) {
    const content = JSON.stringify({ tool: toolName, arguments: args });
    return this.addBlock({
      type: "tool_call",
      content,
      source: toolName,
      parentId: parentId ?? this.currentEpisodeId,
      summary: `Call ${toolName}`,
    });
  }

  /**
   * @param {string} toolName
   * @param {string} content
   * @param {string|null} [parentId]
   * @param {Record<string,unknown>} [metadata]
   * @returns {string}
   */
  addToolResult(toolName, content, parentId = null, metadata = {}) {
    /** @type {BlockType} */
    let blockType = "tool_result";
    if (["read_file", "file_system_read_file", "ide_simulator_read_file"].includes(toolName)) {
      blockType = "file_read";
    } else if (["bash", "run_shell", "compiler", "debugger"].includes(toolName)) {
      blockType = "bash_output";
    } else if (toolName.includes("search")) {
      blockType = "search_result";
    }
    return this.addBlock({
      type: blockType,
      content,
      source: toolName,
      parentId: parentId ?? this.currentEpisodeId,
      metadata,
    });
  }

  // -------------------------------------------------------------------------
  // Queries
  // -------------------------------------------------------------------------

  currentGoal() {
    if (this.currentEpisodeId) {
      const ep = this.blocks.get(this.currentEpisodeId);
      if (ep) return ep.summary || ep.content;
    }
    return "";
  }

  visibleBlocks() {
    return [...this.blocks.values()].filter(
      (b) => b.status === "visible" || b.status === "pinned"
    );
  }

  hiddenBlocks() {
    return [...this.blocks.values()].filter((b) => b.status === "hidden");
  }

  visibleTokenCount() {
    return this.visibleBlocks().reduce((s, b) => s + b.tokens, 0);
  }

  // -------------------------------------------------------------------------
  // Dashboard
  // -------------------------------------------------------------------------

  _tokenBreakdown(blocks) {
    /** @type {Record<string,number>} */
    const out = {};
    for (const b of blocks) out[b.type] = (out[b.type] ?? 0) + b.tokens;
    return out;
  }

  _blockSummaryText(block, fallback = "-") {
    if (block.summary) return block.summary;
    if (block.type === "episode") return short(block.content, 18);
    return fallback;
  }

  _formatBreakdownLines(blocks) {
    const entries = Object.entries(this._tokenBreakdown(blocks)).sort(([a], [b]) => a.localeCompare(b));
    if (entries.length === 0) return ["- (none)"];
    const width = Math.max(...entries.map(([key]) => key.length));
    return entries.map(([key, value]) => `- ${key.padEnd(width)} : ${value}`);
  }

  _formatKeyValueLines(pairs) {
    const width = Math.max(...pairs.map(([key]) => key.length));
    return pairs.map(([key, value]) => `- ${key.padEnd(width)} : ${value}`);
  }

  _dashboardTable(blocks, includeSummary = true) {
    if (blocks.length === 0) return ["(none)"];
    const rows = blocks.map((block) => ({
      id: block.id,
      type: block.type,
      tok: `${block.tokens} tok`,
      state: block.status,
      source: block.source || "n/a",
      summary: includeSummary ? this._blockSummaryText(block) : "-",
    }));
    const widths = {
      id: Math.max(2, ...rows.map((row) => row.id.length)),
      type: Math.max(4, ...rows.map((row) => row.type.length)),
      tok: Math.max(3, ...rows.map((row) => row.tok.length)),
      state: Math.max(5, ...rows.map((row) => row.state.length)),
      source: Math.max(6, ...rows.map((row) => row.source.length)),
    };
    const header = [
      "ID".padEnd(widths.id),
      "Type".padEnd(widths.type),
      "Tok".padEnd(widths.tok),
      "State".padEnd(widths.state),
      "Source".padEnd(widths.source),
      "Summary",
    ].join(" | ");
    const divider = [
      "-".repeat(widths.id),
      "-".repeat(widths.type),
      "-".repeat(widths.tok),
      "-".repeat(widths.state),
      "-".repeat(widths.source),
      "-------",
    ].join("-+-");
    const lines = [header, divider];
    for (const row of rows) {
      lines.push([
        row.id.padEnd(widths.id),
        row.type.padEnd(widths.type),
        row.tok.padStart(widths.tok),
        row.state.padEnd(widths.state),
        row.source.padEnd(widths.source),
        row.summary,
      ].join(" | "));
    }
    return lines;
  }

  miniDashboard() {
    const visible = this.visibleBlocks();
    const hidden = this.hiddenBlocks();
    const largest = [...visible].sort((a, b) => b.tokens - a.tokens).slice(0, this.dashboardTopK);
    const unsummarized = largest.filter((b) => !b.summary && b.type !== "episode");
    const archived = [...hidden].filter((b) => b.summary).slice(-this.dashboardTopK);

    const lines = [
      "# Context Dashboard",
      "",
      "Budget",
      ...this._formatKeyValueLines([
        ["Visible tokens", `${this.visibleTokenCount()} / ${this.tokenBudget}`],
        ["Visible blocks", `${visible.length}`],
        ["Archived blocks", `${hidden.length}`],
      ]),
      "",
      "Visible token spend by type",
      ...this._formatBreakdownLines(visible),
      "",
      "Largest visible blocks",
      ...this._dashboardTable(largest, true),
    ];
    if (unsummarized.length > 0) {
      lines.push("", "Needs summary", ...unsummarized.map((b) => `- ${b.id} (${b.type}, ${b.tokens} tok, source=${b.source || "n/a"})`));
    }
    if (archived.length > 0) {
      lines.push("", "Recently archived", ...this._dashboardTable(archived, true));
    }
    return lines.join("\n");
  }

  fullDashboard() {
    const visible = this.visibleBlocks();
    const hidden = this.hiddenBlocks();
    const scratchpad = [...this.blocks.values()].find((b) => b.type === "scratchpad");
    const usedTokens = this.visibleTokenCount();
    const pct = Math.round((usedTokens / this.tokenBudget) * 100);
    const lines = ["# Context Dashboard", ""];

    // ── 1. Scratchpad first — agent's own mental model ──────────────────────
    lines.push("## Scratchpad");
    if (scratchpad) {
      lines.push(scratchpad.content);
    } else {
      lines.push("(empty — use context_workspace_scratchpad to track your current goals and decisions)");
    }
    lines.push("");

    // ── 2. Situation — orientation at a glance ──────────────────────────────
    const barWidth = 20;
    const filled = Math.round((Math.min(pct, 100) / 100) * barWidth);
    const bar = "█".repeat(filled) + "░".repeat(barWidth - filled);
    lines.push(
      "## Situation",
      `Episode : ${this.currentEpisodeId || "none"}`,
      `Budget  : [${bar}] ${pct}%  (${usedTokens} / ${this.tokenBudget} tokens)`,
      `Archive : ${hidden.length} block${hidden.length !== 1 ? "s" : ""}`,
      "",
    );

    // ── 3. Active context — visible blocks with summary health ──────────────
    const activeBlocks = visible.filter(
      (b) => b.type !== "scratchpad" && b.type !== "session" && b.type !== "episode",
    );
    if (activeBlocks.length > 0) {
      lines.push("## Active Context");
      for (const b of activeBlocks) {
        const summaryPart = b.summary
          ? `"${this.summarizer._truncate(b.summary, 12)}"`
          : "⚠ no summary";
        const src = b.source ? ` | ${b.source}` : "";
        lines.push(`  ${b.id} | ${b.type}${src} | ${summaryPart}`);
      }
      lines.push("");
    }

    // ── 4. Archive grouped by parent episode ────────────────────────────────
    if (hidden.length > 0) {
      lines.push("## Archive");

      // Group hidden non-episode blocks by their parent episode
      const episodeGroups = new Map(); // episodeId -> { ep, blocks[] }
      const ungrouped = [];

      for (const b of hidden) {
        if (b.type === "episode") {
          if (!episodeGroups.has(b.id)) episodeGroups.set(b.id, { ep: b, blocks: [] });
          else episodeGroups.get(b.id).ep = b;
          continue;
        }
        const parentEp = b.parentId ? this.blocks.get(b.parentId) : null;
        if (parentEp && parentEp.type === "episode") {
          if (!episodeGroups.has(parentEp.id)) episodeGroups.set(parentEp.id, { ep: parentEp, blocks: [] });
          episodeGroups.get(parentEp.id).blocks.push(b);
        } else {
          ungrouped.push(b);
        }
      }

      for (const [, { ep, blocks }] of episodeGroups) {
        const epSummary = ep.summary || this.summarizer._truncate(ep.content, 10);
        const sources = [...new Set(blocks.map((b) => b.source).filter(Boolean))].join(", ");
        lines.push(`  ${ep.id} "${epSummary}"${sources ? `  [${sources}]` : ""}`);
      }
      for (const b of ungrouped) {
        const snippet = this.summarizer._truncate(b.summary || b.content, 10);
        lines.push(`  ${b.id} | ${b.type} | ${snippet}`);
      }

      lines.push("  → use recall(query) to search, show_block(id) to read", "");
    }

    // ── 5. Needs Attention — only when there is something to act on ─────────
    const alerts = [];

    const unsummarized = activeBlocks.filter(
      (b) => !b.summary && b.tokens > 20 && b.type !== "episode",
    );
    if (unsummarized.length > 0) {
      alerts.push(
        `${unsummarized.length} block${unsummarized.length > 1 ? "s have" : " has"} no summary` +
        ` — archive with a summary when done: ${unsummarized.map((b) => b.id).join(", ")}`,
      );
    }

    if (pct >= 70) {
      alerts.push(`Budget at ${pct}% — archive completed work to free space`);
    }

    const currentChildren = this.currentEpisodeId
      ? (this.children.get(this.currentEpisodeId) ?? []).length
      : 0;
    if (currentChildren > 8) {
      alerts.push(
        `Current episode has ${currentChildren} blocks — consider archiving completed parts`,
      );
    }

    if (alerts.length > 0) {
      lines.push("## ⚠ Needs Attention");
      for (const a of alerts) lines.push(`  - ${a}`);
    }

    return lines.join("\n");
  }

  _dashboardRow(block, includeSummary = true) {
    const summary = includeSummary ? this._blockSummaryText(block) : "-";
    return `${block.id} | ${block.type} | ${block.tokens} tok | ${block.status} | ${block.source || "n/a"} | ${summary}`;
  }

  assembleVisibleContext(includeDashboard = true, maxBlockTokens = null) {
    const parts = [];
    if (includeDashboard) parts.push(this.miniDashboard());
    for (const block of this.visibleBlocks()) {
      let content = block.content;
      if (maxBlockTokens && block.tokens > maxBlockTokens) {
        content = block.content.split(/\s+/).slice(0, maxBlockTokens).join(" ") + " ... [truncated in visible context]";
      }
      parts.push(`\n<${block.id} type=${block.type} source=${block.source} status=${block.status}>\n${content}\n</${block.id}>`);
    }
    return parts.join("\n");
  }

  // -------------------------------------------------------------------------
  // Actions
  // -------------------------------------------------------------------------

  /**
   * @param {unknown[]} actions
   * @returns {unknown[]}
   */
  async applyActions(actions) {
    const results = [];
    for (const a of actions) {
      results.push(await this.applyAction(typeof a === "object" && a !== null && "action" in a ? a : parseAction(a)));
    }
    return results;
  }

  /**
   * @param {{action: string, target?: string, content?: string, query?: string, reason?: string, maxTokens?: number}} action
   */
  async applyAction(action) {
    switch (action.action) {
      case ActionType.ABSTRACT:
        return this.abstract(action.target, action.content, action.reason);
      case ActionType.HIDE:
        return this.hide(action.target, action.reason);
      case ActionType.ASK_ARCHIVE:
        return this.askArchive(action.query || action.content || "", action.maxTokens);
      case ActionType.SHOW_BLOCK:
        return this.showBlock(action.target, action.maxTokens);
      case ActionType.RESTORE:
        return this.restore(action.target);
      case ActionType.RECALL:
        if (action.target) return this.showBlock(action.target, action.maxTokens);
        return this.askArchive(action.query || "", action.maxTokens);
      case ActionType.ABSTRACT_AND_HIDE: {
        const abstractResult = this.abstract(action.target, action.content, action.reason);
        const hideResult = this.hide(action.target, action.reason || "abstracted and hidden");
        return { action: "ABSTRACT_AND_HIDE", abstract: abstractResult, hide: hideResult };
      }
      default:
        throw new ContextActionError(`Unsupported action: ${action.action}`);
    }
  }

  abstract(target, content, reason = "") {
    if (!target || !this.blocks.has(target)) {
      throw new ContextActionError(`ABSTRACT target not found: ${target}`);
    }
    if (!content || !content.trim()) {
      throw new ContextActionError("ABSTRACT requires non-empty content.");
    }
    const block = this.blocks.get(target);
    block.summary = content.trim();
    block.updatedAt = Date.now();
    block.metadata.abstractReason = reason;
    return { ok: true, action: "ABSTRACT", target, summary: block.summary };
  }

  hide(target, reason = "") {
    if (!target || !this.blocks.has(target)) {
      throw new ContextActionError(`HIDE target not found: ${target}`);
    }
    const block = this.blocks.get(target);
    if (block.status === "pinned") {
      throw new ContextActionError(`Cannot hide pinned block: ${target}`);
    }
    if (block.type === "session") {
      throw new ContextActionError(`Cannot hide block type ${block.type}: ${target}`);
    }
    block.status = "hidden";
    block.updatedAt = Date.now();
    block.metadata.hideReason = reason;
    // Fire-and-forget: compute embedding for future semantic recall
    this._embedBlock(block).catch(() => {});
    return { ok: true, action: "HIDE", target, status: block.status };
  }

  async _embedBlock(block) {
    const embedFn = await getEmbedFn();
    if (!embedFn) return;
    const text = [block.summary, block.content.slice(0, 1500)].filter(Boolean).join(" ");
    block.metadata.embedding = await embedFn(text);
  }

  async askArchive(query, maxTokens = null) {
    if (!query) throw new ContextActionError("ASK_ARCHIVE requires a query.");
    const matches = await this.searchArchive(query, 3);
    const budget = maxTokens ?? 700;
    const answers = [];
    let used = 0;
    for (const block of matches) {
      const snippet = block.summary || short(block.content, 70);
      const snippetTokens = estimateTokens(snippet);
      if (answers.length > 0 && used + snippetTokens > budget) continue;
      used += snippetTokens;
      answers.push({
        blockId: block.id,
        type: block.type,
        source: block.source,
        tokens: block.tokens,
        summaryOrExcerpt: snippet,
        status: block.status,
      });
    }
    return { ok: true, action: "ASK_ARCHIVE", query, answers };
  }

  showBlock(target, maxTokens = null) {
    if (!target || !this.blocks.has(target)) {
      throw new ContextActionError(`SHOW_BLOCK target not found: ${target}`);
    }
    const block = this.blocks.get(target);
    const budget = maxTokens ?? block.tokens;
    const words = block.content.split(/\s+/);
    const content = words.slice(0, budget).join(" ") + (words.length > budget ? " ... [truncated]" : "");
    return {
      ok: true,
      action: "SHOW_BLOCK",
      target,
      block: {
        id: block.id,
        type: block.type,
        source: block.source,
        tokens: block.tokens,
        status: block.status,
        summary: block.summary,
        content,
      },
    };
  }

  restore(target) {
    if (!target || !this.blocks.has(target)) {
      throw new ContextActionError(`RESTORE target not found: ${target}`);
    }
    const block = this.blocks.get(target);
    block.status = "visible";
    block.updatedAt = Date.now();
    return { ok: true, action: "RESTORE", target, status: block.status };
  }

  /**
   * Archive an entire episode and all its children in one operation.
   * @param {string} episodeId
   * @param {string} [summary]
   */
  archiveEpisode(episodeId, summary = "") {
    if (!this.blocks.has(episodeId)) {
      throw new ContextActionError(`Episode not found: ${episodeId}`);
    }
    const ep = this.blocks.get(episodeId);
    if (ep.type !== "episode") {
      throw new ContextActionError(`Block ${episodeId} is not an episode (type=${ep.type})`);
    }
    if (summary) ep.summary = summary;

    const childIds = this.children.get(episodeId) ?? [];
    let hiddenChildren = 0;
    for (const childId of childIds) {
      const child = this.blocks.get(childId);
      if (child && child.status !== "pinned") {
        child.status = "hidden";
        child.updatedAt = Date.now();
        this._embedBlock(child).catch(() => {});
        hiddenChildren++;
      }
    }
    ep.status = "hidden";
    ep.updatedAt = Date.now();
    this._embedBlock(ep).catch(() => {});
    return { ok: true, action: "ARCHIVE_EPISODE", target: episodeId, childrenArchived: hiddenChildren };
  }

  /**
   * Update (or create) the agent's pinned scratchpad block.
   * @param {string} content
   */
  updateScratchpad(content) {
    if (!content || !content.trim()) {
      throw new ContextActionError("Scratchpad content cannot be empty.");
    }
    const existing = [...this.blocks.values()].find((b) => b.type === "scratchpad");
    if (existing) {
      existing.content = content.trim();
      existing.tokens = estimateTokens(content);
      existing.updatedAt = Date.now();
      return { ok: true, action: "SCRATCHPAD", blockId: existing.id };
    }
    const id = this.addBlock({
      type: "scratchpad",
      content: content.trim(),
      source: "agent",
      status: "pinned",
      summary: "Agent working notes",
    });
    return { ok: true, action: "SCRATCHPAD", blockId: id };
  }

  // -------------------------------------------------------------------------
  // Search
  // -------------------------------------------------------------------------

  /**
   * @param {string} query
   * @param {number} [topK]
   * @returns {ContextBlock[]}
   */
  async searchArchive(query, topK = 5) {
    const candidates = [...this.blocks.values()].filter((b) => b.status === "hidden");
    if (candidates.length === 0) return [];

    const embedFn = await getEmbedFn();
    if (embedFn) {
      // Semantic vector search
      const qEmb = await embedFn(query);
      const scored = [];
      for (const block of candidates) {
        // Compute embedding on-the-fly if not yet stored (e.g. blocks hidden before this feature)
        if (!block.metadata.embedding) {
          const text = [block.summary, block.content.slice(0, 1500)].filter(Boolean).join(" ");
          block.metadata.embedding = await embedFn(text);
        }
        scored.push([cosine(qEmb, block.metadata.embedding), block]);
      }
      scored.sort((a, b) => b[0] - a[0]);
      return scored.slice(0, topK).map(([, b]) => b);
    }

    // Keyword fallback (when @xenova/transformers is not installed)
    const qTerms = terms(query);
    const scored = [];
    for (const block of candidates) {
      const text = [block.source, block.summary, block.content.slice(0, 2000)].join(" ");
      const blockTerms = terms(text);
      const overlap = [...qTerms].filter((t) => blockTerms.has(t)).length;
      if (overlap === 0) continue;
      let score = overlap / Math.sqrt(Math.max(1, blockTerms.size));
      if (query.toLowerCase().includes(block.source.toLowerCase()) && block.source) score += 2;
      if (block.summary) score += 0.25;
      scored.push([score, block]);
    }
    scored.sort((a, b) => b[0] - a[0]);
    return scored.slice(0, topK).map(([, b]) => b);
  }

  suggestActions() {
    const notes = [];
    const visible = [...this.visibleBlocks()].sort((a, b) => b.tokens - a.tokens);
    for (const block of visible) {
      if (block.type === "episode") continue;
      if (!block.summary) {
        notes.push(`${block.id} (${block.type}, ${block.tokens} tok) has no summary yet`);
      } else {
        notes.push(`${block.id} (${block.type}, ${block.tokens} tok) has a summary and could be archived if no longer needed`);
      }
      if (notes.length >= this.dashboardTopK) break;
    }
    return notes;
  }

  // -------------------------------------------------------------------------
  // Internal helpers
  // -------------------------------------------------------------------------

  _episodeForBlock(blockId) {
    let current = this.blocks.get(blockId);
    while (current && current.parentId) {
      const parent = this.blocks.get(current.parentId);
      if (parent && parent.type === "episode") return parent.id;
      current = parent;
    }
    return current && current.type === "episode" ? current.id : null;
  }

  _refreshEpisodeSummary(episodeId) {
    const childIds = this.children.get(episodeId) ?? [];
    const children = childIds.map((id) => this.blocks.get(id)).filter(Boolean);
    const episode = this.blocks.get(episodeId);
    if (!episode) return;
    // summarizeEpisode prefers the last assistant_message (agent's final answer).
    // Falls back to child-summary aggregation only when no assistant reply exists yet.
    episode.summary = this.summarizer.summarizeEpisode(children, episode.content);
    episode.updatedAt = Date.now();
  }


  // -------------------------------------------------------------------------
  // Persistence
  // -------------------------------------------------------------------------

  toDict() {
    return {
      tokenBudget: this.tokenBudget,
      nextId: this._nextId,
      currentEpisodeId: this.currentEpisodeId,
      blocks: Object.fromEntries(this.blocks),
      children: Object.fromEntries(this.children),
    };
  }

  /**
   * @param {Record<string,unknown>} data
   * @returns {ContextWorkspace}
   */
  static fromDict(data) {
    const ws = new ContextWorkspace({ tokenBudget: data.tokenBudget ?? 32_000 });
    ws._nextId = data.nextId ?? 1;
    ws.currentEpisodeId = data.currentEpisodeId ?? null;
    ws.blocks = new Map(Object.entries(data.blocks ?? {}));
    ws.children = new Map(
      Object.entries(data.children ?? {}).map(([k, v]) => [k, Array.isArray(v) ? v : []])
    );
    return ws;
  }

  save(path) {
    writeFileSync(path, JSON.stringify(this.toDict(), null, 2), "utf-8");
  }

  static load(path) {
    return ContextWorkspace.fromDict(JSON.parse(readFileSync(path, "utf-8")));
  }
}

// ---------------------------------------------------------------------------
// Action parsing helpers
// ---------------------------------------------------------------------------

/**
 * @param {Record<string,unknown>} obj
 */
export function parseAction(obj) {
  if (!obj || typeof obj !== "object" || !("action" in obj)) {
    throw new ContextActionError("Action object missing 'action'.");
  }
  const actionStr = String(obj.action).toUpperCase();
  if (!Object.values(ActionType).includes(actionStr)) {
    throw new ContextActionError(`Unknown action: ${obj.action}`);
  }
  return {
    action: actionStr,
    target: obj.target ?? obj.block_id ?? null,
    content: obj.content ?? obj.summary ?? null,
    query: obj.query ?? null,
    reason: obj.reason ?? "",
    maxTokens: obj.max_tokens ?? null,
  };
}

/**
 * @param {string} text
 * @returns {ReturnType<typeof parseAction>[]}
 */
export function parseContextActions(text) {
  let raw = text.trim();
  const match = raw.match(/```(?:json)?\s*([\s\S]*?)```/);
  if (match) raw = match[1].trim();
  let obj = JSON.parse(raw);
  if (obj && typeof obj === "object" && "context_actions" in obj) obj = obj.context_actions;
  if (!Array.isArray(obj)) obj = [obj];
  return obj.map(parseAction);
}

/**
 * @param {ContextWorkspace} workspace
 * @returns {string}
 */
export function contextActionPrompt(workspace) {
  return `
Managing context is your primary responsibility during long tasks — a
well-maintained workspace lets you reason accurately without losing track
of what matters. Use the tools below at your own discretion.

Block IDs (e.g. B3, B12) are shown in the Context tree. Always pass the
exact ID string as block_id — never a file path or name.

Context management tools:
- context_workspace_archive(block_id, summary?)
    Archive a block as soon as you finish with it. Do NOT archive blocks
    you are still actively referencing — restoring from archive costs
    more than keeping them visible.

- context_workspace_archive_episode(episode_id, summary?)
    Archive an entire episode immediately after the sub-task is complete,
    before starting the next one. Do not let finished work linger.

- context_workspace_recall(query)
    Semantic search over the archive. Returns summaries and IDs. Follow
    up with context_workspace_show_block if you need the full content.

- context_workspace_show_block(block_id)
    Read the full content of a block without changing its visibility.
    Prefer this over restore when you only need a one-time reference.

- context_workspace_restore(block_id)
    Bring an archived block back to visible context. Use when you need
    to actively reference the block across multiple steps.

- context_workspace_scratchpad(content)
    Update your persistent working notes (always visible in the dashboard).
    Update it whenever you complete a step, make a decision, or shift
    focus — not just at the start or end of a task. Replaces previous
    version each time.

Guidelines:
- Archive immediately after finishing with a block — do not batch.
- Archive episodes before starting the next sub-task.
- Update the scratchpad when direction or decisions change.
- When you need something from the past: recall first, then show_block.

Current workspace state:
${workspace.fullDashboard()}
`.trim();
}

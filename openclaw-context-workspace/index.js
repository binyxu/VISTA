/**
 * OpenClaw Context Workspace Plugin — main entry point.
 *
 * Fixes over the initial prototype:
 * - Tool registration happens at top-level register(api), not inside the context-engine factory.
 * - Session state is stored in a module-level map keyed by sessionId/sessionKey.
 * - assemble()/afterTurn() both sync transcript messages into the workspace so the
 *   plugin still works even if ingest() is skipped or engine instances are short-lived.
 */

import { existsSync } from "fs";
import { delegateCompactionToRuntime, buildMemorySystemPromptAddition } from "openclaw/plugin-sdk/core";
import { inferToolMetaFromArgs } from "openclaw/plugin-sdk/agent-harness-runtime";
import { prepareSimpleCompletionModel, completeWithPreparedSimpleCompletionModel } from "openclaw/plugin-sdk/agent-runtime";
import { ContextWorkspace, estimateTokens, contextActionPrompt } from "./workspace.js";
import { makeArchiveTool, makeArchiveEpisodeTool, makeRecallTool, makeRestoreTool, makeScratchpadTool, makeShowBlockTool } from "./tools.js";

/** @type {Map<string, { workspace: ContextWorkspace, ingestedIds: Set<string>, statePath: string | null }>} */
const sessionStores = new Map();

function resolvePluginConfig(config, pluginConfig = {}) {
  const entryConfig = config?.plugins?.entries?.["context-workspace"]?.config ?? {};
  const merged = { ...(pluginConfig ?? {}), ...entryConfig };
  return {
    mode: merged.mode === "legacy" ? "legacy" : "workspace",
    tokenBudget: typeof merged.tokenBudget === "number" ? merged.tokenBudget : 32_000,
    dashboardTopK: typeof merged.dashboardTopK === "number" ? merged.dashboardTopK : 6,
    summarizerModel: typeof merged.summarizerModel === "string" ? merged.summarizerModel : null,
  };
}

// Cached live reference to OpenClawConfig — set in register(), used by afterTurn.
let _runtimeConfig = null;

function getSessionRef({ sessionId, sessionKey }) {
  return sessionId ?? sessionKey ?? "__default__";
}

function getOrCreateSessionStore(sessionRef, config) {
  let store = sessionStores.get(sessionRef);
  if (!store) {
    store = {
      workspace: new ContextWorkspace({
        tokenBudget: config.tokenBudget,
        dashboardTopK: config.dashboardTopK,
      }),
      ingestedIds: new Set(),
      statePath: null,
    };
    sessionStores.set(sessionRef, store);
  } else {
    store.workspace.tokenBudget = config.tokenBudget;
    store.workspace.dashboardTopK = config.dashboardTopK;
  }
  return store;
}

function ensureLoaded(store, sessionFile, logger) {
  const statePath = sessionFile + ".workspace.json";
  if (store.statePath === statePath) return;

  if (existsSync(statePath)) {
    try {
      const loaded = ContextWorkspace.load(statePath);
      store.workspace = loaded;
      store.ingestedIds = new Set(
        [...loaded.blocks.values()]
          .map((block) => block?.metadata?.originalMessage)
          .filter(Boolean)
          .map((message) => JSON.stringify(message))
      );
      logger?.info?.(`[context-workspace] Loaded workspace state from ${statePath}`);
    } catch (err) {
      logger?.warn?.(`[context-workspace] Failed to load workspace state from ${statePath}: ${err.message}`);
    }
  }

  store.statePath = statePath;
}

/**
 * @param {import("@mariozechner/pi-agent-core").AgentMessage} message
 * @returns {{ kind: "episode" | "block", type?: import("./workspace.js").BlockType, content: string, source: string, originalMessage: unknown }}
 */
function classifyMessage(message) {
  const role = message?.role;

  if (role === "user") {
    const content = message.content;
    if (Array.isArray(content)) {
      // Anthropic format: tool results nested inside a user message
      const toolResults = content.filter((c) => c.type === "tool_result");
      if (toolResults.length > 0) {
        const toolName = toolResults[0]?.tool_use_id ?? "tool";
        const text = toolResults
          .map((c) => {
            if (typeof c.content === "string") return c.content;
            if (Array.isArray(c.content)) return c.content.map((x) => x.text ?? "").join("\n");
            return "";
          })
          .join("\n");
        return {
          kind: "block",
          type: inferToolResultType(toolName),
          content: text,
          source: toolName,
          originalMessage: message,
        };
      }
      const text = content.map((c) => c.text ?? "").join("\n");
      return { kind: "episode", content: text, source: "user", originalMessage: message };
    }
    return {
      kind: "episode",
      content: typeof content === "string" ? content : "",
      source: "user",
      originalMessage: message,
    };
  }

  // OpenClaw format: tool results are top-level messages with role "toolResult"
  if (role === "toolResult") {
    const toolName = message.toolName ?? "tool";
    const content = message.content ?? [];
    const text = Array.isArray(content)
      ? content.map((c) => c.text ?? c.content ?? "").join("\n")
      : String(content);
    return {
      kind: "block",
      type: inferToolResultType(toolName),
      content: text,
      source: toolName,
      originalMessage: message,
    };
  }

  if (role === "assistant") {
    const content = message.content;
    if (Array.isArray(content)) {
      // Support both Anthropic format (type: "tool_use") and OpenClaw format (type: "toolCall")
      const toolCalls = content.filter((c) => c.type === "tool_use" || c.type === "toolCall");
      if (toolCalls.length > 0) {
        const first = toolCalls[0];
        const toolName = first.name ?? "tool";
        // Normalize to {name, input} so autoSummary / inferToolMetaFromArgs works uniformly
        const normalized = toolCalls.map((c) => ({
          name: c.name ?? "tool",
          input: c.input ?? c.arguments ?? {},
        }));
        return {
          kind: "block",
          type: "tool_call",
          content: JSON.stringify(normalized),
          source: toolName,
          originalMessage: message,
        };
      }
      return {
        kind: "block",
        type: "assistant_message",
        content: content.map((c) => c.text ?? "").join("\n"),
        source: "assistant",
        originalMessage: message,
      };
    }
    return {
      kind: "block",
      type: "assistant_message",
      content: typeof content === "string" ? content : "",
      source: "assistant",
      originalMessage: message,
    };
  }

  return {
    kind: "block",
    type: "session",
    content: JSON.stringify(message),
    source: role ?? "system",
    originalMessage: message,
  };
}

function inferToolResultType(toolName) {
  const name = String(toolName ?? "tool").toLowerCase();
  // file reads: OpenClaw uses "file_fetch", "read", "dir_fetch"; Anthropic uses "read"
  if (name === "file_fetch" || name === "dir_fetch" || name.includes("read") || name === "file") return "file_read";
  // bash/exec: OpenClaw uses "exec", "process"; Anthropic uses "bash"
  if (name === "exec" || name === "process" || name.includes("bash") || name.includes("shell") || name.includes("run")) return "bash_output";
  // search: OpenClaw uses "dir_list", "web_search"; Anthropic uses "glob", "grep"
  if (name === "dir_list" || name.includes("search") || name.includes("grep") || name.includes("glob")) return "search_result";
  // writes/edits
  if (name === "file_write" || name.includes("write") || name.includes("edit") || name.includes("patch")) return "file_write";
  return "tool_result";
}

/**
 * Summarize episodes with a real LLM using the configured cheap model.
 *
 * Only episodes that have at least one assistant_message child and have not
 * been LLM-summarized yet (metadata.llmSummarized !== true) are processed.
 * Runs after each turn so the living state stays fresh.
 *
 * Model ref format: "provider/modelId"  e.g. "anthropic/claude-haiku-4-5-20251001"
 *
 * @param {{ workspace: import("./workspace.js").ContextWorkspace }} store
 * @param {import("openclaw/plugin-sdk/core").OpenClawConfig} cfg
 * @param {string} modelRef
 * @param {unknown} logger
 */
async function summarizeEpisodesWithLlm(store, cfg, modelRef, logger) {
  const [provider, modelId] = modelRef.includes("/")
    ? modelRef.split("/", 2)
    : ["anthropic", modelRef];

  let prepared;
  try {
    prepared = await prepareSimpleCompletionModel({ cfg, provider, modelId });
    if ("error" in prepared) {
      logger?.warn?.(`[context-workspace] summarizer model unavailable (${prepared.error})`);
      return;
    }
  } catch (err) {
    logger?.warn?.(`[context-workspace] summarizer model prep failed: ${err.message}`);
    return;
  }

  const episodes = [...store.workspace.blocks.values()].filter((b) => b.type === "episode");
  let updated = false;

  for (const episode of episodes) {
    if (episode.metadata?.llmSummarized) continue;

    const childIds = store.workspace.children.get(episode.id) ?? [];
    const children = childIds.map((id) => store.workspace.blocks.get(id)).filter(Boolean);
    const assistantMsgs = children.filter((b) => b.type === "assistant_message");
    if (assistantMsgs.length === 0) continue;

    const lastAnswer = assistantMsgs[assistantMsgs.length - 1];
    const content = lastAnswer.content.slice(0, 3000);

    try {
      const result = await completeWithPreparedSimpleCompletionModel({
        model: prepared.model,
        auth: prepared.auth,
        cfg,
        context: {
          systemPrompt:
            "You are a concise summarizer. Summarize the assistant response in 1–2 sentences, capturing the key answer or outcome. Reply with only the summary, no preamble.",
          messages: [{ role: "user", content: [{ type: "text", text: content }] }],
        },
        options: { maxTokens: 80 },
      });

      const summaryText = result.content
        .filter((c) => c.type === "text")
        .map((c) => c.text)
        .join("")
        .trim();

      if (summaryText) {
        episode.summary = summaryText;
        episode.metadata = { ...episode.metadata, llmSummarized: true };
        episode.updatedAt = Date.now();
        updated = true;
        logger?.info?.(`[context-workspace] LLM episode summary for ${episode.id}: ${summaryText.slice(0, 80)}`);
      }
    } catch (err) {
      logger?.warn?.(`[context-workspace] LLM summary failed for ${episode.id}: ${err.message}`);
    }
  }

}

/**
 * Auto-generate a summary for a classified block before it is stored.
 * - tool_call  → inferToolMetaFromArgs (OpenClaw's compact label, e.g. "Read (~/foo.py)")
 * - other types → SimpleSummarizer's type-aware heuristics
 * Agent ABSTRACT actions will override these with LLM-quality summaries later.
 *
 * @param {{ type: string, content: string, source: string }} classified
 * @param {import("./summarizer.js").SimpleSummarizer} summarizer
 * @returns {string | undefined}
 */
function autoSummary(classified, summarizer) {
  if (classified.type === "tool_call") {
    try {
      const calls = JSON.parse(classified.content);
      const first = calls[0];
      if (first?.name != null && first?.input != null) {
        const label = inferToolMetaFromArgs(first.name, first.input);
        if (label) return label;
      }
    } catch {
      // fall through
    }
    return `Call ${classified.source}`;
  }

  // For result/message blocks, use the type-aware summarizer heuristics.
  const AUTO_SUMMARIZE_TYPES = new Set([
    "assistant_message", "bash_output", "file_read", "file_write", "search_result", "tool_result",
  ]);
  if (AUTO_SUMMARIZE_TYPES.has(classified.type) && classified.content?.trim()) {
    return summarizer.summarizeBlock(
      { type: classified.type, content: classified.content, source: classified.source },
    );
  }

  return undefined;
}

function ingestOneMessage(store, message) {
  const msgKey = JSON.stringify(message);
  if (store.ingestedIds.has(msgKey)) return false;
  store.ingestedIds.add(msgKey);

  const classified = classifyMessage(message);
  if (!classified.content?.trim() && classified.kind !== "block") return false;

  if (classified.kind === "episode") {
    const episodeId = store.workspace.startEpisode(classified.content || "(empty)", classified.source);
    const block = store.workspace.blocks.get(episodeId);
    if (block) block.metadata = { ...block.metadata, originalMessage: classified.originalMessage };
    return true;
  }

  if (!classified.content?.trim() && classified.type !== "tool_call" && classified.type !== "session") {
    return false;
  }

  store.workspace.addBlock({
    type: classified.type,
    content: classified.content || "(empty)",
    source: classified.source,
    parentId: store.workspace.currentEpisodeId,
    summary: autoSummary(classified, store.workspace.summarizer),
    metadata: { originalMessage: classified.originalMessage },
  });
  return true;
}

function syncMessagesToWorkspace(store, messages = []) {
  let ingestedCount = 0;
  for (const message of messages) {
    if (ingestOneMessage(store, message)) ingestedCount += 1;
  }
  return ingestedCount;
}

function buildEngine(config, logger) {
  const ownsCompaction = config.mode === "workspace";

  return {
    info: {
      id: "context-workspace",
      name: "Context Workspace Engine",
      version: "0.1.1",
      ownsCompaction,
    },

    async bootstrap({ sessionId, sessionKey, sessionFile }) {
      if (config.mode === "legacy") return { bootstrapped: false, reason: "legacy mode" };
      const sessionRef = getSessionRef({ sessionId, sessionKey });
      const store = getOrCreateSessionStore(sessionRef, config);
      ensureLoaded(store, sessionFile, logger);
      return { bootstrapped: true, importedMessages: store.workspace.blocks.size };
    },

    async ingest({ sessionId, sessionKey, message, isHeartbeat }) {
      if (isHeartbeat || config.mode === "legacy") return { ingested: false };
      const sessionRef = getSessionRef({ sessionId, sessionKey });
      const store = getOrCreateSessionStore(sessionRef, config);
      return { ingested: ingestOneMessage(store, message) };
    },

    async ingestBatch({ sessionId, sessionKey, messages, isHeartbeat }) {
      if (isHeartbeat || config.mode === "legacy") return { ingestedCount: 0 };
      const sessionRef = getSessionRef({ sessionId, sessionKey });
      const store = getOrCreateSessionStore(sessionRef, config);
      return { ingestedCount: syncMessagesToWorkspace(store, messages) };
    },

    async assemble({ sessionId, sessionKey, sessionFile, messages, tokenBudget, availableTools, citationsMode }) {
      if (config.mode === "legacy") {
        const estimatedTokens = messages.reduce((sum, message) => sum + estimateTokens(JSON.stringify(message)), 0);
        return { messages, estimatedTokens };
      }

      const sessionRef = getSessionRef({ sessionId, sessionKey });
      const store = getOrCreateSessionStore(sessionRef, config);
      if (sessionFile) ensureLoaded(store, sessionFile, logger);
      syncMessagesToWorkspace(store, messages);

      const effectiveBudget = tokenBudget ?? store.workspace.tokenBudget;
      while (store.workspace.visibleTokenCount() > effectiveBudget) {
        const candidates = store.workspace
          .visibleBlocks()
          .filter((block) => block.type !== "session" && block.status !== "pinned")
          .sort((a, b) => a.createdAt - b.createdAt);
        if (candidates.length === 0) break;
        store.workspace.hide(candidates[0].id, "auto-compaction: over token budget");
      }

      const assembledMessages = store.workspace
        .visibleBlocks()
        .filter((block) => block.metadata?.originalMessage)
        .sort((a, b) => a.createdAt - b.createdAt)
        .map((block) => block.metadata.originalMessage);

      const memoryAddition = buildMemorySystemPromptAddition({
        availableTools: availableTools ?? new Set(),
        citationsMode,
      });
      const workspaceInstructions = contextActionPrompt(store.workspace);
      const systemPromptAddition = [workspaceInstructions, memoryAddition].filter(Boolean).join("\n\n");

      return {
        messages: assembledMessages,
        estimatedTokens: store.workspace.visibleTokenCount(),
        systemPromptAddition,
        promptAuthority: "assembled",
      };
    },

    async compact({ sessionId, sessionKey, sessionFile, ...rest }) {
      if (config.mode === "legacy") {
        return delegateCompactionToRuntime({ sessionId, sessionKey, sessionFile, ...rest });
      }

      const sessionRef = getSessionRef({ sessionId, sessionKey });
      const store = getOrCreateSessionStore(sessionRef, config);
      if (sessionFile) ensureLoaded(store, sessionFile, logger);

      const tokensBefore = store.workspace.visibleTokenCount();
      const episodes = [...store.workspace.blocks.values()]
        .filter((block) => block.type === "episode")
        .sort((a, b) => a.createdAt - b.createdAt);
      const recentEpisodeIds = new Set(episodes.slice(-3).map((episode) => episode.id));

      let hidCount = 0;
      for (const block of store.workspace.visibleBlocks()) {
        if (block.type === "session" || block.status === "pinned") continue;
        if (block.type === "episode" && recentEpisodeIds.has(block.id)) continue;
        const episodeId = store.workspace._episodeForBlock(block.id);
        if (episodeId && recentEpisodeIds.has(episodeId)) continue;
        try {
          store.workspace.hide(block.id, "compact: old episode");
          hidCount += 1;
        } catch {
          // ignore blocks that cannot be hidden
        }
      }

      return {
        ok: true,
        compacted: hidCount > 0,
        reason: hidCount > 0 ? `Hid ${hidCount} blocks from old episodes` : "Nothing to compact",
        result: {
          tokensBefore,
          tokensAfter: store.workspace.visibleTokenCount(),
        },
      };
    },

    async afterTurn({ sessionId, sessionKey, sessionFile, messages }) {
      if (config.mode === "legacy") return;
      const sessionRef = getSessionRef({ sessionId, sessionKey });
      const store = getOrCreateSessionStore(sessionRef, config);
      ensureLoaded(store, sessionFile, logger);
      syncMessagesToWorkspace(store, messages);

      if (config.summarizerModel && _runtimeConfig) {
        await summarizeEpisodesWithLlm(store, _runtimeConfig, config.summarizerModel, logger);
      }

      try {
        store.workspace.save(sessionFile + ".workspace.json");
      } catch (err) {
        logger?.warn?.(`[context-workspace] Failed to save workspace state: ${err.message}`);
      }
    },
  };
}

function resolveToolStore(toolCtx, pluginConfig) {
  const config = resolvePluginConfig(toolCtx.getRuntimeConfig?.() ?? toolCtx.runtimeConfig ?? toolCtx.config, pluginConfig);
  if (config.mode !== "workspace") return null;
  const sessionRef = getSessionRef({ sessionId: toolCtx.sessionId, sessionKey: toolCtx.sessionKey });
  return getOrCreateSessionStore(sessionRef, config);
}

/**
 * @param {import("openclaw/plugin-sdk/core").OpenClawPluginApi} api
 */
export function register(api) {
  _runtimeConfig = api.config;
  const tools = [
    makeArchiveTool,
    makeArchiveEpisodeTool,
    makeRecallTool,
    makeRestoreTool,
    makeScratchpadTool,
    makeShowBlockTool,
  ];
  for (const factory of tools) {
    api.registerTool((toolCtx) => {
      const store = resolveToolStore(toolCtx, api.pluginConfig);
      return store ? factory(() => store.workspace) : null;
    });
  }

  api.registerContextEngine("context-workspace", (ctx) => {
    const config = resolvePluginConfig(ctx.config, api.pluginConfig);
    api.logger?.info?.(`[context-workspace] Starting in ${config.mode} mode (budget: ${config.tokenBudget} tokens)`);
    return buildEngine(config, api.logger);
  });
}

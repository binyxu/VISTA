/**
 * Agent-callable context management tools.
 *
 * Tool set (6 tools):
 *  1. context_workspace_archive        — summarize + hide a block (was ABSTRACT+HIDE)
 *  2. context_workspace_archive_episode — collapse an entire episode
 *  3. context_workspace_recall         — semantic search in archive
 *  4. context_workspace_restore        — bring a hidden block back to visible
 *  5. context_workspace_scratchpad     — update the agent's working notes
 *  6. context_workspace_show_block     — read full content of a known block ID
 */

import { ContextActionError } from "./workspace.js";

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

function okResult(result) {
  return {
    content: [{ type: "text", text: JSON.stringify(result, null, 2) }],
    details: result,
  };
}

function toolError(message) {
  throw new Error(message);
}

function wrap(fn) {
  return async (_toolCallId, params) => {
    try {
      return okResult(await fn(params));
    } catch (err) {
      if (err instanceof ContextActionError) toolError(err.message);
      throw err;
    }
  };
}

// ---------------------------------------------------------------------------
// Tool factories
// ---------------------------------------------------------------------------

/**
 * context_workspace_archive — summarize and archive a block in one step.
 * Replaces the previous ABSTRACT + HIDE two-step pattern.
 */
export function makeArchiveTool(getWorkspace) {
  return {
    name: "context_workspace_archive",
    label: "Archive context block",
    description:
      "Summarize and archive a block in one step. " +
      "Use immediately after finishing with a block — do not wait. " +
      "Do NOT archive blocks you are still actively referencing; restoring from archive costs more than keeping them visible. " +
      "The summary and full content are both preserved; use context_workspace_recall to find it later.",
    parameters: {
      type: "object",
      properties: {
        block_id: {
          type: "string",
          description: "ID of the block to archive (e.g. B3, B12). Find IDs in the Context tree.",
        },
        summary: {
          type: "string",
          description: "Concise summary of the key facts in this block (1-3 sentences). If omitted, an automatic summary is used.",
        },
      },
      required: ["block_id"],
    },
    execute: wrap(({ block_id, summary = "" }) => {
      if (!block_id) toolError("block_id is required");
      const ws = getWorkspace();
      if (summary) ws.abstract(block_id, summary);
      return ws.hide(block_id, "archived");
    }),
  };
}

/**
 * context_workspace_archive_episode — collapse an entire episode to one summary line.
 * Use when a sub-task is fully complete and you no longer need its detail in view.
 */
export function makeArchiveEpisodeTool(getWorkspace) {
  return {
    name: "context_workspace_archive_episode",
    label: "Archive entire episode",
    description:
      "Archive an entire episode and all its child blocks in one operation. " +
      "Use immediately after a sub-task is complete, before starting the next one — do not let finished work linger in visible context. " +
      "The episode collapses to a single summary line in the dashboard; all content remains searchable via context_workspace_recall.",
    parameters: {
      type: "object",
      properties: {
        episode_id: {
          type: "string",
          description: "ID of the episode block to archive (e.g. B1, B9). Must be a block of type 'episode'.",
        },
        summary: {
          type: "string",
          description: "One-sentence summary of what was accomplished in this episode.",
        },
      },
      required: ["episode_id"],
    },
    execute: wrap(({ episode_id, summary = "" }) => {
      if (!episode_id) toolError("episode_id is required");
      return getWorkspace().archiveEpisode(episode_id, summary);
    }),
  };
}

/**
 * context_workspace_recall — semantic search over the archive.
 */
export function makeRecallTool(getWorkspace) {
  return {
    name: "context_workspace_recall",
    label: "Recall from archive",
    description:
      "Search the archive semantically for information relevant to a query. " +
      "Returns the most relevant archived block summaries and their IDs. " +
      "Always recall before restoring — you may only need show_block, not a full restore.",
    parameters: {
      type: "object",
      properties: {
        query: {
          type: "string",
          description: "Natural language description of what you are looking for.",
        },
      },
      required: ["query"],
    },
    execute: wrap(({ query }) => {
      if (!query) toolError("query is required");
      return getWorkspace().askArchive(query);
    }),
  };
}

/**
 * context_workspace_restore — bring an archived block back to visible context.
 */
export function makeRestoreTool(getWorkspace) {
  return {
    name: "context_workspace_restore",
    label: "Restore archived block",
    description:
      "Move an archived block back into your visible context. " +
      "Use when you need to actively reference the block across multiple steps. " +
      "For a one-time read, use context_workspace_show_block instead — it is cheaper.",
    parameters: {
      type: "object",
      properties: {
        block_id: {
          type: "string",
          description: "ID of the archived block to restore.",
        },
      },
      required: ["block_id"],
    },
    execute: wrap(({ block_id }) => {
      if (!block_id) toolError("block_id is required");
      return getWorkspace().restore(block_id);
    }),
  };
}

/**
 * context_workspace_scratchpad — update the agent's persistent working notes.
 */
export function makeScratchpadTool(getWorkspace) {
  return {
    name: "context_workspace_scratchpad",
    label: "Update working notes",
    description:
      "Update your persistent project scratchpad — always visible in the dashboard. " +
      "Update it whenever you complete a step, make a decision, or shift focus — not just at the start or end of a task. " +
      "Write the full updated content each time (replaces the previous version).",
    parameters: {
      type: "object",
      properties: {
        content: {
          type: "string",
          description: "Full content of your working notes. Replace the entire scratchpad with this text.",
        },
      },
      required: ["content"],
    },
    execute: wrap(({ content }) => {
      if (!content) toolError("content is required");
      return getWorkspace().updateScratchpad(content);
    }),
  };
}

/**
 * context_workspace_show_block — read the full content of a known block ID.
 */
export function makeShowBlockTool(getWorkspace) {
  return {
    name: "context_workspace_show_block",
    label: "Read block content",
    description:
      "Read the full content of a specific block by its ID, without changing its visibility. " +
      "Prefer this over context_workspace_restore for one-time references — restore only when you need the block visible across multiple steps.",
    parameters: {
      type: "object",
      properties: {
        block_id: {
          type: "string",
          description: "ID of the block to read.",
        },
        max_tokens: {
          type: "integer",
          description: "Maximum tokens of content to return (default: full block).",
        },
      },
      required: ["block_id"],
    },
    execute: wrap(({ block_id, max_tokens }) => {
      if (!block_id) toolError("block_id is required");
      return getWorkspace().showBlock(block_id, max_tokens ?? null);
    }),
  };
}

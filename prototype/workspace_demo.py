#!/usr/bin/env python3
"""Demonstrate Agentic Context Workspace actions."""

from __future__ import annotations

from context_workspace import ContextWorkspace


def main() -> None:
    ws = ContextWorkspace(token_budget=10_000)

    ep = ws.start_episode(
        "Trace how request logging works in the ChromaCanvas API Gateway."
    )
    ws.add_tool_call("list_directory", {"path": "chromacanvas_api_gateway/src/http"}, parent_id=ep)
    ws.add_tool_result(
        "list_directory",
        "request_handler.c\nrouter.c\nserver.c\n",
        parent_id=ep,
    )
    read_id = ws.add_tool_result(
        "read_file",
        """
// request_handler.c
#include "logger.h"

static int parse_request(...) { /* parse method, path, headers */ }
static int authenticate_request(...) { /* set user_id */ }

int handle_request(request_t *req, response_t *res) {
    parse_request(req);
    authenticate_request(req);
    route_request(req, res);
    log_request(req->client_ip, req->method, req->path, req->user_id, res->status);
    return res->status;
}

// Many more implementation details...
"""
        + "\n".join(f"line {i}: internal implementation detail" for i in range(250)),
        parent_id=ep,
        metadata={"path": "chromacanvas_api_gateway/src/http/request_handler.c"},
    )

    print("\n=== Initial dashboard ===")
    print(ws.full_dashboard())
    print("\nVisible tokens:", ws.visible_token_count())

    ws.apply_actions([
        {
            "action": "ABSTRACT",
            "target": read_id,
            "content": "request_handler.c parses and authenticates the request, routes it, then calls log_request with client_ip, method, path, user_id, and response status.",
            "reason": "Captured the request logging flow; raw implementation details can leave active context.",
        },
        {
            "action": "HIDE",
            "target": read_id,
            "reason": "Key facts abstracted; raw file body is no longer needed in visible context.",
        },
    ])

    print("\n=== After ABSTRACT + HIDE ===")
    print(ws.full_dashboard())
    print("\nVisible tokens:", ws.visible_token_count())
    print("\nVisible context preview:\n", ws.assemble_visible_context(max_block_tokens=120)[:1200])

    answer = ws.apply_actions([
        {
            "action": "ASK_ARCHIVE",
            "query": "where is log_request called in request_handler.c",
            "max_tokens": 300,
        }
    ])

    print("\n=== ASK_ARCHIVE(query), no visibility change ===")
    print(answer)
    print(ws.full_dashboard())
    print("\nVisible tokens:", ws.visible_token_count())

    shown = ws.apply_actions([
        {
            "action": "SHOW_BLOCK",
            "target": read_id,
            "max_tokens": 80,
        }
    ])
    print("\n=== SHOW_BLOCK(id), no visibility change ===")
    print(shown)
    print("\nVisible tokens:", ws.visible_token_count())

    ws.apply_actions([
        {
            "action": "RESTORE",
            "target": read_id,
        }
    ])

    print("\n=== RESTORE(id), raw block visible again ===")
    print(ws.full_dashboard())
    print("\nVisible tokens:", ws.visible_token_count())


if __name__ == "__main__":
    main()

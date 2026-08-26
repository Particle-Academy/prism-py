"""MCP server for prism.py, over HTTP, with no dependencies.

It implements exactly the three methods prism-mcp's client speaks in protocol
2026-07-28 - `server/discover`, `tools/list`, `tools/call`. That revision
removed `initialize` and the session, so there is no handshake to implement
and none is offered.

Binds to loopback. An agent that can run this repo's test suite and spend
tokens is remote code execution wearing a friendly name; it has no business
being reachable from anywhere else.
"""

from __future__ import annotations

import json
import os
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Any

from agent import LANGUAGE, TOOLS

PROTOCOL_VERSION = "2026-07-28"
PORT = int(os.environ.get("PRISM_AGENT_PORT", "7412"))
HOST = "127.0.0.1"

PARSE_ERROR = -32700
INVALID_REQUEST = -32600
METHOD_NOT_FOUND = -32601
INVALID_PARAMS = -32602
INTERNAL_ERROR = -32603

MAX_BODY = 8 * 1024 * 1024


class RpcError(Exception):
    def __init__(self, code: int, message: str) -> None:
        super().__init__(message)
        self.code = code


def _definitions() -> list[dict[str, Any]]:
    return [
        {"name": name, "description": tool["description"], "inputSchema": tool["inputSchema"]}
        for name, tool in TOOLS.items()
    ]


def dispatch(method: str | None, params: dict[str, Any] | None) -> dict[str, Any]:
    if method == "server/discover":
        return {
            "protocolVersion": PROTOCOL_VERSION,
            "supportedVersions": [PROTOCOL_VERSION],
            "capabilities": {"tools": {}},
            "serverInfo": {"name": f"prism.{LANGUAGE}", "version": "0.1.0"},
        }

    if method == "tools/list":
        # No pagination: four tools fit in one page and always will.
        # `nextCursor` is null rather than omitted so a client paging
        # generically sees an explicit end.
        return {"tools": _definitions(), "nextCursor": None}

    if method == "tools/call":
        name = (params or {}).get("name")
        tool = TOOLS.get(name or "")

        if tool is None:
            raise RpcError(INVALID_PARAMS, f"unknown tool: {name}")

        try:
            result = tool["handler"]((params or {}).get("arguments") or {})
        except Exception as error:
            # A tool that fails returns isError on the RESULT, not a JSON-RPC
            # error. The distinction matters: the call succeeded, the work did
            # not.
            return {"content": [{"type": "text", "text": str(error)}], "isError": True}

        return {
            # Structured AND text. The structured form is what a caller should
            # act on; the text form is what survives being pasted into a model
            # that only reads content parts.
            "content": [{"type": "text", "text": json.dumps(result, indent=2)}],
            "structuredContent": result,
            "isError": False,
        }

    raise RpcError(METHOD_NOT_FOUND, f"unknown method: {method}")


class Handler(BaseHTTPRequestHandler):
    protocol_version = "HTTP/1.1"

    def log_message(self, *_: Any) -> None:
        """Quiet by default. The tools report what they did; the transport need not."""

    def _send(self, status: int, body: dict[str, Any]) -> None:
        payload = json.dumps(body).encode("utf-8")
        self.send_response(status)
        self.send_header("content-type", "application/json")
        self.send_header("content-length", str(len(payload)))
        self.send_header("MCP-Protocol-Version", PROTOCOL_VERSION)
        self.end_headers()
        self.wfile.write(payload)

    def do_POST(self) -> None:
        length = int(self.headers.get("content-length") or 0)

        if length > MAX_BODY:
            # An unbounded read is a memory exhaustion away from taking the
            # lane down.
            self._send(
                413,
                {
                    "jsonrpc": "2.0",
                    "id": None,
                    "error": {"code": INVALID_REQUEST, "message": "body too large"},
                },
            )
            return

        try:
            message = json.loads(self.rfile.read(length) or b"{}")
        except json.JSONDecodeError:
            self._send(
                200,
                {
                    "jsonrpc": "2.0",
                    "id": None,
                    "error": {"code": PARSE_ERROR, "message": "invalid JSON"},
                },
            )
            return

        request_id = message.get("id")

        try:
            self._send(
                200,
                {
                    "jsonrpc": "2.0",
                    "id": request_id,
                    "result": dispatch(message.get("method"), message.get("params")),
                },
            )
        except RpcError as error:
            self._send(
                200,
                {
                    "jsonrpc": "2.0",
                    "id": request_id,
                    "error": {"code": error.code, "message": str(error)},
                },
            )
        except Exception as error:
            self._send(
                200,
                {
                    "jsonrpc": "2.0",
                    "id": request_id,
                    "error": {"code": INTERNAL_ERROR, "message": str(error)},
                },
            )

    def do_GET(self) -> None:
        self._send(
            405,
            {
                "jsonrpc": "2.0",
                "id": None,
                "error": {"code": INVALID_REQUEST, "message": "POST only"},
            },
        )


def main() -> None:
    server = ThreadingHTTPServer((HOST, PORT), Handler)
    print(
        f"prism.{LANGUAGE} listening on http://{HOST}:{PORT}/mcp (MCP {PROTOCOL_VERSION})",
        flush=True,
    )
    server.serve_forever()


if __name__ == "__main__":
    main()

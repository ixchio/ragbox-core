"""Zero-dependency Model Context Protocol server for local RAGBox indexes.

The stdio transport uses newline-delimited JSON-RPC 2.0. It intentionally does
not require the optional MCP SDK, so ``ragbox-mcp --dir .`` works with the base
RAGBox installation.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import sys
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path
from typing import Any, BinaryIO, Mapping, TextIO

from loguru import logger

from ragbox import RAGBox

JSONRPC_VERSION = "2.0"
CURRENT_PROTOCOL_VERSION = "2026-07-28"
LEGACY_PROTOCOL_VERSION = "2025-11-25"
MAX_MESSAGE_BYTES = 4 * 1024 * 1024
SERVER_NAME = "ragbox-core"
VALID_SEARCH_MODES = frozenset({"fast", "tri_modal_fast", "full"})

try:
    SERVER_VERSION = version("ragbox-core")
except PackageNotFoundError:
    SERVER_VERSION = "3.0.2"

SERVER_INFO = {"name": SERVER_NAME, "version": SERVER_VERSION}
SERVER_INFO_META_KEY = "io.modelcontextprotocol/serverInfo"


class JsonRpcError(Exception):
    """A JSON-RPC error that can be returned without exposing a traceback."""

    def __init__(self, code: int, message: str, data: Any = None) -> None:
        super().__init__(message)
        self.code = code
        self.message = message
        self.data = data


TOOLS: tuple[dict[str, Any], ...] = (
    {
        "name": "ragbox_search",
        "description": (
            "Search the live local RAGBox index and return revision-safe citations."
        ),
        "inputSchema": {
            "type": "object",
            "properties": {
                "query": {"type": "string", "minLength": 1},
                "mode": {
                    "type": "string",
                    "enum": ["fast", "tri_modal_fast", "full"],
                    "default": "tri_modal_fast",
                },
                "top_k": {
                    "type": "integer",
                    "minimum": 1,
                    "maximum": 50,
                    "default": 5,
                },
            },
            "required": ["query"],
            "additionalProperties": False,
        },
        "outputSchema": {
            "type": "object",
            "properties": {
                "query": {"type": "string"},
                "mode": {"type": "string"},
                "citations": {"type": "array", "items": {"type": "object"}},
            },
            "required": ["query", "mode", "citations"],
        },
    },
    {
        "name": "ragbox_graph_context",
        "description": (
            "Return matched knowledge-graph entities, relationships, and communities."
        ),
        "inputSchema": {
            "type": "object",
            "properties": {"query": {"type": "string", "minLength": 1}},
            "required": ["query"],
            "additionalProperties": False,
        },
        "outputSchema": {
            "type": "object",
            "properties": {
                "query": {"type": "string"},
                "entities": {"type": "array", "items": {"type": "object"}},
                "relationships": {
                    "type": "array",
                    "items": {"type": "object"},
                },
                "community_summaries": {
                    "type": "array",
                    "items": {"type": "object"},
                },
                "synthesized_context": {"type": "string"},
            },
            "required": [
                "query",
                "entities",
                "relationships",
                "community_summaries",
                "synthesized_context",
            ],
        },
    },
    {
        "name": "ragbox_status",
        "description": "Report RAGBox readiness, index size, graph size, and watcher stats.",
        "inputSchema": {
            "type": "object",
            "properties": {},
            "additionalProperties": False,
        },
        "outputSchema": {
            "type": "object",
            "properties": {
                "is_ready": {"type": "boolean"},
                "status": {"type": "string"},
                "active_indexed_documents": {"type": "integer"},
                "graph": {"type": "object"},
                "watcher": {"type": "object"},
            },
            "required": [
                "is_ready",
                "status",
                "active_indexed_documents",
                "graph",
                "watcher",
            ],
        },
    },
)


class RAGBoxMCPServer:
    """Expose one RAGBox instance as MCP tools over JSON-RPC."""

    def __init__(self, ragbox: RAGBox) -> None:
        self.ragbox = ragbox

    @staticmethod
    def _normalize_query(query: Any) -> str:
        if not isinstance(query, str) or not query.strip():
            raise ValueError("query must be a non-empty string")
        query = query.strip()
        if len(query) > 20_000:
            raise ValueError("query exceeds the 20,000 character limit")
        return query

    @classmethod
    def _query_argument(cls, arguments: Mapping[str, Any]) -> str:
        return cls._normalize_query(arguments.get("query"))

    @staticmethod
    def _reject_unknown_arguments(
        arguments: Mapping[str, Any], allowed: set[str]
    ) -> None:
        unknown = sorted(set(arguments) - allowed)
        if unknown:
            raise ValueError(f"unknown argument(s): {', '.join(unknown)}")

    async def ragbox_search(
        self, query: str, mode: str = "tri_modal_fast", top_k: int = 5
    ) -> dict[str, Any]:
        query = self._normalize_query(query)
        if mode not in VALID_SEARCH_MODES:
            valid = ", ".join(sorted(VALID_SEARCH_MODES))
            raise ValueError(f"mode must be one of: {valid}")
        if (
            isinstance(top_k, bool)
            or not isinstance(top_k, int)
            or not 1 <= top_k <= 50
        ):
            raise ValueError("top_k must be an integer between 1 and 50")

        # Readiness plus flush_pending form the freshness barrier before search.
        await self.ragbox._prepare_query()
        sources = await self.ragbox.retriever.retrieve(query, top_k=top_k, mode=mode)

        active_by_revision: dict[str, str] = {}
        for (
            active_source_path,
            active_revision_id,
        ) in self.ragbox.self_healer.cas.active_revisions().items():
            active_by_revision.setdefault(active_revision_id, active_source_path)

        citations = []
        for source in sources:
            citation_revision_id: str | None = source.revision_id
            if (
                citation_revision_id is None
                and source.document_id in active_by_revision
            ):
                citation_revision_id = source.document_id
            citation_source_path: str | None = source.source_path or (
                active_by_revision.get(citation_revision_id)
                if citation_revision_id
                else None
            )
            span = None
            if (
                citation_source_path
                and source.start_line is not None
                and source.end_line is not None
            ):
                span = f"{citation_source_path}:{source.start_line}-{source.end_line}"
            citations.append(
                {
                    "document_id": source.document_id,
                    "source_path": citation_source_path,
                    "revision_id": citation_revision_id,
                    "start_line": source.start_line,
                    "end_line": source.end_line,
                    "span": span,
                    "score": source.score,
                    "text": source.text,
                }
            )
        return {"query": query, "mode": mode, "citations": citations}

    async def ragbox_graph_context(self, query: str) -> dict[str, Any]:
        query = self._normalize_query(query)
        await self.ragbox._prepare_query()
        result = await self.ragbox.knowledge_graph.query(query)
        return {
            "query": query,
            "entities": [
                item.model_dump(mode="json") for item in result.relevant_entities
            ],
            "relationships": [
                item.model_dump(mode="json") for item in result.relevant_relations
            ],
            "community_summaries": [
                item.model_dump(mode="json") for item in result.relevant_communities
            ],
            "synthesized_context": result.synthesized_context,
        }

    async def ragbox_status(self) -> dict[str, Any]:
        graph = self.ragbox.knowledge_graph
        with graph.mutation_lock():
            graph_stats = {
                "nodes": len(graph.graph.nodes),
                "edges": len(graph.graph.edges),
                "communities": len(graph.communities),
            }

        watcher = self.ragbox.self_healer.watcher
        watcher_stats: dict[str, Any]
        if watcher is None:
            watcher_stats = {
                "events_received": 0,
                "events_deduplicated": 0,
                "events_processed": 0,
                "queue_size": 0,
                "deduplication_rate": 0.0,
            }
        else:
            watcher_stats = dict(watcher.get_stats())
        watcher_stats["running"] = self.ragbox.self_healer._watchdog_started

        return {
            "is_ready": self.ragbox.is_ready,
            "status": self.ragbox.status,
            "active_indexed_documents": len(
                self.ragbox.self_healer.cas.active_revisions()
            ),
            "graph": graph_stats,
            "watcher": watcher_stats,
        }

    async def call_tool(
        self, name: str, arguments: Mapping[str, Any]
    ) -> dict[str, Any]:
        if name == "ragbox_search":
            self._reject_unknown_arguments(arguments, {"query", "mode", "top_k"})
            query = self._query_argument(arguments)
            mode = arguments.get("mode", "tri_modal_fast")
            top_k = arguments.get("top_k", 5)
            if not isinstance(mode, str):
                raise ValueError("mode must be a string")
            return await self.ragbox_search(query, mode, top_k)
        if name == "ragbox_graph_context":
            self._reject_unknown_arguments(arguments, {"query"})
            return await self.ragbox_graph_context(self._query_argument(arguments))
        if name == "ragbox_status":
            self._reject_unknown_arguments(arguments, set())
            return await self.ragbox_status()
        raise ValueError(f"unknown tool: {name}")

    @staticmethod
    def _tool_result(
        payload: dict[str, Any], *, is_error: bool = False
    ) -> dict[str, Any]:
        return {
            "content": [
                {
                    "type": "text",
                    "text": json.dumps(payload, ensure_ascii=False, sort_keys=True),
                }
            ],
            "structuredContent": payload,
            "isError": is_error,
        }

    @staticmethod
    def _server_meta() -> dict[str, Any]:
        return {SERVER_INFO_META_KEY: dict(SERVER_INFO)}

    def _result_response(
        self, request_id: Any, result: dict[str, Any]
    ) -> dict[str, Any]:
        result.setdefault("_meta", self._server_meta())
        return {"jsonrpc": JSONRPC_VERSION, "id": request_id, "result": result}

    @staticmethod
    def _error_response(
        request_id: Any, code: int, message: str, data: Any = None
    ) -> dict[str, Any]:
        error: dict[str, Any] = {"code": code, "message": message}
        if data is not None:
            error["data"] = data
        return {"jsonrpc": JSONRPC_VERSION, "id": request_id, "error": error}

    async def handle_message(self, message: Any) -> dict[str, Any] | None:
        if not isinstance(message, dict):
            return self._error_response(None, -32600, "Invalid Request")

        request_id = message.get("id")
        is_notification = "id" not in message
        if message.get("jsonrpc") != JSONRPC_VERSION:
            if is_notification:
                return None
            return self._error_response(request_id, -32600, "Invalid Request")

        method = message.get("method")
        if not isinstance(method, str):
            if is_notification:
                return None
            return self._error_response(request_id, -32600, "Invalid Request")

        if method.startswith("notifications/"):
            return None

        params = message.get("params", {})
        if not isinstance(params, dict):
            if is_notification:
                return None
            return self._error_response(request_id, -32602, "Invalid params")

        try:
            if method == "initialize":
                requested = params.get("protocolVersion")
                protocol_version = (
                    requested
                    if isinstance(requested, str)
                    and requested != CURRENT_PROTOCOL_VERSION
                    else LEGACY_PROTOCOL_VERSION
                )
                result: dict[str, Any] = {
                    "protocolVersion": protocol_version,
                    "capabilities": {"tools": {"listChanged": False}},
                    "serverInfo": dict(SERVER_INFO),
                    "instructions": (
                        "Search the live local RAGBox index and inspect its graph."
                    ),
                }
                return self._result_response(request_id, result)
            if method == "server/discover":
                result = {
                    "protocolVersions": [
                        CURRENT_PROTOCOL_VERSION,
                        LEGACY_PROTOCOL_VERSION,
                    ],
                    "capabilities": {"tools": {"listChanged": False}},
                    "instructions": (
                        "Search the live local RAGBox index and inspect its graph."
                    ),
                    "ttlMs": 0,
                    "cacheScope": "private",
                }
                return self._result_response(request_id, result)
            if method == "ping":
                return self._result_response(request_id, {})
            if method == "tools/list":
                result = {
                    "tools": [dict(tool) for tool in TOOLS],
                    "ttlMs": 0,
                    "cacheScope": "private",
                }
                return self._result_response(request_id, result)
            if method == "tools/call":
                name = params.get("name")
                arguments = params.get("arguments", {})
                if not isinstance(name, str) or not isinstance(arguments, dict):
                    raise JsonRpcError(-32602, "Invalid tools/call params")
                try:
                    payload = await self.call_tool(name, arguments)
                    result = self._tool_result(payload)
                except ValueError as exc:
                    result = self._tool_result({"error": str(exc)}, is_error=True)
                except Exception as exc:
                    logger.exception(f"MCP tool {name} failed: {exc}")
                    result = self._tool_result({"error": str(exc)}, is_error=True)
                return self._result_response(request_id, result)
            raise JsonRpcError(-32601, "Method not found", {"method": method})
        except JsonRpcError as exc:
            if is_notification:
                return None
            return self._error_response(
                request_id, exc.code, exc.message, data=exc.data
            )

    async def serve_stdio(
        self,
        input_stream: BinaryIO | None = None,
        output_stream: TextIO | None = None,
    ) -> None:
        """Serve newline-delimited JSON-RPC until the client closes stdin."""
        reader = input_stream or sys.stdin.buffer
        writer = output_stream or sys.stdout
        try:
            while True:
                raw_line = await asyncio.to_thread(
                    reader.readline, MAX_MESSAGE_BYTES + 1
                )
                if not raw_line:
                    return
                if len(raw_line) > MAX_MESSAGE_BYTES:
                    oversized_response = self._error_response(
                        None, -32700, "Message exceeds 4 MiB limit"
                    )
                    await self._write_response(writer, oversized_response)
                    return
                if not raw_line.strip():
                    continue
                try:
                    message = json.loads(raw_line)
                except (UnicodeDecodeError, json.JSONDecodeError):
                    await self._write_response(
                        writer, self._error_response(None, -32700, "Parse error")
                    )
                    continue
                handled_response = await self.handle_message(message)
                if handled_response is not None:
                    await self._write_response(writer, handled_response)
        finally:
            await self.ragbox.aclose()

    @staticmethod
    async def _write_response(writer: TextIO, response: dict[str, Any]) -> None:
        payload = json.dumps(response, ensure_ascii=False, separators=(",", ":"))

        def _write() -> None:
            writer.write(payload + "\n")
            writer.flush()

        await asyncio.to_thread(_write)


def _argument_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="ragbox-mcp",
        description="Serve a live local RAGBox index over MCP stdio.",
    )
    parser.add_argument(
        "--dir",
        "-d",
        dest="document_dir",
        type=Path,
        default=Path.cwd(),
        help="Directory to index and watch (default: current directory).",
    )
    parser.add_argument(
        "--print-mcp-config",
        action="store_true",
        help="Print a Claude Code/Cursor-compatible MCP configuration and exit.",
    )
    parser.add_argument(
        "--write-mcp-json",
        action="store_true",
        help="Write the MCP configuration to <dir>/.mcp.json and exit.",
    )
    return parser


def _mcp_config(document_dir: Path) -> dict[str, Any]:
    """Return the project-scoped stdio config shared by Claude Code and Cursor."""
    return {
        "mcpServers": {
            "ragbox": {
                "command": "ragbox-mcp",
                "args": ["--dir", str(document_dir)],
                "env": {},
            }
        }
    }


def main() -> None:
    parser = _argument_parser()
    args = parser.parse_args()
    document_dir = args.document_dir.expanduser().resolve()
    if not document_dir.is_dir():
        parser.error(f"document directory does not exist: {document_dir}")

    if args.print_mcp_config or args.write_mcp_json:
        payload = json.dumps(_mcp_config(document_dir), indent=2) + "\n"
        if args.write_mcp_json:
            output_path = document_dir / ".mcp.json"
            output_path.write_text(payload, encoding="utf-8")
            print(output_path)
        if args.print_mcp_config:
            print(payload, end="")
        return

    server = RAGBoxMCPServer(RAGBox(document_dir))
    try:
        asyncio.run(server.serve_stdio())
    except KeyboardInterrupt:
        pass


__all__ = ["RAGBoxMCPServer", "_mcp_config", "main"]


if __name__ == "__main__":
    main()

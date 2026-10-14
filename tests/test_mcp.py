import asyncio
import io
import json
import time
from contextlib import contextmanager
from types import SimpleNamespace
from unittest.mock import AsyncMock

import networkx as nx
import pytest

from ragbox.core.self_healing import FileEvent, ProductionFileWatcher
from ragbox.mcp_server import RAGBoxMCPServer, _mcp_config
from ragbox.models.graph import Community, Entity, GraphQueryResult, Relation
from ragbox.models.queries import Source


class FakeCAS:
    def __init__(self, active=None):
        self.active = active or {}

    def active_revisions(self):
        return dict(self.active)


class FakeKnowledgeGraph:
    def __init__(self, result=None):
        self.graph = nx.Graph()
        self.communities = {}
        self.result = result or GraphQueryResult()
        self.queries = []

    @contextmanager
    def mutation_lock(self):
        yield

    async def query(self, query):
        self.queries.append(query)
        return self.result


def fake_ragbox(*, sources=None, graph=None, active=None, watcher=None):
    retriever = SimpleNamespace(
        retrieve=AsyncMock(return_value=list(sources or [])),
    )
    ragbox = SimpleNamespace(
        _prepare_query=AsyncMock(),
        retriever=retriever,
        knowledge_graph=graph or FakeKnowledgeGraph(),
        self_healer=SimpleNamespace(
            cas=FakeCAS(active),
            watcher=watcher,
            _watchdog_started=watcher is not None,
        ),
        is_ready=True,
        status="ready",
        aclose=AsyncMock(),
    )
    return ragbox


@pytest.mark.asyncio
@pytest.mark.parametrize("mode", ["fast", "tri_modal_fast", "full"])
async def test_ragbox_search_modes_return_structured_citations(mode):
    source = Source(
        document_id="revision-1",
        chunk_id="chunk-1",
        source_path="/docs/policy.md",
        revision_id="revision-1",
        start_line=8,
        end_line=12,
        score=0.75,
        text="The current policy text.",
    )
    ragbox = fake_ragbox(sources=[source], active={"/docs/policy.md": "revision-1"})
    server = RAGBoxMCPServer(ragbox)

    result = await server.ragbox_search("policy", mode=mode, top_k=3)

    ragbox._prepare_query.assert_awaited_once_with()
    ragbox.retriever.retrieve.assert_awaited_once_with("policy", top_k=3, mode=mode)
    assert result == {
        "query": "policy",
        "mode": mode,
        "citations": [
            {
                "document_id": "revision-1",
                "source_path": "/docs/policy.md",
                "revision_id": "revision-1",
                "start_line": 8,
                "end_line": 12,
                "span": "/docs/policy.md:8-12",
                "score": 0.75,
                "text": "The current policy text.",
            }
        ],
    }


def test_mcp_config_targets_the_selected_project(tmp_path):
    project = tmp_path.resolve()

    assert _mcp_config(project) == {
        "mcpServers": {
            "ragbox": {
                "command": "ragbox-mcp",
                "args": ["--dir", str(project)],
                "env": {},
            }
        }
    }


@pytest.mark.asyncio
async def test_ragbox_search_flushes_live_file_update_before_retrieval(tmp_path):
    source_file = tmp_path / "live.md"
    source_file.write_text("old content", encoding="utf-8")
    state = {"revision": "old-revision", "text": "old content"}

    async def apply_event(event_type, path):
        assert event_type == "created_or_modified"
        state["revision"] = "fresh-revision"
        state["text"] = path.read_text(encoding="utf-8")

    watcher = ProductionFileWatcher(
        index_callback=apply_event,
        loop=asyncio.get_running_loop(),
        debounce_seconds=0,
        batch_interval=60,
    )
    source_file.write_text("fresh content", encoding="utf-8")
    watcher._accept_event(
        FileEvent(
            path=str(source_file),
            event_type="created_or_modified",
            timestamp=time.time(),
        )
    )

    async def prepare_query():
        await watcher.flush()

    async def retrieve(query, top_k=5, mode=None):
        assert state["revision"] == "fresh-revision"
        return [
            Source(
                document_id=state["revision"],
                source_path=str(source_file),
                revision_id=state["revision"],
                score=1.0,
                text=state["text"],
            )
        ]

    cas = SimpleNamespace(
        active_revisions=lambda: {str(source_file): state["revision"]}
    )
    ragbox = SimpleNamespace(
        _prepare_query=prepare_query,
        retriever=SimpleNamespace(retrieve=retrieve),
        self_healer=SimpleNamespace(cas=cas, watcher=watcher, _watchdog_started=True),
    )
    server = RAGBoxMCPServer(ragbox)

    try:
        result = await server.ragbox_search("fresh", mode="fast")

        assert result["citations"][0]["revision_id"] == "fresh-revision"
        assert result["citations"][0]["text"] == "fresh content"
        assert watcher.get_stats()["events_processed"] == 1
    finally:
        watcher.shutdown()


@pytest.mark.asyncio
async def test_ragbox_graph_context_returns_structured_graph_data():
    graph_result = GraphQueryResult(
        relevant_entities=[
            Entity(
                id="payments",
                name="Payments",
                entity_type="SERVICE",
                description="Payment service",
            )
        ],
        relevant_relations=[
            Relation(
                source_id="checkout",
                target_id="payments",
                relation_type="CALLS",
                context="Checkout authorizes payments",
            )
        ],
        relevant_communities=[
            Community(
                id="community_1",
                level=1,
                entity_ids=["checkout", "payments"],
                summary="Checkout and payment processing",
            )
        ],
        synthesized_context="Checkout calls Payments.",
    )
    graph = FakeKnowledgeGraph(graph_result)
    ragbox = fake_ragbox(graph=graph)
    server = RAGBoxMCPServer(ragbox)

    result = await server.ragbox_graph_context("checkout dependencies")

    ragbox._prepare_query.assert_awaited_once_with()
    assert graph.queries == ["checkout dependencies"]
    assert result["entities"][0]["name"] == "Payments"
    assert result["relationships"][0]["relation_type"] == "CALLS"
    assert result["community_summaries"][0]["summary"] == (
        "Checkout and payment processing"
    )


@pytest.mark.asyncio
async def test_ragbox_status_reports_index_graph_and_watcher_stats():
    graph = FakeKnowledgeGraph()
    graph.graph.add_edge("checkout", "payments")
    graph.graph.add_node("ledger")
    graph.communities = {0: {"checkout", "payments"}, 1: {"ledger"}}
    watcher = SimpleNamespace(
        get_stats=lambda: {
            "events_received": 4,
            "events_deduplicated": 1,
            "events_processed": 3,
            "queue_size": 0,
            "deduplication_rate": 0.25,
        }
    )
    ragbox = fake_ragbox(
        graph=graph,
        watcher=watcher,
        active={"/docs/a.md": "rev-a", "/docs/b.md": "rev-b"},
    )
    server = RAGBoxMCPServer(ragbox)

    result = await server.ragbox_status()

    assert result["is_ready"] is True
    assert result["status"] == "ready"
    assert result["active_indexed_documents"] == 2
    assert result["graph"] == {"nodes": 3, "edges": 1, "communities": 2}
    assert result["watcher"]["running"] is True
    assert result["watcher"]["events_processed"] == 3


@pytest.mark.asyncio
async def test_jsonrpc_stdio_initialize_list_and_call():
    ragbox = fake_ragbox(active={"/docs/a.md": "rev-a"})
    server = RAGBoxMCPServer(ragbox)
    requests = [
        {"jsonrpc": "2.0", "id": 0, "method": "server/discover"},
        {
            "jsonrpc": "2.0",
            "id": 1,
            "method": "initialize",
            "params": {"protocolVersion": "2025-11-25"},
        },
        {"jsonrpc": "2.0", "method": "notifications/initialized"},
        {"jsonrpc": "2.0", "id": 2, "method": "tools/list"},
        {
            "jsonrpc": "2.0",
            "id": 3,
            "method": "tools/call",
            "params": {"name": "ragbox_status", "arguments": {}},
        },
        {
            "jsonrpc": "2.0",
            "id": 4,
            "method": "tools/call",
            "params": {
                "name": "ragbox_search",
                "arguments": {"query": "policy"},
            },
        },
    ]
    stdin = io.BytesIO(
        b"".join((json.dumps(request) + "\n").encode() for request in requests)
    )
    stdout = io.StringIO()

    await server.serve_stdio(stdin, stdout)

    responses = [json.loads(line) for line in stdout.getvalue().splitlines()]
    assert [response["id"] for response in responses] == [0, 1, 2, 3, 4]
    assert responses[0]["result"]["protocolVersions"][0] == "2026-07-28"
    assert responses[1]["result"]["protocolVersion"] == "2025-11-25"
    assert [tool["name"] for tool in responses[2]["result"]["tools"]] == [
        "ragbox_search",
        "ragbox_graph_context",
        "ragbox_status",
    ]
    assert responses[3]["result"]["isError"] is False
    assert responses[3]["result"]["structuredContent"]["active_indexed_documents"] == 1
    assert responses[4]["result"]["structuredContent"]["mode"] == ("tri_modal_fast")
    ragbox.aclose.assert_awaited_once_with()

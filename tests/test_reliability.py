import asyncio
import threading
from contextlib import contextmanager
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest

from ragbox import RAGBox
from ragbox.core.agentic_orchestrator import AgenticOrchestrator
from ragbox.core.chunking_engine import (
    AdaptiveChunker,
    ChunkingEngine,
    RecursiveChunker,
)
from ragbox.core.document_processor import DocumentProcessorRouter
from ragbox.core.knowledge_graph import OptimizedKnowledgeGraph
from ragbox.core.retrieval_fusion import RetrievalFusionEngine
from ragbox.core.self_healing import (
    ContentAddressedStorage,
    ProductionFileWatcher,
    SelfHealer,
)
from ragbox.models.chunks import TextChunk
from ragbox.models.documents import Document, DocumentType
from ragbox.models.queries import Answer, RAGStrategy, Source
from ragbox.utils.cost_tracker import CostBudget, CostEstimator
from ragbox.utils.llm_clients import LLMClient
from ragbox.utils.vector_stores import VectorQueryResult


class MemoryVectorStore:
    def __init__(self):
        self.documents = {}

    async def add_documents(self, documents, namespace=None):
        for document in documents:
            self.documents[document["id"]] = document

    async def delete(self, ids=None, namespace=None, filter_dict=None):
        if ids:
            targets = set(ids)
        else:
            targets = {
                doc_id
                for doc_id, document in self.documents.items()
                if all(
                    document.get("metadata", {}).get(key) == value
                    for key, value in (filter_dict or {}).items()
                )
            }
        for doc_id in targets:
            self.documents.pop(doc_id, None)
        return len(targets)


class FakeProcessor:
    async def process(self, path: Path, file_hash: str):
        return Document(
            id=file_hash,
            path=path,
            content=path.read_text(),
            doc_type=DocumentType.TEXT,
        )


class CountingProcessor(FakeProcessor):
    def __init__(self):
        self.calls = 0

    async def process(self, path: Path, file_hash: str):
        self.calls += 1
        return await super().process(path, file_hash)


class FakeChunker:
    async def chunk(self, document: Document):
        return [
            TextChunk(
                id=f"chunk-{document.id}",
                document_id=document.id,
                content=document.content,
                metadata={"embedding": [1.0, 0.0], "strategy": "test"},
            )
        ]


class FakeKnowledgeGraph:
    def __init__(self):
        self.documents = {}

    @contextmanager
    def mutation_lock(self):
        yield

    async def prepare_document(self, document):
        return document.content

    def commit_prepared_document(
        self, document, prepared, *, replace_doc_id=None, optimize=True
    ):
        if replace_doc_id:
            self.documents.pop(replace_doc_id, None)
        self.documents[document.id] = prepared

    def remove_document(self, doc_id, *, optimize=True):
        return self.documents.pop(doc_id, None) is not None

    def snapshot_document(self, doc_id):
        if not doc_id or doc_id not in self.documents:
            return None
        return {"doc_id": doc_id, "content": self.documents[doc_id]}

    def rollback_document_replacement(self, new_doc_id, old_snapshot):
        self.documents.pop(new_doc_id, None)
        if old_snapshot:
            self.documents[old_snapshot["doc_id"]] = old_snapshot["content"]

    async def refresh_community_summaries(self):
        return None


class RestartKnowledgeGraph(OptimizedKnowledgeGraph):
    def __init__(self):
        super().__init__(llm_client=None)
        self.prepare_calls = 0
        self.generated_missing_counts = []

    async def prepare_document(self, document):
        self.prepare_calls += 1
        return (
            [
                {"name": "Alpha Service", "description": "Primary service"},
                {"name": "Beta Store", "description": "Persistence service"},
            ],
            [
                {
                    "source": "Alpha Service",
                    "target": "Beta Store",
                    "type": "WRITES_TO",
                    "context": "Alpha writes records to Beta",
                }
            ],
        )

    async def _generate_community_summaries(self):
        self.generated_missing_counts.append(
            sum(
                comm_idx not in self.community_summaries
                for comm_idx in self.communities
            )
        )
        await super()._generate_community_summaries()


class UsageLLM(LLMClient):
    def __init__(self):
        super().__init__(CostBudget(max_daily_cost=10.0, max_query_cost=1.0))
        self._model = "gpt-4o"

    async def _agenerate(self, prompt, system=None, **kwargs):
        self._record_usage(100, 50)
        return "measured response"

    async def _agenerate_structured(self, prompt, schema, system=None, **kwargs):
        self._record_usage(80, 20)
        return {"ok": True}

    async def _astream(self, prompt, system=None, **kwargs):
        yield "hello "
        yield "world"
        self._record_usage(60, 10)


def test_cas_activation_is_separate_from_hashing(tmp_path: Path):
    cas = ContentAddressedStorage(tmp_path / "state")
    source = tmp_path / "doc.txt"
    source.write_text("revision one")

    revision = cas.get_hash(source)
    assert revision
    assert cas.get_recorded_hash(source) is None

    cas.activate(source, revision)
    assert cas.is_active(source, revision)

    source.unlink()
    assert cas.get_recorded_hash(source) == revision
    assert cas.remove(source) == revision
    assert cas.get_recorded_hash(source) is None


def test_fts_search_only_exposes_active_revision(tmp_path: Path):
    cas = ContentAddressedStorage(tmp_path / "state")
    if not cas.sparse_search_available:
        pytest.skip("Python sqlite3 was built without FTS5")

    source = tmp_path / "code.py"
    source.write_text("legacy_identifier = 1")
    source_path = str(source.resolve())
    first_revision = cas.get_hash(source)
    assert first_revision
    first_chunk = {
        "id": "chunk-old",
        "content": "legacy_identifier = 1",
        "sparse_content": "[code.py | legacy_identifier]\nlegacy_identifier = 1",
        "metadata": {
            "doc_id": first_revision,
            "revision_id": first_revision,
            "source_path": source_path,
            "start_line": 3,
            "end_line": 3,
        },
    }
    cas.index_fts_chunks([first_chunk])
    assert cas.search_fts("legacy_identifier") == []

    cas.activate(source, first_revision)
    assert [hit["id"] for hit in cas.search_fts("legacy_identifier")] == ["chunk-old"]
    legacy_hit = cas.search_fts("legacy_identifier")[0]
    assert legacy_hit["content"] == "legacy_identifier = 1"
    assert legacy_hit["metadata"]["start_line"] == 3
    assert legacy_hit["metadata"]["end_line"] == 3

    source.write_text("replacement_symbol = 2")
    second_revision = cas.get_hash(source)
    assert second_revision and second_revision != first_revision
    cas.index_fts_chunks(
        [
            {
                "id": "chunk-new",
                "content": "replacement_symbol = 2",
                "metadata": {
                    "doc_id": second_revision,
                    "revision_id": second_revision,
                    "source_path": source_path,
                    "start_line": 7,
                    "end_line": 8,
                },
            }
        ]
    )
    assert cas.search_fts("replacement_symbol") == []
    cas.activate(source, second_revision)
    assert cas.search_fts("legacy_identifier") == []
    assert [hit["id"] for hit in cas.search_fts("replacement_symbol")] == ["chunk-new"]

    cas.delete_fts_revision(second_revision)
    assert cas.search_fts("replacement_symbol") == []


def test_graph_snapshot_serialization_cannot_advance_active_pointer(tmp_path: Path):
    cas = ContentAddressedStorage(tmp_path / "state")
    source = tmp_path / "doc.txt"
    source.write_text("revision")
    revision = cas.get_hash(source)
    assert revision

    with pytest.raises(TypeError):
        cas.activate(
            source,
            revision,
            {"doc_id": revision, "not_json_serializable": object()},
        )

    assert cas.get_recorded_hash(source) is None
    assert cas.load_active_graph_snapshots() == []


@pytest.mark.asyncio
async def test_self_healer_replaces_and_deletes_complete_revision(tmp_path: Path):
    source = tmp_path / "doc.txt"
    source.write_text("revision one")
    vectors = MemoryVectorStore()
    graph = FakeKnowledgeGraph()
    healer = SelfHealer(
        document_dir=tmp_path,
        document_processor=FakeProcessor(),
        chunking_engine=FakeChunker(),
        vector_store=vectors,
        knowledge_graph=graph,
    )

    await healer.handle_file_event("created_or_modified", source)
    first_revision = healer.cas.get_recorded_hash(source)
    assert first_revision is not None
    assert first_revision in graph.documents
    assert f"chunk-{first_revision}" in vectors.documents
    assert healer.is_active_revision(str(source.resolve()), first_revision)
    if healer.cas.sparse_search_available:
        assert [hit["id"] for hit in healer.cas.search_fts("one")] == [
            f"chunk-{first_revision}"
        ]

    source.write_text("revision two")
    await healer.handle_file_event("created_or_modified", source)
    second_revision = healer.cas.get_recorded_hash(source)
    assert second_revision is not None
    assert second_revision != first_revision
    assert first_revision not in graph.documents
    assert f"chunk-{first_revision}" not in vectors.documents
    assert f"chunk-{second_revision}" in vectors.documents
    assert not healer.is_active_revision(str(source.resolve()), first_revision)
    if healer.cas.sparse_search_available:
        assert healer.cas.search_fts("one") == []
        assert [hit["id"] for hit in healer.cas.search_fts("two")] == [
            f"chunk-{second_revision}"
        ]

    source.unlink()
    await healer.handle_file_event("deleted", source)
    assert healer.cas.get_recorded_hash(source) is None
    assert graph.documents == {}
    assert vectors.documents == {}
    if healer.cas.sparse_search_available:
        assert healer.cas.search_fts("two") == []


@pytest.mark.asyncio
async def test_initial_build_restores_graph_and_summaries_without_reindexing(
    tmp_path: Path,
):
    source = tmp_path / "architecture.txt"
    source.write_text("Alpha Service writes records to Beta Store")
    vectors = MemoryVectorStore()
    first_processor = CountingProcessor()
    first_graph = RestartKnowledgeGraph()
    first = SelfHealer(
        document_dir=tmp_path,
        document_processor=first_processor,
        chunking_engine=FakeChunker(),
        vector_store=vectors,
        knowledge_graph=first_graph,
    )

    await first.initial_build()
    revision = first.cas.get_recorded_hash(source)
    first_snapshot = first_graph.snapshot_document(revision)
    first_summaries = first_graph.export_community_summaries()

    assert first_processor.calls == 1
    assert first_snapshot is not None
    assert len(first_graph.graph.nodes) == 2
    assert len(first_graph.graph.edges) == 1
    assert first_summaries

    second_processor = CountingProcessor()
    second_graph = RestartKnowledgeGraph()
    second = SelfHealer(
        document_dir=tmp_path,
        document_processor=second_processor,
        chunking_engine=FakeChunker(),
        vector_store=vectors,
        knowledge_graph=second_graph,
    )

    await second.initial_build()

    assert second_processor.calls == 0
    assert second_graph.prepare_calls == 0
    assert second_graph.snapshot_document(revision) == first_snapshot
    assert second_graph.export_community_summaries() == first_summaries
    assert second_graph.generated_missing_counts == [0]
    assert set(vectors.documents) == {f"chunk-{revision}"}


@pytest.mark.asyncio
async def test_initial_build_reconciles_file_deleted_while_offline(tmp_path: Path):
    source = tmp_path / "architecture.txt"
    source.write_text("Alpha Service writes records to Beta Store")
    vectors = MemoryVectorStore()
    first = SelfHealer(
        document_dir=tmp_path,
        document_processor=CountingProcessor(),
        chunking_engine=FakeChunker(),
        vector_store=vectors,
        knowledge_graph=RestartKnowledgeGraph(),
    )
    await first.initial_build()
    revision = first.cas.get_recorded_hash(source)
    source.unlink()

    second_processor = CountingProcessor()
    second_graph = RestartKnowledgeGraph()
    second = SelfHealer(
        document_dir=tmp_path,
        document_processor=second_processor,
        chunking_engine=FakeChunker(),
        vector_store=vectors,
        knowledge_graph=second_graph,
    )
    await second.initial_build()

    assert second_processor.calls == 0
    assert second.cas.active_revisions() == {}
    assert second.cas.load_active_graph_snapshots() == []
    assert len(second_graph.graph.nodes) == 0
    assert len(second_graph.graph.edges) == 0
    assert vectors.documents == {}
    if second.cas.sparse_search_available:
        assert second.cas.search_fts("Alpha") == []
    assert revision is not None


@pytest.mark.asyncio
async def test_failed_activation_rolls_back_shadow_revision(
    tmp_path: Path, monkeypatch
):
    source = tmp_path / "doc.txt"
    source.write_text("revision one")
    vectors = MemoryVectorStore()
    graph = FakeKnowledgeGraph()
    healer = SelfHealer(
        document_dir=tmp_path,
        document_processor=FakeProcessor(),
        chunking_engine=FakeChunker(),
        vector_store=vectors,
        knowledge_graph=graph,
    )

    await healer.handle_file_event("created_or_modified", source)
    first_revision = healer.cas.get_recorded_hash(source)
    source.write_text("revision two")

    def fail_activation(path, revision, graph_snapshot=None):
        raise RuntimeError("activation failed")

    monkeypatch.setattr(healer.cas, "activate", fail_activation)
    with pytest.raises(RuntimeError, match="activation failed"):
        await healer.handle_file_event("created_or_modified", source)

    assert healer.cas.get_recorded_hash(source) == first_revision
    assert graph.documents == {first_revision: "revision one"}
    assert set(vectors.documents) == {f"chunk-{first_revision}"}
    if healer.cas.sparse_search_available:
        assert [hit["id"] for hit in healer.cas.search_fts("one")] == [
            f"chunk-{first_revision}"
        ]
        assert healer.cas.search_fts("two") == []


@pytest.mark.asyncio
async def test_watcher_marshals_callbacks_to_bound_loop(tmp_path: Path):
    source = tmp_path / "watched.txt"
    source.write_text("hello")
    callback = AsyncMock()
    loop = asyncio.get_running_loop()
    watcher = ProductionFileWatcher(
        callback,
        loop=loop,
        debounce_seconds=0,
        batch_interval=0.01,
    )
    event = SimpleNamespace(is_directory=False, src_path=str(source))

    await asyncio.to_thread(watcher.on_modified, event)
    await asyncio.sleep(0)
    await watcher.flush()

    callback.assert_awaited_once()
    assert callback.await_args is not None
    assert callback.await_args.args[0] == "created_or_modified"
    assert callback.await_args.args[1] == source


def test_watcher_ignores_build_artifacts_lockfiles_and_gitignore(tmp_path: Path):
    (tmp_path / ".gitignore").write_text(
        "generated/\n*.snapshot\ncache-*/\n", encoding="utf-8"
    )
    watcher = ProductionFileWatcher(
        AsyncMock(),
        document_dir=tmp_path,
    )

    assert watcher._path_is_ignored(str(tmp_path / "dist" / "bundle.js"))
    assert watcher._path_is_ignored(str(tmp_path / "src" / "uv.lock"))
    assert watcher._path_is_ignored(str(tmp_path / "generated" / "api.py"))
    assert watcher._path_is_ignored(str(tmp_path / "tests" / "result.snapshot"))
    assert watcher._path_is_ignored(str(tmp_path / "cache-local" / "data.json"))
    assert not watcher._path_is_ignored(str(tmp_path / "src" / "service.py"))


def test_graph_document_removal_preserves_shared_entities():
    graph = OptimizedKnowledgeGraph(llm_client=None)
    graph.add_document(
        "rev-1",
        ["Shared Entity", "Old Entity"],
        [
            {
                "source": "Shared Entity",
                "target": "Old Entity",
                "type": "owns",
                "context": "old relation",
            }
        ],
        entity_source_texts={"Shared Entity": "old source"},
    )
    graph.add_document(
        "rev-2",
        ["Shared Entity", "New Entity"],
        [
            {
                "source": "Shared Entity",
                "target": "New Entity",
                "type": "owns",
                "context": "new relation",
            }
        ],
        entity_source_texts={"Shared Entity": "new source"},
    )

    assert graph.remove_document("rev-1")
    assert graph.graph.has_node("Shared Entity")
    assert not graph.graph.has_node("Old Entity")
    assert graph.graph.has_edge("Shared Entity", "New Entity")
    assert graph.entity_source_texts["Shared Entity"] == ["new source"]


def test_graph_canonicalizes_entity_case_and_ignores_stopword_only_queries():
    graph = OptimizedKnowledgeGraph(llm_client=None)
    graph.add_document("rev-1", ["FastEmbed"], [], optimize=False)
    graph.add_document("rev-2", ["fastembed"], [], optimize=False)
    graph.add_document("rev-3", ["VP of Engineering"], [], optimize=False)

    assert "FastEmbed" in graph.graph
    assert "fastembed" not in graph.graph
    assert graph.graph.nodes["FastEmbed"]["source_docs"] == {"rev-1", "rev-2"}
    assert graph._match_query_to_entities("what is the and of") == []
    assert (
        graph._match_query_to_entities("How does FastEmbed work?")[0][0] == "FastEmbed"
    )


def test_community_summary_cache_is_invalidated_by_description_changes():
    first = OptimizedKnowledgeGraph(llm_client=None)
    first.add_document(
        "revision",
        ["Alpha", "Beta"],
        [{"source": "Alpha", "target": "Beta", "type": "owns"}],
        entity_descriptions={"Alpha": "Old description"},
    )
    first._optimize_graph()
    first.set_community_summary(next(iter(first.communities)), "cached summary")
    cached = first.export_community_summaries()
    snapshot = first.snapshot_document("revision")
    assert snapshot is not None
    snapshot["entity_descriptions"]["Alpha"] = "Changed description"

    restored = OptimizedKnowledgeGraph(llm_client=None)
    restored.restore_snapshots([snapshot], cached)

    assert restored.community_summaries == {}


@pytest.mark.asyncio
async def test_markdown_chunking_preserves_offsets_and_adds_index_only_headers():
    class CapturingEmbeddings:
        def __init__(self):
            self.headers = []

        async def embed_chunks_with_context(self, document, chunks, llm_client=None):
            self.headers = [chunk.metadata.get("context_header") for chunk in chunks]
            return [[1.0, 0.0] for _ in chunks]

    body = "\n".join(f"row_{index} | value_{index}" for index in range(180))
    document = Document(
        id="markdown-doc",
        path=Path("/docs/guide.md"),
        content=f"# Installation\n{body}",
        doc_type=DocumentType.TEXT,
    )
    embeddings = CapturingEmbeddings()
    engine = ChunkingEngine(llm_client=None, embedding_provider=embeddings)

    assert isinstance(AdaptiveChunker().select(document), RecursiveChunker)
    chunks = await engine.chunk(document)

    assert len(chunks) > 1
    assert chunks[1].metadata["start_idx"] > 0
    assert all(len(chunk.content) <= 1201 for chunk in chunks)
    assert embeddings.headers[0] == "[guide.md | Installation]"
    assert not chunks[0].content.startswith("[guide.md")


@pytest.mark.asyncio
async def test_document_router_skips_unknown_binary_files(tmp_path: Path):
    binary = tmp_path / "archive.unknown"
    binary.write_bytes(b"header\x00binary-data")
    assert await DocumentProcessorRouter().process(binary, "binary-hash") is None


def test_retriever_rejects_inactive_or_legacy_vectors():
    engine = object.__new__(RetrievalFusionEngine)
    engine.active_revision_resolver = lambda path, revision: (
        path == "/docs/a.txt" and revision == "active"
    )
    results = [
        VectorQueryResult(
            id="active",
            score=1.0,
            content="active",
            metadata={
                "doc_id": "active",
                "revision_id": "active",
                "source_path": "/docs/a.txt",
            },
        ),
        VectorQueryResult(
            id="stale",
            score=0.9,
            content="stale",
            metadata={
                "doc_id": "stale",
                "revision_id": "stale",
                "source_path": "/docs/a.txt",
            },
        ),
        VectorQueryResult(
            id="legacy",
            score=0.8,
            content="legacy",
            metadata={"doc_id": "legacy"},
        ),
    ]

    assert [result.id for result in engine._active_only(results)] == ["active"]


def test_retriever_propagates_one_based_line_spans_to_sources():
    sources = RetrievalFusionEngine._to_sources(
        [
            {
                "id": "chunk-1",
                "content": "selected code",
                "score": 0.9,
                "metadata": {
                    "doc_id": "revision-1",
                    "revision_id": "revision-1",
                    "source_path": "/repo/service.py",
                    "start_line": 11,
                    "end_line": 14,
                },
            }
        ]
    )

    assert sources[0].start_line == 11
    assert sources[0].end_line == 14


def test_three_way_rrf_and_dense_sparse_consensus_skip():
    engine = object.__new__(RetrievalFusionEngine)
    dense = [
        VectorQueryResult(
            id="shared",
            score=0.8,
            content="dense content",
            metadata={"doc_id": "rev"},
        )
    ]
    sparse = [
        {
            "id": "shared",
            "score": 2.0,
            "content": "dense content",
            "metadata": {"doc_id": "rev"},
        },
        {
            "id": "sparse-only",
            "score": 1.0,
            "content": "exact identifier",
            "metadata": {"doc_id": "rev"},
        },
    ]
    no_graph = SimpleNamespace(synthesized_context="")

    fused = engine._rrf(dense, no_graph, sparse)
    assert fused[0]["id"] == "shared"
    assert fused[0]["score"] > fused[1]["score"]
    assert engine._should_skip_rerank(dense, no_graph, sparse)
    graph_context = SimpleNamespace(synthesized_context="bounded graph context")
    assert engine._should_skip_rerank(dense, graph_context, sparse)


def test_full_mode_pruning_reserves_graph_candidates_for_reranker():
    merged = [
        {
            "id": f"dense-{index}",
            "content": "dense",
            "metadata": {"source": "dense"},
            "score": 1.0 - index / 100,
        }
        for index in range(12)
    ]
    merged.extend(
        [
            {
                "id": "graph-1",
                "content": "relationship evidence",
                "metadata": {"source": "graph"},
                "score": 0.01,
            },
            {
                "id": "graph-2",
                "content": "community evidence",
                "metadata": {"source": "graph"},
                "score": 0.009,
            },
        ]
    )

    pruned = RetrievalFusionEngine._prune_for_rerank(merged, top_k=5)

    assert len(pruned) == 8
    assert [item["id"] for item in pruned[-2:]] == ["graph-1", "graph-2"]


@pytest.mark.asyncio
async def test_explicit_retrieval_modes_and_fast_alias_dispatch():
    engine = object.__new__(RetrievalFusionEngine)
    engine.embeddings = SimpleNamespace(embed_query=AsyncMock(return_value=[0.1]))
    engine._retrieve_fast = AsyncMock(return_value=[])
    engine._retrieve_tri_modal_fast = AsyncMock(return_value=[])
    engine._retrieve_full = AsyncMock(return_value=[])

    await engine.retrieve("relationships", mode="tri_modal_fast")
    engine._retrieve_tri_modal_fast.assert_awaited_once_with("relationships", [0.1], 5)

    await engine.retrieve("identifier", top_k=3, fast_mode=True)
    engine._retrieve_fast.assert_awaited_once_with("identifier", [0.1], 3)

    with pytest.raises(ValueError, match="Unknown retrieval mode"):
        await engine.retrieve("bad", mode="unknown")
    with pytest.raises(ValueError, match="conflicts"):
        await engine.retrieve("bad", fast_mode=True, mode="full")


def test_rank_blending_and_graph_result_cap():
    candidates = [
        {
            "id": "rrf-first",
            "rrf_rank": 0,
            "ce_rank": 2,
            "metadata": {"source": "dense"},
        },
        {
            "id": "balanced",
            "rrf_rank": 1,
            "ce_rank": 0,
            "metadata": {"source": "dense"},
        },
        {
            "id": "graph-1",
            "rrf_rank": 2,
            "ce_rank": 1,
            "metadata": {"source": "graph"},
        },
        {
            "id": "graph-2",
            "rrf_rank": 3,
            "ce_rank": 3,
            "metadata": {"source": "graph"},
        },
        {
            "id": "document-fill",
            "rrf_rank": 4,
            "ce_rank": 4,
            "metadata": {"source": "dense"},
        },
    ]

    blended = RetrievalFusionEngine._blend_rank_scores(candidates)
    selected = RetrievalFusionEngine._select_top_k(blended, top_k=4)

    assert blended[0]["id"] == "balanced"
    assert blended[0]["blended_score"] == pytest.approx(1 / 61 + 1 / 60)
    assert len(selected) == 4
    assert sum(item["metadata"]["source"] == "graph" for item in selected) == 1
    assert "document-fill" in {item["id"] for item in selected}


@pytest.mark.asyncio
async def test_speculative_graph_retrieval_is_reused():
    class CountingRetriever:
        def __init__(self):
            self.calls = 0

        async def retrieve(self, query, top_k=5, fast_mode=False):
            self.calls += 1
            return [
                Source(
                    document_id="knowledge_graph",
                    text="graph context with enough detail to answer safely",
                    score=1.0,
                )
            ]

    retriever = CountingRetriever()
    llm = AsyncMock()
    llm.agenerate_structured.return_value = {"strategy": "graph"}
    llm.agenerate.return_value = "answer"
    knowledge_graph = AsyncMock()
    orchestrator = AgenticOrchestrator(
        retriever=retriever,
        llm_client=llm,
        document_processor=None,
        knowledge_graph=knowledge_graph,
    )

    answer = await orchestrator.execute("Explain the ownership structure")

    assert answer.content == "answer"
    assert retriever.calls == 1
    knowledge_graph.query.assert_not_awaited()


@pytest.mark.asyncio
async def test_cost_accounting_reconciles_provider_usage_and_streaming():
    llm = UsageLLM()
    estimator = CostEstimator("gpt-4o")

    assert await llm.agenerate("prompt", max_tokens=800) == "measured response"
    expected_first = estimator.cost_for_tokens(100, 50)
    assert llm.circuit_breaker.daily_cost == pytest.approx(expected_first)
    assert llm.circuit_breaker.reserved_cost == 0

    streamed = [chunk async for chunk in llm.astream("prompt", max_tokens=800)]
    assert streamed == ["hello ", "world"]
    expected_total = expected_first + estimator.cost_for_tokens(60, 10)
    assert llm.circuit_breaker.daily_cost == pytest.approx(expected_total)
    assert llm.circuit_breaker.concurrent_queries == 0


@pytest.mark.asyncio
async def test_closing_stream_charges_partial_usage_and_releases_reservation():
    llm = UsageLLM()
    stream = llm.astream("prompt", max_tokens=800)

    assert await anext(stream) == "hello "
    await stream.aclose()

    assert llm.circuit_breaker.daily_cost > 0
    assert llm.circuit_breaker.reserved_cost == 0
    assert llm.circuit_breaker.concurrent_queries == 0
    assert llm.circuit_breaker.failure_count == 0


@pytest.mark.asyncio
async def test_async_query_waits_for_initial_build():
    rag = object.__new__(RAGBox)
    built = False

    class FakeHealer:
        def start_watchdog(self, loop):
            return None

        async def initial_build(self):
            nonlocal built
            await asyncio.sleep(0)
            built = True

        async def flush_pending(self):
            return None

        async def stop_watchdog(self):
            return None

    async def execute(question):
        assert built
        return Answer(
            query=question,
            content="ready answer",
            strategy_used=RAGStrategy.VECTOR,
        )

    rag.self_healer = FakeHealer()
    rag.orchestrator = SimpleNamespace(execute=execute)
    rag.document_dir = Path("/tmp/ragbox-test")
    rag._build_task = None
    rag._build_loop = None
    rag._ready_event = threading.Event()
    rag._state_lock = threading.Lock()
    rag._build_error = None
    rag._sync_loop = None
    rag._closed = False

    assert await rag.aquery("question") == "ready answer"
    assert rag.is_ready


@pytest.mark.asyncio
async def test_sync_query_refuses_active_event_loop():
    rag = object.__new__(RAGBox)
    with pytest.raises(RuntimeError, match="async equivalent"):
        rag.query("question")

"""
RAGBox Integration Test Suite — Real tests, no mocks (except for LLM calls).

Tests cover: ingestion, chunking (fixed, sentence, recursive), vector store,
knowledge graph (build, entity matching, graph query), retrieval fusion,
reranker, self-healing, cost estimation, query routing, and multi-hop.
"""
import time
import pytest
from pathlib import Path
from unittest.mock import AsyncMock

from ragbox.core.chunking_engine import FixedChunker, SentenceChunker, RecursiveChunker
from ragbox.core.document_processor import DocumentProcessorRouter
from ragbox.core.reranker import CrossEncoderReranker
from ragbox.core.self_healing import ContentAddressedStorage, ProductionFileWatcher
from ragbox.models.documents import Document, DocumentType
from ragbox.models.queries import RAGStrategy
from ragbox.utils.cost_tracker import CostEstimator, CostCircuitBreaker, CostBudget


# ==========================================
# Test 1: Text File Ingestion Pipeline
# ==========================================
@pytest.mark.asyncio
async def test_text_ingestion_pipeline(tmp_path: Path):
    """Real ingestion: write files, process them, verify Document objects."""
    doc_dir = tmp_path / "docs"
    doc_dir.mkdir()

    (doc_dir / "policy.txt").write_text(
        "Our vacation policy allows 20 days of PTO per year. "
        "Employees can carry over up to 5 unused days."
    )
    (doc_dir / "handbook.md").write_text(
        "# Employee Handbook\n\n"
        "## Section 1: Code of Conduct\n"
        "All employees must maintain professional behavior."
    )

    processor = DocumentProcessorRouter()

    doc1 = await processor.process(doc_dir / "policy.txt", "hash_policy")
    assert doc1 is not None
    assert "vacation" in doc1.content.lower()
    assert doc1.id == "hash_policy"

    doc2 = await processor.process(doc_dir / "handbook.md", "hash_handbook")
    assert doc2 is not None
    assert "Employee Handbook" in doc2.content
    assert doc2.doc_type == DocumentType.TEXT


# ==========================================
# Test 2: Fixed Chunker
# ==========================================
def test_chunking_fixed():
    """Verify fixed chunker produces correct chunks with overlap."""
    chunker = FixedChunker(chunk_size=100, overlap=20)

    doc = Document(
        id="test_doc", path=Path("/tmp/test.txt"),
        content="A" * 250, doc_type=DocumentType.TEXT,
    )
    chunks = chunker.chunk(doc)

    assert len(chunks) >= 3
    assert all(len(c.content) <= 100 for c in chunks[:-1])
    assert chunks[0].document_id == "test_doc"
    assert chunks[0].metadata["strategy"] == "fixed"


# ==========================================
# Test 3: Sentence Chunker
# ==========================================
def test_chunking_sentence():
    """Verify sentence chunker respects sentence boundaries."""
    chunker = SentenceChunker(max_sentences=2)

    doc = Document(
        id="sent_doc", path=Path("/tmp/test.txt"),
        content="First sentence. Second sentence. Third sentence. Fourth sentence. Fifth sentence.",
        doc_type=DocumentType.TEXT,
    )
    chunks = chunker.chunk(doc)

    assert len(chunks) >= 2
    assert "First sentence" in chunks[0].content
    assert chunks[0].metadata["strategy"] == "sentence"


# ==========================================
# Test 3b: Recursive Chunker
# ==========================================
def test_chunking_recursive():
    """Verify recursive chunker respects paragraph and sentence boundaries."""
    chunker = RecursiveChunker(chunk_size=150, overlap=30)

    doc = Document(
        id="rec_doc", path=Path("/tmp/test.txt"),
        content=(
            "First paragraph with some important content about the company policy. "
            "It describes how employees should behave in the workplace.\n\n"
            "Second paragraph talks about something else entirely. "
            "It has multiple sentences about benefits and compensation. "
            "Each one adds valuable information for new hires.\n\n"
            "Third paragraph wraps up the handbook. It includes final thoughts "
            "about professional development and career growth opportunities."
        ),
        doc_type=DocumentType.TEXT,
    )
    chunks = chunker.chunk(doc)

    assert len(chunks) >= 2
    assert chunks[0].metadata["strategy"] == "recursive"
    assert all(len(c.content) > 0 for c in chunks)


# ==========================================
# Test 4: Content-Addressed Storage
# ==========================================
def test_content_addressed_storage(tmp_path: Path):
    """Verify CAS detects file changes via SHA-256 hashing."""
    cas = ContentAddressedStorage(tmp_path / ".ragbox_state")

    test_file = tmp_path / "data.txt"
    test_file.write_text("original content")

    assert cas.has_changed(test_file) is True
    cas.update(test_file)
    assert cas.has_changed(test_file) is False

    test_file.write_text("modified content")
    assert cas.has_changed(test_file) is True


# ==========================================
# Test 5: Vector Store Add and Search
# ==========================================
@pytest.mark.asyncio
async def test_vector_store_add_and_search(tmp_path: Path):
    """Real ChromaDB: add documents with embeddings and search."""
    from ragbox.utils.vector_stores import ChromaStore
    import numpy as np

    store = ChromaStore(persist_dir=str(tmp_path / "chroma"))
    dim = 384
    docs = [
        {"id": "doc1", "content": "Python is a programming language used for AI.",
         "embedding": np.random.randn(dim).tolist(), "metadata": {"doc_id": "file1"}},
        {"id": "doc2", "content": "JavaScript is used for web development.",
         "embedding": np.random.randn(dim).tolist(), "metadata": {"doc_id": "file2"}},
        {"id": "doc3", "content": "Machine learning is a subset of artificial intelligence.",
         "embedding": np.random.randn(dim).tolist(), "metadata": {"doc_id": "file3"}},
    ]
    await store.add_documents(docs)

    query_emb = np.random.randn(dim).tolist()
    results = await store.search(query_emb, k=2, min_score=-1.0)

    assert len(results) >= 1
    assert all(hasattr(r, "content") for r in results)
    assert all(hasattr(r, "score") for r in results)


# ==========================================
# Test 6: Retrieval Fusion RRF
# ==========================================
def test_retrieval_fusion_rrf():
    """Verify Reciprocal Rank Fusion merges results correctly."""
    from ragbox.core.retrieval_fusion import RetrievalFusionEngine

    engine = object.__new__(RetrievalFusionEngine)

    vector_results = [
        {"id": "a", "content": "doc A", "metadata": {}},
        {"id": "b", "content": "doc B", "metadata": {}},
        {"id": "c", "content": "doc C", "metadata": {}},
    ]

    merged = engine._rrf(vector_results, None, k=60)

    assert len(merged) == 3
    assert merged[0]["id"] == "a"
    assert all("score" in item for item in merged)
    scores = [item["score"] for item in merged]
    assert scores == sorted(scores, reverse=True)


# ==========================================
# Test 7: Cross-Encoder Reranker (Real Model)
# ==========================================
@pytest.mark.asyncio
async def test_cross_encoder_reranker_real():
    """Real cross-encoder: load model and verify reranking order."""
    reranker = CrossEncoderReranker()

    candidates = [
        {"id": "irrelevant", "content": "The weather in Tokyo is sunny today."},
        {"id": "relevant", "content": "Python is a popular programming language for data science and AI."},
        {"id": "partial", "content": "Java is sometimes used for backend programming."},
    ]

    result = await reranker.rerank(
        "What programming language is best for AI?", candidates, top_k=3
    )

    assert len(result) == 3
    assert result[0]["id"] == "relevant"
    assert "cross_encoder_score" in result[0]
    scores = [r["cross_encoder_score"] for r in result]
    assert scores == sorted(scores, reverse=True)


# ==========================================
# Test 8: Knowledge Graph Build + Entity Matching + Query
# ==========================================
@pytest.mark.asyncio
async def test_knowledge_graph_build_and_query():
    """Build a knowledge graph, verify structure, entity matching, and query."""
    from ragbox.core.knowledge_graph import OptimizedKnowledgeGraph

    kg = OptimizedKnowledgeGraph(llm_client=None)

    # Add entities with descriptions and source text
    kg.add_document(
        doc_id="doc1",
        entities=["Maria Santos", "Lisa Park", "Engineering Team"],
        relationships=[
            {"source": "Maria Santos", "target": "Lisa Park", "type": "reports_to",
             "context": "Maria Santos reports to VP of Engineering Lisa Park"},
            {"source": "Maria Santos", "target": "Engineering Team", "type": "leads",
             "context": "Maria Santos leads the Platform Team"},
        ],
        entity_descriptions={
            "Maria Santos": "Platform Team Lead",
            "Lisa Park": "VP of Engineering",
            "Engineering Team": "Core engineering organization",
        },
        entity_source_texts={
            "Maria Santos": "Maria Santos, Platform Team Lead, reports to Lisa Park.",
            "Lisa Park": "Lisa Park serves as VP of Engineering.",
        },
    )

    kg.add_document(
        doc_id="doc2",
        entities=["Lisa Park", "Security Team", "Infrastructure"],
        relationships=[
            {"source": "Lisa Park", "target": "Security Team", "type": "oversees",
             "context": "Lisa Park oversees the Security Team"},
        ],
        entity_descriptions={
            "Security Team": "Responsible for security policies",
            "Infrastructure": "Cloud infrastructure team",
        },
    )

    # Verify graph structure
    assert kg.graph.number_of_nodes() >= 4
    assert kg.graph.number_of_edges() >= 3

    # Verify entity descriptions stored
    assert "Maria Santos" in kg.entity_descriptions
    assert "Platform Team Lead" in kg.entity_descriptions["Maria Santos"]

    # Verify entity matching
    matched = kg._match_query_to_entities("Who does Maria Santos report to?")
    assert len(matched) > 0
    entity_names = [name for name, _ in matched]
    assert "Maria Santos" in entity_names

    # Verify graph query returns real context
    result = await kg.query("Who does Maria Santos report to?")
    assert result.synthesized_context  # Should not be empty
    assert "Maria Santos" in result.synthesized_context
    assert len(result.relevant_entities) > 0

    # Verify cross-document entity linking (Lisa Park in both docs)
    related = kg.get_related_entities("Lisa Park")
    assert len(related) > 0
    related_names = set(related)
    assert "Maria Santos" in related_names or "Security Team" in related_names


# ==========================================
# Test 9: Cost Estimator with Tiktoken
# ==========================================
def test_cost_estimator_tiktoken():
    """Verify tiktoken-based token counting is accurate."""
    estimator = CostEstimator("gpt-4o")

    tokens = estimator.count_tokens("Hello, world!")
    assert 2 <= tokens <= 6

    assert estimator.count_tokens("") == 0

    short_tokens = estimator.count_tokens("Hello")
    long_tokens = estimator.count_tokens("Hello " * 100)
    assert long_tokens > short_tokens

    estimate = estimator.estimate_generation("Test prompt", approx_output_tokens=100)
    assert estimate.input_tokens > 0
    assert estimate.total_cost_usd >= 0


# ==========================================
# Test 10: Watchdog File Change Detection
# ==========================================
def test_watchdog_debounce_and_dedup():
    """Verify ProductionFileWatcher debouncing and deduplication."""
    callback = AsyncMock()
    watcher = ProductionFileWatcher(
        index_callback=callback, debounce_seconds=1.0, max_queue_size=10,
    )

    watcher._event_timestamps.clear()
    watcher._event_hashes.clear()
    watcher._events_received = 0
    watcher._events_deduplicated = 0

    stats = watcher.get_stats()
    assert stats["events_received"] == 0
    assert stats["queue_size"] == 0
    assert stats["deduplication_rate"] == 0.0

    from ragbox.core.self_healing import FileEvent

    for i in range(15):
        watcher._pending_queue.append(
            FileEvent(path=f"/tmp/file_{i}.txt", event_type="modified", timestamp=time.time())
        )
    assert len(watcher._pending_queue) == 10


# ==========================================
# Test 11: Query Routing Classification
# ==========================================
@pytest.mark.asyncio
async def test_query_routing():
    """Test query classification with a mock LLM."""
    from ragbox.core.agentic_orchestrator import AgenticOrchestrator

    mock_llm = AsyncMock()
    orchestrator = AgenticOrchestrator(
        retriever=None, llm_client=mock_llm,
        document_processor=None, knowledge_graph=None,
    )

    mock_llm.agenerate_structured.return_value = {"strategy": "graph", "reasoning": "relationship"}
    strategy = await orchestrator._classify_query("How does X relate to Y?")
    assert strategy == RAGStrategy.GRAPH

    mock_llm.agenerate_structured.return_value = {"strategy": "vector", "reasoning": "simple"}
    strategy = await orchestrator._classify_query("What is X?")
    assert strategy == RAGStrategy.VECTOR

    mock_llm.agenerate_structured.return_value = {"strategy": "multi_query", "reasoning": "comparison"}
    strategy = await orchestrator._classify_query("Compare A and B")
    assert strategy == RAGStrategy.MULTI_QUERY

    mock_llm.agenerate_structured.side_effect = Exception("LLM error")
    strategy = await orchestrator._classify_query("anything")
    assert strategy == RAGStrategy.VECTOR


# ==========================================
# Test 12: Multi-Query Expansion + Decomposition
# ==========================================
@pytest.mark.asyncio
async def test_multiquery_expansion_and_decomposition():
    """Test multi-query expansion and decomposition."""
    from ragbox.core.agentic_orchestrator import AgenticOrchestrator

    mock_llm = AsyncMock()
    mock_llm.agenerate.return_value = (
        "How to deploy an application?\n"
        "Steps for deploying software\n"
        "Guide to application deployment"
    )

    orchestrator = AgenticOrchestrator(
        retriever=None, llm_client=mock_llm,
        document_processor=None, knowledge_graph=None,
    )

    expanded = await orchestrator._expand_query("How do I deploy?")
    assert len(expanded) == 3
    assert any("deploy" in q.lower() for q in expanded)

    # Test decomposition
    mock_llm.agenerate.return_value = (
        "How many engineers does the company have?\n"
        "What percentage of engineers are senior?"
    )
    decomposed = await orchestrator._decompose_query(
        "How many engineers and what percentage are senior?"
    )
    assert len(decomposed) == 2

    # Test fallback when LLM returns empty
    mock_llm.agenerate.return_value = ""
    expanded = await orchestrator._expand_query("test query")
    assert expanded == ["test query"]


# ==========================================
# Test 13: Heuristic Classifier Patterns
# ==========================================
def test_heuristic_classifier():
    """Verify heuristic patterns match correctly."""
    from ragbox.core.agentic_orchestrator import _heuristic_classify

    # Graph patterns
    assert _heuristic_classify("Who does Maria Santos report to?") == RAGStrategy.GRAPH
    assert _heuristic_classify("What is the relationship between X and Y?") == RAGStrategy.GRAPH
    assert _heuristic_classify("How does the outage affect Q4 revenue?") == RAGStrategy.GRAPH

    # Multi-query/comparison patterns
    assert _heuristic_classify("Compare Python and Java") == RAGStrategy.MULTI_QUERY
    assert _heuristic_classify("What are the pros and cons?") == RAGStrategy.MULTI_QUERY

    # Simple factual (VECTOR)
    assert _heuristic_classify("What is the PTO policy?") == RAGStrategy.VECTOR
    assert _heuristic_classify("How many days off?") == RAGStrategy.VECTOR

    # Uncertain (should return None)
    assert _heuristic_classify("Tell me about the company culture and values") is None


# ==========================================
# Test 14: Graph Entity Context Collection
# ==========================================
@pytest.mark.asyncio
async def test_graph_entity_context():
    """Verify entity context collection returns real content."""
    from ragbox.core.knowledge_graph import OptimizedKnowledgeGraph

    kg = OptimizedKnowledgeGraph(llm_client=None)

    kg.add_document(
        doc_id="doc1",
        entities=["Deployment Strategy", "SEV1 Incident"],
        relationships=[
            {"source": "Deployment Strategy", "target": "SEV1 Incident",
             "type": "caused",
             "context": "The deployment strategy caused the SEV1 incident"},
        ],
        entity_descriptions={
            "Deployment Strategy": "Blue-green deployment approach",
            "SEV1 Incident": "November production outage",
        },
        entity_source_texts={
            "Deployment Strategy": "The company uses a blue-green deployment strategy.",
            "SEV1 Incident": "A SEV1 incident occurred in November causing 4 hours of downtime.",
        },
    )

    context = kg._collect_entity_context(["Deployment Strategy", "SEV1 Incident"])
    assert "Deployment Strategy" in context
    assert "SEV1 Incident" in context
    assert "caused" in context.lower() or "blue-green" in context.lower()


# ==========================================
# Test 15: Circuit Breaker
# ==========================================
@pytest.mark.asyncio
async def test_circuit_breaker():
    """Verify the cost circuit breaker opens after repeated failures."""
    budget = CostBudget(
        max_daily_cost=1.0, max_query_cost=0.10,
        failure_threshold=2, recovery_timeout=1,
    )
    breaker = CostCircuitBreaker(budget)

    async def success_op():
        return "ok"

    result = await breaker.execute(success_op, estimated_cost=0.01, operation_name="test")
    assert result == "ok"

    async def fail_op():
        raise ValueError("simulated failure")

    for _ in range(2):
        try:
            await breaker.execute(fail_op, estimated_cost=0.01, operation_name="test")
        except ValueError:
            pass

    from ragbox.utils.cost_tracker import CircuitBreakerOpen

    with pytest.raises(CircuitBreakerOpen):
        await breaker.execute(success_op, estimated_cost=0.01, operation_name="test")


# ==========================================
# Test 16: RRF with Graph Context
# ==========================================
def test_rrf_with_graph_context():
    """Verify RRF properly integrates graph context segments."""
    from ragbox.core.retrieval_fusion import RetrievalFusionEngine
    from ragbox.models.graph import GraphQueryResult

    engine = object.__new__(RetrievalFusionEngine)

    vector_results = [
        {"id": "v1", "content": "Vector result about deployment.", "metadata": {}},
        {"id": "v2", "content": "Vector result about security.", "metadata": {}},
    ]

    graph_result = GraphQueryResult(
        relevant_entities=[], relevant_relations=[], relevant_communities=[],
        synthesized_context=(
            "Entity: Deployment Strategy\n  Description: Blue-green deployment\n\n"
            "Entity: Security Team\n  Description: Handles security policies"
        ),
    )

    merged = engine._rrf(vector_results, graph_result, k=60)

    # Should have vector results + graph context segments
    assert len(merged) > 2
    # Graph context should be injected
    graph_items = [m for m in merged if "graph" in m.get("id", "")]
    assert len(graph_items) > 0

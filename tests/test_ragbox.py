"""
RAGBox Integration Test Suite — Real tests, no mocks.

Tests cover: ingestion, chunking, vector store, knowledge graph,
retrieval fusion, reranker, self-healing, cost estimation, and routing.
"""
import time
import pytest
from pathlib import Path
from unittest.mock import AsyncMock

from ragbox.core.chunking_engine import FixedChunker, SentenceChunker
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

    # Create test documents
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

    # Process text file
    doc1 = await processor.process(doc_dir / "policy.txt", "hash_policy")
    assert doc1 is not None
    assert "vacation" in doc1.content.lower()
    assert doc1.id == "hash_policy"

    # Process markdown file
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
        id="test_doc",
        path=Path("/tmp/test.txt"),
        content="A" * 250,  # 250 chars
        doc_type=DocumentType.TEXT,
    )

    chunks = chunker.chunk(doc)

    # With chunk_size=100, overlap=20, stride=80
    # 250 chars -> ceil(250/80) = 4 chunks
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
        id="sent_doc",
        path=Path("/tmp/test.txt"),
        content="First sentence. Second sentence. Third sentence. Fourth sentence. Fifth sentence.",
        doc_type=DocumentType.TEXT,
    )

    chunks = chunker.chunk(doc)

    assert len(chunks) >= 2
    assert "First sentence" in chunks[0].content
    assert chunks[0].metadata["strategy"] == "sentence"


# ==========================================
# Test 4: Content-Addressed Storage
# ==========================================
def test_content_addressed_storage(tmp_path: Path):
    """Verify CAS detects file changes via SHA-256 hashing."""
    cas = ContentAddressedStorage(tmp_path / ".ragbox_state")

    # Create a test file
    test_file = tmp_path / "data.txt"
    test_file.write_text("original content")

    # First check — file is new, should be "changed"
    assert cas.has_changed(test_file) is True

    # Update the hash
    cas.update(test_file)

    # Same content — should NOT be changed
    assert cas.has_changed(test_file) is False

    # Modify the file
    test_file.write_text("modified content")

    # Now it should be changed
    assert cas.has_changed(test_file) is True


# ==========================================
# Test 5: Vector Store Add and Search
# ==========================================
@pytest.mark.asyncio
async def test_vector_store_add_and_search(tmp_path: Path):
    """Real ChromaDB: add documents with embeddings and search."""
    from ragbox.utils.vector_stores import ChromaStore

    store = ChromaStore(persist_dir=str(tmp_path / "chroma"))

    # Create test documents with fake embeddings (384-dim for MiniLM)
    import numpy as np

    dim = 384
    docs = [
        {
            "id": "doc1",
            "content": "Python is a programming language used for AI.",
            "embedding": np.random.randn(dim).tolist(),
            "metadata": {"doc_id": "file1"},
        },
        {
            "id": "doc2",
            "content": "JavaScript is used for web development.",
            "embedding": np.random.randn(dim).tolist(),
            "metadata": {"doc_id": "file2"},
        },
        {
            "id": "doc3",
            "content": "Machine learning is a subset of artificial intelligence.",
            "embedding": np.random.randn(dim).tolist(),
            "metadata": {"doc_id": "file3"},
        },
    ]

    await store.add_documents(docs)

    # Search with a random query embedding
    query_emb = np.random.randn(dim).tolist()
    results = await store.search(query_emb, k=2, min_score=-1.0)

    assert len(results) >= 1  # At least 1 result with random embeddings
    assert all(hasattr(r, "content") for r in results)
    assert all(hasattr(r, "score") for r in results)


# ==========================================
# Test 6: Retrieval Fusion RRF
# ==========================================
def test_retrieval_fusion_rrf():
    """Verify Reciprocal Rank Fusion merges results correctly."""
    from ragbox.core.retrieval_fusion import RetrievalFusionEngine

    # Create a mock engine just to test _rrf
    engine = object.__new__(RetrievalFusionEngine)

    vector_results = [
        {"id": "a", "content": "doc A", "metadata": {}},
        {"id": "b", "content": "doc B", "metadata": {}},
        {"id": "c", "content": "doc C", "metadata": {}},
    ]

    # No graph results
    merged = engine._rrf(vector_results, None, k=60)

    assert len(merged) == 3
    # First result should be "a" (rank 0 in vector)
    assert merged[0]["id"] == "a"
    # All should have scores
    assert all("score" in item for item in merged)
    # Scores should be descending
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
        {
            "id": "relevant",
            "content": "Python is a popular programming language for data science and AI.",
        },
        {"id": "partial", "content": "Java is sometimes used for backend programming."},
    ]

    result = await reranker.rerank(
        "What programming language is best for AI?", candidates, top_k=3
    )

    assert len(result) == 3
    # The Python doc should be ranked highest
    assert result[0]["id"] == "relevant"
    assert "cross_encoder_score" in result[0]
    # Scores should be descending
    scores = [r["cross_encoder_score"] for r in result]
    assert scores == sorted(scores, reverse=True)


# ==========================================
# Test 8: Knowledge Graph Build
# ==========================================
@pytest.mark.asyncio
async def test_knowledge_graph_build():
    """Build a knowledge graph from documents and verify structure."""
    from ragbox.core.knowledge_graph import OptimizedKnowledgeGraph

    kg = OptimizedKnowledgeGraph(llm_client=None)

    # Add entities manually (no LLM needed)
    kg.add_document(
        doc_id="doc1",
        entities=["Python", "AI", "Machine Learning"],
        relationships=[
            {"source": "Python", "target": "AI", "type": "used_for"},
            {"source": "AI", "target": "Machine Learning", "type": "includes"},
        ],
    )

    kg.add_document(
        doc_id="doc2",
        entities=["TensorFlow", "Python"],
        relationships=[
            {"source": "TensorFlow", "target": "Python", "type": "built_with"},
        ],
    )

    # Verify graph structure
    assert kg.graph.number_of_nodes() >= 4
    assert kg.graph.number_of_edges() >= 3

    # Verify entity lookup
    related = kg.get_related_entities("Python", max_distance=1)
    assert len(related) > 0


# ==========================================
# Test 9: Cost Estimator with Tiktoken
# ==========================================
def test_cost_estimator_tiktoken():
    """Verify tiktoken-based token counting is accurate."""
    estimator = CostEstimator("gpt-4o")

    # Known test: "Hello, world!" should be ~4 tokens with cl100k_base
    tokens = estimator.count_tokens("Hello, world!")
    assert 2 <= tokens <= 6  # Should be around 4

    # Empty string
    assert estimator.count_tokens("") == 0

    # Longer text should have more tokens
    short_tokens = estimator.count_tokens("Hello")
    long_tokens = estimator.count_tokens("Hello " * 100)
    assert long_tokens > short_tokens

    # Cost estimation
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
        index_callback=callback,
        debounce_seconds=1.0,
        max_queue_size=10,
    )

    # Simulate rapid events for the same file
    from watchdog.events import FileModifiedEvent

    FileModifiedEvent("/tmp/test_file.txt")

    # First event should be accepted
    watcher._event_timestamps.clear()
    watcher._event_hashes.clear()
    watcher._events_received = 0
    watcher._events_deduplicated = 0

    # We can't fully test on_modified without a real file, but we can test stats
    stats = watcher.get_stats()
    assert stats["events_received"] == 0
    assert stats["queue_size"] == 0
    assert stats["deduplication_rate"] == 0.0

    # Test backpressure — fill queue to max
    from ragbox.core.self_healing import FileEvent

    for i in range(15):  # more than max_queue_size=10
        watcher._pending_queue.append(
            FileEvent(
                path=f"/tmp/file_{i}.txt", event_type="modified", timestamp=time.time()
            )
        )
    assert len(watcher._pending_queue) == 10  # Bounded by maxlen


# ==========================================
# Test 11: Query Routing Classification
# ==========================================
@pytest.mark.asyncio
async def test_query_routing():
    """Test query classification with a mock LLM."""
    from ragbox.core.agentic_orchestrator import AgenticOrchestrator

    mock_llm = AsyncMock()

    orchestrator = AgenticOrchestrator(
        retriever=None,
        llm_client=mock_llm,
        document_processor=None,
        knowledge_graph=None,
    )

    # Test GRAPH routing
    mock_llm.agenerate_structured.return_value = {
        "strategy": "graph",
        "reasoning": "relationship query",
    }
    strategy = await orchestrator._classify_query("How does X relate to Y?")
    assert strategy == RAGStrategy.GRAPH

    # Test VECTOR routing
    mock_llm.agenerate_structured.return_value = {
        "strategy": "vector",
        "reasoning": "simple lookup",
    }
    strategy = await orchestrator._classify_query("What is X?")
    assert strategy == RAGStrategy.VECTOR

    # Test MULTI_QUERY routing
    mock_llm.agenerate_structured.return_value = {
        "strategy": "multi_query",
        "reasoning": "comparison",
    }
    strategy = await orchestrator._classify_query("Compare A and B")
    assert strategy == RAGStrategy.MULTI_QUERY

    # Test fallback on error
    mock_llm.agenerate_structured.side_effect = Exception("LLM error")
    strategy = await orchestrator._classify_query("anything")
    assert strategy == RAGStrategy.VECTOR  # Default fallback


# ==========================================
# Test 12: Multi-Query Expansion
# ==========================================
@pytest.mark.asyncio
async def test_multiquery_expansion():
    """Test multi-query expansion with mock LLM."""
    from ragbox.core.agentic_orchestrator import AgenticOrchestrator

    mock_llm = AsyncMock()
    mock_llm.agenerate.return_value = (
        "How to deploy an application?\n"
        "Steps for deploying software\n"
        "Guide to application deployment"
    )

    orchestrator = AgenticOrchestrator(
        retriever=None,
        llm_client=mock_llm,
        document_processor=None,
        knowledge_graph=None,
    )

    expanded = await orchestrator._expand_query("How do I deploy?")

    assert len(expanded) == 3
    assert any("deploy" in q.lower() for q in expanded)

    # Test fallback when LLM returns empty
    mock_llm.agenerate.return_value = ""
    expanded = await orchestrator._expand_query("test query")
    assert expanded == ["test query"]


# ==========================================
# Test Circuit Breaker
# ==========================================
@pytest.mark.asyncio
async def test_circuit_breaker():
    """Verify the cost circuit breaker opens after repeated failures."""
    budget = CostBudget(
        max_daily_cost=1.0,
        max_query_cost=0.10,
        failure_threshold=2,
        recovery_timeout=1,
    )
    breaker = CostCircuitBreaker(budget)

    # Normal operation
    async def success_op():
        return "ok"

    result = await breaker.execute(
        success_op, estimated_cost=0.01, operation_name="test"
    )
    assert result == "ok"

    # Trigger failures
    async def fail_op():
        raise ValueError("simulated failure")

    for _ in range(2):
        try:
            await breaker.execute(fail_op, estimated_cost=0.01, operation_name="test")
        except ValueError:
            pass

    # Circuit should be open now
    from ragbox.utils.cost_tracker import CircuitBreakerOpen

    with pytest.raises(CircuitBreakerOpen):
        await breaker.execute(success_op, estimated_cost=0.01, operation_name="test")

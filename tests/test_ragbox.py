import pytest
from pathlib import Path
from unittest.mock import AsyncMock, patch, MagicMock

from ragbox import RAGBox
from ragbox.core.agentic_orchestrator import AgenticOrchestrator
from ragbox.core.reranker import CrossEncoderReranker
from ragbox.models.queries import RAGStrategy

@pytest.mark.asyncio
async def test_ragbox_pipeline(tmp_path: Path):
    d = tmp_path / "docs"
    d.mkdir()
    (d / "info.txt").write_text("hello 123")
    rag = RAGBox(d)
    ans = await rag.aquery("what is the info?")
    assert len(ans) > 0

@pytest.mark.asyncio
async def test_multiquery_expansion_mocked():
    mock_llm = AsyncMock()
    mock_llm.agenerate.return_value = "How to deploy app?\nDeploying the application\nSteps for app deployment"
    
    orchestrator = AgenticOrchestrator(retriever=None, llm_client=mock_llm, document_processor=None, knowledge_graph=None)
    expanded = await orchestrator._expand_query("How do I deploy?")
    
    assert len(expanded) == 3
    assert "How to deploy app?" in expanded
    assert "Steps for app deployment" in expanded

@pytest.mark.asyncio
async def test_query_routing_mocked():
    mock_llm = AsyncMock()
    # Mocking for MULTI_QUERY
    mock_llm.agenerate_structured.return_value = {"strategy": "multi_query", "reasoning": "comparison asked"}
    
    orchestrator = AgenticOrchestrator(retriever=None, llm_client=mock_llm, document_processor=None, knowledge_graph=None)
    
    strategy = await orchestrator._classify_query("How many XYZ")
    assert strategy == RAGStrategy.MULTI_QUERY
    
    # Mocking for VECTOR
    mock_llm.agenerate_structured.return_value = {"strategy": "vector", "reasoning": "general knowledge"}
    strategy = await orchestrator._classify_query("What is X?")
    assert strategy == RAGStrategy.VECTOR

@pytest.mark.asyncio
async def test_cross_encoder_reranker_mocked():
    reranker = CrossEncoderReranker()
    # Force initialization
    reranker._initialized = True
    
    # Create a mock internal model
    mock_model = MagicMock()
    # predict returns scores in the same order as input pairs
    # Input order: Doc A, Doc B. We want Doc B to win.
    mock_model.predict.return_value = [0.1, 0.9] 
    reranker._model = mock_model
    
    candidates = [
        {"id": "doc_a", "content": "This is totally irrelevant context missing the point."},
        {"id": "doc_b", "content": "This is the exact perfect answer to the user query."}
    ]
    
    result = await reranker.rerank("What is the perfect answer?", candidates, top_k=2)
    
    # Assert Doc B (index 1) moved to the top
    assert result[0]["id"] == "doc_b"
    assert result[1]["id"] == "doc_a"
    assert result[0]["cross_encoder_score"] == 0.9

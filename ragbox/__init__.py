"""
RAGBox: Zero-Configuration Self-Building Agentic RAG System.

Provides the public RAGBox class for one-line integrations.
"""
import asyncio
from pathlib import Path
from typing import Any, Dict, Optional

from loguru import logger

from ragbox.config.defaults import Settings
from ragbox.core.agentic_orchestrator import AgenticOrchestrator
from ragbox.core.chunking_engine import ChunkingEngine
from ragbox.core.document_processor import DocumentProcessorRouter
from ragbox.core.knowledge_graph import OptimizedKnowledgeGraph
from ragbox.core.retrieval_fusion import RetrievalFusionEngine
from ragbox.core.self_healing import SelfHealer
from ragbox.models.queries import Answer
from ragbox.utils.embeddings import EmbeddingAutoDetector
from ragbox.utils.llm_clients import LLMAutoDetector
from ragbox.utils.vector_stores import VectorStoreAutoDetector


class RAGBox:
    """
    RAG-in-a-Box: Zero-Configuration Self-Building Agentic RAG System.
    """

    def __init__(
        self, document_dir: str | Path, config: Optional[Settings] = None
    ) -> None:
        """
        Initialize a complete RAG system pointing to a directory.

        Args:
            document_dir: Path to the directory containing documents.
            config: Optional configuration overrides.
        """
        self.document_dir = Path(document_dir)
        self.config = config or Settings()

        # Initialize Core Layers Auto-magically
        self.llm_client = LLMAutoDetector.detect(self.config)
        self.embedding_provider = EmbeddingAutoDetector.detect(self.config)
        self.vector_store = VectorStoreAutoDetector.detect(
            self.config, self.embedding_provider
        )

        self.document_processor = DocumentProcessorRouter()
        self.chunking_engine = ChunkingEngine(self.llm_client, self.embedding_provider)
        self.knowledge_graph = OptimizedKnowledgeGraph(llm_client=self.llm_client)

        self.retriever = RetrievalFusionEngine(
            vector_store=self.vector_store,
            knowledge_graph=self.knowledge_graph,
            embedding_provider=self.embedding_provider,
            llm_client=self.llm_client,
        )
        self.orchestrator = AgenticOrchestrator(
            retriever=self.retriever,
            llm_client=self.llm_client,
            document_processor=self.document_processor,
            knowledge_graph=self.knowledge_graph,
        )
        self.self_healer = SelfHealer(
            document_dir=self.document_dir,
            document_processor=self.document_processor,
            chunking_engine=self.chunking_engine,
            vector_store=self.vector_store,
            knowledge_graph=self.knowledge_graph,
        )

        # Start initial build and self-healing daemon
        logger.info(f"Initializing RAGBox for {self.document_dir}")
        self._ensure_built()

    def _ensure_built(self) -> None:
        """Start the build process without blocking the main thread."""
        try:
            loop = asyncio.get_running_loop()
        except RuntimeError:
            logger.warning("RAGBox initialized outside an active event loop. Build delayed until queried.")
            return

        # Fire and forget the initial build in the background
        asyncio.create_task(self.self_healer.initial_build())
        
        # Start watchdog
        self.self_healer.start_watchdog()

    def query(self, question: str) -> str:
        """
        Query the RAGBox system synchronously.

        Args:
            question: The user's query.

        Returns:
            The direct answer string.
        """
        try:
            loop = asyncio.get_running_loop()
            if loop.is_running():
                raise RuntimeError(
                    "Cannot call synchronous `query` inside an active event loop. Use `aquery`."
                )
        except RuntimeError:
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)

        answer: Answer = loop.run_until_complete(self.orchestrator.execute(question))
        return answer.content

    async def aquery(self, question: str) -> str:
        """
        Async evaluation of a query.

        Args:
            question: The user's query.

        Returns:
            The direct answer string.
        """
        answer: Answer = await self.orchestrator.execute(question)
        return answer.content

    def estimate_cost(self, query: Optional[str] = None) -> str:
        """
        Estimate the cost of a query or indexing the corpus upfront.

        Args:
            query: The query to run. If None, estimates the cost of indexing the entire corpus.

        Returns:
            A string detailing the estimated cost.
        """
        from ragbox.utils.cost_tracker import CostEstimator

        # We assume the default model for estimating cost
        estimator = CostEstimator(
            self.llm_client.model_name
            if hasattr(self.llm_client, "model_name")
            else "gpt-4o"
        )

        if query is None:
            # Estimate cost of indexing the corpus based on file size
            total_bytes = 0
            for ext in self.document_processor.processors.keys():
                for p in self.document_dir.rglob(f"*{ext}"):
                    if p.is_file():
                        total_bytes += p.stat().st_size

            # Wild guess: 1 byte ~ 0.25 tokens
            estimated_tokens = int(total_bytes * 0.25)
            # Embedding cost
            embed_cost = (
                estimated_tokens / 1_000_000
            ) * 0.13  # large embedding assumption
            return f"Corpus Indexing Estimate: ~{estimated_tokens} tokens. Embedding Cost: ${embed_cost:.4f}"

        else:
            # Simulate retrieval and estimation for query
            try:
                loop = asyncio.get_running_loop()
                if loop.is_running():
                    raise RuntimeError(
                        "Cannot call synchronous `estimate_cost` inside loop."
                    )
            except RuntimeError:
                loop = asyncio.new_event_loop()
                asyncio.set_event_loop(loop)

            # Retrieve mock sources
            sources = loop.run_until_complete(self.retriever.retrieve(query))
            context_text = "\n".join([s.text for s in sources])

            estimate = estimator.estimate_generation(
                prompt=f"{context_text}\n{query}", approx_output_tokens=500
            )
            return estimate.__str__()


__all__ = ["RAGBox"]

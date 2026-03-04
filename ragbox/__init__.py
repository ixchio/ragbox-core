"""
RAGBox: Batteries-Included, Developer-Friendly Agentic RAG System.

The RAG framework for people who don't want to think about RAG.
Provides the public RAGBox class for one-line integrations.
"""
import asyncio
from pathlib import Path
from typing import AsyncIterator, Optional

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
    RAG-in-a-Box: Batteries-Included, Developer-Friendly Agentic RAG System.

    Usage::

        rag = RAGBox("./my-docs")
        answer = rag.query("What is our vacation policy?")

        # Async streaming
        async for chunk in rag.astream("Summarize all reports"):
            print(chunk, end="")
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
            asyncio.get_running_loop()
        except RuntimeError:
            logger.warning(
                "RAGBox initialized outside an active event loop. Build delayed until queried."
            )
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

    async def astream(self, question: str) -> AsyncIterator[str]:
        """
        Stream the answer token-by-token using async iteration.

        Usage::

            async for chunk in rag.astream("What is X?"):
                print(chunk, end="", flush=True)

        Args:
            question: The user's query.

        Yields:
            String chunks of the answer as they are generated.
        """
        async for chunk in self.orchestrator.stream_execute(question):
            yield chunk

    def estimate_cost(self, query: Optional[str] = None) -> str:
        """
        Estimate the cost of a query or indexing the corpus.

        Uses tiktoken for accurate token counting instead of byte-level estimation.

        Args:
            query: The query to run. If None, estimates the cost of indexing the entire corpus.

        Returns:
            A string detailing the estimated cost.
        """
        from ragbox.utils.cost_tracker import CostEstimator

        model_name = (
            self.llm_client._model if hasattr(self.llm_client, "_model") else "gpt-4o"
        )
        estimator = CostEstimator(
            model_name if isinstance(model_name, str) else "gpt-4o"
        )

        if query is None:
            # Read actual file contents and tokenize for accurate count
            total_tokens = 0
            total_files = 0
            for ext in self.document_processor.processors.keys():
                for p in self.document_dir.rglob(f"*{ext}"):
                    if p.is_file():
                        try:
                            content = p.read_text(encoding="utf-8", errors="ignore")
                            total_tokens += estimator.count_tokens(content)
                            total_files += 1
                        except Exception:
                            # Fallback for binary files (images, etc.)
                            total_tokens += int(p.stat().st_size * 0.25)
                            total_files += 1

            # Embedding cost
            embed_cost = (total_tokens / 1_000_000) * 0.13
            return (
                f"Corpus Indexing Estimate: ~{total_tokens:,} tokens "
                f"across {total_files} files. "
                f"Embedding Cost: ${embed_cost:.4f}"
            )

        else:
            try:
                loop = asyncio.get_running_loop()
                if loop.is_running():
                    raise RuntimeError(
                        "Cannot call synchronous `estimate_cost` inside loop."
                    )
            except RuntimeError:
                loop = asyncio.new_event_loop()
                asyncio.set_event_loop(loop)

            sources = loop.run_until_complete(self.retriever.retrieve(query))
            context_text = "\n".join([s.text for s in sources])

            estimate = estimator.estimate_generation(
                prompt=f"{context_text}\n{query}", approx_output_tokens=500
            )
            return str(estimate)


__all__ = ["RAGBox"]

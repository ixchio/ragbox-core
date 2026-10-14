"""Live, citation-first repository retrieval for applications and coding agents."""

import asyncio
import threading
from pathlib import Path
from typing import AsyncIterator, Optional, Any, Coroutine

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
    A live hybrid retrieval and answer-generation engine over a local directory.

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
        self.retriever.set_active_revision_resolver(self.self_healer.is_active_revision)
        if self.self_healer.cas.sparse_search_available:
            self.retriever.set_sparse_search(self.self_healer.cas.search_fts)

        self._build_task: Optional[asyncio.Task] = None
        self._build_loop: Optional[asyncio.AbstractEventLoop] = None
        self._ready_event = threading.Event()
        self._state_lock = threading.Lock()
        self._build_error: Optional[BaseException] = None
        self._sync_loop: Optional[asyncio.AbstractEventLoop] = None
        self._closed = False

        logger.info(f"Initializing RAGBox for {self.document_dir}")
        self._ensure_built()

    def _ensure_built(self) -> None:
        """Start initialization when constructed inside a running async app."""
        try:
            loop = asyncio.get_running_loop()
        except RuntimeError:
            logger.info(
                "RAGBox initialized synchronously; the first query or ready() "
                "call will complete indexing."
            )
            return
        self._start_build(loop)

    def _start_build(self, loop: asyncio.AbstractEventLoop) -> asyncio.Task:
        with self._state_lock:
            if self._closed:
                raise RuntimeError("RAGBox is closed")
            if self._build_task is not None and not self._build_task.done():
                return self._build_task
            if self._ready_event.is_set() and self._build_error is None:

                async def already_ready() -> None:
                    return None

                return loop.create_task(already_ready())
            self._build_loop = loop
            self._build_task = loop.create_task(self._initialize())
            return self._build_task

    async def _initialize(self) -> None:
        try:
            self.self_healer.start_watchdog(asyncio.get_running_loop())
            await self.self_healer.initial_build()
            self._build_error = None
            logger.info(f"RAGBox ready for {self.document_dir}")
        except BaseException as exc:
            self._build_error = exc
            logger.exception(f"RAGBox initialization failed: {exc}")
            try:
                await self.self_healer.stop_watchdog()
            except Exception as shutdown_error:
                logger.error(
                    f"Failed to stop watcher after initialization error: "
                    f"{shutdown_error}"
                )
            raise
        finally:
            self._ready_event.set()

    @property
    def is_ready(self) -> bool:
        return self._ready_event.is_set() and self._build_error is None

    @property
    def status(self) -> str:
        if self._closed:
            return "closed"
        if self._build_error is not None:
            return "error"
        if self.is_ready:
            return "ready"
        return "building" if self._build_task is not None else "not_started"

    @property
    def build_error(self) -> Optional[str]:
        return str(self._build_error) if self._build_error is not None else None

    async def aready(self) -> None:
        """Wait until the initial index is complete or raise its build error."""
        if self._closed:
            raise RuntimeError("RAGBox is closed")
        if self.is_ready:
            return

        loop = asyncio.get_running_loop()
        task = self._build_task
        if task is None:
            task = self._start_build(loop)

        if task.get_loop() is loop:
            await task
        else:
            # The owning application loop performs the build; this loop waits
            # on a thread-safe completion signal rather than awaiting its Task.
            await asyncio.to_thread(self._ready_event.wait)

        if self._build_error is not None:
            raise RuntimeError("RAGBox initialization failed") from self._build_error

    def ready(self) -> None:
        """Synchronous readiness barrier."""
        self._run_sync(self.aready())

    def _run_sync(self, operation: Coroutine[Any, Any, Any]) -> Any:
        try:
            asyncio.get_running_loop()
        except RuntimeError:
            pass
        else:
            operation.close()
            raise RuntimeError(
                "Cannot call a synchronous RAGBox method inside an active "
                "event loop. Use the async equivalent."
            )

        if self._sync_loop is None or self._sync_loop.is_closed():
            self._sync_loop = asyncio.new_event_loop()
        asyncio.set_event_loop(self._sync_loop)
        return self._sync_loop.run_until_complete(operation)

    async def _prepare_query(self) -> None:
        await self.aready()
        # Let thread-safe watcher callbacks already queued on this loop land,
        # then make freshness a query-time barrier for synchronous callers.
        await asyncio.sleep(0)
        await self.self_healer.flush_pending()

    def query(self, question: str) -> str:
        """
        Query the RAGBox system synchronously.

        Args:
            question: The user's query.

        Returns:
            The direct answer string.
        """
        return self._run_sync(self.aquery(question))

    async def aquery(self, question: str) -> str:
        """
        Async evaluation of a query.

        Args:
            question: The user's query.

        Returns:
            The direct answer string.
        """
        await self._prepare_query()
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
        await self._prepare_query()
        async for chunk in self.orchestrator.stream_execute(question):
            yield chunk

    async def aestimate_cost(self, query: Optional[str] = None) -> str:
        """
        Estimate the cost of a query or indexing the corpus.

        Uses tiktoken for accurate token counting instead of byte-level estimation.

        Args:
            query: Query to run. If None, estimates indexing the entire corpus.

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
            await self._prepare_query()
            sources = await self.retriever.retrieve(query)
            context_text = "\n".join([s.text for s in sources])

            estimate = estimator.estimate_generation(
                prompt=f"{context_text}\n{query}", approx_output_tokens=500
            )
            return str(estimate)

    def estimate_cost(self, query: Optional[str] = None) -> str:
        """Synchronous wrapper for :meth:`aestimate_cost`."""
        return self._run_sync(self.aestimate_cost(query))

    async def aclose(self) -> None:
        if self._closed:
            return
        task = self._build_task
        if task is not None and not task.done():
            owner_loop = task.get_loop()
            current_loop = asyncio.get_running_loop()
            if owner_loop is current_loop:
                task.cancel()
                try:
                    await task
                except asyncio.CancelledError:
                    pass
            elif owner_loop.is_running():
                owner_loop.call_soon_threadsafe(task.cancel)
                await asyncio.to_thread(self._ready_event.wait, 5.0)
        await self.self_healer.stop_watchdog()
        self._closed = True

    def close(self) -> None:
        if self._closed:
            return
        self._run_sync(self.aclose())
        if self._sync_loop is not None and not self._sync_loop.is_closed():
            self._sync_loop.close()
        self._sync_loop = None


__all__ = ["RAGBox"]

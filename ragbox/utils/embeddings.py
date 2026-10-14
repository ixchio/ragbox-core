"""
Unified embedding interface for RAGBox.
Automatically falls back between local models and OpenAI based on env.
"""

import os
import asyncio
from abc import ABC, abstractmethod
from typing import TYPE_CHECKING, Any, List
from loguru import logger

from ragbox.config.defaults import Settings

if TYPE_CHECKING:
    from ragbox.models.chunks import Chunk
    from ragbox.models.documents import Document


def _cpu_thread_count() -> int:
    override = os.getenv("RAGBOX_ONNX_THREADS")
    if override:
        try:
            return max(1, int(override))
        except ValueError:
            logger.warning("Ignoring invalid RAGBOX_ONNX_THREADS value")
    return max(1, (os.cpu_count() or 2) // 2)


class EmbeddingProvider(ABC):
    """Abstract Base Class for embedding providers."""

    @abstractmethod
    async def embed_query(self, text: str) -> List[float]:
        """Embed a single query text."""
        pass

    @abstractmethod
    async def embed_documents(self, texts: List[str]) -> List[List[float]]:
        """Embed a batch of document texts."""
        pass

    @abstractmethod
    async def embed_chunks_with_context(
        self, document: "Document", chunks: List["Chunk"], llm_client: Any = None
    ) -> List[List[float]]:
        """
        Embed chunks while preserving document-level context.
        Local models: surrounding-window context enrichment.
        Cloud models: Contextual Retrieval (prepend document summary).
        """
        pass

    @property
    @abstractmethod
    def dimension(self) -> int:
        """Embedding vector width."""
        pass


class FastEmbedProvider(EmbeddingProvider):
    """Local CPU embeddings via FastEmbed's ONNX Runtime backend."""

    def __init__(self, model_name: str = "BAAI/bge-small-en-v1.5") -> None:
        try:
            from fastembed import TextEmbedding

            self._model_name = model_name
            self._model = TextEmbedding(
                model_name=model_name,
                threads=_cpu_thread_count(),
                providers=["CPUExecutionProvider"],
            )
            logger.debug(f"Initialized FastEmbedProvider with {model_name} on CPU")
        except ImportError:
            logger.error("Failed to import fastembed.")
            raise

    async def embed_query(self, text: str) -> List[float]:
        return await asyncio.to_thread(self._query_embed_sync, text)

    async def embed_documents(self, texts: List[str]) -> List[List[float]]:
        return await asyncio.to_thread(self._passage_embed_sync, texts)

    def _query_embed_sync(self, text: str) -> List[float]:
        return next(iter(self._model.query_embed(text))).tolist()

    def _passage_embed_sync(self, texts: List[str]) -> List[List[float]]:
        return [embedding.tolist() for embedding in self._model.passage_embed(texts)]

    async def embed_chunks_with_context(
        self, document: "Document", chunks: List["Chunk"], llm_client: Any = None
    ) -> List[List[float]]:
        """
        Context-enriched embedding: embeds each chunk with its surrounding
        document context via a sliding window, improving retrieval quality
        for chunks that reference entities or concepts defined elsewhere.
        """
        if not chunks:
            return []

        def _context_embed_sync() -> List[List[float]]:
            logger.info(f"Context-enriched embedding for {document.path}")
            doc_text = document.content
            embedded_chunks = []

            for chunk in chunks:
                start = chunk.metadata.get("start_idx", 0)
                end = chunk.metadata.get("end_idx", len(chunk.content))

                # Expand context window by 800 chars on each side
                context_start = max(0, start - 800)
                context_end = min(len(doc_text), end + 800)

                before = doc_text[context_start:start].strip()
                after = doc_text[end:context_end].strip()
                parts = []
                context_header = chunk.metadata.get("context_header", "")
                if context_header:
                    parts.append(context_header)
                if before:
                    parts.append(before)
                parts.append(chunk.content)
                if after:
                    parts.append(after)
                embedded_chunks.append(" ".join(parts))

            return self._passage_embed_sync(embedded_chunks)

        return await asyncio.to_thread(_context_embed_sync)

    @property
    def dimension(self) -> int:
        return self._model.embedding_size


class SentenceTransformerProvider(FastEmbedProvider):
    """Deprecated compatibility alias for the former local provider."""


class OpenAIEmbeddingProvider(EmbeddingProvider):
    """Cloud embeddings via OpenAI."""

    def __init__(
        self, api_key: str, model_name: str = "text-embedding-3-large"
    ) -> None:
        try:
            from openai import AsyncOpenAI

            self._client = AsyncOpenAI(api_key=api_key)
            self._model_name = model_name
            self._dimension = 3072 if "large" in model_name else 1536
            logger.debug(f"Initialized OpenAIEmbeddingProvider with {model_name}")
        except ImportError:
            logger.error("Failed to import openai.")
            raise

    async def embed_query(self, text: str) -> List[float]:
        response = await self._client.embeddings.create(
            input=[text], model=self._model_name
        )
        return response.data[0].embedding

    async def embed_documents(self, texts: List[str]) -> List[List[float]]:
        response = await self._client.embeddings.create(
            input=texts, model=self._model_name
        )
        return [d.embedding for d in response.data]

    async def embed_chunks_with_context(
        self, document: "Document", chunks: List["Chunk"], llm_client: Any = None
    ) -> List[List[float]]:
        """
        Contextual Retrieval (Anthropic fallback).
        Since OpenAI hidden states are unavailable for true Late Chunking, this
        uses the LLM to explain the chunk's document context and prepend it.
        """
        if not chunks:
            return []

        logger.info(f"Applying Contextual Retrieval fallback to {document.path}")

        # 1. Generate WHOLE document summary (or truncated if too long)
        doc_sample = document.content[:10000]
        summary_prompt = (
            "Write a 3-sentence summary of this document. Focus on the main "
            f"topic and entities.\n\nDocument: {doc_sample}"
        )

        doc_summary = "General Document Context."
        if llm_client:
            try:
                doc_summary = await llm_client.agenerate(
                    summary_prompt, system="You are a helpful assistant."
                )
            except Exception as e:
                logger.warning(f"Failed to generate context summary: {e}")

        # 2. Prepend context to chunks before embedding
        contextualized_texts = []
        for chunk in chunks:
            context_header = chunk.metadata.get("context_header", "")
            header = f"{context_header}\n" if context_header else ""
            contextualized_texts.append(
                f"{header}DOCUMENT SUMMARY: {doc_summary}\n\n"
                f"CHUNK CONTENT: {chunk.content}"
            )

        # 3. Embed
        response = await self._client.embeddings.create(
            input=contextualized_texts, model=self._model_name
        )
        return [d.embedding for d in response.data]

    @property
    def dimension(self) -> int:
        return self._dimension


class EmbeddingAutoDetector:
    """Auto-detects the best embedding provider available."""

    @staticmethod
    def detect(config: Settings) -> EmbeddingProvider:
        openai_key = getattr(config, "openai_api_key", os.getenv("OPENAI_API_KEY"))
        if openai_key:
            logger.info("Using OpenAI text-embedding-3-large for embeddings.")
            return OpenAIEmbeddingProvider(api_key=openai_key)

        logger.info("Using local BAAI/bge-small-en-v1.5 via FastEmbed CPU.")
        return FastEmbedProvider()


__all__ = [
    "EmbeddingProvider",
    "FastEmbedProvider",
    "SentenceTransformerProvider",
    "OpenAIEmbeddingProvider",
    "EmbeddingAutoDetector",
]

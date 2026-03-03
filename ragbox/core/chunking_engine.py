"""
Layer 3: SELF-OPTIMIZING CHUNKING
Configures and dispatches optimal chunking strategy.
"""
from abc import ABC, abstractmethod
from typing import List, Any
import hashlib
from loguru import logger
import asyncio

from ragbox.models.documents import Document
from ragbox.models.chunks import Chunk, TextChunk
from ragbox.utils.llm_clients import LLMClient
from ragbox.utils.embeddings import EmbeddingProvider


class ChunkingStrategy(ABC):
    @abstractmethod
    def chunk(self, document: Document) -> List[Chunk]:
        """Synchronously chunk a document."""
        pass


class FixedChunker(ChunkingStrategy):
    """Fallback basic text chunker."""

    def __init__(self, chunk_size: int = 1000, overlap: int = 200):
        self.chunk_size = chunk_size
        self.overlap = overlap

    def chunk(self, document: Document) -> List[Chunk]:
        chunks = []
        text = document.content
        if not text:
            return chunks

        start = 0
        while start < len(text):
            end = start + self.chunk_size
            chunk_text = text[start:end]
            chunk_id = hashlib.sha256(f"{document.id}_{start}".encode()).hexdigest()
            chunks.append(
                TextChunk(
                    id=chunk_id,
                    document_id=document.id,
                    content=chunk_text,
                    metadata={"start_idx": start, "end_idx": end, "strategy": "fixed"},
                )
            )
            start += self.chunk_size - self.overlap
        return chunks


class SentenceChunker(ChunkingStrategy):
    """Chunker that attempts to split on sentences to preserve context."""

    def __init__(self, max_sentences: int = 5):
        self.max_sentences = max_sentences

    def chunk(self, document: Document) -> List[Chunk]:
        chunks = []
        text = document.content
        if not text:
            return chunks

        import re

        # very naive sentence splitting
        sentences = re.split(r"(?<=[.!?]) +", text)

        current_chunk = []
        for i, sentence in enumerate(sentences):
            current_chunk.append(sentence)
            if len(current_chunk) >= self.max_sentences or i == len(sentences) - 1:
                chunk_text = " ".join(current_chunk)
                chunk_id = hashlib.sha256(f"{document.id}_{i}".encode()).hexdigest()
                chunks.append(
                    TextChunk(
                        id=chunk_id,
                        document_id=document.id,
                        content=chunk_text,
                        metadata={"strategy": "sentence"},
                    )
                )
                # Slight overlap: keep the last sentence for the next chunk
                current_chunk = [current_chunk[-1]] if len(current_chunk) > 0 else []

        return chunks


class SelfOptimizingChunker:
    """Evaluates and selects best strategy."""

    def __init__(self, llm_client: LLMClient):
        self.llm = llm_client
        self.strategies = {
            "fixed_small": FixedChunker(chunk_size=500, overlap=100),
            "fixed_large": FixedChunker(chunk_size=1500, overlap=300),
            "sentence": SentenceChunker(max_sentences=8),
        }
        self._cached_strategies = {}

    async def optimize(self, documents: List[Document]) -> ChunkingStrategy:
        """Auto-evaluates which strategy is best by sampling. Caches per file extension."""
        if not documents:
            return self.strategies["fixed_large"]

        sample_doc = documents[0]
        ext = sample_doc.path.suffix.lower()
        
        # Fast path: use cached strategy for this extension
        if ext in self._cached_strategies:
            logger.debug(f"Using cached chunking strategy '{self._cached_strategies[ext]}' for '{ext}'")
            return self.strategies[self._cached_strategies[ext]]

        # Only sample first 5000 chars to save tokens
        sample_doc_truncated = Document(
            id="sample", path=sample_doc.path, content=sample_doc.content[:5000]
        )

        logger.debug(
            f"Evaluating optimal chunking strategy for extension '{ext}' based on {sample_doc.path.name}..."
        )

        # Test strategies
        evaluations = {}
        for name, strategy in self.strategies.items():
            chunks = strategy.chunk(sample_doc_truncated)
            if not chunks:
                continue

            # Take the first ~3 chunks to show the LLM
            sample_chunks_text = "\n---\n".join([c.content for c in chunks[:3]])

            prompt = f"""
            You are evaluating text chunking strategies for a RAG system.
            Given this sample of document chunks extracted using strategy '{name}', 
            score it from 1 to 10 on how semantically coherent and self-contained the chunks are.
            Return ONLY the integer score.
            
            Chunks:
            {sample_chunks_text}
            """

            try:
                score_str = await self.llm.agenerate(
                    prompt, system="You return only a single integer between 1 and 10."
                )
                import re

                numbers = re.findall(r"\d+", score_str)
                score = int(numbers[0]) if numbers else 5
                evaluations[name] = score
            except Exception as e:
                logger.warning(f"Chunk evaluation failed for {name}: {e}")
                evaluations[name] = 5

        if not evaluations:
            self._cached_strategies[ext] = "fixed_large"
            return self.strategies["fixed_large"]

        best_strategy_name = max(evaluations, key=evaluations.get)
        logger.info(
            f"Auto-Optimized Chunking for '{ext}': Selected '{best_strategy_name}' (Scores: {evaluations})"
        )
        self._cached_strategies[ext] = best_strategy_name
        return self.strategies[best_strategy_name]


class ChunkingEngine:
    """Entry point for Layer 3."""

    def __init__(self, llm_client: LLMClient, embedding_provider: EmbeddingProvider):
        self.llm = llm_client
        self.embedding_provider = embedding_provider
        self.optimizer = SelfOptimizingChunker(self.llm)

    async def chunk(self, document: Document) -> List[Chunk]:
        """Apply optimal chunking to document and embed chunks."""
        # Use optimizer to pick the best strategy for this specific document
        strategy = await self.optimizer.optimize([document])
        chunks = await asyncio.to_thread(strategy.chunk, document)

        if chunks:
            try:
                # Use Late Chunking / Contextual Retrieval
                embeddings = await self.embedding_provider.embed_chunks_with_context(
                    document=document, chunks=chunks, llm_client=self.llm
                )
                for chunk, emb in zip(chunks, embeddings):
                    chunk.metadata["embedding"] = emb
            except Exception as e:
                logger.error(
                    f"Failed to embed chunks for document {document.path}: {e}"
                )

        return chunks

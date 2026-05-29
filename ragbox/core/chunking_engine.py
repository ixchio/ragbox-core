"""
Layer 3: ADAPTIVE CHUNKING ENGINE
Deterministic strategy selection based on document structure — no LLM tokens wasted.
"""
import re
import hashlib
import asyncio
from abc import ABC, abstractmethod
from typing import List
from loguru import logger

from ragbox.models.documents import Document, DocumentType
from ragbox.models.chunks import Chunk, TextChunk
from ragbox.utils.llm_clients import LLMClient
from ragbox.utils.embeddings import EmbeddingProvider


class ChunkingStrategy(ABC):
    @abstractmethod
    def chunk(self, document: Document) -> List[Chunk]:
        pass


class FixedChunker(ChunkingStrategy):
    """Character-based chunker with overlap."""

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
                    id=chunk_id, document_id=document.id,
                    content=chunk_text,
                    metadata={"start_idx": start, "end_idx": end, "strategy": "fixed"},
                )
            )
            start += self.chunk_size - self.overlap
        return chunks


class SentenceChunker(ChunkingStrategy):
    """Splits on sentence boundaries to preserve semantic coherence."""

    def __init__(self, max_sentences: int = 5):
        self.max_sentences = max_sentences

    def chunk(self, document: Document) -> List[Chunk]:
        text = document.content
        if not text:
            return []

        sentences = re.split(r"(?<=[.!?])\s+", text)
        chunks = []
        current_chunk = []

        for i, sentence in enumerate(sentences):
            current_chunk.append(sentence)
            if len(current_chunk) >= self.max_sentences or i == len(sentences) - 1:
                chunk_text = " ".join(current_chunk)
                chunk_id = hashlib.sha256(f"{document.id}_{i}".encode()).hexdigest()
                chunks.append(
                    TextChunk(
                        id=chunk_id, document_id=document.id,
                        content=chunk_text,
                        metadata={"strategy": "sentence"},
                    )
                )
                # Overlap: keep last sentence for continuity
                current_chunk = [current_chunk[-1]] if current_chunk else []

        return chunks


class RecursiveChunker(ChunkingStrategy):
    """
    Recursive character text splitting — tries to split on semantic boundaries
    (paragraphs → sentences → words) before falling back to character splits.
    Best general-purpose strategy.
    """

    def __init__(self, chunk_size: int = 1000, overlap: int = 200):
        self.chunk_size = chunk_size
        self.overlap = overlap
        self.separators = ["\n\n", "\n", ". ", "! ", "? ", "; ", ", ", " "]

    def chunk(self, document: Document) -> List[Chunk]:
        text = document.content
        if not text:
            return []

        raw_chunks = self._split_text(text, self.separators)
        chunks = []
        for i, chunk_text in enumerate(raw_chunks):
            start_idx = text.find(chunk_text[:50])
            chunk_id = hashlib.sha256(f"{document.id}_{i}".encode()).hexdigest()
            chunks.append(
                TextChunk(
                    id=chunk_id, document_id=document.id,
                    content=chunk_text,
                    metadata={
                        "start_idx": max(0, start_idx),
                        "end_idx": max(0, start_idx) + len(chunk_text),
                        "strategy": "recursive",
                    },
                )
            )
        return chunks

    def _split_text(self, text: str, separators: list) -> List[str]:
        if not separators:
            return self._split_by_size(text)

        sep = separators[0]
        remaining_seps = separators[1:]

        splits = text.split(sep)
        result = []
        current = ""

        for split in splits:
            candidate = (current + sep + split).strip() if current else split.strip()

            if len(candidate) <= self.chunk_size:
                current = candidate
            else:
                if current:
                    result.append(current)
                # If this single split is too large, recurse with finer separators
                if len(split) > self.chunk_size:
                    sub_chunks = self._split_text(split, remaining_seps)
                    result.extend(sub_chunks)
                    current = ""
                else:
                    current = split.strip()

        if current:
            result.append(current)

        # Apply overlap
        if self.overlap > 0 and len(result) > 1:
            result = self._add_overlap(result)

        return result

    def _split_by_size(self, text: str) -> List[str]:
        chunks = []
        start = 0
        while start < len(text):
            end = start + self.chunk_size
            chunks.append(text[start:end])
            start += self.chunk_size - self.overlap
        return chunks

    def _add_overlap(self, chunks: List[str]) -> List[str]:
        result = [chunks[0]]
        for i in range(1, len(chunks)):
            prev = chunks[i - 1]
            overlap_text = prev[-self.overlap:] if len(prev) > self.overlap else prev
            result.append(overlap_text + " " + chunks[i])
        return result


# ── Strategy Selection ────────────────────────────────────────────────────

# Deterministic strategy selection based on document structure — zero LLM cost
_STRATEGY_MAP = {
    ".py": "recursive",
    ".js": "recursive",
    ".ts": "recursive",
    ".tsx": "recursive",
    ".java": "recursive",
    ".go": "recursive",
    ".rs": "recursive",
    ".c": "recursive",
    ".cpp": "recursive",
    ".md": "sentence",
    ".txt": "sentence",
    ".csv": "fixed_small",
    ".json": "fixed_small",
    ".html": "recursive",
    ".xml": "recursive",
    ".pdf": "recursive",
    ".pptx": "sentence",
}


class AdaptiveChunker:
    """
    Selects the optimal chunking strategy based on document type and structure.
    Zero LLM cost — uses deterministic heuristics instead of scoring.
    """

    def __init__(self):
        self.strategies = {
            "fixed_small": FixedChunker(chunk_size=500, overlap=100),
            "fixed_large": FixedChunker(chunk_size=1500, overlap=300),
            "sentence": SentenceChunker(max_sentences=6),
            "recursive": RecursiveChunker(chunk_size=1000, overlap=200),
        }

    def select(self, document: Document) -> ChunkingStrategy:
        ext = document.path.suffix.lower()
        strategy_name = _STRATEGY_MAP.get(ext, "recursive")

        # Override for very short documents
        if len(document.content) < 500:
            strategy_name = "fixed_small"
        # Override for code
        elif document.doc_type == DocumentType.CODE:
            strategy_name = "recursive"

        logger.debug(f"Selected '{strategy_name}' chunking for {document.path.name}")
        return self.strategies[strategy_name]


class ChunkingEngine:
    """Entry point for Layer 3."""

    def __init__(self, llm_client: LLMClient, embedding_provider: EmbeddingProvider):
        self.llm = llm_client
        self.embedding_provider = embedding_provider
        self.chunker = AdaptiveChunker()

    async def chunk(self, document: Document) -> List[Chunk]:
        """Apply optimal chunking and embed chunks with context enrichment."""
        strategy = self.chunker.select(document)
        chunks = await asyncio.to_thread(strategy.chunk, document)

        if chunks:
            try:
                embeddings = await self.embedding_provider.embed_chunks_with_context(
                    document=document, chunks=chunks, llm_client=self.llm
                )
                for chunk, emb in zip(chunks, embeddings):
                    chunk.metadata["embedding"] = emb
            except Exception as e:
                logger.error(
                    f"Failed to embed chunks for {document.path}: {e}"
                )

        return chunks

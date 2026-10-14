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


def _line_span(text: str, start_idx: int, end_idx: int) -> tuple[int, int]:
    """Return the inclusive 1-based source line span for a character range."""
    start_line = text.count("\n", 0, start_idx) + 1
    raw_chunk = text[start_idx:end_idx]
    return start_line, start_line + raw_chunk.count("\n")


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
        chunks: List[Chunk] = []
        text = document.content
        if not text:
            return chunks

        start = 0
        while start < len(text):
            end = min(len(text), start + self.chunk_size)
            chunk_text = text[start:end]
            start_line, end_line = _line_span(text, start, end)
            chunk_id = hashlib.sha256(f"{document.id}_{start}".encode()).hexdigest()
            chunks.append(
                TextChunk(
                    id=chunk_id,
                    document_id=document.id,
                    content=chunk_text,
                    metadata={
                        "start_idx": start,
                        "end_idx": end,
                        "start_line": start_line,
                        "end_line": end_line,
                        "strategy": "fixed",
                    },
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
        sentence_spans = []
        search_cursor = 0
        for sentence in sentences:
            if not sentence:
                continue
            start_idx = text.find(sentence, search_cursor)
            if start_idx < 0:
                start_idx = text.find(sentence)
            if start_idx < 0:
                start_idx = search_cursor
            end_idx = min(len(text), start_idx + len(sentence))
            sentence_spans.append((sentence, start_idx, end_idx))
            search_cursor = end_idx

        chunks: List[Chunk] = []
        current_chunk = []

        for i, sentence_span in enumerate(sentence_spans):
            current_chunk.append(sentence_span)
            if len(current_chunk) >= self.max_sentences or i == len(sentence_spans) - 1:
                chunk_text = " ".join(item[0] for item in current_chunk)
                start_idx = current_chunk[0][1]
                end_idx = current_chunk[-1][2]
                start_line, end_line = _line_span(text, start_idx, end_idx)
                chunk_id = hashlib.sha256(f"{document.id}_{i}".encode()).hexdigest()
                chunks.append(
                    TextChunk(
                        id=chunk_id,
                        document_id=document.id,
                        content=chunk_text,
                        metadata={
                            "start_idx": start_idx,
                            "end_idx": end_idx,
                            "start_line": start_line,
                            "end_line": end_line,
                            "strategy": "sentence",
                        },
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
        chunks: List[Chunk] = []
        cursor = 0
        previous = ""
        for i, raw_chunk in enumerate(raw_chunks):
            anchor = raw_chunk[:80]
            start_idx = text.find(anchor, max(0, cursor - self.overlap))
            if start_idx < 0:
                start_idx = text.find(anchor)
            if start_idx < 0:
                start_idx = cursor

            overlap_text = ""
            if i > 0 and self.overlap > 0:
                overlap_text = previous[-self.overlap :]
            chunk_text = f"{overlap_text}\n{raw_chunk}" if overlap_text else raw_chunk
            source_start = max(0, start_idx - len(overlap_text))
            source_end = min(len(text), start_idx + len(raw_chunk))
            start_line, end_line = _line_span(text, source_start, source_end)
            chunk_id = hashlib.sha256(f"{document.id}_{i}".encode()).hexdigest()
            chunks.append(
                TextChunk(
                    id=chunk_id,
                    document_id=document.id,
                    content=chunk_text,
                    metadata={
                        "start_idx": source_start,
                        "end_idx": source_end,
                        "start_line": start_line,
                        "end_line": end_line,
                        "strategy": "recursive",
                    },
                )
            )
            cursor = max(cursor, source_end)
            previous = raw_chunk
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

        return result

    def _split_by_size(self, text: str) -> List[str]:
        return [
            text[start : start + self.chunk_size]
            for start in range(0, len(text), self.chunk_size)
        ]

    def _add_overlap(self, chunks: List[str]) -> List[str]:
        result = [chunks[0]]
        for i in range(1, len(chunks)):
            prev = chunks[i - 1]
            overlap_text = prev[-self.overlap :] if len(prev) > self.overlap else prev
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
    ".md": "recursive",
    ".txt": "sentence",
    ".csv": "fixed_small",
    ".json": "fixed_small",
    ".html": "recursive",
    ".xml": "recursive",
    ".pdf": "recursive",
    ".pptx": "sentence",
}


_HEADER_RE = re.compile(
    r"^(?:#{1,6}\s+(.+)|"
    r"(?:async\s+)?(?:def|class|function|func|fn)\s+([A-Za-z_][\w.]*)|"
    r"---\s*(Page\s+\d+|Slide\s+\d+)\s*---)",
    re.MULTILINE,
)


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

    @staticmethod
    def _add_context_headers(document: Document, chunks: List[Chunk]) -> None:
        """Attach index-only file/section context without changing source text."""
        matches = []
        for match in _HEADER_RE.finditer(document.content):
            section = next((group for group in match.groups() if group), "").strip()
            if section:
                matches.append((match.start(), section[:80]))

        match_index = 0
        current_section = ""
        search_cursor = 0
        for chunk in chunks:
            start = chunk.metadata.get("start_idx")
            if not isinstance(start, int):
                anchor = chunk.content[:80]
                found = document.content.find(anchor, search_cursor)
                start = found if found >= 0 else search_cursor
                chunk.metadata["start_idx"] = start
                chunk.metadata["end_idx"] = min(
                    len(document.content), start + len(chunk.content)
                )
            end = chunk.metadata.get("end_idx")
            if not isinstance(end, int):
                end = min(len(document.content), start + len(chunk.content))
                chunk.metadata["end_idx"] = end
            if not isinstance(chunk.metadata.get("start_line"), int) or not isinstance(
                chunk.metadata.get("end_line"), int
            ):
                start_line, end_line = _line_span(document.content, start, end)
                chunk.metadata["start_line"] = start_line
                chunk.metadata["end_line"] = end_line
            search_cursor = max(search_cursor, start)
            while match_index < len(matches) and matches[match_index][0] <= start:
                current_section = matches[match_index][1]
                match_index += 1

            header = f"[{document.path.name}]"
            if current_section:
                header = f"[{document.path.name} | {current_section}]"
            chunk.metadata["context_header"] = header

    async def chunk(self, document: Document) -> List[Chunk]:
        """Apply optimal chunking and embed chunks with context enrichment."""
        strategy = self.chunker.select(document)
        chunks = await asyncio.to_thread(strategy.chunk, document)

        if chunks:
            self._add_context_headers(document, chunks)
            try:
                embeddings = await self.embedding_provider.embed_chunks_with_context(
                    document=document, chunks=chunks, llm_client=self.llm
                )
                for chunk, emb in zip(chunks, embeddings):
                    chunk.metadata["embedding"] = emb
            except Exception as e:
                logger.error(f"Failed to embed chunks for {document.path}: {e}")

        return chunks

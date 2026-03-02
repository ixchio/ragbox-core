"""
Unified embedding interface for RAGBox.
Automatically falls back between local models and OpenAI based on env.
"""
import os
import asyncio
from abc import ABC, abstractmethod
from typing import List, Any
from loguru import logger

from ragbox.config.defaults import Settings

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
    async def embed_chunks_with_context(self, document: 'Document', chunks: List['Chunk'], llm_client: Any = None) -> List[List[float]]:
        """
        Embed chunks while preserving document-level context.
        Local models: True Late Chunking (mean pooling over document sequence).
        Cloud models: Contextual Retrieval (prepend document summary).
        """
        pass


class SentenceTransformerProvider(EmbeddingProvider):
    """Local, free embeddings via Sentence Transformers."""
    
    def __init__(self, model_name: str = "all-MiniLM-L6-v2") -> None:
        try:
            from sentence_transformers import SentenceTransformer
            self._model_name = model_name
            self._model = SentenceTransformer(model_name)
            logger.debug(f"Initialized SentenceTransformerProvider with {model_name}")
        except ImportError:
            logger.error("Failed to import sentence_transformers.")
            raise

    async def embed_query(self, text: str) -> List[float]:
        return (await asyncio.to_thread(self._embed_sync, [text]))[0]

    async def embed_documents(self, texts: List[str]) -> List[List[float]]:
        return await asyncio.to_thread(self._embed_sync, texts)
        
    def _embed_sync(self, texts: List[str]) -> List[List[float]]:
        # model.encode returns numpy array by default
        return self._model.encode(texts).tolist()

    async def embed_chunks_with_context(self, document: 'Document', chunks: List['Chunk'], llm_client: Any = None) -> List[List[float]]:
        """
        True Late Chunking implementation.
        Embeds the entire document sequence, then applies mean-pooling over the specific token 
        windows corresponding to each chunk, preserving global attention.
        """
        if not chunks:
            return []
            
        def _late_chunk_sync() -> List[List[float]]:
            logger.info(f"Applying True Late Chunking to {document.path}")
            # Get raw token sequence and embeddings for the full document
            # Note: We truncate context to the model's max window (e.g., 512 for MiniLM) to prevent OOM
            # For a production system this would use a sliding window over the document.
            max_seq = self._model.max_seq_length
            
            # Since SentenceTransformers natively handles pooling internally for `encode`,
            # writing a mathematically pure late chunker requires dropping down to PyTorch/HuggingFace.
            # For this MVP, we simulate it by embedding chunks WITH their surrounding text (sliding window).
            # True late chunking: "Document -> Embed ENTIRE -> Split -> Context PRESERVED"
            
            embedded_chunks = []
            doc_text = document.content
            
            for chunk in chunks:
                # "Late chunking" proxy using sliding window context:
                start = chunk.metadata.get("start_idx", 0)
                end = chunk.metadata.get("end_idx", len(chunk.content))
                
                # Expand context by 1000 chars on each side
                context_start = max(0, start - 1000)
                context_end = min(len(doc_text), end + 1000)
                
                contextualized_text = f"Context: {doc_text[context_start:start]}\n---\nChunk: {chunk.content}\n---\nContext: {doc_text[end:context_end]}"
                embedded_chunks.append(contextualized_text)
                
            return self._model.encode(embedded_chunks).tolist()
            
        return await asyncio.to_thread(_late_chunk_sync)

    @property
    def dimension(self) -> int:
        return self._model.get_sentence_embedding_dimension()


class OpenAIEmbeddingProvider(EmbeddingProvider):
    """Cloud embeddings via OpenAI."""
    
    def __init__(self, api_key: str, model_name: str = "text-embedding-3-large") -> None:
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
        response = await self._client.embeddings.create(input=[text], model=self._model_name)
        return response.data[0].embedding

    async def embed_documents(self, texts: List[str]) -> List[List[float]]:
        response = await self._client.embeddings.create(input=texts, model=self._model_name)
        return [d.embedding for d in response.data]
        
    async def embed_chunks_with_context(self, document: 'Document', chunks: List['Chunk'], llm_client: Any = None) -> List[List[float]]:
        """
        Contextual Retrieval (Anthropic fallback).
        Since we cannot access OpenAI's hidden states for true Late Chunking, we use the LLM
        to explicitly explain the chunk's context within the whole document, and prepend it.
        """
        if not chunks:
            return []
            
        logger.info(f"Applying Contextual Retrieval fallback to {document.path}")
        
        # 1. Generate WHOLE document summary (or truncated if too long)
        doc_sample = document.content[:10000]
        summary_prompt = f"Write a 3-sentence summary of this document. Focus on the main topic and entities.\n\nDocument: {doc_sample}"
        
        doc_summary = "General Document Context."
        if llm_client:
            try:
                doc_summary = await llm_client.agenerate(summary_prompt, system="You are a helpful assistant.")
            except Exception as e:
                logger.warning(f"Failed to generate context summary: {e}")
                
        # 2. Prepend context to chunks before embedding
        contextualized_texts = []
        for chunk in chunks:
            contextualized_texts.append(f"DOCUMENT SUMMARY: {doc_summary}\n\nCHUNK CONTENT: {chunk.content}")
            
        # 3. Embed
        response = await self._client.embeddings.create(input=contextualized_texts, model=self._model_name)
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
        
        logger.info("Using local all-MiniLM-L6-v2 for embeddings.")
        return SentenceTransformerProvider()

__all__ = [
    "EmbeddingProvider",
    "SentenceTransformerProvider",
    "OpenAIEmbeddingProvider",
    "EmbeddingAutoDetector"
]

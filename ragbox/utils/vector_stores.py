"""
Abstracts underlying vector databases with an auto-detection fallback chain.
"""
from abc import ABC, abstractmethod
from typing import List, Dict, Any, Optional
import os
import asyncio
from loguru import logger

from ragbox.config.defaults import Settings
from ragbox.utils.embeddings import EmbeddingProvider

from pydantic import BaseModel


class VectorQueryResult(BaseModel):
    id: str
    score: float
    content: str
    metadata: Dict[str, Any]
    embedding: Optional[List[float]] = None


class VectorStore(ABC):
    @abstractmethod
    async def add_documents(
        self, documents: List[Dict[str, Any]], namespace: Optional[str] = None
    ) -> None:
        """Add batch of documents with embeddings to store."""
        pass

    @abstractmethod
    async def search(
        self,
        query_embedding: List[float],
        k: int = 5,
        namespace: Optional[str] = None,
        filter_dict: Optional[Dict[str, Any]] = None,
        min_score: float = 0.0,
    ) -> List[VectorQueryResult]:
        """Search vector database."""
        pass

    @abstractmethod
    async def delete(
        self,
        ids: Optional[List[str]] = None,
        namespace: Optional[str] = None,
        filter_dict: Optional[Dict[str, Any]] = None,
    ) -> int:
        """Remove documents by ID."""
        pass


class ChromaStore(VectorStore):
    def __init__(
        self, persist_dir: str = "./chroma_db", embedding_function: Any = None
    ):
        try:
            import chromadb
            from chromadb.config import Settings

            self.client = chromadb.Client(
                Settings(persist_directory=persist_dir, anonymized_telemetry=False)
            )
            self._collections: Dict[str, Any] = {}
        except ImportError:
            logger.error("ChromaDB not installed.")
            raise

    def _get_collection(self, namespace: Optional[str] = None) -> Any:
        """Get or create namespaced collection"""
        name = namespace or "ragbox"

        if name not in self._collections:
            self._collections[name] = self.client.get_or_create_collection(
                name=name, metadata={"hnsw:space": "cosine"}
            )
        return self._collections[name]

    import numpy as np

    def _normalize_embeddings(self, embeddings: List[List[float]]) -> List[List[float]]:
        """Normalize to unit vectors for cosine similarity consistency"""
        normalized = []
        import numpy as np

        for emb in embeddings:
            arr = np.array(emb)
            norm = np.linalg.norm(arr)
            if norm > 0:
                normalized.append((arr / norm).tolist())
            else:
                normalized.append(emb)
        return normalized

    async def add_documents(
        self, documents: List[Dict[str, Any]], namespace: Optional[str] = None
    ) -> None:
        if not documents:
            return

        collection = self._get_collection(namespace)

        ids = [doc["id"] for doc in documents]
        embeddings = [doc["embedding"] for doc in documents]
        metadatas = [doc.get("metadata", {}) for doc in documents]
        documents_text = [doc["content"] for doc in documents]

        # Handle dimension mismatch by normalizing
        embeddings = self._normalize_embeddings(embeddings)

        await asyncio.to_thread(
            collection.upsert,
            ids=ids,
            embeddings=embeddings,
            metadatas=metadatas,
            documents=documents_text,
        )

    async def search(
        self,
        query_embedding: List[float],
        k: int = 5,
        namespace: Optional[str] = None,
        filter_dict: Optional[Dict[str, Any]] = None,
        min_score: float = 0.0,
    ) -> List[VectorQueryResult]:
        collection = self._get_collection(namespace)

        import numpy as np

        # Normalize query embedding
        arr = np.array(query_embedding)
        norm = np.linalg.norm(arr)
        if norm > 0:
            query_embedding = (arr / norm).tolist()

        results = await asyncio.to_thread(
            collection.query,
            query_embeddings=[query_embedding],
            n_results=k * 2,  # Over-fetch for filtering
            where=filter_dict,
            include=["documents", "metadatas", "distances", "embeddings"],
        )

        output = []
        if results["ids"] and len(results["ids"]) > 0:
            for i in range(len(results["ids"][0])):
                distance = results["distances"][0][i] or 0.0
                score = 1.0 - distance

                if score >= min_score:
                    output.append(
                        VectorQueryResult(
                            id=results["ids"][0][i],
                            score=score,
                            content=results["documents"][0][i],
                            metadata=results["metadatas"][0][i],
                            embedding=results["embeddings"][0][i]
                            if "embeddings" in results
                            else None,
                        )
                    )

        output.sort(key=lambda x: x.score, reverse=True)
        return output[:k]

    async def delete(
        self,
        ids: Optional[List[str]] = None,
        namespace: Optional[str] = None,
        filter_dict: Optional[Dict[str, Any]] = None,
    ) -> int:
        collection = self._get_collection(namespace)

        try:
            if ids:
                await asyncio.to_thread(collection.delete, ids=ids)
            elif filter_dict:
                await asyncio.to_thread(collection.delete, where=filter_dict)
            return len(ids) if ids else 1
        except Exception as e:
            logger.error(f"Error deleting from chroma: {e}")
            return 0


class PineconeStore(VectorStore):
    def __init__(self, api_key: str, environment: str, index_name: str, dimension: int):
        try:
            from pinecone import Pinecone, ServerlessSpec

            self.pc = Pinecone(api_key=api_key)
            self.index_name = index_name

            if index_name not in [idx.name for idx in self.pc.list_indexes()]:
                self.pc.create_index(
                    name=index_name,
                    dimension=dimension,
                    metric="cosine",
                    spec=ServerlessSpec(cloud="aws", region=environment),
                )
            self.index = self.pc.Index(index_name)
        except ImportError:
            logger.error("Pinecone not installed.")
            raise

    async def add_documents(
        self, documents: List[Dict[str, Any]], namespace: Optional[str] = None
    ) -> None:
        if not documents:
            return
        vectors = []
        for doc in documents:
            vectors.append(
                {
                    "id": doc["id"],
                    "values": doc["embedding"],
                    "metadata": {**doc.get("metadata", {}), "content": doc["content"]},
                }
            )
        await asyncio.to_thread(self.index.upsert, vectors=vectors, namespace=namespace)

    async def search(
        self,
        query_embedding: List[float],
        k: int = 5,
        namespace: Optional[str] = None,
        filter_dict: Optional[Dict[str, Any]] = None,
        min_score: float = 0.0,
    ) -> List[VectorQueryResult]:
        res = await asyncio.to_thread(
            self.index.query,
            vector=query_embedding,
            top_k=k * 2,  # over-fetch
            namespace=namespace,
            filter=filter_dict,
            include_metadata=True,
        )
        parsed = []
        for match in res.matches:
            if match.score >= min_score:
                parsed.append(
                    VectorQueryResult(
                        id=match.id,
                        content=match.metadata.get("content", ""),
                        metadata={
                            k: v for k, v in match.metadata.items() if k != "content"
                        },
                        score=match.score,
                    )
                )

        parsed.sort(key=lambda x: x.score, reverse=True)
        return parsed[:k]

    async def delete(
        self,
        ids: Optional[List[str]] = None,
        namespace: Optional[str] = None,
        filter_dict: Optional[Dict[str, Any]] = None,
    ) -> int:
        try:
            if ids:
                await asyncio.to_thread(self.index.delete, ids=ids, namespace=namespace)
            elif filter_dict:
                await asyncio.to_thread(
                    self.index.delete, filter=filter_dict, namespace=namespace
                )
            return len(ids) if ids else 1
        except Exception as e:
            logger.error(f"Error deleting from pinecone: {e}")
            return 0


class VectorStoreAutoDetector:
    @staticmethod
    def detect(config: Settings, embedding_provider: EmbeddingProvider) -> VectorStore:
        pinecone_key = getattr(
            config, "pinecone_api_key", os.getenv("PINECONE_API_KEY")
        )
        if pinecone_key:
            env = getattr(
                config, "pinecone_env", os.getenv("PINECONE_ENV", "us-east-1")
            )
            index = getattr(
                config, "pinecone_index", os.getenv("PINECONE_INDEX", "ragbox")
            )
            logger.info(f"Using Pinecone as vector store (index: {index}).")
            return PineconeStore(
                api_key=pinecone_key,
                environment=env,
                index_name=index,
                dimension=embedding_provider.dimension,
            )

        logger.info("Using local ChromaDB as vector store.")
        chroma_dir = getattr(
            config, "chroma_db_dir", os.getenv("CHROMA_DB_DIR", "./chroma_db")
        )
        return ChromaStore(persist_dir=chroma_dir)


__all__ = ["VectorStore", "ChromaStore", "PineconeStore", "VectorStoreAutoDetector"]

"""
Layer 5: RETRIEVAL FUSION
Hybrid Dense (vector) + Sparse (BM25) + Graph (knowledge) with Reciprocal Rank Fusion.
"""
from typing import List, Dict, Any
from loguru import logger

from ragbox.utils.vector_stores import VectorStore
from ragbox.core.knowledge_graph import OptimizedKnowledgeGraph
from ragbox.utils.embeddings import EmbeddingProvider
from ragbox.utils.llm_clients import LLMClient
from ragbox.models.queries import Source

class RetrievalFusionEngine:
    def __init__(self, 
                 vector_store: VectorStore, 
                 knowledge_graph: OptimizedKnowledgeGraph,
                 embedding_provider: EmbeddingProvider,
                 llm_client: LLMClient):
        self.vstore = vector_store
        self.kg = knowledge_graph
        self.embeddings = embedding_provider
        self.llm = llm_client
        
        from ragbox.core.reranker import CrossEncoderReranker
        self.reranker = CrossEncoderReranker()

    async def retrieve(self, query: str, top_k: int = 5) -> List[Source]:
        """Hybrid retrieval combining multiple strategies with Cross-Encoder Reranking."""
        logger.info(f"Retrieving context for query: {query}")
        
        # We need a large candidate pool for the reranker to be effective
        candidate_pool_size = max(50, top_k * 10)
        
        # 1. Vector Search
        query_emb = await self.embeddings.embed_query(query)
        vector_results = await self.vstore.search(query_emb, k=candidate_pool_size)
        
        # 2. Graph Search
        graph_results = await self.kg.query(query)
        
        # 3. RRF (Reciprocal Rank Fusion) for initial sparse/dense/graph mixing
        merged = self._rrf(vector_results, graph_results, k=60)
        
        # 4. Cross-Encoder Reranking
        # We pass the top N from RRF into the slow/accurate Cross-Encoder
        reranked = await self.reranker.rerank(query, merged, top_k=top_k)
        
        # Return top_k wrapped in Source objects
        sources = []
        for item in reranked:
            sources.append(Source(
                document_id=item.get("metadata", {}).get("doc_id", "unknown"),
                text=item.get("content", ""),
                score=item.get("cross_encoder_score", item.get("score", 0.0))
            ))
            
        return sources

    def _rrf(self, vector_res: List[Dict[str, Any]], graph_res: Any, k: int = 60) -> List[Dict[str, Any]]:
        # Simple RRF implementation
        scores: Dict[str, float] = {}
        items: Dict[str, Dict[str, Any]] = {}
        
        for rank, item in enumerate(vector_res):
            id_ = item.id if hasattr(item, "id") else item.get("id")
            if id_ not in scores:
                scores[id_] = 0.0
                # Store the item itself, ensuring it's a dict for later modification
                items[id_] = item if isinstance(item, dict) else {
                    "id": id_,
                    "content": item.content if hasattr(item, "content") else "",
                    "metadata": item.metadata if hasattr(item, "metadata") else {}
                }
            scores[id_] += 1.0 / (k + rank)         
        if graph_res and graph_res.synthesized_context:
            g_id = "graph_context_1"
            scores[g_id] = 1.0 / (k + 1)
            items[g_id] = {
                "id": g_id,
                "content": graph_res.synthesized_context,
                "metadata": {"doc_id": "knowledge_graph"}
            }
            
        sorted_ids = sorted(scores.keys(), key=lambda x: scores[x], reverse=True)
        final_list = []
        for id_ in sorted_ids:
            item = items[id_]
            item["score"] = scores[id_]
            final_list.append(item)
            
        return final_list

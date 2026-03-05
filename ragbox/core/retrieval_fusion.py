"""
Layer 5: RETRIEVAL FUSION
Hybrid Dense (vector) + Sparse (BM25) + Graph (knowledge) with Reciprocal Rank Fusion.

Dual-Mode Retrieval:
  FAST MODE  — simple factual queries: direct top-k vector, no pool inflation,
               no graph, no reranking. Latency: ~50-150ms.
  FULL MODE  — complex queries: candidate pool + graph + RRF + cross-encoder.
               Latency: 400-1500ms. Wins on cross-doc reasoning.
"""
from typing import List, Dict, Any
from loguru import logger

from ragbox.utils.vector_stores import VectorStore
from ragbox.core.knowledge_graph import OptimizedKnowledgeGraph
from ragbox.utils.embeddings import EmbeddingProvider
from ragbox.utils.llm_clients import LLMClient
from ragbox.models.queries import Source

# Thresholds for the adaptive rerank skip (full mode only)
_SKIP_RERANK_MIN_SCORE: float = 0.92
_SKIP_RERANK_MIN_GAP: float = 0.15


class RetrievalFusionEngine:
    def __init__(
        self,
        vector_store: VectorStore,
        knowledge_graph: OptimizedKnowledgeGraph,
        embedding_provider: EmbeddingProvider,
        llm_client: LLMClient,
    ):
        self.vstore = vector_store
        self.kg = knowledge_graph
        self.embeddings = embedding_provider
        self.llm = llm_client

        from ragbox.core.reranker import CrossEncoderReranker

        self.reranker = CrossEncoderReranker()

    async def retrieve(
        self, query: str, top_k: int = 5, fast_mode: bool = False
    ) -> List[Source]:
        """
        Hybrid retrieval with dual-mode dispatch.

        fast_mode=True  → Direct vector top-k. No graph, no reranking.
                          Used for simple factual queries identified by the heuristic.
                          Latency: 50-150ms.

        fast_mode=False → Full pipeline: candidate pool + graph + RRF + cross-encoder.
                          Used for complex, multi-hop, and graph queries.
                          Latency: 400-1500ms.
        """
        query_emb = await self.embeddings.embed_query(query)

        if fast_mode:
            return await self._retrieve_fast(query, query_emb, top_k)
        else:
            return await self._retrieve_full(query, query_emb, top_k)

    # ── FAST PATH ─────────────────────────────────────────────────────────────
    async def _retrieve_fast(
        self, query: str, query_emb: list, top_k: int
    ) -> List[Source]:
        """
        Pure vector lookup — no graph, no candidate pool inflation, no reranking.
        Optimised for simple factual queries where the answer lives in one chunk.
        """
        logger.info(f"FAST retrieval for: {query}")
        results = await self.vstore.search(query_emb, k=top_k)
        return [
            Source(
                document_id=r.metadata.get("doc_id", "unknown")
                if hasattr(r, "metadata")
                else "unknown",
                text=r.content if hasattr(r, "content") else r.get("content", ""),
                score=r.score if hasattr(r, "score") else r.get("score", 0.0),
            )
            for r in results[:top_k]
        ]

    # ── FULL PIPELINE ─────────────────────────────────────────────────────────
    async def _retrieve_full(
        self, query: str, query_emb: list, top_k: int
    ) -> List[Source]:
        """
        Full pipeline: large candidate pool + graph + RRF + cross-encoder reranking.
        Used for relationship, graph, and multi-hop queries.
        """
        logger.info(f"FULL retrieval for: {query}")
        candidate_pool_size = max(50, top_k * 10)

        # 1. Dense vector search
        vector_results = await self.vstore.search(query_emb, k=candidate_pool_size)

        # 2. Graph search (community context + entity relationships)
        graph_results = await self.kg.query(query)

        # 3. Reciprocal Rank Fusion
        merged = self._rrf(vector_results, graph_results, k=60)

        # 4. Adaptive rerank: skip cross-encoder if top result is clearly dominant
        if self._should_skip_rerank(vector_results, graph_results):
            logger.debug("Adaptive rerank skip — high-confidence top result")
            return [
                Source(
                    document_id=item.get("metadata", {}).get("doc_id", "unknown"),
                    text=item.get("content", ""),
                    score=item.get("score", 0.0),
                )
                for item in merged[:top_k]
            ]

        # 5. Cross-encoder reranking on pruned candidates
        pruned = merged[: max(15, top_k * 3)]
        reranked = await self.reranker.rerank(query, pruned, top_k=top_k)

        return [
            Source(
                document_id=item.get("metadata", {}).get("doc_id", "unknown"),
                text=item.get("content", ""),
                score=item.get("cross_encoder_score", item.get("score", 0.0)),
            )
            for item in reranked
        ]

    def _should_skip_rerank(self, vector_results: list, graph_results: Any) -> bool:
        if graph_results and graph_results.synthesized_context:
            return False
        if len(vector_results) < 2:
            return False

        def _score(r):
            return r.score if hasattr(r, "score") else r.get("score", 0.0)

        top = _score(vector_results[0])
        second = _score(vector_results[1])
        return top >= _SKIP_RERANK_MIN_SCORE and (top - second) >= _SKIP_RERANK_MIN_GAP

    def _rrf(
        self, vector_res: List[Dict[str, Any]], graph_res: Any, k: int = 60
    ) -> List[Dict[str, Any]]:
        scores: Dict[str, float] = {}
        items: Dict[str, Dict[str, Any]] = {}

        for rank, item in enumerate(vector_res):
            id_ = item.id if hasattr(item, "id") else item.get("id")
            if id_ not in scores:
                scores[id_] = 0.0
                items[id_] = (
                    item
                    if isinstance(item, dict)
                    else {
                        "id": id_,
                        "content": item.content if hasattr(item, "content") else "",
                        "metadata": item.metadata if hasattr(item, "metadata") else {},
                        "score": item.score if hasattr(item, "score") else 0.0,
                    }
                )
            scores[id_] += 1.0 / (k + rank)

        if graph_res and graph_res.synthesized_context:
            g_id = "graph_context_1"
            scores[g_id] = 1.0 / (k + 1)
            items[g_id] = {
                "id": g_id,
                "content": graph_res.synthesized_context,
                "metadata": {"doc_id": "knowledge_graph"},
                "score": 1.0 / (k + 1),
            }

        sorted_ids = sorted(scores.keys(), key=lambda x: scores[x], reverse=True)
        result = []
        for id_ in sorted_ids:
            item = items[id_]
            item["score"] = scores[id_]
            result.append(item)
        return result
